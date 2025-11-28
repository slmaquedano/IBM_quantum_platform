using LinearAlgebra
using Convex
using DelimitedFiles
using MosekTools
# using SCS
using Base.Threads
using HDF5

function read_hdf5_data(filename; rep, x, a)
    h5open(filename, "r") do file
        # Acessar a repetição
        rep_str = "repetition_$(rep-1)"
        haskey(file, rep_str) || error("Repetição $rep não encontrada")
        rep_group = file[rep_str]
        
        # Acessar o grupo x
        x_str = "x_$(x-1)"
        haskey(rep_group, x_str) || error("Medida x=$x não encontrada")
        x_group = rep_group[x_str]
        
        # Acessar os datasets
        a_real_str = "a_$(a-1)_elements_real"
        a_imag_str = "a_$(a-1)_elements_imag"
        haskey(x_group, a_real_str) || error("Dataset real não encontrado")
        haskey(x_group, a_imag_str) || error("Dataset imag não encontrado")
        
        # Ler os elementos
        reals = read(x_group[a_real_str])
        imags = read(x_group[a_imag_str])
        elements = complex.(reals, imags)
        
        # Reconstruir a matriz na ordem CORRETA
        matrix = [
            elements[1] elements[2];
            elements[3] elements[4]
        ]
        
        # Ler os ângulos
        angles = read_attribute(rep_group, "angles")
        
        return matrix, angles
    end
end

function KP(A, B)
    kron(A, B)
end

function IM(n)
   Matrix(I, n, n)
end

function partial_trace(rho, dim_A, dim_B, which_subspace)
    dim = dim_A * dim_B
    rho_reshaped = reshape(rho, dim_A, dim_B, dim_A, dim_B)
    rho_traced = zeros(Complex{Float64}, dim_B, dim_B)    
    if      which_subspace == 1 #tr_A() = rho_B
        for j in 1:dim_B
            rho_traced .+= rho_reshaped[:, j, :, j]
        end
    elseif    which_subspace == 2 #tr_B() = rho_A
        for i in 1:dim_A
            rho_traced .+= rho_reshaped[i, :, i, :]
        end
    else   
        println("which_subspace isen't 1 or 2")
    end

    return rho_traced
end

function prob_determ(oa, ma) #return a 3D array containing the deterministic probabiliy, prob_matrix[oa, ma, lamb].
    total_lamb = oa^ma 
    prob_matrix = zeros(Int8, oa, ma, total_lamb)
    for lamb in 1:total_lamb
        base_func = digits(lamb-1, base=oa)
        while length(base_func) < ma
            push!(base_func,0)
        end
        for x in 1:ma
            for a in 1:oa
                if base_func[x] == a-1
                prob_matrix[a, x, lamb] = 1 
                end
            end
        end
    end
    prob_matrix
end

# function M(dim, mub, out) 
#     #dim=2
#     if dim == 2 && mub == 1 && out == 1
#         1/sqrt(2) * [1; 1]
#     elseif dim == 2 && mub == 1 && out == 2
#         1/sqrt(2) * [1; -1]
#     elseif dim == 2 && mub == 2 && out == 1
#         1/sqrt(2) * [1; im]
#     elseif dim == 2 && mub == 2 && out == 2
#         1/sqrt(2) * [1; -im]
#     elseif dim == 2 && mub == 3 && out == 1
#         [1; 0]
#     elseif dim == 2 && mub == 3 && out == 2
#         [0; 1]
#     end
# end


function rho_2qbits(a1,a2,a3,b1,b2,b3,c1,c2,c3)
    rho = (1/4)*[
        1 + a3 + b3 + c3    b1 - b2*im          a1 - a2*im           c1 - c2;
        b1 + b2*im          1 + a3 - b3 - c3    c1 + c2              a1 - a2*im;
        a1 + a2*im          c1 + c2             1 - a3 + b3 - c3     b1 - b2*im;
        c1 - c2             a1 + a2*im          b1 + b2*im           1 - a3 - b3 + c3]
    return rho
end

#NEW FUNCTION
function prob_determ_classical_bit(oa,ma,n)
    if n == 1
        if  oa == 2 && ma == 2
            # Cada coluna tem 4 possibilidades (onde o 1 estará)
            combinations = []   
            choices = 1:oa*n

            # Gerar todas as combinações (3 colunas, cada coluna com uma escolha de 1 entre 4 linhas)
            for lines_col1 in choices, lines_col2 in choices
                # for lines_col1 in choices, lines_col2 in choices, lines_col3 in choices

                # Inicializa uma matriz 4x3 cheia de zeros
                matrix = zeros(Int, oa*n, ma)
                
                # Preenche a matriz com 1s de acordo com as combinações
                matrix[lines_col1, 1] = 1
                matrix[lines_col2, 2] = 1
                # Adiciona a combinação gerada na lista
                push!(combinations, matrix)
            end
            # println(combinations) #ok!

            total_lamb = (oa*n)^ma 
            # prob_matrix = zeros(Int8, oa*n, ma, total_lamb)
            prob_matrix = zeros(Int8, oa, ma, 1, total_lamb)


            for lamb in 1:total_lamb
                for x in 1:ma
                    # for am in 1:oa*n
                        # prob_matrix[am, x, lamb] = combinations[lamb][am,x]            
                    # end

                    # prob_matrix[a, x, m, lamb] = combinations[lamb][am,x]
                    # prob_matrix[a, x, m, lamb] = combinations[lamb][linha,coluna]
                    for a in 1:oa
                        prob_matrix[a, x, 1, lamb] = combinations[lamb][a,x]
                        # prob_matrix[2, x, 1, lamb] = combinations[lamb][2,x]
                        # println("prob_matrix[",a,", ",x,", ",1,", ",lamb,"]= ", prob_matrix[a, x, 1, lamb])#ok!
                    end
                end
            end
        elseif  oa == 2 && ma == 3
            # Cada coluna tem 4 possibilidades (onde o 1 estará)
            combinations = []   
            choices = 1:oa*n

            # Gerar todas as combinações (3 colunas, cada coluna com uma escolha de 1 entre 4 linhas)
            for lines_col1 in choices, lines_col2 in choices, lines_col3 in choices
                # for lines_col1 in choices, lines_col2 in choices, lines_col3 in choices

                # Inicializa uma matriz 4x3 cheia de zeros
                matrix = zeros(Int, oa*n, ma)
                
                # Preenche a matriz com 1s de acordo com as combinações
                matrix[lines_col1, 1] = 1
                matrix[lines_col2, 2] = 1
                matrix[lines_col3, 3] = 1
                # Adiciona a combinação gerada na lista
                push!(combinations, matrix)
            end
            # println(combinations) #ok!

            total_lamb = (oa*n)^ma 
            # prob_matrix = zeros(Int8, oa*n, ma, total_lamb)
            prob_matrix = zeros(Int8, oa, ma, 1, total_lamb)


            for lamb in 1:total_lamb
                for x in 1:ma
                    # for am in 1:oa*n
                        # prob_matrix[am, x, lamb] = combinations[lamb][am,x]            
                    # end

                    # prob_matrix[a, x, m, lamb] = combinations[lamb][am,x]
                    # prob_matrix[a, x, m, lamb] = combinations[lamb][linha,coluna]
                    for a in 1:oa
                        prob_matrix[a, x, 1, lamb] = combinations[lamb][a,x]
                        # prob_matrix[2, x, 1, lamb] = combinations[lamb][2,x]
                        # println("prob_matrix[",a,", ",x,", ",1,", ",lamb,"]= ", prob_matrix[a, x, 1, lamb])#ok!
                    end
                end
            end

        elseif  oa == 2 && ma == 4
            # Cada coluna tem 4 possibilidades (onde o 1 estará)
            combinations = []   
            choices = 1:oa*n

            # Gerar todas as combinações (3 colunas, cada coluna com uma escolha de 1 entre 4 linhas)
            for lines_col1 in choices, lines_col2 in choices, lines_col3 in choices, lines_col4 in choices
                # for lines_col1 in choices, lines_col2 in choices, lines_col3 in choices

                # Inicializa uma matriz 4x3 cheia de zeros
                matrix = zeros(Int, oa*n, ma)
                
                # Preenche a matriz com 1s de acordo com as combinações
                matrix[lines_col1, 1] = 1
                matrix[lines_col2, 2] = 1
                matrix[lines_col3, 3] = 1
                matrix[lines_col4, 4] = 1
                # Adiciona a combinação gerada na lista
                push!(combinations, matrix)
            end
            # println(combinations) #ok!

            total_lamb = (oa*n)^ma 
            # prob_matrix = zeros(Int8, oa*n, ma, total_lamb)
            prob_matrix = zeros(Int8, oa, ma, 1, total_lamb)


            for lamb in 1:total_lamb
                for x in 1:ma
                    # for am in 1:oa*n
                        # prob_matrix[am, x, lamb] = combinations[lamb][am,x]            
                    # end

                    # prob_matrix[a, x, m, lamb] = combinations[lamb][am,x]
                    # prob_matrix[a, x, m, lamb] = combinations[lamb][linha,coluna]
                    for a in 1:oa
                        prob_matrix[a, x, 1, lamb] = combinations[lamb][a,x]
                        # prob_matrix[2, x, 1, lamb] = combinations[lamb][2,x]
                        # println("prob_matrix[",a,", ",x,", ",1,", ",lamb,"]= ", prob_matrix[a, x, 1, lamb])#ok!
                    end
                end
            end
        elseif  oa == 2 && ma == 6
            # Cada coluna tem 4 possibilidades (onde o 1 estará)
            combinations = []   
            choices = 1:oa*n

            # Gerar todas as combinações (3 colunas, cada coluna com uma escolha de 1 entre 4 linhas)
            for lines_col1 in choices, lines_col2 in choices, lines_col3 in choices, lines_col4 in choices, lines_col5 in choices, lines_col6 in choices
                # for lines_col1 in choices, lines_col2 in choices, lines_col3 in choices

                # Inicializa uma matriz 4x3 cheia de zeros
                matrix = zeros(Int, oa*n, ma)
                
                # Preenche a matriz com 1s de acordo com as combinações
                matrix[lines_col1, 1] = 1
                matrix[lines_col2, 2] = 1
                matrix[lines_col3, 3] = 1
                matrix[lines_col4, 4] = 1
                matrix[lines_col5, 5] = 1
                matrix[lines_col6, 6] = 1
                # Adiciona a combinação gerada na lista
                push!(combinations, matrix)
            end
            # println(combinations) #ok!

            total_lamb = (oa*n)^ma 
            # prob_matrix = zeros(Int8, oa*n, ma, total_lamb)
            prob_matrix = zeros(Int8, oa, ma, 1, total_lamb)


            for lamb in 1:total_lamb
                for x in 1:ma
                    # for am in 1:oa*n
                        # prob_matrix[am, x, lamb] = combinations[lamb][am,x]            
                    # end

                    # prob_matrix[a, x, m, lamb] = combinations[lamb][am,x]
                    # prob_matrix[a, x, m, lamb] = combinations[lamb][linha,coluna]
                    for a in 1:oa
                        prob_matrix[a, x, 1, lamb] = combinations[lamb][a,x]
                        # prob_matrix[2, x, 1, lamb] = combinations[lamb][2,x]
                        # println("prob_matrix[",a,", ",x,", ",1,", ",lamb,"]= ", prob_matrix[a, x, 1, lamb])#ok!
                    end
                end
            end
        end

    elseif n == 2
        # Vamos criar uma lista para armazenar as 1024 combinações
        combinations = []   
        # Cada coluna tem 4 possibilidades (onde o 1 estará)
        choices = 1:oa*n
        if  oa == 2 && ma == 2
            for lines_col1 in choices, lines_col2 in choices
                matrix = zeros(Int, oa*n, ma)
                # Preenche a matriz com 1s de acordo com as combinações
                matrix[lines_col1, 1] = 1
                matrix[lines_col2, 2] = 1
                # Adiciona a combinação gerada na lista
                push!(combinations, matrix)
            end
            # println(combinations) #ok!
            total_lamb = (oa*n)^ma 
            # prob_matrix = zeros(Int8, oa*n, ma, total_lamb)
            prob_matrix = zeros(Int8, oa, ma, n, total_lamb)
            for lamb in 1:total_lamb
                for x in 1:ma
                    i = 1
                    for c in 1:n
                    for a in 1:oa
                            # prob_matrix[a, x, m, lamb] = combinations[lamb][am,x]
                            prob_matrix[a, x, c, lamb] = combinations[lamb][i,x]
                            # println("prob_matrix[",a,", ",x,", ",c,", ",lamb,"]= ", prob_matrix[a, x, c, lamb])#ok!
                            i += 1
                        end
                    end
                    
                end
            end
        elseif  oa == 2 && ma == 3
            for lines_col1 in choices, lines_col2 in choices, lines_col3 in choices
                matrix = zeros(Int, oa*n, ma)
                # Preenche a matriz com 1s de acordo com as combinações
                matrix[lines_col1, 1] = 1
                matrix[lines_col2, 2] = 1
                matrix[lines_col3, 3] = 1
                # Adiciona a combinação gerada na lista
                push!(combinations, matrix)
            end
            # println(combinations) #ok!
            total_lamb = (oa*n)^ma 
            # prob_matrix = zeros(Int8, oa*n, ma, total_lamb)
            prob_matrix = zeros(Int8, oa, ma, n, total_lamb)
            for lamb in 1:total_lamb
                for x in 1:ma
                    i = 1
                    for c in 1:n
                    for a in 1:oa
                            # prob_matrix[a, x, m, lamb] = combinations[lamb][am,x]
                            prob_matrix[a, x, c, lamb] = combinations[lamb][i,x]
                            # println("prob_matrix[",a,", ",x,", ",c,", ",lamb,"]= ", prob_matrix[a, x, c, lamb])#ok!
                            i += 1
                        end
                    end
                    
                end
            end  
        elseif  oa == 2 && ma == 4
            for lines_col1 in choices, lines_col2 in choices, lines_col3 in choices,  lines_col4 in choices
                matrix = zeros(Int, oa*n, ma)

                # Preenche a matriz com 1s de acordo com as combinações
                matrix[lines_col1, 1] = 1
                matrix[lines_col2, 2] = 1
                matrix[lines_col3, 3] = 1
                matrix[lines_col4, 4] = 1
                # Adiciona a combinação gerada na lista
                push!(combinations, matrix)
            end
            total_lamb = (oa*n)^ma 
            # prob_matrix = zeros(Int8, oa*n, ma, total_lamb)
            prob_matrix = zeros(Int8, oa, ma, n, total_lamb)

            for lamb in 1:total_lamb
                for x in 1:ma
                    i = 1
                    for c in 1:n
                        for a in 1:oa
                            # prob_matrix[a, x, m, lamb] = combinations[lamb][am,x]
                            prob_matrix[a, x, c, lamb] = combinations[lamb][i,x]
                            i += 1
                        end
                    end
                    
                end
            end  
        elseif  oa == 2 && ma == 6
            for lines_col1 in choices, lines_col2 in choices, lines_col3 in choices,  lines_col4 in choices, lines_col5 in choices,lines_col6 in choices
                matrix = zeros(Int, oa*n, ma)

                # Preenche a matriz com 1s de acordo com as combinações
                matrix[lines_col1, 1] = 1
                matrix[lines_col2, 2] = 1
                matrix[lines_col3, 3] = 1
                matrix[lines_col4, 4] = 1
                matrix[lines_col5, 5] = 1
                matrix[lines_col6, 6] = 1
                # Adiciona a combinação gerada na lista
                push!(combinations, matrix)
            end
            total_lamb = (oa*n)^ma 
            # prob_matrix = zeros(Int8, oa*n, ma, total_lamb)
            prob_matrix = zeros(Int8, oa, ma, n, total_lamb)

            for lamb in 1:total_lamb
                for x in 1:ma
                    i = 1
                    for c in 1:n
                        for a in 1:oa
                            # prob_matrix[a, x, m, lamb] = combinations[lamb][am,x]
                            prob_matrix[a, x, c, lamb] = combinations[lamb][i,x]
                            i += 1
                        end
                    end
                    
                end
            end  
        end
    end
    return prob_matrix
end

# TO TEST prob_determ_classical_bit:
# oa = num_c = 2
# ma = 3
# num_lamb = (num_c*oa)^ma
# prob_ac = prob_determ_classical_bit(oa,ma,num_c)
# for l in 1:num_lamb
#     println("prob_ac[",l,"]: ", prob_ac[:, :, :, l])
# end


function generate_hermitian_psd_matrix(dim)
    # Gerar elementos na diagonal principal em torno de 1.0
    d2d = 0.1 * rand()
    # Construir a matriz Hermitiana
    if dim == 2
        H = [rand() d2d; d2d rand()]
        # Garantir que a matriz seja positiva semidefinida
        eigenvalues = eigen(H).values
        if any(eigenvalues .< 0)
            H += abs(minimum(eigenvalues)) * I
        end
        return H
    else 
        H = [rand() d2d d2d;d2d rand() d2d;d2d d2d rand()]
        # Garantir que a matriz seja positiva semidefinida
        eigenvalues = eigen(H).values
        if any(eigenvalues .< 0)
            H += abs(minimum(eigenvalues)) * [1 0 0;0 1 0;0 0 1]
        end
        return H
    end
end

# isotropic state
function rho_noisy(dim, alpha)
    if dim == 2
        rho = alpha/2 * ( (KP([1; 0], [1; 0])+KP([0; 1], [0; 1])) * (KP([1; 0], [1; 0])+KP([0; 1], [0; 1]))') + (1 - alpha)/4 * IM(4)
    elseif dim ==3
        rho = alpha/3 * ( (KP([1; 0;0], [1; 0;0])+KP([0; 1;0], [0; 1;0])+KP([0; 0;1], [0; 0;1])) * (KP([1; 0;0], [1; 0;0])+KP([0; 1;0], [0; 1;0])+KP([0; 0;1], [0; 0;1]))') + (1 - alpha)/9 * IM(9)
    elseif dim ==4
        rho = alpha/4 * ( (KP([1; 0;0;0], [1; 0;0;0])+KP([0; 1;0;0], [0; 1;0;0])+KP([0; 0;1;0], [0; 0;1;0])+KP([0; 0;0;1], [0; 0;0;1])) * (KP([1; 0;0;0], [1; 0;0;0])+KP([0; 1;0;0], [0; 1;0;0])+KP([0; 0;1;0], [0; 0;1;0])+KP([0; 0;0;1], [0; 0;0;1]))') + (1 - alpha)/16 * IM(16)
    elseif dim ==5
        rho = alpha/5 * ( (KP([1; 0;0;0;0], [1; 0;0;0;0])+KP([0; 1;0;0;0], [0; 1;0;0;0])+KP([0; 0;1;0;0], [0; 0;1;0;0])+KP([0; 0;0;1;0], [0; 0;0;1;0])+KP([0; 0;0;0;1], [0; 0;0;0;1])) * (KP([1; 0;0;0;0], [1; 0;0;0;0])+KP([0; 1;0;0;0], [0; 1;0;0;0])+KP([0; 0;1;0;0], [0; 0;1;0;0])+KP([0; 0;0;1;0], [0; 0;0;1;0])+KP([0; 0;0;0;1], [0; 0;0;0;1]))') + (1 - alpha)/25 * IM(25) 
    end
end


# M[dim,mub,out]
function M(dim, mub, out)
    #d=2
    if dim == 2 && mub == 1 && out == 1
        1/sqrt(2) * [1; 1]
    elseif dim == 2 && mub == 1 && out == 2
        1/sqrt(2) * [1; -1]
    elseif dim == 2 && mub == 2 && out == 1
        1/sqrt(2) * [1; im]
    elseif dim == 2 && mub == 2 && out == 2
        1/sqrt(2) * [1; -im]
    elseif dim == 2 && mub == 3 && out == 1
        [1; 0]
    elseif dim == 2 && mub == 3 && out == 2
        [0; 1]
    #d=3

    elseif dim == 3 && mub == 1 && out ==1
        [1;0;0] 
    elseif dim == 3 && mub == 1 && out ==2
        [0;1;0] 
    elseif dim == 3 && mub == 1 && out ==3
        [0;0;1] 

    elseif dim == 3 && mub == 2 && out ==1
        1/sqrt(3) * [1;1;1] 
    elseif dim == 3 && mub == 2 && out ==2
        1/sqrt(3) * [1;exp(2im * π / 3);exp(-2im * π / 3)] 
    elseif dim == 3 && mub == 2 && out ==3
        1/sqrt(3) * [1;exp(-2im * π / 3);exp(2im * π / 3)] 
        
    elseif dim == 3 && mub == 3 && out ==1
        1/sqrt(3) * [1;1;exp(-2im * π / 3)]
    elseif dim == 3 && mub == 3 && out ==2
        1/sqrt(3) * [1;exp(2im * π / 3);exp(2im * π / 3)] 
    elseif dim == 3 && mub == 3 && out ==3
        1/sqrt(3) * [1;exp(-2im * π / 3);1] 

    elseif dim == 3 && mub == 4 && out ==1
        1/sqrt(3) * [1;1;exp(2im * π / 3)] 
    elseif dim == 3 && mub == 4 && out ==2
        1/sqrt(3) * [1;exp(2im * π / 3);1] 
    elseif dim == 3 && mub == 4 && out ==3
        1/sqrt(3) * [1;exp(-2im * π / 3);exp(-2im * π / 3)] 

    #d=4
    elseif dim == 4 && mub == 1 && out ==1
        [1;0;0;0]
    elseif dim == 4 && mub == 1 && out ==2
        [0;1;0;0]
    elseif dim == 4 && mub == 1 && out ==3
        [0;0;1;0]
    elseif dim == 4 && mub == 1 && out ==4
        [0;0;0;1]
    elseif dim == 4 && mub == 2 && out ==1
        1/2 * [1;1;1;1]
    elseif dim == 4 && mub == 2 && out ==2
        1/2 * [1;1;-1;-1]
    elseif dim == 4 && mub == 2 && out ==3
        1/2 * [1;-1;-1;1]
    elseif dim == 4 && mub == 2 && out ==4
        1/2 * [1;-1;1;-1]
    elseif dim == 4 && mub == 3 && out ==1
        1/2 * [1;-1;-im;-im]
    elseif dim == 4 && mub == 3 && out ==2
        1/2 * [1;-1;im;im]
    elseif dim == 4 && mub == 3 && out ==3
        1/2 * [1;1;im;-im]
    elseif dim == 4 && mub == 3 && out ==4
        1/2 * [1;1;-im;im]
    elseif dim == 4 && mub == 4 && out ==1
        1/2 * [1;-im;-im;-1]
    elseif dim == 4 && mub == 4 && out ==2
        1/2 * [1;-im;im;1]
    elseif dim == 4 && mub == 4 && out ==3
        1/2 * [1;im;im;-1]
    elseif dim == 4 && mub == 4 && out ==4
        1/2 * [1;im;-im;1]
    elseif dim == 4 && mub == 5 && out ==1
        1/2 * [1;-im;-1;-im]
    elseif dim == 4 && mub == 5 && out ==2
        1/2 * [1;-im;1;im]
    elseif dim == 4 && mub == 5 && out ==3
        1/2 * [1;im;-1;im]
    elseif dim == 4 && mub == 5 && out ==4
        1/2 * [1;im;1;-im]
    #d=5
    elseif dim == 5 && mub == 1 && out ==1
        [1;0;0;0;0]
    elseif dim == 5 && mub == 1 && out ==2
        [0;1;0;0;0]
    elseif dim == 5 && mub == 1&& out ==3
        [0;0;1;0;0]
    elseif dim == 5 && mub == 1 && out ==4
        [0;0;0;1;0]
    elseif dim == 5 && mub == 1 && out ==5
        [0;0;0;0;1]

    elseif dim == 5 && mub == 2 && out ==1
        1/sqrt(5) * [1;1;1;1;1]
    elseif dim == 5 && mub == 2 && out ==2
        1/sqrt(5) * [1;exp(2im * π / 5);exp(4im * π / 5);exp(-4im * π / 5);exp(-2im * π / 5)]
    elseif dim == 5 && mub == 2 && out ==3
        1/sqrt(5) * [1;exp(4im * π / 5);exp(-2im * π / 5);exp(2im * π / 5);exp(-4im * π / 5)]
    elseif dim == 5 && mub == 2 && out ==4
        1/sqrt(5) * [1;exp(-4im * π / 5);exp(2im * π / 5);exp(-2im * π / 5);exp(4im * π / 5)]
    elseif dim == 5 && mub == 2 && out ==5
        1/sqrt(5) * [1;exp(-2im * π / 5);exp(-4im * π / 5);exp(4im * π / 5);exp(2im * π / 5)]

    elseif dim == 5 && mub == 3 && out ==1
        1/sqrt(5) * [1;1;exp(2im * π / 5);exp(-4im * π / 5);exp(2im * π / 5)]
    elseif dim == 5 && mub == 3 && out ==2
        1/sqrt(5) * [1;exp(2im * π / 5);exp(-4im * π / 5);exp(2im * π / 5);1]
    elseif dim == 5 && mub == 3 && out ==3
        1/sqrt(5) * [1;exp(4im * π / 5);1;exp(-2im * π / 5);exp(-2im * π / 5)]
    elseif dim == 5 && mub == 3 && out ==4
        1/sqrt(5) * [1;exp(-4im * π / 5);exp(4im * π / 5);exp(4im * π / 5);exp(-4im * π / 5)]
    elseif dim == 5 && mub == 3 && out ==5
        1/sqrt(5) * [1;exp(-2im * π / 5);exp(-2im * π / 5);1;exp(4im * π / 5)]

    elseif dim == 5 && mub == 4 && out ==1
        1/sqrt(5) * [1;1;exp(4im * π / 5);exp(2im * π / 5);exp(4im * π / 5)]
    elseif dim == 5 && mub == 4 && out ==2
        1/sqrt(5) * [1;exp(2im * π / 5);exp(-2im * π / 5);exp(-2im * π / 5);exp(2im * π / 5)]
    elseif dim == 5 && mub == 4 && out ==3
        1/sqrt(5) * [1;exp(4im * π / 5);exp(2im * π / 5);exp(4im * π / 5);1]
    elseif dim == 5 && mub == 4 && out ==4
        1/sqrt(5) * [1;exp(-4im * π / 5);exp(-4im * π / 5);1;exp(-2im * π / 5)]
    elseif dim == 5 && mub == 4 && out ==5
        1/sqrt(5) * [1;exp(-2im * π / 5);1;exp(-4im * π / 5);exp(-4im * π / 5)]

    elseif dim == 5 && mub == 5 && out ==1
        1/sqrt(5) * [1;1;exp(-4im * π / 5);exp(-2im * π / 5);exp(-4im * π / 5)]
    elseif dim == 5 && mub == 5 && out ==2
        1/sqrt(5) * [1;exp(2im * π / 5);1;exp(4im * π / 5);exp(4im * π / 5)]
    elseif dim == 5 && mub == 5 && out ==3
        1/sqrt(5) * [1;exp(4im * π / 5);exp(4im * π / 5);1;exp(2im * π / 5)]
    elseif dim == 5 && mub == 5 && out ==4
        1/sqrt(5) * [1;exp(-4im * π / 5);exp(-2im * π / 5);exp(-4im * π / 5);1]
    elseif dim == 5 && mub == 5 && out ==5
        1/sqrt(5) * [1;exp(-2im * π / 5);exp(2im * π / 5);exp(2im * π / 5);exp(-2im * π / 5)]

    elseif dim == 5 && mub == 6 && out ==1
        1/sqrt(5) * [1;1;exp(-2im * π / 5);exp(4im * π / 5);exp(-2im * π / 5)]
    elseif dim == 5 && mub == 6 && out ==2
        1/sqrt(5) * [1;exp(2im * π / 5);exp(2im * π / 5);1;exp(-4im * π / 5)]
    elseif dim == 5 && mub == 6 && out ==3
        1/sqrt(5) * [1;exp(4im * π / 5);exp(-4im * π / 5);exp(-4im * π / 5);exp(4im * π / 5)]
    elseif dim == 5 && mub == 6 && out ==4
        1/sqrt(5) * [1;exp(-4im * π / 5);1;exp(2im * π / 5);exp(2im * π / 5)]
    elseif dim == 5 && mub == 6 && out ==5
        1/sqrt(5) * [1;exp(-2im * π / 5);exp(4im * π / 5);exp(-2im * π / 5);1]
    end
end

# M[dim,mub,out] FOR dim == 2, MORE POVM (PLATONIC SOLID) OPTIONS:
# function M(dim, mub, out)
#     # if dim == 2 && mub == 1 && out == 1
#     #     1/2 * [1+1/(2*sqrt(2))  (1-1*im)/(2*sqrt(2)); (1+1*im)/(2*sqrt(2)) 1-1/(2*sqrt(2)) ]
#     # elseif dim == 2 && mub == 1 && out == 2
#     #     1/2 * [1-1/(2*sqrt(2))  (1+1*im)/(2*sqrt(2)); (1-1*im)/(2*sqrt(2)) 1-1/(2*sqrt(2))]
#     # elseif dim == 2 && mub == 2 && out == 1
#     #     1/2 * [1-1/(2*sqrt(2))  -(1+1*im)/(2*sqrt(2)); -(1-1*im)/(2*sqrt(2)) 1+1/(2*sqrt(2))]
#     # elseif dim == 2 && mub == 2 && out == 2
#     #     1/2 * [1+1/(2*sqrt(2))  -(1-1*im)/(2*sqrt(2)); -(1+1*im)/(2*sqrt(2)) 1-1/(2*sqrt(2)) ]
#     # end

#     # if dim == 2 && mub == 1 && out == 1
#     #     1/2 * [1; 1]*adjoint([1; 1])
#     # elseif dim == 2 && mub == 1 && out == 2
#     #     1/2 * [1; -1]*adjoint([1; -1])
#     # elseif dim == 2 && mub == 2 && out == 1
#     #     1/2 * [1; im]*adjoint([1; im])
#     # elseif dim == 2 && mub == 2 && out == 2
#     #     1/2 * [1; -im]*adjoint([1; -im])
#     # elseif dim == 2 && mub == 3 && out == 1
#     #     [1; 0]*adjoint([1; 0])
#     # elseif dim == 2 && mub == 3 && out == 2
#     #     [0; 1]*adjoint([0; 1])
#     # end
    

#     # if dim == 2 && mub == 1 && out == 1
#     #     1/2 * (IM(2)+(1/sqrt(3))*((1)*[0 1;1 0]+(1)*[0 -im;im 0]+(1)*[1 0;0 -1]))
#     # elseif dim == 2 && mub == 1 && out == 2
#     #     1/2 * (IM(2)-(1/sqrt(3))*((1)*[0 1;1 0]+(1)*[0 -im;im 0]+(1)*[1 0;0 -1]))    
#     # elseif dim == 2 && mub == 2 && out == 1
#     #     1/2 * (IM(2)+(1/sqrt(3))*((1)*[0 1;1 0]+(-1)*[0 -im;im 0]+(-1)*[1 0;0 -1]))
#     # elseif dim == 2 && mub == 2 && out == 2
#     #     1/2 * (IM(2)-(1/sqrt(3))*((1)*[0 1;1 0]+(-1)*[0 -im;im 0]+(-1)*[1 0;0 -1]))    
#     # elseif dim == 2 && mub == 3 && out == 1
#     #     1/2 * (IM(2)+(1/sqrt(3))*((-1)*[0 1;1 0]+(1)*[0 -im;im 0]+(-1)*[1 0;0 -1]))
#     # elseif dim == 2 && mub == 3 && out == 2
#     #     1/2 * (IM(2)-(1/sqrt(3))*((-1)*[0 1;1 0]+(1)*[0 -im;im 0]+(-1)*[1 0;0 -1]))
#     # elseif dim == 2 && mub == 4 && out == 1
#     #     1/2 * (IM(2)+(1/sqrt(3))*((-1)*[0 1;1 0]+(-1)*[0 -im;im 0]+(1)*[1 0;0 -1]))
#     # elseif dim == 2 && mub == 4 && out == 2
#     #     1/2 * (IM(2)-(1/sqrt(3))*((-1)*[0 1;1 0]+(-1)*[0 -im;im 0]+(1)*[1 0;0 -1]))
#     # end

#     phi = (1+sqrt(5))/2
#     if dim == 2 && mub == 1 && out == 1
#         1/2 * (IM(2)+(1/sqrt((1/2)*(5+sqrt(5))))*((0)*[0 1;1 0]+(1)*[0 -im;im 0]+(phi)*[1 0;0 -1]))
#     elseif dim == 2 && mub == 1 && out == 2
#         1/2 * (IM(2)-(1/sqrt((1/2)*(5+sqrt(5))))*((0)*[0 1;1 0]+(1)*[0 -im;im 0]+(phi)*[1 0;0 -1]))    
#     elseif dim == 2 && mub == 2 && out == 1
#         1/2 * (IM(2)+(1/sqrt((1/2)*(5+sqrt(5))))*((0)*[0 1;1 0]+(1)*[0 -im;im 0]+(-phi)*[1 0;0 -1]))
#     elseif dim == 2 && mub == 2 && out == 2
#         1/2 * (IM(2)-(1/sqrt((1/2)*(5+sqrt(5))))*((0)*[0 1;1 0]+(1)*[0 -im;im 0]+(-phi)*[1 0;0 -1]))    
#     elseif dim == 2 && mub == 3 && out == 1
#         1/2 * (IM(2)+(1/sqrt((1/2)*(5+sqrt(5))))*((1)*[0 1;1 0]+(phi)*[0 -im;im 0]+(0)*[1 0;0 -1]))
#     elseif dim == 2 && mub == 3 && out == 2
#         1/2 * (IM(2)-(1/sqrt((1/2)*(5+sqrt(5))))*((1)*[0 1;1 0]+(phi)*[0 -im;im 0]+(0)*[1 0;0 -1]))
#     elseif dim == 2 && mub == 4 && out == 1
#         1/2 * (IM(2)+(1/sqrt((1/2)*(5+sqrt(5))))*((1)*[0 1;1 0]-(phi)*[0 -im;im 0]+(0)*[1 0;0 -1]))
#     elseif dim == 2 && mub == 4 && out == 2
#         1/2 * (IM(2)-(1/sqrt((1/2)*(5+sqrt(5))))*((1)*[0 1;1 0]-(phi)*[0 -im;im 0]+(0)*[1 0;0 -1]))
#     elseif dim == 2 && mub == 5 && out == 1
#         1/2 * (IM(2)+(1/sqrt((1/2)*(5+sqrt(5))))*((phi)*[0 1;1 0]+(0)*[0 -im;im 0]+(1)*[1 0;0 -1]))    
#     elseif dim == 2 && mub == 5 && out == 2
#         1/2 * (IM(2)-(1/sqrt((1/2)*(5+sqrt(5))))*((phi)*[0 1;1 0]+(0)*[0 -im;im 0]+(1)*[1 0;0 -1])) 
#     elseif dim == 2 && mub == 6 && out == 1
#         1/2 * (IM(2)+(1/sqrt((1/2)*(5+sqrt(5))))*((-phi)*[0 1;1 0]+(0)*[0 -im;im 0]+(1)*[1 0;0 -1])) 
#     elseif dim == 2 && mub == 6 && out == 2
#         1/2 * (IM(2)-(1/sqrt((1/2)*(5+sqrt(5))))*((-phi)*[0 1;1 0]+(0)*[0 -im;im 0]+(1)*[1 0;0 -1])) 
#     end
# end