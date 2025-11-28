include("func_SDP.jl")

function main()

    valores_desejados = [10, 50, 100, 200, 400, 600, 800, 1000, 2500, 5000, 7500, 10000]

    for num_repetitions in valores_desejados

        dim = 2
        ma = 3
        num_c = 1
        # vec_lenght = 1000 #Deve coincidir com o numero 'total/passo' da variavel 'rep' 
        vec_lenght = 10

        oa = dim 
        ob = oa 
        mb = ma 
        num_lamb_Smu = oa^ma
        num_lamb = (num_c*oa)^ma

        prob_ac = prob_determ_classical_bit(oa,ma,num_c)

        save_SDP_NS_same_as_mu = zeros(vec_lenght)
        save_SDP_LHS_plus_mu = zeros(vec_lenght)
        save_SDP_Pg = zeros(vec_lenght)
        save_SDP_LHS = zeros(vec_lenght)
        save_SDP_SRr = zeros(vec_lenght)

        steps = 1
        for line in 1:vec_lenght
            sigma_ruido = Array{Matrix{ComplexF64}}(undef, oa, ma)        
            for x in 1:ma 
                for a in 1:oa
                    saves = read_hdf5_data("../../h5_simulations/werner08/sigma_werner08werner_shots_$num_repetitions.h5", rep = line, x=x, a=a)
                    sigma_ruido[a,x] = saves[1]
                    # println("sigma_ruido[",a,",",x,"]",sigma_ruido[a,x])
                end
            end

            sigma_a = Array{Matrix{ComplexF64}}(undef, ma)
            for x in 1:ma 
                sigma_a[x] = zeros(oa,oa)
            end

            for x in 1:ma 
                for a in 1:oa
                    sigma_a[x] += sigma_ruido[a,x]
                end
            end

            sigma_ax = []
            for x in 1:ma 
                for a in 1:oa
                    push!(sigma_ax, sigma_ruido[a,x]/tr(sigma_a[x]))
                end
            end

            sigma_exp = Array{Matrix{ComplexF64}}(undef, oa, ma)
            i = 1
            for x in 1:ma 
                for a in 1:oa
                    sigma_exp[a,x] = sigma_ax[i]
                    i += 1
                end
            end

            opt_sigma = 1 
            opt_M = 0
            
            #------------------------|Pg|-------------------->|
            Mpg = []
            for x in 1:ma
                push!(Mpg, ComplexVariable(oa,oa))
            end 

            obj_func_pguess = 0
            for x in 1:ma
                sum_a = zeros(oa,oa)
                for a in 1:oa
                    sum_a += sigma_exp[a,x]
                end
                obj_func_pguess += (1/ma)*real(tr(sum_a*Mpg[x]))
            end 

            opt_val_pg = maximize(obj_func_pguess)
            sum_M_a = zeros(oa,oa)
            for x in 1:ma
                sum_M_a += Mpg[x]
            end 
            opt_val_pg.constraints += sum_M_a == IM(oa)

            for x in 1:ma
                opt_val_pg.constraints += Mpg[x] in :SDP
            end 

            solve!(opt_val_pg, Mosek.Optimizer; silent = true) 
            opt_sigma = opt_val_pg.optval
            #|<---------------------|Pg|--------------------

            #-----------------------|SR|------------------->|
            mu = 0

            F_ax = Array{ComplexVariable}(undef, oa, ma) 
            for x in 1:ma 
                for a in 1:oa
                    F_ax[a,x] = HermitianSemidefinite(oa)
                end
            end

            obj_fc_dual_SRsig = 0
            for x in 1:ma 
                for a in 1:oa
                    obj_fc_dual_SRsig += tr(F_ax[a,x]*sigma_exp[a,x])
                end
            end

            omega = real(obj_fc_dual_SRsig-1)
            opt_val_US_Smu_zero = maximize(omega)

            D = prob_determ(oa, ma)
            for l in 1:num_lamb_Smu
                st1_SRsig = 0
                for x in 1:ma 
                    for a in 1:oa
                        st1_SRsig += D[a,x,l]*F_ax[a,x]
                    end
                end
                opt_val_US_Smu_zero.constraints += IM(oa)-(1+mu)*st1_SRsig in :SDP
            end

            solve!(opt_val_US_Smu_zero, Mosek.Optimizer; silent = true)
            #|<-----------------------|SR|-------------------


            #-------------------------|Rsig|---------------->|
            eta_ax = Array{ComplexVariable}(undef, oa, ma) 
            for x in 1:ma 
                for a in 1:oa
                    eta_ax[a,x] = HermitianSemidefinite(oa)
                end
            end
            
            gamma = ComplexVariable(oa,oa) 
            arg_obj_fuc_i = 0
            for x in 1:ma
                for a in 1:oa
                    arg_obj_fuc_i += real(tr(eta_ax[a,x]))
                end
            end 
            arg_obj_fuc = arg_obj_fuc_i*1/ma -1
            opt_val_ns = minimize(arg_obj_fuc) #SDP otimization

            opt_val_ns.constraints += arg_obj_fuc >= 0
            opt_val_ns.constraints += gamma in :SDP # gamma >= 0

            for x in 1:ma # eta - sigma-exp >=0
                for a in 1:oa
                    opt_val_ns.constraints += eta_ax[a,x] - sigma_exp[a,x] in :SDP 
                end
            end

            for x in 1:ma # Dual NS problem  
                sum_a_eta_ax = zeros(oa,oa)
                for a in 1:oa
                    sum_a_eta_ax += eta_ax[a,x]
                end
                opt_val_ns.constraints += gamma - sum_a_eta_ax*1/ma in :SDP
            end

            for x in 1:ma
                sum_a_eta_ax = 0
                for a in 1:oa
                    sum_a_eta_ax += real(tr(eta_ax[a,x]))
                end

                #HERE WE NEED TO CHANGE THE 1 BIT BOUND
                opt_val_ns.constraints += sum_a_eta_ax*(1/3) - tr(gamma) in :SDP
            end 

            solve!(opt_val_ns, Mosek.Optimizer; silent_solver = true)
            #|<-------------------------|Rsig|----------------

            #---------------------|SRsig_DUAL|-------------->|
            mu = evaluate(opt_val_ns.optval)

            F_ax = Array{ComplexVariable}(undef, oa, ma) 
            for x in 1:ma 
                for a in 1:oa
                    F_ax[a,x] = HermitianSemidefinite(oa)
                end
            end

            obj_fc_dual_SRsig = 0
            for x in 1:ma 
                for a in 1:oa
                    obj_fc_dual_SRsig += tr(F_ax[a,x]*sigma_exp[a,x])
                end
            end

            omega = real(obj_fc_dual_SRsig-1)
            opt_val_US_Smu = maximize(omega)

            D = prob_determ(oa, ma)
            for l in 1:num_lamb_Smu
                st1_SRsig = 0
                for x in 1:ma 
                    for a in 1:oa
                        st1_SRsig += D[a,x,l]*F_ax[a,x]
                    end
                end
                opt_val_US_Smu.constraints += IM(oa)-(1+mu)*st1_SRsig in :SDP
            end

            solve!(opt_val_US_Smu, Mosek.Optimizer; silent = true)
            #|<---------------------|SRsig_DUAL|--------------

            #---------------------|SR(sigma, r)|------------>|
            mu = evaluate(opt_val_ns.optval)

            LHSr_ax = Array{ComplexVariable}(undef, oa, ma) 
            for x in 1:ma 
                for a in 1:oa
                    LHSr_ax[a,x] = HermitianSemidefinite(oa)
                end
            end

            LHS_lamb = Array{ComplexVariable}(undef, num_lamb_Smu ) 
            for l in 1:num_lamb_Smu
                LHS_lamb[l] = HermitianSemidefinite(oa)
            end

            tau_ax = Array{ComplexVariable}(undef, oa, ma) 
            for x in 1:ma 
                for a in 1:oa
                    tau_ax[a,x] = HermitianSemidefinite(oa)
                end
            end

            obj_fc_dual_SRr = 0
            for x in 1:ma 
                for a in 1:oa
                    obj_fc_dual_SRr += tr(LHSr_ax[a,x])
                end
            end

            SRr = real((1/ma)*obj_fc_dual_SRr-1)
            opt_val_SRr= minimize(SRr)

            D = prob_determ(oa, ma)
            for x in 1:ma 
                for a in 1:oa
                    st1_SRsig  = 0
                    for l in 1:num_lamb_Smu
                        st1_SRsig += D[a,x,l]*LHS_lamb[l]
                    end
                    opt_val_SRr.constraints += (1+mu)*st1_SRsig-mu*tau_ax[a,x]-sigma_exp[a,x] in :SDP
                end
            end

            for x in 1:ma 
                for x_i in 1:ma 
                    sum_a_LHS  = 0
                    sum_a_tau  = 0
                    sum_a_LHSr  = 0
                    for a in 1:oa
                        for l in 1:num_lamb_Smu
                            sum_a_LHS += D[a,x,l]*LHS_lamb[l]
                        end
                        sum_a_tau += tau_ax[a,x_i]
                        sum_a_LHSr += LHSr_ax[a,x]
                    end
                    opt_val_SRr.constraints += tr(sum_a_LHS)==tr(sum_a_LHSr)
                    opt_val_SRr.constraints += tr(sum_a_tau)==tr(sum_a_LHSr)
                end
            end

            solve!(opt_val_SRr, Mosek.Optimizer; silent = true)
            #|<---------------------|SR(sigma, r)|------------

            save_SDP_NS_same_as_mu[steps] = opt_val_ns.optval
            save_SDP_LHS_plus_mu[steps] = opt_val_US_Smu.optval
            save_SDP_Pg[steps] = opt_val_pg.optval  
            save_SDP_LHS[steps] = opt_val_US_Smu_zero.optval
            save_SDP_SRr[steps] = opt_val_SRr.optval

            steps += 1
        end

        save_shots = collect(1:vec_lenght)
        writedlm("../dat/sim/werner08/TEST_17_nov-sigma_werner08_shots_$num_repetitions.dat", [save_shots save_SDP_NS_same_as_mu save_SDP_LHS_plus_mu save_SDP_Pg save_SDP_LHS save_SDP_SRr])
    end
end 
@time main()