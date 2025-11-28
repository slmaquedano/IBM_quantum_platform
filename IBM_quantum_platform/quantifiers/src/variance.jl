# Importa o módulo DelimitedFiles para facilitar a leitura de arquivos com colunas
using DelimitedFiles

# --- Função para calcular a Variância Populacional ---
# A função recebe um vetor (uma coluna de dados) como entrada
function calcular_variancia_populacional(dados::Vector{Float64})
    # Calcula o número de elementos (n)
    n = length(dados)

    # Se não houver dados, retorna 0 para evitar divisão por zero
    if n == 0
        return 0.0
    end

    # Calcula a média (x_avg)
    media = sum(dados) / n

    # Calcula a soma dos desvios ao quadrado: Σ (media - x_i)^2
    # Usamos "broadcasting" aqui: (media .- dados) subtrai a média de cada elemento.
    # O .^2 eleva cada uma dessas diferenças ao quadrado.
    soma_diferencas_quadrado = sum((media .- dados).^2)

    # Divide a soma pelo número de elementos (1/n * Σ(...))
    variancia = soma_diferencas_quadrado / n

    return variancia
end 


# --- Função Principal do Programa ---
function main()
    valores_desejados = [10,
    600,
    2500,
    10000 ]

    w = 10

    # Criar o loop for
    for num_repetitions in valores_desejados

        # Tenta ler o arquivo. Se não conseguir, mostra uma mensagem de erro.
        try
            # readdlm lê o arquivo delimitado.
            # Ele retorna uma matriz com todos os dados.
            # ' ' é o delimitador (espaço), e '\n' é a quebra de linha.
            # O 'comments=true' ignora linhas que começam com '#'
            dados_matriz = readdlm("../dat/werner_$w/quantifiers/werner_$(w)_N_$num_repetitions.dat", '\t', Float64, '\n', comments=true)

            # Pega a segunda e a terceira colunas.
            # A sintaxe [:, 2] significa "todas as linhas, coluna 2".
            coluna2 = dados_matriz[:, 2]
            coluna3 = dados_matriz[:, 3]
            # println("Dados da coluna 2 lidos com sucesso. Total de pontos: ", length(coluna2))
            # println("Dados da coluna 3 lidos com sucesso. Total de pontos: ", length(coluna3))
            # println("-"^30)

            # Calcula a variância para cada coluna usando a função que criamos
            variancia_col2 = calcular_variancia_populacional(coluna2)
            variancia_col3 = calcular_variancia_populacional(coluna3)

            # Imprime os resultados formatados
            # println("Resultados do Estimador (Variância Populacional):", num_repetitions)
            println(num_repetitions, "\t", w,"\t", round(variancia_col2, digits=4),"\t", round(variancia_col3, digits=6))

        catch e
            # Se ocorrer um erro (ex: arquivo não encontrado), esta mensagem será exibida.
            println("Erro ao ler o arquivo '$num_repetitions'.")
            println("Por favor, verifique se o arquivo existe no mesmo diretório e tem o formato correto.")
            println("Detalhes do erro: ", e)
        end
    end
end

# Executa a função principal
main()