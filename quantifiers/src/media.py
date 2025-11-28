import sys
import statistics

def calcular_media_e_variancia(nome_arquivo="arquivo.dat"):
    """
    Lê um arquivo .dat, extrai os valores da terceira coluna e calcula
    a média e a variância amostral desses valores.

    Args:
        nome_arquivo (str): O nome do arquivo a ser lido. Padrão é 'arquivo.dat'.

    Returns:
        tuple: Uma tupla contendo (media, variancia). Retorna (None, None)
               se não for possível calcular (ex: erro ou dados insuficientes).
    """
    valores = []
    numero_linha = 0

    try:
        # Abre o arquivo para leitura ('r')
        with open(nome_arquivo, 'r') as f:
            # Itera sobre cada linha do arquivo
            for linha in f:
                numero_linha += 1
                # Remove espaços/quebras de linha e divide em colunas
                colunas = linha.strip().split()

                # Verifica se há pelo menos 3 colunas
                if len(colunas) >= 3:
                    try:
                        # Pega o 5to elemento (índice 1) e converte para float
                        valor = float(colunas[4])
                        valores.append(valor)
                    except ValueError:
                        # Avisa se o valor não for um número
                        print(f"Aviso: Linha {numero_linha}: O valor '{colunas[2]}' "
                              f"não é numérico e será ignorado.", file=sys.stderr)
                else:
                    # Avisa se não houver colunas suficientes
                     print(f"Aviso: Linha {numero_linha}: Não possui 3 colunas e será ignorada.",
                           file=sys.stderr)

        # Verifica se temos dados suficientes para os cálculos
        if len(valores) >= 2:
            # Calcula a média usando statistics.mean()
            media = statistics.mean(valores)
            # Calcula a variância amostral usando statistics.variance()
            variancia = statistics.variance(valores)
            return media, variancia
        elif len(valores) == 1:
            # Se houver apenas um valor, podemos calcular a média, mas a variância é 0
            media = statistics.mean(valores)
            print("Aviso: Apenas um valor encontrado. A variância será 0.", file=sys.stderr)
            return media, 0.0
        else:
            # Se nenhum valor foi encontrado
            print("Nenhum valor numérico foi encontrado na terceira coluna.", file=sys.stderr)
            return None, None

    except FileNotFoundError:
        print(f"Erro: O arquivo '{nome_arquivo}' não foi encontrado.", file=sys.stderr)
        return None, None
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}", file=sys.stderr)
        return None, None

# --- Como usar o código ---

# 1. Crie (ou certifique-se que existe) um arquivo 'arquivo.dat'.
#    Coloque nele o conteúdo esperado (4 linhas, N colunas).
#    Exemplo 'arquivo.dat':
#
#    10  25  100.5  A
#    12  30  250.0  B
#    15  35  300.2  C
#    20  40  150.3  D
#
# try:
#     with open("arquivo.dat", "w") as f:
#         f.write("10  25  100.5  A\n")
#         f.write("12  30  250.0  B\n")
#         f.write("15  35  300.2  C\n")
#         f.write("20  40  150.3  D\n")
#     print("Arquivo 'arquivo.dat' de exemplo criado/sobrescrito.")
# except Exception as e:
#     print(f"Erro ao criar arquivo de exemplo: {e}")

# Número de repetições
valores_desejados = [10, 50, 100, 200, 400, 600, 800, 1000, 2500, 5000, 7500, 10000]

# Criar o loop for
for num_repetitions in valores_desejados:

    # 2. Chame a função e guarde os resultados
    nome_do_arquivo = f"../dat/sim/werner05/sigma_werner05_shots_{num_repetitions}.dat"
    media_calculada, variancia_calculada = calcular_media_e_variancia(nome_do_arquivo)

    # 3. Exibe os resultados
    if media_calculada is not None:
        print(f"{num_repetitions}\t{media_calculada:.4f}\t{variancia_calculada:.4f}")
    else:
        print("\nNão foi possível calcular a média e a variância.")