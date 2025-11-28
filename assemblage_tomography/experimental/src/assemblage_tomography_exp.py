import numpy as np
from qiskit_aer import AerSimulator
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import QuantumCircuit, transpile, assemble
import matplotlib.pyplot as plt
from qiskit_ibm_runtime.fake_provider import FakePerth
from qiskit_experiments.framework import ParallelExperiment
from qiskit_experiments.library import StateTomography
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.visualization import plot_state_city
from qiskit.primitives import Sampler
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix
import random
import h5py
import pandas as pd


# Método 1: Usar o Sampler padrão (sem backend explícito)
sampler = Sampler()  # Usa o simulador padrão automaticamente

# Método 2: Para usar um backend específico, crie uma instância do AerSimulator primeiro
backend = AerSimulator()
sampler = Sampler(options={"backend": backend})  # Novo formato pós-Qiskit 0.24

# Número de repetições
# valores_desejados = [10, 50, 100, 200, 400, 600, 800, 1000, 2500, 5000, 7500, 10000]
valores_desejados = [10000]
vec_lenght = 1

# Criar o loop for
for num_repetitions in valores_desejados:

    # Listas para armazenar resultados
    angles_list = []
    sigma_total = []
    for rep in range(1, vec_lenght+1, 1):
        
        # # Gerar ângulos aleatórios entre -pi e pi
        # alpha = random.uniform(0, 2*np.pi)
        # beta = random.uniform(0, 2*np.pi)
        # gamma = random.uniform(0, 2*np.pi)

        # w = 0.5
        # alpha = -2.82743
        # beta = -1.01722
        # gamma = 1.01722

        # # w = 0.6
        # alpha = -2.80619
        # beta = -0.882262
        # gamma = 0.882262

        # w = 0.8
        alpha =-2.82397 
        beta = -0.56959 
        gamma = 0.56959

        # # w = 1.0
        # alpha =-3.141592
        # beta =  0.0 
        # gamma = 0.0

        angles_list.append([alpha, beta, gamma])
        
        def qc_state():

            G = QuantumCircuit(2, name='G')
            G.ry(alpha, 0)
            G.cx(0,1)
            G.ry(beta, 0)
            G.ry(gamma, 1)

            #Rodando sigma_z sigma_z
            qc = QuantumCircuit(4, name='state')
            qc.append(G, [0,1])
            qc.cx(0,2)
            qc.cx(1,3)

            Bell = QuantumCircuit(2, name='Bell')
            Bell.h(0)
            Bell.cx(0,1)
            qc.append(Bell, [2,3])

            return qc
        
        sigma_x_list = []
        for x in range(3):
            qrt = QuantumRegister(4)
            # crt = ClassicalRegister(2)
            crt = ClassicalRegister(2, name='crt')
            qct = QuantumCircuit(qrt, crt, name='assemblage_tomography')
            (q0,q1,q2,q3) = qrt
            (c0,c1) = crt
            qct.append(qc_state(), [q0,q1,q2,q3]) 
            # qct.barrier()

            if x == 0:  
                # to measure in X basis
                qct.h(q2)
                # qct.measure(q2, c0)
            elif x == 1:
                # to measure in Y basis
                qct.s(2)
                qct.h(2)
                # qct.measure(2, 0)
                
                # # to measure in Z basis
                # qct.measure(2, 0)

            # # qct.if_test((gate, condition)):
            # with qct.if_test((0, 0)): #https://github.com/jonasmaziero/computacao_quantica_qiskit/blob/main/qiskit5/08_q_state_tomography.ipynb

            #     nshots = 2**13
            #     # sampler = Sampler(backend=AerSimulator())
            #     sampler = Sampler(mode=AerSimulator())
            #     avgs = []
            #     for j in range(0,3):
            #         qc_list = []
            #         if j == 0: # medida de X
            #             qct.h(q3)
            #         elif j == 1: # medida de Y
            #             qct.sdg(q3)
            #             qct.h(q3)
            #         qct.measure(q3,c1)

                    
            #         qc_list.append(q3)
            #         job = sampler.run(qc_list, shots=nshots)
            #         counts = job.result().data.c.get_counts()
            #         if '0' in counts:
            #             avg = counts['0']
            #         if '1' in counts:
            #             avg -= counts['1']
            #         avg = avg/nshots
            #         avgs.append(avg)        
            #     s0 = np.array([[1,0],[0,1]]); s1 = np.array([[0,1],[1,0]])
            #     s2 = np.array([[0,-1j],[1j,0]]); s3 = np.array([[1,0],[0,-1]]) 
            #     sigma_ax = 0.5*(s0 + avgs[0]*s1 + avgs[1]*s2 + avgs[2]*s3)
            #     print(sigma_ax)

            from qiskit_ibm_runtime import QiskitRuntimeService
            service = QiskitRuntimeService(channel="ibm_quantum",
                                        token="79df2141a7dcc26e6e146a1e2643e4b13691f5660bacf2403517fcc3d33e7a41dff500367c63dbc13c18a8740daace439dbe16d5e311a947a04712089e42823")
            backend = service.backend(name="ibm_brisbane")
            # backend = service.backend(name="ibm_sherbrooke")
            from qiskit import transpile

            sigma_a_fixed_x_list = []
            for a in range(2):
                if a == 0:  
                    # sampler = Sampler(options={"backend": AerSimulator()})
                    avgs = []
                    for j in range(3):

                        qc_tomo = qct.copy()
                        if j == 0:  # Base X
                            qc_tomo.h(3)
                        elif j == 1:  # Base Y
                            qc_tomo.sdg(3)
                            qc_tomo.h(3)
                        qc_tomo.measure([q2, q3], [c0, c1])  # Mede q2->c0 e q3->c1

                        qc_list = []
                        qc_list.append(qc_tomo.decompose().decompose())
                        # pra otimizar implementação (pra evitar erros)
                        qc_transpiled_list = transpile(qc_list, backend=backend, optimization_level=1)
                        from qiskit_ibm_runtime import SamplerV2 as Sampler
                        sampler = Sampler(mode=backend)
                        sampler.options.default_shots = num_repetitions
                        sampler.options.dynamical_decoupling.enable = True # pra diminuir decoerência
                        sampler.options.dynamical_decoupling.sequence_type = "XY4"
                        job = sampler.run(qc_transpiled_list)
                        # print(job.job_id())
                        job_id = job.job_id()
                        job = service.job(job_id)
                        counts = job.result()[0].data.crt.get_counts()
                                            
                        # job = sampler.run(qc_tomo, shots=num_repetitions)
                        # counts = job.result().quasi_dists[0].binary_probabilities()
                        
                        # Pós-seleção: filtrar c0=0
                        total_c0_0 = counts.get('00', 0) + counts.get('01', 0)
                        # total_c0_0 = counts.get('10', 0) + counts.get('11', 0)

                        if total_c0_0 > 0:
                            avg = (counts.get('00', 0) - counts.get('01', 0)) / total_c0_0
                            # avg = (counts.get('10', 0) - counts.get('11', 0)) / total_c0_0
                        else:
                            avg = 0
                        avgs.append(avg)

                    # Reconstrução da matriz densidade (condicional a c0=0)
                    s0 = np.array([[1, 0], [0, 1]])
                    s1 = np.array([[0, 1], [1, 0]])
                    s2 = np.array([[0, -1j], [1j, 0]])
                    s3 = np.array([[1, 0], [0, -1]])

                    # Reconstrução da matriz densidade (como antes)
                    sigma_a_fixed_x = 0.5 * (s0 + avgs[0]*s1 + avgs[1]*s2 + avgs[2]*s3)
                    # print(rho)

                    # state = DensityMatrix(rho)
                    # plot_state_city(state, title='Matriz Densidade de q3 (Pós-seleção q2=0)')
                    
                    sigma_a_fixed_x_list.append(sigma_a_fixed_x)

                    # # TO PRINT:
                    # # qct.draw('mpl')
                    # qct.decompose().decompose().draw('mpl')
                    # plt.show()

                elif a == 1: 
                    # sampler = Sampler(options={"backend": AerSimulator()})
                    avgs = []
                    for j in range(3):
                        qc_tomo = qct.copy()
                        if j == 0:  # Base X
                            qc_tomo.h(3)
                        elif j == 1:  # Base Y
                            qc_tomo.sdg(3)
                            qc_tomo.h(3)
                        qc_tomo.measure([q2, q3], [c0, c1])  # Mede q2->c0 e q3->c1

                        qc_list = []
                        qc_list.append(qc_tomo.decompose().decompose())
                        # pra otimizar implementação (pra evitar erros)
                        qc_transpiled_list = transpile(qc_list, backend=backend, optimization_level=1)
                        from qiskit_ibm_runtime import SamplerV2 as Sampler
                        sampler = Sampler(mode=backend)
                        sampler.options.default_shots = num_repetitions
                        sampler.options.dynamical_decoupling.enable = True # pra diminuir decoerência
                        sampler.options.dynamical_decoupling.sequence_type = "XY4"
                        job = sampler.run(qc_transpiled_list)
                        # print(job.job_id())
                        job_id = job.job_id()
                        job = service.job(job_id)
                        counts = job.result()[0].data.crt.get_counts()

                        # job = sampler.run(qc_tomo, shots=num_repetitions)
                        # counts = job.result().quasi_dists[0].binary_probabilities()
                        
                        # Pós-seleção: filtrar c0=0
                        # total_c0_0 = counts.get('00', 0) + counts.get('01', 0)
                        total_c0_0 = counts.get('10', 0) + counts.get('11', 0)

                        if total_c0_0 > 0:
                            # avg = (counts.get('00', 0) - counts.get('01', 0)) / total_c0_0
                            avg = (counts.get('10', 0) - counts.get('11', 0)) / total_c0_0
                        else:
                            avg = 0
                        avgs.append(avg)

                    # Reconstrução da matriz densidade (condicional a c0=0)
                    s0 = np.array([[1, 0], [0, 1]])
                    s1 = np.array([[0, 1], [1, 0]])
                    s2 = np.array([[0, -1j], [1j, 0]])
                    s3 = np.array([[1, 0], [0, -1]])

                    # Reconstrução da matriz densidade (como antes)
                    sigma_a_fixed_x = 0.5 * (s0 + avgs[0]*s1 + avgs[1]*s2 + avgs[2]*s3)
                    # print(rho)

                    # state = DensityMatrix(rho)
                    # plot_state_city(state, title='Matriz Densidade de q3 (Pós-seleção q2=0)')
                
                    sigma_a_fixed_x_list.append(sigma_a_fixed_x)
                

                    # # TO PRINT:
                    # # qct.draw('mpl')
                    # qct.decompose().decompose().draw('mpl')
                    # plt.show()
            sigma_x_list.append(sigma_a_fixed_x_list)
        sigma_total.append(sigma_x_list)
    # print(sigma_total)

    with h5py.File(f'IBM_tomo_sigma_werner10_shots_{num_repetitions}_test.h5', 'w') as f:
        for rep_idx, repetition in enumerate(sigma_total):
            rep_group = f.create_group(f'repetition_{rep_idx}')
            rep_group.attrs['angles'] = angles_list[rep_idx]
            
            for x_idx, x_data in enumerate(repetition):
                x_group = rep_group.create_group(f'x_{x_idx}')
                
                for a_idx, matrix in enumerate(x_data):
                    # Salvar os elementos explicitamente na ordem correta
                    element_order = [
                        matrix[0, 0], matrix[0, 1],  # Primeira linha
                        matrix[1, 0], matrix[1, 1]   # Segunda linha
                    ]
                    
                    # Salvar como arrays separados mantendo a ordem explícita
                    x_group.create_dataset(f'a_{a_idx}_elements_real', 
                                        data=[x.real for x in element_order])
                    x_group.create_dataset(f'a_{a_idx}_elements_imag', 
                                        data=[x.imag for x in element_order])
    #-------------------------------------------------------|
    # # Configuração do Sampler
    # sampler = Sampler(options={"backend": AerSimulator()})
    # avgs = []
    # for j in range(3):
    #     qc_tomo = qct.copy()
    #     if j == 0:  # Base X para q3
    #         qc_tomo.h(3)
    #     elif j == 1:  # Base Y para q3
    #         qc_tomo.sdg(3)
    #         qc_tomo.h(3)
    #     # Base Z (j=2) não requer rotação
    #     qc_tomo.measure(3, 1)  # Mede q3 -> c1

    #     job = sampler.run(qc_tomo, shots=100000)
    #     quasi_dist = job.result().quasi_dists[0]
    #     counts = quasi_dist.binary_probabilities() #retorna probabilidades normalizadas (valores entre 0 e 1, soma = 1)
    #     print(counts)

    #     # Pós-seleção: filtrar apenas c0=0 (q2=0)
    #     total_c0_0 = counts.get('00', 0) + counts.get('01', 0)  # Todos os casos onde c0=0
    #     if total_c0_0 > 0:
    #         # Probabilidade condicional P(c1=0|c0=0) e P(c1=1|c0=0)
    #         prob_c1_0_given_c0_0 = counts.get('00', 0) / total_c0_0
    #         prob_c1_1_given_c0_0 = counts.get('01', 0) / total_c0_0
    #         avg = prob_c1_0_given_c0_0 - prob_c1_1_given_c0_0
    #     else:
    #         avg = 0  # Caso não haja eventos com c0=0
    #     avgs.append(avg)

    # # Reconstrução da matriz densidade (condicional a c0=0)
    # s0 = np.array([[1, 0], [0, 1]])
    # s1 = np.array([[0, 1], [1, 0]])
    # s2 = np.array([[0, -1j], [1j, 0]])
    # s3 = np.array([[1, 0], [0, -1]])

    # rho = 0.5 * (s0 + avgs[0]*s1 + avgs[1]*s2 + avgs[2]*s3)
    # print("Matriz densidade de q3 (condicional a q2=0):\n", rho)

    # # TO PRINT:
    # # qct.draw('mpl')
    # qct.decompose().draw('mpl')
    # plt.show()

    # backend = Aer.get_backend('qasm_simulator')

    # transpiled_circuit = transpile(qc, backend)

    # qobj = assemble(transpiled_circuit)

    # job = backend.run(transpiled_circuit, shots=1024)

    # results = job.result()
    # counts_z = results.get_counts()

    # print("Counts in Z basis:", counts_z)