
import numpy as np 
from qiskit_aer import AerSimulator
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
import h5py

# Número de repetições
valores_desejados = [100000]
w = 1.0

# Criar o loop for
for num_repetitions in valores_desejados:

    # Listas para armazenar resultados
    angles_list = []
    sigma_total = []
    for rep in range(1, 2, 1):
        
        # # Gerar ângulos aleatórios entre -pi e pi
        # alpha = random.uniform(0, 2*np.pi)
        # beta = random.uniform(0, 2*np.pi)
        # gamma = random.uniform(0, 2*np.pi)

        if w == 0.5:
        # w = 0.5
            alpha = -2.82743
            beta = -1.01722
            gamma = 1.01722

        elif w == 0.6:
        # w = 0.6
            alpha = -2.80619
            beta = -0.882262
            gamma = 0.882262

        elif w == 0.8: 
        # w = 0.8
            alpha =-2.82397 
            beta = -0.56959 
            gamma = 0.56959

        elif w == 1.0:
        # w = 1.0
            alpha = np.pi
            beta = 0 
            gamma = 0
        else:
            print("angulos para w não atribuidos")
        
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
            crt = ClassicalRegister(2)
            qct = QuantumCircuit(qrt, crt, name='assemblage_tomography')
            (q0,q1,q2,q3) = qrt
            (c0,c1) = crt

            state_circuit = qc_state().decompose()  # Decompor aqui!
            qct.append(state_circuit, [q0,q1,q2,q3]) 

            if x == 0:  
                # to measure in X basis
                qct.h(q2)
            elif x == 1:
                # to measure in Y basis
                qct.s(2)
                qct.h(2)

            sigma_a_fixed_x_list = []
            for a in range(2):
                if a == 0: 
                    backend = AerSimulator()
                    avgs = []
                    for j in range(3):
                        qc_tomo = qct.decompose().copy()
                        if j == 0:  # Base X
                            qc_tomo.h(3)
                        elif j == 1:  # Base Y
                            qc_tomo.sdg(3)
                            qc_tomo.h(3)
                        qc_tomo.measure([2,3], [0,1])  

                        job = backend.run(qc_tomo, shots=num_repetitions)
                        result = job.result()
                        counts = result.get_counts() 

                        
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

                    # Reconstrução da matriz densidade 
                    sigma_a_fixed_x = 0.5 * (s0 + avgs[0]*s1 + avgs[1]*s2 + avgs[2]*s3)
                    sigma_a_fixed_x_list.append(sigma_a_fixed_x)

                    # # TO PRINT:
                    # # qct.draw('mpl')
                    # qct.decompose().decompose().draw('mpl')
                    # plt.show()

                elif a == 1: 
                    backend = AerSimulator()
                    avgs = []
                    for j in range(3):
                        qc_tomo = qct.decompose().copy()
                        if j == 0:  # Base X
                            qc_tomo.h(3)
                        elif j == 1:  # Base Y
                            qc_tomo.sdg(3)
                            qc_tomo.h(3)
                        qc_tomo.measure([2,3], [0,1])  

                        job = backend.run(qc_tomo, shots=num_repetitions)
                        result = job.result()
                        counts = result.get_counts()
                           
                        # Pós-seleção: filtrar c0=0
                        # total_c0_0 = counts.get('00', 0) + counts.get('01', 0)
                        total_c0_0 = counts.get('10', 0) + counts.get('11', 0)

                        if total_c0_0 > 0:
                            # avg = (counts.get('00', 0) - counts.get('01', 0)) / total_c0_0
                            avg = (counts.get('10', 0) - counts.get('11', 0)) / total_c0_0
                        else:
                            avg = 0
                        avgs.append(avg)

                    # Reconstrução da matriz densidade (condicional a c0=1)
                    s0 = np.array([[1, 0], [0, 1]])
                    s1 = np.array([[0, 1], [1, 0]])
                    s2 = np.array([[0, -1j], [1j, 0]])
                    s3 = np.array([[1, 0], [0, -1]])

                    # Reconstrução da matriz densidade 
                    # print(avgs[0],avgs[1],avgs[2])
                    sigma_a_fixed_x = 0.5 * (s0 + avgs[0]*s1 + avgs[1]*s2 + avgs[2]*s3)
                    sigma_a_fixed_x_list.append(sigma_a_fixed_x)
                

                    # # TO PRINT:
                    # # qct.draw('mpl')
                    # qct.decompose().decompose().draw('mpl')
                    # plt.show()
            sigma_x_list.append(sigma_a_fixed_x_list)
        sigma_total.append(sigma_x_list)
    # print(sigma_total)

    with h5py.File(f'../dat/sigma_werner{w}num_shots{num_repetitions}.h5', 'w') as f:
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