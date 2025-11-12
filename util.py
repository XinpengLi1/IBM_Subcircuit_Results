import networkx as nx
import numpy as np
import pickle
import json
from qiskit import qpy
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator
from qiskit_aer import Aer
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import Fake127QPulseV1

import numpy as np
import networkx as nx
from itertools import combinations, groupby
import random
from itertools import permutations

from qiskit_aer import AerSimulator
from scipy.optimize import minimize

import copy

# ============================================================
# New Optcut2 code
# ============================================================

def state_normalization(v):
    shape = v.shape
    if len(shape) == 2:
        # "Vector {i} has shape (1, -1) (row vector)")
        if v[0][0] !=0:
            v = v/v[0][0]
    elif len(shape) == 1:
        if v[0] !=0:
            v = v/v[0]
    else:
        print(f"Vector does not match any of (1, -1), (-1, 1), or (-1)")
    v = v/np.sqrt(np.sum(np.abs(np.square(v))))
    return v

def unitary_normalization(Unitary):
    standard_identity = np.conjugate(Unitary.T) @ Unitary
    lamba = np.trace(standard_identity)/standard_identity.shape[0]
    assert lamba!=0
    return Unitary/np.sqrt(lamba)

def compute_unitary_U(A, B):
    """
    Computes the unitary matrix U such that U A = B.
    
    Parameters:
    A (ndarray): A complex matrix of size (2^n, 2^m).
    B (ndarray): A complex matrix of size (2^n, 2^m).
    
    Returns:
    U (ndarray): The unitary matrix of size (2^m, 2^m) satisfying U A = B.
    """
    if len(A.shape) == 1:
        A = A.reshape(-1,1)
    
    if len(B.shape) == 1:
        B = B.reshape(-1,1)

    # Compute C = A^H B
    C = B @ np.conj(A.T)

    # Perform Singular Value Decomposition on C
    U_svd, _ , Vh_svd = np.linalg.svd(C)
    # V_svd is the conjugate transpose of Vh_svd
    # V_svd = Vh_svd.conj().T

    # U_svd = U_svd.conj().T
    
    # Compute the unitary matrix U = V_svd U_svd^H
    # U = V_svd @ U_svd.conj().T
    U = U_svd @ Vh_svd
    
    return unitary_normalization(U)

def tensor_arrays(array_dict, total_systems):
    """
    Construct a matrix representing the action of the arrays on their respective systems,
    extended to the total Hilbert space by replacing missing systems with identity matrices.

    # Example Usage:
    array1 = np.array([1,2,3,4])  # Vector acting on system 1
    array2 = np.array([1 ,0])
    array_dict = { (0,2): array1,(1,): array2}
    total_systems = 3
    result = tensor_arrays(array_dict, total_systems)
    Result: [[1. 2. 0. 0. 3. 4. 0. 0.]]

    Parameters:
    - array_dict: dict where keys are tuples of system indices and values are arrays (vectors).
    - total_systems: int, total number of systems.

    Returns:
    - A numpy array representing the combined action.
    """
    # Determine the total dimension
    total_dim = 2 ** total_systems

    # Initialize the total operator as None
    total_operator = None

    for key, vector in array_dict.items():
        array_dict[key] = vector.flatten()

    # Identity matrix for missing systems
    identity = np.eye(2)

    # Compute the dimension of the systems the arrays act upon
    array_systems = set()
    for systems in array_dict.keys():
        array_systems.update(systems)
    array_systems = sorted(array_systems)
    system_size = []
    identity_systems = []
    for isys , i in enumerate(range(total_systems)):
        if i in array_systems:
            system_size.append(1)
        else:
            system_size.append(2)
            identity_systems.append(isys)
    array_dim = 2 ** (total_systems-len(array_systems))

    # Initialize the operator with the correct shape
    total_operator = np.zeros((total_dim, array_dim),dtype= complex)

    # Generate all possible basis states for the total systems
    for i in range(total_dim):
        # Represent the basis state as a bit string
        bits = format(i, '0{}b'.format(total_systems))
        # Extract bits corresponding to the systems the arrays act upon
        identity_bits = ''.join(bits[s] for s in identity_systems)
        # Convert bits to indices
        row_idx = 0
        for p,b in enumerate(identity_bits):
            row_idx +=  int(b) * 2**int(len(identity_bits)-p-1)
            # print(system_size[p],int(b),system_size[p]**int(b),row_idx)
        col_idx = i
        # print(row_idx,col_idx,identity_bits,bits)
        # Compute the product of array elements
        value = 1
        for systems, array in array_dict.items():
            # Extract bits for these systems
            bits_subset = ''.join(bits[ s] for s in systems)
            idx = int(bits_subset, 2)
            value *= array[idx]
        total_operator[col_idx,row_idx] = value

    return total_operator

def count_gate(qc: QuantumCircuit):
    count_gate_dict = {q: 0 for q in qc.qubits}
    for gate in qc.data:
        for qubit in gate.qubits:
            count_gate_dict[qubit]+=1
    return count_gate_dict

def remove_unused_wire(qc: QuantumCircuit):
    gate_count = count_gate(qc)
    org_qubit_to_new_index_mapping = {}
    i = 0
    for qubit, count in gate_count.items():
        if count != 0:
            org_qubit_to_new_index_mapping[qubit.index] = i
            i+=1

    new_qc = QuantumCircuit(i)
    for gate in qc.data:
        new_qc.append(gate.operation, [org_qubit_to_new_index_mapping[q.index] for q in gate.qubits])

    return new_qc

def reverse_qubit_order(circuit):
    num_qubits = circuit.num_qubits
    # Create a new quantum circuit with the same number of qubits
    new_circuit = QuantumCircuit(num_qubits)
    
    # Loop over the instructions of the original circuit
    for instr, qargs, cargs in circuit.data:
        # Reverse the qubit index in qargs (map qubit 0 -> n-1, 1 -> n-2, etc.)
        new_qargs = [(num_qubits - 1 - q.index) for q in qargs]
        # Apply the same instruction to the reversed qubits
        new_circuit.append(instr, new_qargs, cargs)
    
    return new_circuit

def circuit_to_unitary (qc: QuantumCircuit) -> np.array:
    new_qc = reverse_qubit_order(copy.deepcopy(qc))
    # new_qc = remove_unused_wire(new_qc)

    full_operator = Operator(new_qc)
    extended_U = full_operator.data 
    return np.array(extended_U).T


def unitary_to_circuit(Unitary):

    num_qubit = int(np.log2(Unitary.shape[0]))

    qubit_list = [num_qubit-i-1 for i in range(num_qubit)]

    new_replace_circuit = QuantumCircuit(num_qubit)

    unitary_gate = UnitaryGate(Unitary)

    new_replace_circuit.append(unitary_gate,qubit_list)

    new_replace_circuit = new_replace_circuit.decompose()

    return new_replace_circuit

def counts_to_probability(counts_dict):
    """
    Convert a counts dictionary to a probability numpy array.
    
    Args:
    counts_dict (dict): A dictionary with keys as bitstrings and values as counts.
    
    Returns:
    numpy.ndarray: A numpy array representing the probabilities in the order from '00000' to '11111'.
    """
    num_qubits = max(len(bitstring) for bitstring in counts_dict.keys())  # Determine the number of qubits
    num_outcomes = 2 ** num_qubits  # Calculate the number of possible outcomes
    total_counts = sum(counts_dict.values())  # Total number of counts (shots)
    
    # Initialize the probability array
    prob_array = np.zeros(num_outcomes)
    
    # Fill the probability array
    for i in range(num_outcomes):
        # Format the index as a bitstring with leading zeros
        bitstring = format(i, f'0{num_qubits}b')
        # Get the count for the bitstring, defaulting to 0 if not present
        count = counts_dict.get(bitstring, 0)
        # Calculate the probability
        prob_array[i] = count / total_counts
    
    return prob_array

def evaluate_circuit(qc,shots):
    # Step 2: Execute the circuit using the qasm_simulator
    circuit = copy.deepcopy(qc)
    if not any(op.name == 'measure' for op, _, _ in qc.data):
        circuit.measure_all()
    # circuit.measure_all()
    fake_backend = AerSimulator()

    job = fake_backend.run(circuit, shots = shots)

    results = job.result()

    counts = results.get_counts(circuit )

    return counts_to_probability(counts)

def evaluate_circuit_with_noise(qc,shots):
    # Step 2: Execute the circuit using the qasm_simulator
    circuit = copy.deepcopy(qc)
    if not any(op.name == 'measure' for op, _, _ in qc.data):
        circuit.measure_all()
    # circuit.measure_all()
    fake_backend = Fake127QPulseV1()

    transpiled_qc = transpile(circuit , fake_backend)

    n_gate=count_two_qubit_gates(transpiled_qc)

    job = fake_backend.run(transpiled_qc, shots = shots)

    result = job.result()

    counts = result.get_counts(circuit)

    return counts_to_probability(counts), n_gate

def partial_trace(rho, keep):
    """
    Compute the partial trace over specified qubits.

    Parameters:
    - rho: Density matrix of the composite system (numpy array of shape (2^n, 2^n)).
    - keep: List of qubit indices to keep (e.g., [0, 3]).
    - dims: List of dimensions of each subsystem (e.g., [2, 2, 2, 2] for 4 qubits).

    Returns:
    - Reduced density matrix after tracing out the qubits not in 'keep'.
    """
    n = int(np.log2(rho.shape[1]))
    dims = [2] * n
    trace = [i for i in range(n) if i not in keep]  # Qubits to trace out

    # Reshape rho into a tensor with 2n indices
    rho = rho.reshape([dims[i] for i in range(n)] + [dims[i] for i in range(n)])

    # Permute the axes to bring the qubits to keep to the front
    keep_indices = keep
    trace_indices = trace
    permute_order = keep_indices + trace_indices + [i + n for i in keep_indices + trace_indices]
    rho = rho.transpose(permute_order)

    # Reshape to group the traced and kept indices
    dim_keep = np.int(np.prod([dims[i] for i in keep]))
    dim_trace = np.int(np.prod([dims[i] for i in trace]))
    rho = rho.reshape([dim_keep, dim_trace, dim_keep, dim_trace])

    # Perform partial trace over the traced qubits
    rho_reduced = np.trace(rho, axis1=1, axis2=3)

    return rho_reduced


# ============================================================
# Old Optcut code
# ============================================================


def get_cut_position_from_cuts_solution(Old_cuts_solution):
    position = []
    for circuit_qubit,subcircuits_list in Old_cuts_solution['complete_path_map'].items():
        if len(subcircuits_list) ==1:
            continue
        else:
            depth = 0
            for subcircuit_dict in subcircuits_list[:-1]:
                subcircuit_idx = subcircuit_dict['subcircuit_idx']
                subcircuit_qubit = subcircuit_dict['subcircuit_qubit']
                for instr, qargs, cargs in Old_cuts_solution['subcircuits'][subcircuit_idx].data:
                    if subcircuit_qubit in qargs:
                        depth +=1
                position.append((circuit_qubit,depth))
    return position

def count_to_array(result,num_qubits):   
    to_list_result = []
    for probs in result:
        prob_array = np.zeros(2**num_qubits)
        for key, prob in probs.items():
            prob_array[int(key,2)]=prob
        to_list_result.append(prob_array/sum(prob_array))
    return to_list_result

def calculate_expectation_value(distribution, measure_indices, expectation_indices):
    """
    Recovers the original probability distribution based on the final layout.

    Parameters:
    - final_layout (list): A list indicating the mapping of original qubits to their new indices.
    - transpiled_prob (np.array): The probability distribution obtained from executing the transpiled circuit.

    Returns:
    - np.array: The recovered original probability distribution.
    """
    num_qubits = int(np.log2(len(distribution)))  # For a 6-qubit system

    # Initialize an array for the recovered probabilities
    
    trace_indices = measure_indices+expectation_indices
    original_prob = np.zeros(2**(num_qubits-len(trace_indices)))

    for state_index, prob in enumerate(distribution):
        # Convert the outcome index to its binary representation
        bin_outcome = format(state_index, f'0{num_qubits}b')
        
        # Rearrange the bits according to the inverse layout
        rearranged_outcome = ''.join(bin_outcome[i] for i in range(num_qubits) if i not in trace_indices)
        
        # Convert the rearranged binary outcome back to an index
        if rearranged_outcome:
            new_idx = int(rearranged_outcome, 2)
        else:
            new_idx= 0
        
        # Update the original probability distribution
        contribution = 1
        for qubit_index in expectation_indices:
            if bin_outcome[qubit_index] == '1':
                contribution *= -1  # Flip the contribution for a |1> state 
        original_prob[new_idx] += contribution*prob
    
    return original_prob

def reorder_qubits(distribution, new_order):
    n_qubits = int(np.log2(len(distribution)))
    # Check if the new order is valid
    if len(new_order) != n_qubits or sorted(new_order) != list(range(n_qubits)):
        raise ValueError("Invalid new order for qubits")
    
    # Create an index map from old to new indices
    index_map = np.zeros(2**n_qubits, dtype=int)
    for index in range(2**n_qubits):
        # Get binary representation of the index
        bin_index = np.binary_repr(index, width=n_qubits)
        # Rearrange the binary representation according to new qubit order
        new_bin_index = ''.join([bin_index[new_order[i]] for i in range(n_qubits)])
        # Convert the new binary index back to an integer
        new_index = int(new_bin_index, 2)
        # Map the old index to the new index
        index_map[index] = new_index
    
    # Reorder the distribution according to the new index map
    new_distribution = distribution[index_map]
    return new_distribution

from random import randint
def generate_random_adder_benchmark(n_bits):
    """
    Generates a quantum circuit for an n-bit adder with random inputs.

    Args:
        n_bits (int): The number of bits for the adder.

    Returns:
        QuantumCircuit: A quantum circuit implementing the adder with random inputs.
    """
    num_qubits = 2 * n_bits + 1  # Two n-bit numbers + carry bit
    qc = QuantumCircuit(num_qubits, n_bits + 1)  # Include classical bits for measurement

    # Generate random inputs for |A> and |B>
    A = randint(0, 2**n_bits - 1)
    B = randint(0, 2**n_bits - 1)
    
    # Encode inputs A and B into the quantum circuit
    for i in range(n_bits):
        if (A >> i) & 1:
            qc.x(i)  # Set bit i of A
        if (B >> i) & 1:
            qc.x(i + n_bits)  # Set bit i of B

    # Ripple-Carry Adder Logic
    for i in range(n_bits):
        # Carry
        qc.ccx(i, i + n_bits, 2 * n_bits)
        # Sum
        qc.cx(i, i + n_bits)
        qc.cx(i + n_bits, i)
        # Update Carry for next stage
        if i < n_bits - 1:
            qc.cx(2 * n_bits, i + n_bits + 1)

    # Measure the sum and carry bits
    for i in range(n_bits):
        qc.measure(i, i)  # Measure sum
    qc.measure(2 * n_bits, n_bits)  # Measure carry-out

    return qc, A, B

def create_qaoa_circuit(graph, p=1):
    """
    Creates a QAOA circuit for a given graph with random parameters.

    Parameters:
    graph (networkx.Graph): The input graph.
    p (int): The number of QAOA layers.

    Returns:
    QuantumCircuit: A QAOA circuit for the graph.
    """
    # Initialize circuit with as many qubits as there are nodes in the graph
    n = graph.number_of_nodes()
    circuit = QuantumCircuit(n)
    
    # Create parameters for the circuit
    gamma = [Parameter(f'γ_{i}') for i in range(p)]
    beta = [Parameter(f'β_{i}') for i in range(p)]

    # Apply Hadamard gates to all qubits
    circuit.h(range(n))

    for layer in range(p):
        # Apply problem unitary
        for u, v in graph.edges():
            u = int(u)
            v = int(v)
            # circuit.cx(u, v)
            # circuit.rz(2 * gamma[layer], v)
            # circuit.cx(u, v)
            circuit.rzz(2 * gamma[layer], u, v)
        
        # Apply mixing unitary
        for qubit in range(n):
            circuit.rx(2 * beta[layer], qubit)

    # Set random values for gamma and beta parameters
    param_values = {gamma[i]: np.random.uniform(0, 2*np.pi) for i in range(p)}
    param_values.update({beta[i]: np.random.uniform(0, 2*np.pi) for i in range(p)})
    circuit = circuit.assign_parameters(param_values)

    return circuit

def hamiltonian_value(bitstring, G):
    """ Calculate the Hamiltonian value for a given bitstring. """
    value = 0
    num_qubit=len(bitstring)
    for u, v in G.edges():
        u = int(u)
        v = int(v)
        if bitstring[u] != bitstring[v]:  # Check if the edge is cut
            value += 1
    return value

def objective_function(output_distribution, G):
    """ Calculate the expectation value of the Hamiltonian over the output distribution.
    
    :param output_distribution: dict, where keys are bitstrings and values are their counts
    :param G: NetworkX graph
    :return: float, expectation value of the Hamiltonian
    """
    total_value = 0
    total_counts = sum(output_distribution.values())  # Total number of samples

    for bitstring, count in output_distribution.items():
        # Calculate the Hamiltonian value for each bitstring
        H_value = hamiltonian_value(bitstring, G)
        # Weighted sum of Hamiltonian values
        total_value += H_value * (count / total_counts)
    
    return total_value

def create_trained_qaoa_circuit(sample_graph, p=1):
    n = sample_graph.number_of_nodes()
    qaoa_circuit = QuantumCircuit(n)
    # Create parameters for the circuit
    gamma = [Parameter(f'γ_{i}') for i in range(p)]
    beta = [Parameter(f'β_{i}') for i in range(p)]

    # Apply Hadamard gates to all qubits
    qaoa_circuit.h(range(n))

    for layer in range(p):
        # Apply problem unitary
        for u, v in sample_graph.edges():
            u = int(u)
            v = int(v)
            qaoa_circuit.rzz(2 * gamma[layer], u, v)
        
        # Apply mixing unitary
        for qubit in range(n):
            qaoa_circuit.rx(2 * beta[layer], qubit)
    
    # Execution function
    def execute_circuit_and_calculate_objective(params, qc, G, backend, shots):
        # Create the QAOA circuit with the given parameters
        param_values = {gamma[i]: params[2*i] for i in range(p)}
        param_values.update({beta[i]: params[2*i+1] for i in range(p)})
        bc = qc.assign_parameters(param_values)
        bc.measure_all()
        result = backend.run(bc, shots=shots).result()
        counts = result.get_counts(0)
        
        # Calculate the objective function value
        objective_value = objective_function(counts, G)
        return -objective_value

    backend_noise_free = AerSimulator()
    x0 = 2 * np.pi * np.random.rand(2*p)
    res = minimize(execute_circuit_and_calculate_objective, x0, args=(qaoa_circuit, sample_graph, backend_noise_free,200000), method="L-BFGS-B")

    param_values = {gamma[i]: res['x'][2*i]  for i in range(p)}
    param_values.update({beta[i]: res['x'][2*i+1]  for i in range(p)})
    qc = qaoa_circuit.assign_parameters(param_values)

    return qc

def calculate_fidelity(P, Q):
    # Ensure both distributions sum to 1
    P = np.array(P) / np.sum(P)
    Q = np.array(Q) / np.sum(Q)
    
    # Calculate the square root of the product of corresponding probabilities
    sqrt_product = np.sqrt(P * Q)
    
    # Sum these values and square the result to obtain the fidelity
    fidelity = np.sum(sqrt_product)**2
    
    return fidelity

import copy
def store_solution(solution,path):
    for key, value in solution.items():
        if key =='subcircuits':
            for i, c in enumerate(value):
                with open(path+key+'/subcircuit'+str(i)+'.qpy', 'wb') as fd:
                    qpy.dump(c, fd)
            with open(path+'num_subcircuits'+'.txt', 'w') as file:
                json.dump(len(value), file)
        elif key =='complete_path_map':
            store_form={}
            for i, map in value.items():
                reform_list=[]
                for submap in map:
                    new_map=copy.deepcopy(submap)
                    if type(submap['subcircuit_qubit']) is not int:
                        new_map['subcircuit_qubit'] = submap['subcircuit_qubit'].index #!!!
                    reform_list.append(new_map)
                if type(i) is int:
                    store_form[i]=reform_list
                else:
                    store_form[i.index]=reform_list
            with open(path+key+'.pkl', 'wb') as fp:
                pickle.dump(store_form, fp)
        # with open('/Users/xinpengli/Desktop/CutOpt/Result/QAOA/8node_1/CutQC_solution/'+'complete_path_map'+'.pkl', 'rb') as fp:
        #         person = pickle.load(fp)
        #         print(person)
        elif key =='counter':
            with open(path+key+'.pkl', 'wb') as fp:
                pickle.dump(value, fp)
        elif key == 'num_cuts' or key == 'classical_cost' or key =='max_subcircuit_width':
            # Writing to a file
            with open(path+key+'.txt', 'w') as file:
                json.dump(value, file)

def read_solution(path,qc):
    solution={}
    key_list = ['subcircuits','complete_path_map','counter','num_cuts','classical_cost','max_subcircuit_width']
    
    with open(path+'num_subcircuits'+'.txt', 'r') as fp:
        num_subcircuits = json.load(fp)
    solution['subcircuits'] = []
    for i in range(num_subcircuits):
        with open(path+'subcircuits'+'/subcircuit'+str(i)+'.qpy', 'rb') as fd:
            circuit = qpy.load(fd)[0]
            solution['subcircuits'].append(circuit)

    for key in key_list:
        if key =='complete_path_map':
            with open(path+key+'.pkl', 'rb') as fp:
                reformed_complete_path_map = pickle.load(fp)
            complete_path_map = {}
            for qubit_index, map in reformed_complete_path_map.items():
                for i, sub_map in enumerate(map):
                    map[i]['subcircuit_qubit'] = solution['subcircuits'][sub_map['subcircuit_idx']].qubits[sub_map['subcircuit_qubit']]
                complete_path_map[qc.qubits[qubit_index]] = map  # Step 2: Assign the value to the new key
            solution[key] = complete_path_map
        elif key =='counter':
            with open(path+key+'.pkl', 'rb') as fp:
                counter = pickle.load(fp)
            solution[key] = counter
        elif key == 'num_cuts' or key == 'classical_cost' or key =='max_subcircuit_width':
            with open(path+key+'.txt', 'r') as file:
                value  = json.load(file)
            solution[key] = value
    return solution

def random_connected_graph(n, p):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi
    graph, but enforcing that the resulting graph is conneted
    """
    edges = combinations(range(n), 2)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(n, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge, weight = random.randint(0, 10) / 10)
        for e in node_edges:
            if random.random() < p:
                G.add_edge(*e, weight = random.randint(0, 10) / 10)
    return G

def count_gates(circuit: QuantumCircuit):
    """
    Counts the total number of gates, single-qubit gates, and multi-qubit gates in a given circuit.
    
    Parameters:
        circuit (QuantumCircuit): The quantum circuit to analyze.
    
    Returns:
        tuple: A tuple containing the total number of gates, number of single-qubit gates, and number of multi-qubit gates.
    """
    total_gates = 0
    single_qubit_gates = 0
    multi_qubit_gates = 0

    # Iterate through each gate in the circuit data
    for instr, qargs, cargs in circuit.data:
        # Check if the instruction is a gate (ignores barriers, measurements, etc.)
        if instr.name not in ['barrier', 'measure']:
            total_gates += 1
            # Check the number of qubits the gate acts on
            if len(qargs) == 1:
                single_qubit_gates += 1
            elif len(qargs) > 1:
                multi_qubit_gates += 1

    return total_gates, single_qubit_gates, multi_qubit_gates

def count_two_qubit_gates(circuit):
    count = 0
    # Iterate through all operations in the circuit
    for instr, qargs, _ in circuit.data:
        # Check if the operation involves exactly two qubits
        if len(qargs) == 2:
            count += 1
    return count

def count_two_qubit_gates_transpiled(circuit):
    fake_backend = Fake127QPulseV1()
    transpiled_qc = transpile(circuit , fake_backend)
    return count_two_qubit_gates(transpiled_qc)

def remove_idle_wires(qc: QuantumCircuit):
    qc_out = qc.copy()
    gate_count = count_gate(qc_out)
    for qubit, count in gate_count.items():
        if count == 0:
            qc_out.qubits.remove(qubit)
    return qc_out

#################### Noise evaluate ####################
from Optpass import opt_pass

def generate_binary_strings(n):
    """
    Generates all possible n-length binary strings.

    Parameters:
        n (int): The length of the binary strings.

    Returns:
        List[str]: A list containing all binary strings of length n.
    """
    binary_strings = []
    for i in range(2 ** n):
        # Convert integer to binary string and remove '0b' prefix
        binary_str = bin(i)[2:]
        # Pad the string with leading zeros
        binary_str = binary_str.zfill(n)
        # Append to the list
        binary_strings.append(binary_str)
    return binary_strings

def prob_normalization(prob):
    return prob/sum(prob)

def reverse_circuit(circuit):
    """
    Reverses the order of gates in a QuantumCircuit.

    Parameters:
        circuit (QuantumCircuit): The original quantum circuit to reverse.

    Returns:
        QuantumCircuit: A new quantum circuit with gates in reversed order.
    """
    # Create a new circuit with the same number of qubits and classical bits
    reversed_circuit = QuantumCircuit(circuit.num_qubits)

    # Reverse the circuit data (gates and instructions)
    for instruction in reversed(circuit.data):
        reversed_circuit.append(instruction.operation, instruction.qubits)

    return reversed_circuit

def measure_opt(circuit, measure_qubits):
    _circuit = copy.deepcopy(circuit)
    reversed_circuit = reverse_circuit(_circuit)
    reversed_opt_circuit = opt_pass(reversed_circuit, measure_qubits)
    opt_circuit = reverse_circuit(reversed_opt_circuit)
    return opt_circuit

def run_subcircuits_with_meas_opt(subcircuits, subcircuit_meas_basis_list, biased_bases):
    subcircuits_prob = [] # a list of prob with subcircuits
    n_gates = 0
    for subcircuit, meas_basis, bb in zip (subcircuits, subcircuit_meas_basis_list, biased_bases):

        meas_basis_qubits_indices = []
        for i in range(len(meas_basis)):
            if meas_basis[i] == 'I' or meas_basis[i] == bb[i]:
                meas_basis_qubits_indices.append(i)
                
        # If there is no measurement in subcircuit
        if len(meas_basis_qubits_indices) == 0:
            opt_subcircuit = opt_pass(subcircuit, subcircuit.qubits)
            subcircuit_prob,n_gate = evaluate_circuit_with_noise(opt_subcircuit, 2**(20))
            subcircuits_prob.append(subcircuit_prob)
            n_gates += n_gate
        else:
            # Generate binary string. each string eg: "01001" represent what state is measure right here eg: "IXIIX".
            # when we extract the result from prob, we simply check all position that is 0. 
            # eg for 8 qubits "bb0b0000" on the position of string "01001" should be all 0, cause we apply x gate before measurement
            binary_list = generate_binary_strings(len(meas_basis_qubits_indices))
            subcircuit_prob = np.zeros(2**(subcircuit.num_qubits))
            for string in binary_list: # string is state on bias meas position

                #Copy subcircuit and add x gate for all string that is 1
                string_subcircuit = copy.deepcopy(subcircuit)
                for bit, qubit in  zip(string, meas_basis_qubits_indices):
                    if bit == '1':
                        string_subcircuit.x(qubit)
                # Optimize string circuit
                opt_subcircuit = opt_pass(string_subcircuit, [q for q in string_subcircuit.qubits])
                meas_basis_qubits = [ opt_subcircuit.qubits[i] for i in meas_basis_qubits_indices]
                measure_opt_string_circuit = measure_opt(opt_subcircuit, meas_basis_qubits)
                # Run this opt_ed circuit
                string_subcircuit_prob,n_gate = evaluate_circuit_with_noise(measure_opt_string_circuit, 2**(20))
                n_gates += n_gate
                # Contruct subcircuit_prob, 
                # if global string has sample bits on bias meas position as string, 
                # then assigning with correct prob, that is as string with all '0' in this positions for string_subcircuit_prob.
                for global_string in generate_binary_strings(subcircuit.num_qubits):
                    subcircuit_prob_sub_string = ''.join([global_string[i] for i in meas_basis_qubits_indices])
                    string_subcircuit_prob_string = ''.join(
                        ['0' if idx in meas_basis_qubits_indices else b for idx, b in enumerate(global_string)])[::-1]  
                    prob_index = int(global_string[::-1], 2)
                    string_subcircuit_prob_index = int(string_subcircuit_prob_string, 2)
                    if subcircuit_prob_sub_string == string:
                        subcircuit_prob[prob_index] = string_subcircuit_prob[string_subcircuit_prob_index]
            subcircuits_prob.append(prob_normalization(subcircuit_prob))
    return subcircuits_prob, n_gates/ (2**len(meas_basis_qubits_indices))