from circuit_knitting.cutting.cutqc.wire_cutting import _generate_metadata
from multiprocessing.pool import ThreadPool
from typing import Sequence, Any
# from Deferred_measurement_Opt import Deferred_measurement_Opt
from qiskit_ibm_runtime import Options
from circuit_knitting.cutting.cutqc.wire_cutting_evaluation import  mutate_measurement_basis,measure_prob
#modify_subcircuit_instance,
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.circuit.library.standard_gates import HGate, SGate, SdgGate, XGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit import QuantumCircuit
from qiskit.primitives import BaseSampler, Sampler as TestSampler
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session, Options
import numpy as np
import copy
from circuit_knitting.utils.conversion import quasi_to_real
from circuit_knitting.utils.metrics import (
    chi2_distance,
    MSE,
    MAPE,
    cross_entropy,
    HOP,
)
from qiskit_aer import Aer
from qiskit import  QuantumCircuit, transpile
# from qiskit.providers.fake_provider import FakeManila
from itertools import permutations

def get_circuit_to_run(cuts):
    _, _, subcircuit_instances = _generate_metadata(cuts)
    subcircuits=cuts["subcircuits"]
    circuits_to_run = []
    subcircuit_label = []
    cutting_downstream_qubits_list=[]
    measurement_qubits_list=[]

    for subcircuit_idx, subcircuit in enumerate(subcircuits):
        subcircuit_instance_probs = []
        subcircuit_instance = subcircuit_instances[subcircuit_idx]
        # For each circuit associated with a given subcircuit
        for init_meas in subcircuit_instance:
            subcircuit_instance_idx = subcircuit_instance[init_meas]
            if subcircuit_instance_idx not in subcircuit_instance_probs:
                modified_subcircuit_instance, cutting_downstream_qubits,measurement_qubits = modify_subcircuit_instance(
                    subcircuit=subcircuit,
                    init=init_meas[0],
                    meas=tuple(init_meas[1]),
                )
                circuits_to_run.append(modified_subcircuit_instance)
                subcircuit_label.append(subcircuit_idx)
                cutting_downstream_qubits_list.append(cutting_downstream_qubits)
                measurement_qubits_list.append(tuple(init_meas[1]))
                mutated_meas = mutate_measurement_basis(meas=tuple(init_meas[1]))
                for meas in mutated_meas:
                    mutated_subcircuit_instance_idx = subcircuit_instance[
                        (init_meas[0], meas)
                    ]
                    # Set a placeholder in the probability dict to prevent duplicate circuits to the Sampler
                    subcircuit_instance_probs.append(mutated_subcircuit_instance_idx)
    return circuits_to_run, subcircuit_label ,cutting_downstream_qubits_list, measurement_qubits_list

def pre_recontruction(circuits_probs, cuts):
    _, _, subcircuit_instances = _generate_metadata(cuts)
    subcircuits=cuts["subcircuits"]
    all_subcircuit_instance_probs = {}

    for subcircuit_idx, subcircuit in enumerate(subcircuits):
            subcircuit_instance_probs = {}
            unique_subcircuit_check = {}
            subcircuit_instance = subcircuit_instances[subcircuit_idx]
            i = 0
            for init_meas in subcircuit_instance:
                subcircuit_instance_idx = subcircuit_instance[init_meas]
                if subcircuit_instance_idx not in unique_subcircuit_check:
                    prob = circuits_probs[subcircuit_idx][i]
                    subcircuit_inst_prob = prob
                    i = i + 1
                    mutated_meas = mutate_measurement_basis(meas=tuple(init_meas[1]))
                    for meas in mutated_meas:
                        measured_prob = measure_prob(
                            unmeasured_prob=subcircuit_inst_prob, meas=meas
                        )
                        mutated_subcircuit_instance_idx = subcircuit_instance[
                            (init_meas[0], meas)
                        ]
                        subcircuit_instance_probs[
                            mutated_subcircuit_instance_idx
                        ] = measured_prob
                        unique_subcircuit_check[mutated_subcircuit_instance_idx] = True
            all_subcircuit_instance_probs[subcircuit_idx] = subcircuit_instance_probs
    return all_subcircuit_instance_probs

def reorder_list(original_list, order):
    return [original_list[i] for i in order]


def modify_subcircuit_instance(
    subcircuit: QuantumCircuit, init: tuple[str, ...], meas: tuple[str, ...]
) -> tuple[QuantumCircuit, list, list]:
    """
    Modify the initialization and measurement bases for a given subcircuit.

    Args:
        subcircuit: The subcircuit to be modified
        init: The current initializations
        meas: The current measement bases

    Returns:
        The updated circuit, modified so the initialziation
        and measurement operators are all in the standard computational basis

    Raises:
        Exeption: One of the inits or meas's are not an acceptable string
    """
    subcircuit_dag = circuit_to_dag(subcircuit)
    subcircuit_instance_dag = copy.deepcopy(subcircuit_dag)

    # Select cutting downstream qubits(wires)
    cutting_downstream_qubits=[]
    # Select measurement qubits(wires)
    measurement_qubits=[]

    for i, x in enumerate(init):
        q = subcircuit.qubits[i]
        if x == "zero":
            continue
        elif x == "one":
            subcircuit_instance_dag.apply_operation_front(
                op=XGate(), qargs=[q], cargs=[]
            )
            cutting_downstream_qubits.append(q)
        elif x == "plus":
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
            cutting_downstream_qubits.append(q)
        elif x == "minus":
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=XGate(), qargs=[q], cargs=[]
            )
            cutting_downstream_qubits.append(q)
        elif x == "plusI":
            subcircuit_instance_dag.apply_operation_front(
                op=SGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
            cutting_downstream_qubits.append(q)
        elif x == "minusI":
            subcircuit_instance_dag.apply_operation_front(
                op=SGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=XGate(), qargs=[q], cargs=[]
            )
            cutting_downstream_qubits.append(q)
        else:
            raise Exception("Illegal initialization :", x)
    for i, x in enumerate(meas):
        q = subcircuit.qubits[i]
        if x == "I" or x == "comp":
            continue
        elif x == "X":
            subcircuit_instance_dag.apply_operation_back(
                op=HGate(), qargs=[q], cargs=[]
            )
            measurement_qubits.append(q)
        elif x == "Y":
            subcircuit_instance_dag.apply_operation_back(
                op=SdgGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_back(
                op=HGate(), qargs=[q], cargs=[]
            )
            measurement_qubits.append(q)
        else:
            raise Exception("Illegal measurement basis:", x)
    subcircuit_instance_circuit = dag_to_circuit(subcircuit_instance_dag)

    return subcircuit_instance_circuit, cutting_downstream_qubits ,measurement_qubits

def run_subcircuits_using_sampler(
    subcircuits: Sequence[QuantumCircuit],
    sampler: BaseSampler,
) -> list[np.ndarray]:
    """
    Execute the subcircuit(s).

    Args:
        subcircuit: The subcircuits to be executed
        sampler: The Sampler to use for executions

    Returns:
        The probability distributions
    """

    """replaced code start"""
    # Original code below~!
    # for subcircuit in subcircuits:
    #     if subcircuit.num_clbits == 0:
    #         subcircuit.measure_all()
    for index, subcircuit in enumerate(subcircuits):
        if subcircuit.num_clbits == 0:
            subcircuit.measure_all()
            # # Extract all the qubits has not been applied 'measure'
            # measure_missing_qubit=[i for i in range(0,subcircuit.num_qubits)]

            # # Create a new circuit with the same number of qubits
            # new_subcircuit = QuantumCircuit(subcircuit.num_qubits, subcircuit.num_qubits)

            # # Copy all operations from the original circuit
            # for instr, qargs, cargs in subcircuit.data:
            #     if instr.name == 'measure':
            #         # Change the order of clbits here if necessary
            #         new_subcircuit.measure(qargs[0], [new_subcircuit.clbits[qargs[0].index]])
            #         measure_missing_qubit.remove(qargs[0].index)
            #     elif instr.condition:
            #         new_instr=instr.c_if(new_subcircuit.clbits[qargs[0].index], 1)
            #         new_subcircuit.append(new_instr, qargs, cargs)
            #     else:
            #         # For other operations, just add them as they are
            #         if cargs:
            #             new_subcircuit.append(instr, qargs, [new_subcircuit.clbits[qargs[0].index]])
            #         else:
            #             new_subcircuit.append(instr, qargs, cargs)

            # for qubit in measure_missing_qubit:
            #     new_subcircuit.measure(new_subcircuit.qubits[qubit],new_subcircuit.clbits[qubit])
            # #
            # subcircuits[index]=new_subcircuit

    """replaced code start"""

    quasi_dists = sampler.run(circuits=subcircuits).result().quasi_dists

    all_probabilities_out = []
    for i, qd in enumerate(quasi_dists):
        probabilities = qd.nearest_probability_distribution()
        probabilities_out = np.zeros(2 ** subcircuits[i].num_clbits, dtype=float)

        for state in probabilities:
            probabilities_out[state] = probabilities[state]
        all_probabilities_out.append(probabilities_out)

    return all_probabilities_out


def verify(
    ground_truth,
    reconstructed_output: np.ndarray,
) -> tuple[dict[str, dict[str, float]], Sequence[float]]:
    """
    Compare the reconstructed probabilities to the ground truth.

    Executes the original circuit, then measures the distributional differences between this exact
    result (ground truth) and the reconstructed result from the subcircuits.
    Provides a variety of metrics to evaluate the differences in the distributions.

    Args:
        full_circuit: The original quantum circuit that was cut
        reconstructed_output: The reconstructed probability distribution from the
            execution of the subcircuits

    Returns:
        A tuple containing metrics for the ground truth and reconstructed distributions
    """
    # ground_truth = _evaluate_circuit(circuit=full_circuit)
    metrics = {}
    for quasi_conversion_mode in ["nearest", "naive"]:
        real_probability = quasi_to_real(
            quasiprobability=reconstructed_output, mode=quasi_conversion_mode
        )

        chi2 = chi2_distance(target=ground_truth, obs=real_probability)
        mse = MSE(target=ground_truth, obs=real_probability)
        mape = MAPE(target=ground_truth, obs=real_probability)
        ce = cross_entropy(target=ground_truth, obs=real_probability)
        hop = HOP(target=ground_truth, obs=real_probability)
        metrics[quasi_conversion_mode] = {
            "chi2": chi2,
            "Mean Squared Error": mse,
            "Mean Absolute Percentage Error": mape,
            "Cross Entropy": ce,
            "HOP": hop,
        }
    return metrics