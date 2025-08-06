from __future__ import annotations

from collections.abc import Sequence, Hashable, Mapping
from collections import defaultdict

import numpy as np
from qiskit.quantum_info import PauliList
from qiskit.primitives import (
    SamplerResult,  # for SamplerV1
    PrimitiveResult,  # for SamplerV2
)

from qiskit_addon_cutting.utils.observable_grouping import CommutingObservableGroup, ObservableCollection
#from qiskit_addon_cutting.utils.bitwise import bit_count
from qiskit_addon_cutting.cutting_decomposition import decompose_observables
#from qiskit_addon_cutting.cutting_experiments import _get_pauli_indices
from qiskit_addon_cutting.qpd import WeightType
from qiskit_addon_cutting.cutting_reconstruction import _process_outcome, _process_outcome_v2

from qiskit.circuit import QuantumCircuit

from qiskit_addon_cutting.utils.iteration import strict_zip
#from qiskit_addon_cutting.utils.observable_grouping import ObservableCollection, CommutingObservableGroup
from qiskit_addon_cutting.qpd import (
    QPDBasis,
    SingleQubitQPDGate,
    TwoQubitQPDGate,
    generate_qpd_weights,
    decompose_qpd_instructions,
)

from qiskit_addon_cutting.cutting_experiments import _remove_final_resets, _remove_resets_in_zero_state, _consolidate_resets, _get_bases, _get_mapping_ids_by_partition, _get_bases_by_partition, _append_measurement_circuit, _append_measurement_register


#Set up logging
import logging

#logging.basicConfig(filename="loop_sizes.log", level=logging.INFO)


# Log reconstruction function for loop sizes

def reconstruct_expectation_values(
    results: (
        SamplerResult
        | PrimitiveResult
        | dict[Hashable, SamplerResult | PrimitiveResult]
    ),
    coefficients: Sequence[tuple[float, WeightType]],
    observables: PauliList | dict[Hashable, PauliList],
) -> list[float]:
    r"""Reconstruct an expectation value from the results of the sub-experiments.

    Args:
        results: The results from running the cutting subexperiments. If the cut circuit
            was not partitioned between qubits and run separately, this argument should be
            a :class:`~qiskit.primitives.SamplerResult` instance or a dictionary mapping
            a single partition to the results. If the circuit was partitioned and its
            pieces were run separately, this argument should be a dictionary mapping partition labels
            to the results from each partition's subexperiments.

            The subexperiment results are expected to be ordered in the same way the subexperiments
            are ordered in the output of :func:`.generate_cutting_experiments` -- one result for every
            sample and observable, as shown below. The Qiskit Sampler primitive will return the results
            in the same order the experiments are submitted, so users who do not use :func:`.generate_cutting_experiments`
            to generate their experiments should take care to order their subexperiments as follows before submitting them
            to the sampler primitive:

            :math:`[sample_{0}observable_{0}, \ldots, sample_{0}observable_{N-1}, sample_{1}observable_{0}, \ldots, sample_{M-1}observable_{N-1}]`

        coefficients: A sequence containing the coefficient associated with each unique subexperiment. Each element is a tuple
            containing the coefficient (a ``float``) together with its :class:`.WeightType`, which denotes
            how the value was generated. The contribution from each subexperiment will be multiplied by
            its corresponding coefficient, and the resulting terms will be summed to obtain the reconstructed expectation value.
        observables: The observable(s) for which the expectation values will be calculated.
            This should be a :class:`~qiskit.quantum_info.PauliList` if ``results`` is a
            :class:`~qiskit.primitives.SamplerResult` instance. Otherwise, it should be a
            dictionary mapping partition labels to the observables associated with that partition.

    Returns:
        A ``list`` of ``float``\ s, such that each float is an expectation
        value corresponding to the input observable in the same position

    Raises:
        ValueError: ``observables`` and ``results`` are of incompatible types.
        ValueError: An input observable has a phase not equal to 1.
    """
    # If circuit was not separated, transform input data structures to
    # dictionary format.  Perform some input validation in either case.
    if isinstance(observables, PauliList):
        if not isinstance(results, (SamplerResult, PrimitiveResult)):
            raise ValueError(
                "If observables is a PauliList, results must be a SamplerResult or PrimitiveResult instance."
            )
        if any(obs.phase != 0 for obs in observables):
            raise ValueError("An input observable has a phase not equal to 1.")
        subobservables_by_subsystem: Mapping[Hashable, PauliList] = (
            decompose_observables(observables, "A" * len(observables[0]))
        )
        results_dict: Mapping[Hashable, SamplerResult | PrimitiveResult] = {
            "A": results
        }
        expvals = np.zeros(len(observables))

    elif isinstance(observables, Mapping):
        if not isinstance(results, Mapping):
            raise ValueError(
                "If observables is a dictionary, results must also be a dictionary."
            )
        if observables.keys() != results.keys():
            raise ValueError(
                "The subsystem labels of the observables and results do not match."
            )
        results_dict = results
        for label, subobservable in observables.items():
            if any(obs.phase != 0 for obs in subobservable):
                raise ValueError("An input observable has a phase not equal to 1.")
        subobservables_by_subsystem = observables
        expvals = np.zeros(len(list(observables.values())[0]))

    else:
        raise ValueError("observables must be either a PauliList or dict.")

    subsystem_observables = {
        label: ObservableCollection(subobservables)
        for label, subobservables in subobservables_by_subsystem.items()
    }

    # Validate that the number of subexperiments executed is consistent with
    # the number of coefficients and observable groups.
    for label, so in subsystem_observables.items():
        current_result = results_dict[label]
        if isinstance(current_result, SamplerResult):
            # SamplerV1 provides a SamplerResult
            current_result = current_result.quasi_dists
        if len(current_result) != len(coefficients) * len(so.groups):
            raise ValueError(
                f"The number of subexperiments performed in subsystem '{label}' "
                f"({len(current_result)}) should equal the number of coefficients "
                f"({len(coefficients)}) times the number of mutually commuting "
                f"subobservable groups ({len(so.groups)}), but it does not."
            )
        
    # log
    logging.info(f"len(coefficients) = {len(coefficients)}")
    logging.info(f"len(subsystem_observables) = {len(subsystem_observables)}")

    loop = 1
    # Reconstruct the expectation values
    for i, coeff in enumerate(coefficients):
        #log
        if loop <= 10:
            logging.info(f"Processing coefficient {i+1}/{len(coefficients)}")

        current_expvals = np.ones((len(expvals),))
        for label, so in subsystem_observables.items():
            subsystem_expvals = [
                np.zeros(len(cog.commuting_observables)) for cog in so.groups
            ]
            #log
            logging.info(
                f"label={label} | len(so.groups)={len(so.groups)}"
            )

            current_result = results_dict[label]
            for k, cog in enumerate(so.groups):
                idx = i * len(so.groups) + k
                if isinstance(current_result, SamplerResult):
                    # SamplerV1 provides a SamplerResult
                    quasi_probs = current_result.quasi_dists[idx]

                    #log
                    if loop <= 10:
                        logging.info(f"SamplerResult label={label}, idx={idx}, "
                                  f"len(quasi_probs)={len(quasi_probs)}")
                    
                    for outcome, quasi_prob in quasi_probs.items():
                        subsystem_expvals[k] += quasi_prob * _process_outcome(
                            cog, outcome
                        )
                else:
                    # SamplerV2 provides a PrimitiveResult
                    data_pub = current_result[idx].data
                    obs_array = data_pub.observable_measurements.array
                    qpd_array = data_pub.qpd_measurements.array
                    shots = qpd_array.shape[0]
                    if loop <= 10:
                        logging.info(f"SamplerResult label={label}, idx={idx}, "
                                  f"shots={shots}")
                    
                    for j in range(shots):
                        obs_outcomes = int.from_bytes(obs_array[j], "big")
                        qpd_outcomes = int.from_bytes(qpd_array[j], "big")
                        subsystem_expvals[k] += (1 / shots) * _process_outcome_v2(
                            cog, obs_outcomes, qpd_outcomes
                        )

            for k, subobservable in enumerate(subobservables_by_subsystem[label]):
                current_expvals[k] *= np.mean(
                    [subsystem_expvals[m][n] for m, n in so.lookup[subobservable]]
                )

        expvals += coeff[0] * current_expvals

    return list(expvals)


#Modified to log relevant quantities
def generate_cutting_experiments(
    circuits: QuantumCircuit | dict[Hashable, QuantumCircuit],
    observables: PauliList | dict[Hashable, PauliList],
    num_samples: int | float,
) -> tuple[
    list[QuantumCircuit] | dict[Hashable, list[QuantumCircuit]],
    list[tuple[float, WeightType]],
]:
    r"""Generate cutting subexperiments and their associated coefficients.

    If the input, ``circuits``, is a :class:`QuantumCircuit` instance, the
    output subexperiments will be contained within a 1D array, and ``observables`` is
    expected to be a :class:`PauliList` instance.

    If the input circuit and observables are specified by dictionaries with partition labels
    as keys, the output subexperiments will be returned as a dictionary which maps each
    partition label to a 1D array containing the subexperiments associated with that partition.

    In both cases, the subexperiment lists are ordered as follows:

        :math:`[sample_{0}observable_{0}, \ldots, sample_{0}observable_{N-1}, sample_{1}observable_{0}, \ldots, sample_{M-1}observable_{N-1}]`

    The coefficients will always be returned as a 1D array -- one coefficient for each unique sample.

    Args:
        circuits: The circuit(s) to partition and separate
        observables: The observable(s) to evaluate for each unique sample
        num_samples: The number of samples to draw from the quasi-probability distribution. If set
            to infinity, the weights will be generated rigorously rather than by sampling from
            the distribution.

    Returns:
        A tuple containing the cutting experiments and their associated coefficients.
        If the input circuits is a :class:`QuantumCircuit` instance, the output subexperiments
        will be a sequence of circuits -- one for every unique sample and observable. If the
        input circuits are represented as a dictionary keyed by partition labels, the output
        subexperiments will also be a dictionary keyed by partition labels and containing
        the subexperiments for each partition.
        The coefficients are always a sequence of length-2 tuples, where each tuple contains the
        coefficient and the :class:`WeightType`. Each coefficient corresponds to one unique sample.

    Raises:
        ValueError: ``num_samples`` must be at least one.
        ValueError: ``circuits`` and ``observables`` are incompatible types
        ValueError: :class:`SingleQubitQPDGate` instances must have their cut ID
            appended to the gate label so they may be associated with other gates belonging
            to the same cut.
        ValueError: :class:`SingleQubitQPDGate` instances are not allowed in unseparated circuits.
    """

    logging.info("=== generate_cutting_experiments ===")

    if isinstance(circuits, dict):
        logging.info(f"Number of partitions: {len(circuits)}")
    else:
        logging.info("Single circuit input")

    if isinstance(observables, dict):
        total_observables = sum(len(v) for v in observables.values())
        logging.info(f"Total observables: {total_observables} across {len(observables)} partitions")
    else:
        logging.info(f"Single observable list of length: {len(observables)}")



    if isinstance(circuits, QuantumCircuit) and not isinstance(observables, PauliList):
        raise ValueError(
            "If the input circuits is a QuantumCircuit, the observables must be a PauliList."
        )
    if isinstance(circuits, dict) and not isinstance(observables, dict):
        raise ValueError(
            "If the input circuits are contained in a dictionary keyed by partition labels, the input observables must also be represented by such a dictionary."
        )
    if not num_samples >= 1:
        raise ValueError("num_samples must be at least 1.")

    # Retrieving the unique bases, QPD gates, and decomposed observables is slightly different
    # depending on the format of the execute_experiments input args, but the 2nd half of this function
    # can be shared between both cases.
    if isinstance(circuits, QuantumCircuit):
        is_separated = False
        subcircuit_dict: dict[Hashable, QuantumCircuit] = {"A": circuits}
        subobservables_by_subsystem = decompose_observables(
            observables, "A" * len(observables[0])
        )
        subsystem_observables = {
            label: ObservableCollection(subobservables)
            for label, subobservables in subobservables_by_subsystem.items()
        }
        # Gather the unique bases from the circuit
        bases, qpd_gate_ids = _get_bases(circuits)
        subcirc_qpd_gate_ids: dict[Hashable, list[list[int]]] = {"A": qpd_gate_ids}

    else:
        is_separated = True
        subcircuit_dict = circuits
        # Gather the unique bases across the subcircuits
        subcirc_qpd_gate_ids, subcirc_map_ids = _get_mapping_ids_by_partition(
            subcircuit_dict
        )
        bases = _get_bases_by_partition(subcircuit_dict, subcirc_qpd_gate_ids)

        # Create the commuting observable groups
        subsystem_observables = {
            label: ObservableCollection(so) for label, so in observables.items()
        }

    logging.info(f"Number of bases: {len(bases)}")

    # Sample the joint quasiprobability decomposition
    random_samples = generate_qpd_weights(bases, num_samples=num_samples)
    logging.info(f"Number of QPD samples: {len(random_samples)}")

    # Calculate terms in coefficient calculation
    kappa = np.prod([basis.kappa for basis in bases])
    num_samples = sum([value[0] for value in random_samples.values()])

    # Sort samples in descending order of frequency
    sorted_samples = sorted(random_samples.items(), key=lambda x: x[1][0], reverse=True)

    # Generate the output experiments and their respective coefficients
    subexperiments_dict: dict[Hashable, list[QuantumCircuit]] = defaultdict(list)
    coefficients: list[tuple[float, WeightType]] = []
    loop = 1
    for z, (map_ids, (redundancy, weight_type)) in enumerate(sorted_samples):
        #log
        logging.debug(f"Sample {z+1}/{len(sorted_samples)} | redundancy={redundancy}, weight_type={weight_type}")
        if loop == 1:
            logging.info(f"Sample {z+1}/{len(sorted_samples)} | redundancy={redundancy}, weight_type={weight_type}")

        actual_coeff = np.prod(
            [basis.coeffs[map_id] for basis, map_id in strict_zip(bases, map_ids)]
        )
        sampled_coeff = (redundancy / num_samples) * (kappa * np.sign(actual_coeff))
        coefficients.append((sampled_coeff, weight_type))
        map_ids_tmp = map_ids
        for label, so in subsystem_observables.items():
            #log
            logging.debug(f"Partition label: {label} | Number of observable groups: {len(so.groups)}")
            if loop == 1:
                logging.info(f"Partition label: {label} | Number of observable groups: {len(so.groups)}")
            loop += 1
            subcircuit = subcircuit_dict[label]
            if is_separated:
                map_ids_tmp = tuple(map_ids[j] for j in subcirc_map_ids[label])
            for j, cog in enumerate(so.groups):
                #log
                logging.debug(f"Group {j+1}/{len(so.groups)} — Adding measurement and decomposition")
                if loop == 2:
                    logging.info(f"Group {j+1}/{len(so.groups)} — Adding measurement and decomposition")
                loop += 1

                new_qc = _append_measurement_register(subcircuit, cog)
                decompose_qpd_instructions(
                    new_qc, subcirc_qpd_gate_ids[label], map_ids_tmp, inplace=True
                )
                _append_measurement_circuit(new_qc, cog, inplace=True)
                subexperiments_dict[label].append(new_qc)
        loop += 1

    #log
    logging.info(f"Total number of generated coefficients: {len(coefficients)}")

    #log
    for label, experiments in subexperiments_dict.items():
        logging.info(f"Partition {label} generated {len(experiments)} subexperiments")

    # Remove initial and final resets from the subexperiments.  This will
    # enable the `Move` operation to work on backends that don't support
    # `Reset`, as long as qubits are not re-used.  See
    # https://github.com/Qiskit/qiskit-addon-cutting/issues/452.
    # While we are at it, we also consolidate each run of multiple resets
    # (which can arise when re-using qubits) into a single reset.
    for subexperiments in subexperiments_dict.values():
        for circ in subexperiments:
            _remove_resets_in_zero_state(circ)
            _remove_final_resets(circ)
            _consolidate_resets(circ)

    # If the input was a single quantum circuit, return the subexperiments as a list
    subexperiments_out: list[QuantumCircuit] | dict[Hashable, list[QuantumCircuit]] = (
        dict(subexperiments_dict)
    )
    assert isinstance(subexperiments_out, dict)
    if isinstance(circuits, QuantumCircuit):
        assert len(subexperiments_out.keys()) == 1
        subexperiments_out = list(subexperiments_dict.values())[0]

    return subexperiments_out, coefficients