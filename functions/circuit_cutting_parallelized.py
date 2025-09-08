# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Composite set of modified functions from the circuit cutting toolkit, adjusted to involve parallelization.
Generation of cutting experiments, reconstruction of expectation values."""

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


from joblib import Parallel, delayed, parallel_config, wrap_non_picklable_objects
from joblib.externals.loky import set_loky_pickler

#Reconstruct expectation values option 2 - concurrent.futures & ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor
#from typing import Sequence, Mapping, Hashable

import os
import traceback
from debugging.pickle_debug import diagnose_joblib_pickle



## Expectation Value Reconstruction
## Using joblib

def _compute_expval_for_coefficient(
    i: int,
    coeff: tuple[float, 'WeightType'],
    subsystem_observables,
    results_dict,
    subobservables_by_subsystem,
) -> np.ndarray:
    """Helper function to compute the partial expectation value for one coefficient."""
    num_observables = len(next(iter(subobservables_by_subsystem.values())))
    current_expvals = np.ones((num_observables,))

    for label, so in subsystem_observables.items():
        subsystem_expvals = [
            np.zeros(len(cog.commuting_observables)) for cog in so.groups
        ]
        current_result = results_dict[label]
        for k, cog in enumerate(so.groups):
            idx = i * len(so.groups) + k
            if isinstance(current_result, SamplerResult):
                quasi_probs = current_result.quasi_dists[idx]
                for outcome, quasi_prob in quasi_probs.items():
                    subsystem_expvals[k] += quasi_prob * _process_outcome(cog, outcome)
            else:
                data_pub = current_result[idx].data
                obs_array = data_pub.observable_measurements.array
                qpd_array = data_pub.qpd_measurements.array
                shots = qpd_array.shape[0]
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

    return coeff[0] * current_expvals



def reconstruct_expectation_values(
    results,
    coefficients: Sequence[tuple[float, 'WeightType']],
    observables: PauliList | dict[Hashable, PauliList],
) -> list[float]:
    if isinstance(observables, PauliList):
        if not isinstance(results, (SamplerResult, PrimitiveResult)):
            raise ValueError(
                "If observables is a PauliList, results must be a SamplerResult or PrimitiveResult instance."
            )
        if any(obs.phase != 0 for obs in observables):
            raise ValueError("An input observable has a phase not equal to 1.")
        subobservables_by_subsystem = decompose_observables(observables, "A" * len(observables[0]))
        results_dict = {"A": results}
        num_expvals = len(observables)
    elif isinstance(observables, Mapping):
        if not isinstance(results, Mapping):
            raise ValueError("If observables is a dictionary, results must also be a dictionary.")
        if observables.keys() != results.keys():
            raise ValueError("The subsystem labels of the observables and results do not match.")
        results_dict = results
        for label, subobservable in observables.items():
            if any(obs.phase != 0 for obs in subobservable):
                raise ValueError("An input observable has a phase not equal to 1.")
        subobservables_by_subsystem = observables
        num_expvals = len(list(observables.values())[0])
    else:
        raise ValueError("observables must be either a PauliList or dict.")

    subsystem_observables = {
        label: ObservableCollection(subobservables)
        for label, subobservables in subobservables_by_subsystem.items()
    }

    for label, so in subsystem_observables.items():
        current_result = results_dict[label]
        if isinstance(current_result, SamplerResult):
            current_result = current_result.quasi_dists
        if len(current_result) != len(coefficients) * len(so.groups):
            raise ValueError(
                f"The number of subexperiments performed in subsystem '{label}' "
                f"({len(current_result)}) should equal the number of coefficients "
                f"({len(coefficients)}) times the number of mutually commuting "
                f"subobservable groups ({len(so.groups)}), but it does not."
            )

    dummy_env = "0"
    #os.getenv("DIAG_PICKLE", "0")
    # --- DIAGNOSTIC BLOCK ---
    if dummy_env == "1":
        try:
            for i0, coeff0 in enumerate(coefficients):
                diag = diagnose_joblib_pickle(
                    _compute_expval_for_coefficient,
                    i0, coeff0, results_dict, subobservables_by_subsystem, subsystem_observables
                )
                if not diag["call_tuple"]:
                    print("[pickle-diag fail at i]", i0, diag)
                    break
        except Exception:
            
            print("[pickle-diag] Exception while diagnosing:\n", traceback.format_exc())
            raise
    # --- END DIAGNOSTIC BLOCK ---


    # 🔀 Parallel execution
    partials = Parallel(n_jobs=-1, backend="threading")(
        delayed(_compute_expval_for_coefficient)(
            i, coeff, subsystem_observables, results_dict, subobservables_by_subsystem
        )
        for i, coeff in enumerate(coefficients)
    )

    total_expvals = np.sum(partials, axis=0)
    return list(total_expvals)

## Using concurrent.futures

def _compute_single_coefficient_contribution(
    i: int,
    coeff: tuple[float, WeightType],
    results_dict: dict,
    subobservables_by_subsystem: dict,
    subsystem_observables: dict,
) -> np.ndarray:
    current_expvals = np.ones(len(list(subobservables_by_subsystem.values())[0]))
    for label, so in subsystem_observables.items():
        subsystem_expvals = [np.zeros(len(cog.commuting_observables)) for cog in so.groups]
        current_result = results_dict[label]

        for k, cog in enumerate(so.groups):
            idx = i * len(so.groups) + k
            if isinstance(current_result, SamplerResult):
                quasi_probs = current_result.quasi_dists[idx]
                for outcome, quasi_prob in quasi_probs.items():
                    subsystem_expvals[k] += quasi_prob * _process_outcome(cog, outcome)
            else:
                data_pub = current_result[idx].data
                obs_array = data_pub.observable_measurements.array
                qpd_array = data_pub.qpd_measurements.array
                shots = qpd_array.shape[0]
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

    return coeff[0] * current_expvals


def reconstruct_expectation_values_concurrent(
    results: SamplerResult | PrimitiveResult | dict[Hashable, SamplerResult | PrimitiveResult],
    coefficients: Sequence[tuple[float, WeightType]],
    observables: PauliList | dict[Hashable, PauliList],
) -> list[float]:
    if isinstance(observables, PauliList):
        if not isinstance(results, (SamplerResult, PrimitiveResult)):
            raise ValueError(...)
        if any(obs.phase != 0 for obs in observables):
            raise ValueError(...)
        subobservables_by_subsystem = decompose_observables(observables, "A" * len(observables[0]))
        results_dict = {"A": results}
        expvals = np.zeros(len(observables))
    elif isinstance(observables, Mapping):
        if not isinstance(results, Mapping) or observables.keys() != results.keys():
            raise ValueError(...)
        results_dict = results
        for label, subobservable in observables.items():
            if any(obs.phase != 0 for obs in subobservable):
                raise ValueError(...)
        subobservables_by_subsystem = observables
        expvals = np.zeros(len(list(observables.values())[0]))
    else:
        raise ValueError("observables must be either a PauliList or dict.")

    subsystem_observables = {
        label: ObservableCollection(subobservables)
        for label, subobservables in subobservables_by_subsystem.items()
    }

    for label, so in subsystem_observables.items():
        current_result = results_dict[label]
        if isinstance(current_result, SamplerResult):
            current_result = current_result.quasi_dists
        if len(current_result) != len(coefficients) * len(so.groups):
            raise ValueError(...)

    # Use multiprocessing to compute contributions in parallel
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                _compute_single_coefficient_contribution,
                i,
                coeff,
                results_dict,
                subobservables_by_subsystem,
                subsystem_observables,
            )
            for i, coeff in enumerate(coefficients)
        ]
        for f in futures:
            expvals += f.result()

    return list(expvals)



## Generate Cutting Experiments parallelization

def _process_sample(
    sample_index,
    map_ids,
    redundancy,
    weight_type,
    bases,
    kappa,
    num_samples,
    is_separated,
    subcircuit_dict,
    subsystem_observables,
    subcirc_qpd_gate_ids,
    subcirc_map_ids,
):
    from copy import deepcopy

    actual_coeff = np.prod([basis.coeffs[map_id] for basis, map_id in zip(bases, map_ids)])
    sampled_coeff = (redundancy / num_samples) * (kappa * np.sign(actual_coeff))
    result_circuits = defaultdict(list)

    for label, so in subsystem_observables.items():
        subcircuit = subcircuit_dict[label]
        map_ids_tmp = map_ids if not is_separated else tuple(map_ids[j] for j in subcirc_map_ids[label])

        for cog in so.groups:
            new_qc = _append_measurement_register(deepcopy(subcircuit), cog)
            decompose_qpd_instructions(
                new_qc, subcirc_qpd_gate_ids[label], map_ids_tmp, inplace=True
            )
            _append_measurement_circuit(new_qc, cog, inplace=True)
            result_circuits[label].append(new_qc)

    return sampled_coeff, weight_type, result_circuits


def generate_cutting_experiments(
    circuits: QuantumCircuit | dict[Hashable, QuantumCircuit],
    observables: PauliList | dict[Hashable, PauliList],
    num_samples: int | float,
) -> tuple[
    list[QuantumCircuit] | dict[Hashable, list[QuantumCircuit]],
    list[tuple[float, WeightType]],
]:
    # [Existing validation and setup code remains unchanged...]

    if isinstance(circuits, QuantumCircuit):
        is_separated = False
        subcircuit_dict = {"A": circuits}
        subobservables_by_subsystem = decompose_observables(
            observables, "A" * len(observables[0])
        )
        subsystem_observables = {
            label: ObservableCollection(subobservables)
            for label, subobservables in subobservables_by_subsystem.items()
        }
        bases, qpd_gate_ids = _get_bases(circuits)
        subcirc_qpd_gate_ids: dict[Hashable, list[list[int]]] = {"A": qpd_gate_ids}
        #The subcirc_map_ids line below is introduced to allow use of joblib with this variable when is_seperated == False.
        subcirc_map_ids = {"A": list(range(len(qpd_gate_ids)))}
    else:
        is_separated = True
        subcircuit_dict = circuits
        subcirc_qpd_gate_ids, subcirc_map_ids = _get_mapping_ids_by_partition(subcircuit_dict)
        bases = _get_bases_by_partition(subcircuit_dict, subcirc_qpd_gate_ids)
        subsystem_observables = {
            label: ObservableCollection(so) for label, so in observables.items()
        }

    random_samples = generate_qpd_weights(bases, num_samples=num_samples)
    kappa = np.prod([basis.kappa for basis in bases])
    num_samples = sum([value[0] for value in random_samples.values()])
    sorted_samples = sorted(random_samples.items(), key=lambda x: x[1][0], reverse=True)

    # ✅ Parallel processing of experiments
    results = Parallel(n_jobs=-1)(
        delayed(_process_sample)(
            idx,
            map_ids,
            redundancy,
            weight_type,
            bases,
            kappa,
            num_samples,
            is_separated,
            subcircuit_dict,
            subsystem_observables,
            subcirc_qpd_gate_ids,
            subcirc_map_ids,
        )
        for idx, (map_ids, (redundancy, weight_type)) in enumerate(sorted_samples)
    )

    # ✅ Consolidate results
    subexperiments_dict: dict[Hashable, list[QuantumCircuit]] = defaultdict(list)
    coefficients: list[tuple[float, WeightType]] = []

    for coeff, wtype, circuits_by_label in results:
        coefficients.append((coeff, wtype))
        for label, circ_list in circuits_by_label.items():
            subexperiments_dict[label].extend(circ_list)

    # ✅ Post-process: Remove resets
    for subexperiments in subexperiments_dict.values():
        for circ in subexperiments:
            _remove_resets_in_zero_state(circ)
            _remove_final_resets(circ)
            _consolidate_resets(circ)

    subexperiments_out: list[QuantumCircuit] | dict[Hashable, list[QuantumCircuit]] = (
        dict(subexperiments_dict)
    )
    if isinstance(circuits, QuantumCircuit):
        subexperiments_out = list(subexperiments_dict.values())[0]

    return subexperiments_out, coefficients



