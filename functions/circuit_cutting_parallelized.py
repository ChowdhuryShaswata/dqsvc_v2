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

from joblib import Parallel, delayed

def _compute_single_coefficient_contribution_joblib(
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


def reconstruct_expectation_values(
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

    # Parallel computation using joblib
    contributions = Parallel(n_jobs=-1, backend="loky")(
        delayed(_compute_single_coefficient_contribution_joblib)(
            i, coeff, results_dict, subobservables_by_subsystem, subsystem_observables
        )
        for i, coeff in enumerate(coefficients)
    )

    expvals = np.sum(contributions, axis=0)
    return list(expvals)