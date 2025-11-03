from __future__ import annotations

from collections.abc import Sequence
from typing import List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
#from ..state_fidelities import BaseStateFidelity, ComputeUncompute
from cutting_CompUncomp import cutting_CompUncomp

#from .base_kernel import BaseKernel
from qiskit_machine_learning.kernels.fidelity_quantum_kernel import FidelityQuantumKernel

import io
from qiskit import qpy

KernelIndices = List[Tuple[int, int]]

_worker_ctx = {
    "feature_map_bytes": None,
    "feature_map": None,
    "fidelity_ctor_payload": None,
}

def _run_chunk(left_right_chunk):
    """Execute one chunk and return the fidelities list."""
    feature_map = _load_feature_map()
    fidelity = _build_fidelity()
    n = left_right_chunk.shape[0]
    job = fidelity.run(
        [feature_map] * n,
        [feature_map] * n,
        left_right_chunk[0],
        left_right_chunk[1],
    )
    return job.result().fidelities  # list[float]

def _build_fidelity():
    ctor = _worker_ctx["fidelity_ctor_payload"]
    #`ctor()` should return an initialized fidelity primitive.
    return ctor()

def _worker_init(feature_map_bytes, fidelity_ctor_payload):
    _worker_ctx["feature_map_bytes"] = feature_map_bytes
    _worker_ctx["fidelity_ctor_payload"] = fidelity_ctor_payload
    _worker_ctx["feature_map"] = None  # lazy-load

def _load_feature_map():
    if _worker_ctx["feature_map"] is None:
        bio = io.BytesIO(_worker_ctx["feature_map_bytes"])
        print(_worker_ctx["feature_map_bytes"])
        circuits = list(qpy.load(bio))
        if len(circuits) != 1:
            raise RuntimeError("Expected exactly one feature map circuit in QPY payload.")
        _worker_ctx["feature_map"] = circuits[0]
    return _worker_ctx["feature_map"]

class ParallelizedFidelityQuantumKernel(FidelityQuantumKernel):
    """
    Parallelization of Qiskit ML's Fidelity Quantum Kernel, designed for use on HPCs. 
    Currently parallelizes the kernel entry computation.
    """
    
    def _make_fidelity_for_worker(self):
        """Return a newly constructed fidelity primitive equivalent to self._fidelity."""

        sampler = Sampler()
        fidelity = cutting_CompUncomp(sampler=sampler)
        return fidelity

    
    def _get_kernel_entries(self, left_parameters, right_parameters):
        import io
        from qiskit import qpy
        from concurrent.futures import ProcessPoolExecutor

        if left_parameters.shape[0] == 0:
            return []

        # Serialize feature map
        buf = io.BytesIO()
        qpy.dump([self._feature_map], buf)
        feature_map_bytes = buf.getvalue()

        # Build constructor payload
        fidelity_ctor_payload = self._make_fidelity_for_worker

        if self.max_circuits_per_job is None:
            job = self._fidelity.run(
                [self._feature_map]*len(left_parameters),
                [self._feature_map]*len(right_parameters),
                left_parameters,
                right_parameters,
            )
            return list(job.result().fidelities)

        # Chunking
        m = self.max_circuits_per_job
        tasks = [
            (left_parameters[i:i+m], right_parameters[i:i+m])
            for i in range(0, len(left_parameters), m)
        ]

        with ProcessPoolExecutor(
            initializer=_worker_init,
            initargs=(feature_map_bytes, fidelity_ctor_payload),
        ) as ex:
            results = list(ex.map(_run_chunk, tasks))

        return [f for chunk in results for f in chunk]
