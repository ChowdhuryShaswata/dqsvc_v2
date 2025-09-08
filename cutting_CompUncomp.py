from __future__ import annotations
from collections.abc import Sequence
#from copy import copy
from qiskit_machine_learning.state_fidelities import ComputeUncompute
import numpy as np
from qiskit import QuantumCircuit
#from qiskit.primitives import BaseSampler, BaseSamplerV1, SamplerResult
#from qiskit.primitives.base import BaseSamplerV2
#from qiskit.transpiler.passmanager import PassManager
#from qiskit.result import QuasiDistribution
#from qiskit.primitives.primitive_job import PrimitiveJob
#from qiskit.providers import Options
#from qiskit_ibm_runtime import EstimatorV2 as Estimator
#from qiskit_aer import AerSimulator
from qiskit_machine_learning.algorithm_job import AlgorithmJob
from qiskit.quantum_info import SparsePauliOp
import itertools
from copy import copy
import time

#circuit cutting
from qiskit_addon_cutting import partition_problem
from qiskit_ibm_runtime.fake_provider import FakeManilaV2, FakeManhattanV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2, Batch

##Choices

#cutting
#from qiskit_addon_cutting import generate_cutting_experiments #default
from functions.circuit_cutting_functions import generate_cutting_experiments #logging
#from functions.circuit_cutting_parallelized import generate_cutting_experiments #parallelized

#Reconstruction
#from qiskit_addon_cutting import reconstruct_expectation_values #default
from functions.circuit_cutting_functions import reconstruct_expectation_values #logging
#from functions.circuit_cutting_parallelized import reconstruct_expectation_values as reconstruct_expectation_values #parallelized


class cutting_CompUncomp(ComputeUncompute):
    r"""
    This class modifies the ComputeUncompute state fidelity operation to predict the fidelity by
    measuring the expectation value of |0><0| as a SparsePauliOp using the circuit cutting framework, instead of sampling |0><0| directly."""

    # def __init__(self, sampler, *, options = None, local = False, pass_manager = None, estimator = Estimator(AerSimulator(), options={"default_shots": int(1e6)})):
    #     super().__init__(sampler, options=options, local=local, pass_manager=pass_manager)
    #     self.estimator = estimator
    #     self.operator = None

    def __init__(self, sampler, *, options = None, local = False, pass_manager = None):
        super().__init__(sampler, options=options, local=local, pass_manager=pass_manager)
        self.operator = None

    def create_fidelity_circuit(self, circuit_1, circuit_2):
        fid_circuit = super().create_fidelity_circuit(circuit_1, circuit_2)
        fid_circuit.remove_final_measurements()
        return fid_circuit

    def generate_operator(self, circuit: QuantumCircuit):

        if self.operator is None:
            #Define the observable |0><0| as a Pauli Observable for circuit.num_qubits number of gates.

            pauli_labels = [''.join(p) for p in itertools.product('IZ', repeat=circuit.num_qubits)]

            # Assign 1/(2^n) coefficient
            coeff = 1/ (2**circuit.num_qubits)

            coeffs = np.full(len(pauli_labels), coeff, dtype=np.float64)

            #make the SparsePauliOp 

            self.operator = SparsePauliOp.from_list(list(zip(pauli_labels, coeffs)))

            #print(self.operator)

        return self.operator
            
    def partitioning_strategy(self, circuit: QuantumCircuit, operator):
        #generate labels according to half-half split.
        num_A = circuit.num_qubits // 2
        num_B = circuit.num_qubits - num_A

        partition_list = ['A'] * num_A + ['B'] * num_B
        partition_str = ''.join(partition_list)
        print(partition_str)

        partitioned_problem = partition_problem(circuit=circuit, partition_labels=partition_str, observables=operator.paulis)
        return partitioned_problem
    

    def _run(
        self,
        circuits_1: QuantumCircuit | Sequence[QuantumCircuit],
        circuits_2: QuantumCircuit | Sequence[QuantumCircuit],
        values_1: Sequence[float] | Sequence[Sequence[float]] | None = None,
        values_2: Sequence[float] | Sequence[Sequence[float]] | None = None,
        **options,
    ):
        # comp_params = np.concatenate((values_1, values_2))


        circuits = self._construct_circuits(circuits_1, circuits_2)
        #print("circuit construction error")

        print(f'the number of circuits is {len(circuits)}')

        if len(circuits) == 0:
            raise ValueError(
                "At least one pair of circuits must be defined to calculate the state overlap."
            )
        
        values = self._construct_value_list(circuits_1, circuits_2, values_1, values_2)

        print(f'the number of values is {len(values)}')

        opts = copy(self._default_options)
        opts.update_options(**options)

        ## Construct circuits produces a list of QuantumCircuits.

        #print(f'Circuit parameters: {circuits.parameters}')
        operator = self.generate_operator(circuits[0])

        #print(f'the {0} circuit parameters are {circuits[0].parameters}')
        #print(f'the {1} circuit parameters are {circuits[1].parameters}')

        # #Assign parameters to circuits prior to cutting.
        # for i in range(len(circuits)):
        #     #TODO: does assign parameters require values to be a single list?
        #     #print(f'the {i+1} circuit parameters are {circuits[i+1].parameters}')
        #     #print(f'the {i} circuit values are {values[i]}')
        #     circuits[i].assign_parameters(values[i], inplace=True)
        #     #print(f'the {i+1} circuit parameters are {circuits[i+1].parameters}')
        #     #circuits[i].assign_parameters(values[i][0].concatenate(values[i][1]))

        #Alternate for-loop - if in-place is false, a copy of the circuit with the assigned parameters is returned.
        for i in range(len(circuits)):
            #print(f'the {i+1} circuit parameters are {circuits[i+1].parameters}')
            circuits[i] = circuits[i].assign_parameters(values[i], inplace=False)
            #print(f'the {i+1} circuit parameters are {circuits[i+1].parameters}')

        # Now we have a list of circuits and corresponding list of parameter vectors.
        # We need to partition each circuit by the partitioning strategy to create the partitioning problems.

        # create list of partitioning problems and generated cutting experiments

        # transpile circuits

        backend = FakeManhattanV2()
        #FakeManilaV2 is a 5-qubit backend.

        pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)

        partitioned_problems = []
        generated_experiments = []
        isa_subexperiment_sets = []
        for i in range(len(circuits)):
            #label partition for the circuits
            #print(circuits[i])
            #print(f'number of qubits in circuit {i} is {circuits[i].num_qubits}')
            part_problem = self.partitioning_strategy(circuits[i], operator)
            partitioned_problems.append(part_problem)

            print(f"Circuit {i}: partitioned.")

            #generate subexperiments
            subexperiments, coefficients = generate_cutting_experiments(
                circuits=part_problem.subcircuits, observables=part_problem.subobservables, num_samples=np.inf
                )
            generated_experiments.append([subexperiments, coefficients])

            print(f"Circuit {i}: cutting experiments generated.")

            #transpile subexperiments into ISA using pass_manager.
            isa_subexperiments = {
                label: pass_manager.run(partition_subexpts)
                for label, partition_subexpts in subexperiments.items()}

            isa_subexperiment_sets.append(isa_subexperiments)

            print(f"Circuit {i}: ISA transpiled subexperiments generated.")

        print("Circuits prepped.")

        


        #Adjusted isa for multiple circuits.

        #All circuit execution using runtime sampler.
        #TODO: check if this can be integrated with AlgorithmJob and _call. Sampler comparision with regular version.

        jobs_set = []
        results_set = []
        for isa_subexperiments_set in isa_subexperiment_sets:
            with Batch(backend=backend) as batch:
                sampler = SamplerV2(mode=batch)
                jobs = {
                    label: sampler.run(subsystem_subexpts, shots=2**12)
                    for label, subsystem_subexpts in isa_subexperiments_set.items()
                }
                jobs_set.append(jobs)
            
            print("A set of jobs sent to the Sampler have been processed.")

            start = time.time()
            # Retrieve results
            results = {label: job.result() for label, job in jobs.items()}
            end = time.time()
            print(f'Result retrieval time: {end-start}s')
            results_set.append(results)
        
        print("All jobs sent to the Sampler have been processed.")


        #jobs are dicts, jobs_set is a list.

        start = time.time()
        reconstructed_fidelities = []
        #post process
        for i in range(len(jobs_set)):
            reconstructed_fidelity_terms = reconstruct_expectation_values(
                results_set[i],
                #modify this to take coefficients from generated experiments instead of just coefficients.
                generated_experiments[i][1],
                partitioned_problems[i].subobservables,
            )

            reconstructed_fidelity = np.dot(reconstructed_fidelity_terms, operator.coeffs).real

            reconstructed_fidelities.append(reconstructed_fidelity)

            print(f"Job {i} post-processed.")

        print("All jobs post-processed.")

        end = time.time()

        print(f'Reconstruction time: {end-start}s')
            

        #create a job.results().fidelities to store resultant list of fidelities produced by the samplers.


        return reconstructed_fidelities




        

        # bd_circuits = []
        # isa_circuits = []
        # isa_observables = []
        # for i in range(len(circuits)):
        #     bd_circuits.append(circuits[i].assign_parameters(comp_params))
        #     isa_circuits.append(self._pass_manager.run(bd_circuits[i]))
        #     isa_observables.append(operator.apply_layout(isa_circuits[i].layout))

        # estimator_job = self.estimator.run([(isa_circuits, isa_observables)])

        # take estimator results (all fidelities for inputs) and have them be added to the kernel? - store as job.result().fidelities.
        # overwrite _call method to do so? or just return samplerjob and give it a fidelities parameter?

        ##pub_result = estimator_job.result()[0]
        
        #estimator_job.result().fidelities = job.result()[0].data.evs

        #AlgorithmJob is a wrapper for RunTimeJobV2, which both sampler and estimator output.

        #local_opts = self._get_local_options(opts.__dict__)

        #return AlgorithmJob(ComputeUncompute._call, estimator_job, circuits, self._local, local_opts)

    def run(
        self,
        circuits_1: QuantumCircuit | Sequence[QuantumCircuit],
        circuits_2: QuantumCircuit | Sequence[QuantumCircuit],
        values_1: Sequence[float] | Sequence[Sequence[float]] | None = None,
        values_2: Sequence[float] | Sequence[Sequence[float]] | None = None,
        **options,
    ) -> AlgorithmJob:
        
        fidelities = self._run(circuits_1, circuits_2, values_1, values_2)

        job = DummyJob(fidelities)

        return job


class DummyResult:
    def __init__(self, fidelities):
        self.fidelities = fidelities

class DummyJob:
    def __init__(self, fidelities):
        self._result = DummyResult(fidelities)

    def result(self):
        return self._result



        

