import numpy as np
# calculating fidelity between two states using the kernel
from qiskit_aer import AerSimulator
#from qiskit.primitives import Sampler
from cutting_CompUncomp import cutting_CompUncomp
from qiskit_machine_learning.state_fidelities import ComputeUncompute

from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel
import numpy as np

from qiskit import QuantumCircuit


from qiskit_machine_learning.utils import algorithm_globals

algorithm_globals.random_seed = 12345

## Dataset

NUM_QUBITS = 4

dataset = np.array([
    np.random.rand(NUM_QUBITS),
    np.random.rand(NUM_QUBITS)
])
print(dataset)

#generate circuit
 
manual_kernel = QuantumCircuit(NUM_QUBITS)

U_map = ZZFeatureMap(feature_dimension=NUM_QUBITS, reps=2, entanglement="linear")

for i in range(NUM_QUBITS):
    manual_kernel.h(i)


manual_kernel = manual_kernel.compose(U_map)

manual_kernel = manual_kernel.decompose()

print(f'Manual kernel parameters: {manual_kernel.parameters}')
manual_kernel.draw("mpl")

fidelity_computer = cutting_CompUncomp(sampler=Sampler())

operator = fidelity_computer.generate_operator(manual_kernel)

print(len(operator))

print(len(operator.paulis))


fidelities = fidelity_computer._run(
    [manual_kernel],
    [manual_kernel],
    dataset[0],
    dataset[1]
)

print(fidelities)

fidelities_real = fidelities[0].real

print(fidelities_real)
