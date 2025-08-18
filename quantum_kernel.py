## import statements
import pandas as pd
from sklearn.model_selection import train_test_split
from qiskit_machine_learning.utils import algorithm_globals
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap
from cutting_CompUncomp import cutting_CompUncomp
import numpy as np
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import SVC
import matplotlib.pyplot as plt

#logging
import logging

logging.basicConfig(
    filename="analysis_loop_sizes.log",  # Or any desired name
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"  # Overwrite each run; use "a" to append instead
)

logging.info("=== Starting full execution ===")


## Data Preparation

#seeding
algorithm_globals.random_seed = 78910

# File paths
features_file = "carc_16comp_x.csv"
labels_file = "carc_16comp_y.csv"

# Load the data
features_df = pd.read_csv(features_file).iloc[:, :4]
labels_df = pd.read_csv(labels_file)

labels = labels_df.values.ravel()

# # Split the data into training (80%) and testing (20%) sets
# X_train, X_test, y_train, y_test = train_test_split(features_df, labels, test_size=0.2, random_state=42)

# test partition
X_train, X_test, y_train, y_test = train_test_split(features_df, labels, train_size=3, test_size=2, random_state=42)

# Print shape of data
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


## Quantum Feature Map

# Build quantum feature map

NUM_QUBITS = 4

adme_feature_map = QuantumCircuit(NUM_QUBITS)

U_map = ZZFeatureMap(feature_dimension=NUM_QUBITS, reps=1, entanglement="linear")

for i in range(NUM_QUBITS):
    adme_feature_map.h(i)


adme_feature_map = adme_feature_map.compose(U_map)

adme_feature_map = adme_feature_map.decompose()

print(f'Manual kernel parameters: {adme_feature_map.parameters}')

## Quantum Kernel

# # Monolithic ComputeUncompute
# sampler = Sampler()

# fidelity = ComputeUncompute(sampler=sampler)

# Cutting ComputeUncompute

sampler = Sampler()
fidelity = cutting_CompUncomp(sampler=sampler)

# Define Quantum Kernel object.

quantum_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=adme_feature_map)

## Evaluate Quantum Kernel to get kernel matrix


adhoc_matrix_train = quantum_kernel.evaluate(x_vec=X_train)
adhoc_matrix_test = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(
    np.asmatrix(adhoc_matrix_train), interpolation="nearest", origin="upper", cmap="Blues"
)
axs[0].set_title("Ad hoc training kernel matrix")

axs[1].imshow(np.asmatrix(adhoc_matrix_test), interpolation="nearest", origin="upper", cmap="Reds")
axs[1].set_title("Ad hoc testing kernel matrix")

plt.show()

## Pass to classical SVC

adhoc_svc = SVC(kernel="precomputed")

adhoc_svc.fit(adhoc_matrix_train, y_train)

adhoc_score_precomputed_kernel = adhoc_svc.score(adhoc_matrix_test, y_test)

print(f"Precomputed kernel classification test score: {adhoc_score_precomputed_kernel}")



