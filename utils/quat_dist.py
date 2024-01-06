from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import numpy as np

# Number of examples
n_examples = 1000000

# Generate random rotation matrices and convert them to quaternions
quaternions = np.array([R.from_matrix(R.random().as_matrix().T).as_quat() for _ in range(n_examples)])


import pdb; pdb.set_trace()

# Adjust to scalar-first format (w, x, y, z)
quaternions = quaternions[:, [3, 0, 1, 2]]

# Plotting the distribution of quaternion values
plt.figure(figsize=(12, 8))

# Plot each component of the quaternion
components = ['w', 'x', 'y', 'z']
for i, component in enumerate(components):
    plt.subplot(2, 2, i+1)
    plt.hist(quaternions[:, i], bins=50, alpha=0.7)
    plt.title(f'Distribution of {component}-component')
    plt.xlabel(f'{component} value')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
