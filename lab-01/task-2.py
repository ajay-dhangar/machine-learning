import tensorflow as tf

# Create a 3x4 matrix (Rank 2 Tensor)
matrix_a = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(matrix_a.shape)
# Output: (3, 4)

# Creating a tensor of Rank 4 (e.g., batch of images)
tensor_4d = tf.zeros(shape=(16, 224, 224, 3))
print(tensor_4d.ndim) # Number of dimensions (rank)
# Output: 4
