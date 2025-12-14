import torch

# Create a 3x4 matrix (Rank 2 Tensor)
matrix_a = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(matrix_a.shape)
# Output: torch.Size([3, 4])

# Create a Rank 3 Tensor (e.g., a batch of 2, 3x3 images with 1 channel)
tensor_3d = torch.zeros(2, 3, 3, 1)
print(tensor_3d.ndim) # Number of dimensions (rank)
# Output: 4
