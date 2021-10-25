import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"

# my_tensor = torch.tensor(
#     [[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True
# )
# print(my_tensor.shape)  # Prints shape, in this case 2x3


x = torch.empty(size=(3, 3))  # Tensor of shape 3x3 with uninitialized data
x = torch.zeros((3, 3))  # Tensor of shape 3x3 with values of 0
x = torch.rand( (3, 3) )  # Tensor of shape 3x3 with values from uniform distribution in interval [0,1)
x = torch.ones((3, 3))  # Tensor of shape 3x3 with values of 1
x = torch.eye(5, 5)  # Returns Identity Matrix I, (I <-> Eye), matrix of shape 2x3

x = torch.arange( start=0, end=5, step=1 )  # Tensor [0, 1, 2, 3, 4], note, can also do: torch.arange(11)
x = torch.linspace(start=0.1, end=1, steps=10)  # x = [0.1, 0.2, ..., 1]
x = torch.empty(size=(1, 5)).normal_( mean=0, std=1 ) # Normally distributed with mean=0, std=1

x = torch.empty(size=(1, 5)).uniform_(0,1)  # Values from a uniform distribution low=0, high=1
x = torch.diag(torch.ones(3))  # Diagonal matrix of shape 3x3

t = torch.arange(4)
t.short()
t.float()
t.half()

# ------------------------------ array to tensor conversion ------------------------------
import numpy as np 
array = np.zeros((5,5))
tensor = torch.from_numpy(array)
array = tensor.numpy()
# print(tensor)

# ----------------------------------- Math operations ------------------------------
x = torch.tensor([1,2,3])
y = torch.tensor([1,2,3])

z = torch.add(x,y)
z = x + y
z = x-y
z = torch.true_divide(x,y)
x.add_(y) # x+=y

x = torch.rand((2,5))
y = torch.rand((5,3))
z = torch.mm(x,y) ## z = x.mm(y)
print(z)
# -----------------------------------  comparison ------------------------------
z = x<0

print(z)