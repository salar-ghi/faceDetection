import numpy as np
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
print(device)
torch.device(device)

# torch.set_default_device(device)


print("Is CUDA supported by this system? ",
      {torch.cuda.is_available()})
print("CUDA version:", {torch.version.cuda})
 
# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print("ID of current CUDA device:",
      {torch.cuda.current_device()})
       
print("Name of current CUDA device:",
      {torch.cuda.get_device_name(cuda_id)})

