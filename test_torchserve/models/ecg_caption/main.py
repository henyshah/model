import torch

from models.ecg_caption.model import TheModelClass

model = TheModelClass()

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

torch.save(model.state_dict(), 'model_weights.pth')
