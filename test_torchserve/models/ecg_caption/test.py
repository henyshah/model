import torch

from models.ecg_caption.main import model

models = model()
models.load_state_dict(torch.load(PATH))
models.eval()
PATH = "entire_model.pt"