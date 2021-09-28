import torch
from transformers import BartModel, BartConfig

class LogBart(torch.nn.Module):
    # A Bart model for abnormaly detection
    def __init__(self, name = 'Log2Log'):
        super(LogBart, self).__init__()
        pass

    def forward(self, inputs):
        return None
