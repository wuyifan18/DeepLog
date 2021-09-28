import torch
from transformers import BertModel, XLNetModel, BertConfig, XLNetConfig

class LogBERT(torch.nn.Module):
    # A BERT model for abnormaly detection on logs
    def __init__(self, name = 'LogBERT'):
        super(LogBERT, self).__init__()
        pass

    def forward(self, inputs):
        return None

class LogXLNet(torch.nn.Module):
    # A XLNet model for abnormaly detection on logs
    def __init__(self, name = 'LogXLNet'):
        super(LogXLNet, self).__init__()
        pass

    def forward(self, inputs):
        return None
