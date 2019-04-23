import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self,input_size,output_size,hidden_layers,drop_p=0.5):
        super().__init__()
        
        self.hidden_layers =nn.ModuleList([nn.Linear(input_size,hidden_layers[0])])
        
        layer_sizes=zip(hidden_layers[:-1],hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1,h2) for h1,h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1],output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, M):
        for linear in self.hidden_layers:
            M = F.relu(linear(M))
            M = self.dropout(M)
        M = self.output(M)
        
        return F.log_softmax(M,dim=1)