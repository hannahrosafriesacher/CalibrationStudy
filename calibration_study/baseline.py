import torch

class Baseline(torch.nn.Module):
    def __init__(self, input_features, hidden_sizes, output_features, dropout):
        super().__init__()
        self.input_features = input_features
        self.hidden_sizes = hidden_sizes
        self.output_features = output_features
        self.input = torch.nn.Linear(in_features=input_features, out_features=hidden_sizes)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.output = torch.nn.Linear(in_features=hidden_sizes, out_features=output_features)
        
    def forward(self, x, return_hidden):
        fc = self.input(x)
        a = self.relu(fc)
        dr = self.dropout(a)
        out = self.output(dr)
        
        if return_hidden == 1:
            return out, a
        elif return_hidden == 0:
            return out