import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim = -1, keepdim=True) + self.eps)
        return x / rms * self.scale
    
x =  torch.rand(2,512)
normal = RMSNorm(512)

out = normal(x)
print(x)
print(out)



#swish
#swish = x* sigmoid(x)
#swiglu = swish(xW +b) pointwise* (xV + C)

#correct my swiglu implenentation
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = find_multiple(hidden_dim, multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    
    def swish(self, x) :
        return x * torch.sigmoid(x)

    def forward(self, x):
        x1 = self.w1(x)
        x3 = self.w3(x)
        swiglu = self.swish(x1) * x3
        out = self.w2(swiglu)
        
        return out

