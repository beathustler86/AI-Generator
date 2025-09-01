import math, torch, torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8, elementwise_affine=True, **_):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None
    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms
        return x * self.weight if self.weight is not None else x

class SinCosEmbedding(nn.Module):
    def __init__(self, dim, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe)
    def forward(self, x):
        return self.pe[:x.size(1)].unsqueeze(0)

operations = type("Ops", (), {
    "RMSNorm": RMSNorm,
    "LayerNorm": nn.LayerNorm,
    "Linear": nn.Linear,
    "sincos": SinCosEmbedding
})
