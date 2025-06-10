import torch
import torch.nn as nn

#----------------------------------------------------------------------------

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    pass