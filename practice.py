import torch
import numpy as np

if __name__ == '__main__':
    x = np.random.randn(4, 1,28,28)
    y=torch.cat(x,0)
    print(y.shape)
