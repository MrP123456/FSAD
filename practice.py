import torch
import numpy as np

if __name__ == '__main__':
    x = np.random.randn(8, 4)
    print(np.cov(x).shape)
