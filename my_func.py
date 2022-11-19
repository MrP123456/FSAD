import torch


def cov(x):
    '''
    输入: x [m,n]
    输出: y [n,n] (x的协方差矩阵)
    用于代替 torch.cov
    '''
    x = x - torch.mean(x, dim=0)
    y = torch.matmul(x, x.T) / (x.shape[0] - 1)
    return y


if __name__ == '__main__':
    x = torch.randn([3, 4])
    y = cov(x)
    print(y.shape)
