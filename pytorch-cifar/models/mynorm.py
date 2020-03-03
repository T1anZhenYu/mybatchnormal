import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x,running_mean,running_var, eps, momentum):
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        # print('mean size:', mean.size())
        # use biased var in train
        var = (x - mean).pow(2).mean(dim=(0, 2, 3), keepdim=True)
        mean = mean.squeeze()
        var = var.squeeze()
        n = x.numel() / (x.size(1))
        with torch.no_grad():
            running_mean.copy_(momentum * mean\
                                + (1 - momentum) * running_mean)
            # update running_var with unbiased var
            running_var.copy_(momentum * var * n / (n - 1) \
                               + (1 - momentum) * running_var)
        y = (x - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + eps))
        ctx.eps = eps
        ctx.save_for_backward(y, var, )
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        y, var= ctx.saved_variables

        g = grad_output
        # print("g:",g[:,0,:,:])
        gy = (g * y).mean(dim=(0,2,3),keepdim=True)*y
        # print("g*y",(g * y).mean(dim=(0,2,3),keepdim=True)[:,0,:,:])
        # print("gy:",gy[:,0,:,:])
        g1 = g.mean(dim=(0,2,3),keepdim=True)
        # print("g1:",g1[:,0,:,:])
        gx_ = g - g1 -gy
        # print("g - g1",(g-g1)[:,0,:,:])
        # print("gx_:",gx_[:,0,:,:])
        gx = 1. / torch.sqrt(var[None, :, None, None] + eps) * (gx_)
        # print("gx:",gx[:,0,:,:])
        return gx, None,None,None,None

class MyBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(MyBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
    def forward(self,x):
        self._check_input_dim(x)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        if self.training:
            x = BatchNormFunction.apply(x,self.running_mean,self.running_var,self.eps,exponential_average_factor)
        else:
            mean = self.running_mean
            var = self.running_var
            x = (x - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return x