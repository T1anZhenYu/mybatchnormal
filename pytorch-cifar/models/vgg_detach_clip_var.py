'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import numpy as np
class BatchNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x,running_mean,running_var, eps, momentum):
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        # mean = torch.clamp(mean,min=0,max=4)
        # print('mean size:', mean.size())
        # use biased var in train
        var = (x - mean).pow(2).mean(dim=(0, 2, 3), keepdim=True)
        var = torch.clamp(var,min=0.05,max=4)
        mean = mean.squeeze()
        var = var.squeeze()
        n = x.numel() / (x.size(1))

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
        # gy = (g * y).mean(dim=(0,2,3),keepdim=True)*y
        # print("g*y",(g * y).mean(dim=(0,2,3),keepdim=True)[:,0,:,:])
        # print("gy:",gy[:,0,:,:])
        g1 = g.mean(dim=(0,2,3),keepdim=True)
        # print("g1:",g1[:,0,:,:])
        gx_ = g -g1
        # print("g - g1",(g-g1)[:,0,:,:])
        # print("gx_:",gx_[:,0,:,:])
        gx = 1. / torch.sqrt(var[None, :, None, None] + eps) * (gx_)
        # print("gx:",gx[:,0,:,:])
        return gx, None,None,None,None

class GradBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(GradBatchNorm, self).__init__(
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
            y = BatchNormFunction.apply(x,self.running_mean,self.running_var,self.eps,exponential_average_factor)
        else:
            mean = self.running_mean
            var = self.running_var
            y = (x - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            y = y * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return y
class DetachClipVar(nn.BatchNorm2d):
    def __init__(self, num_features, eps=0.01, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(DetachClipVar, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.total = 1
    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean(dim=(0, 2, 3), keepdim=True)
            # print('mean size:', mean.size())
            # use biased var in train
            var = (input-mean).pow(2).mean(dim=(0,2, 3), keepdim=True)
            var = torch.clamp(var,min=0.05,max=4)
            mean = mean.squeeze()
            var = var.squeeze()

            n = input.numel() / (input.size(1))
            # self.total = self.total + 1
            # if n==4 and self.total %300 == 1 :
            #     print("saving")
            #     dic = {}
            #     dic['var']=var.cpu().detach().numpy()
            #     dic['mean']=mean.cpu().detach().numpy()
            #
            #     np.savez("./npz/"+str(self.total)+"tempiter",**dic)
            # print("n:",n)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
                # for i in range(var.size(0)):
                #     self.running_var = exponential_average_factor * var[i] * n / (n - 1)\
                #         + (1 - exponential_average_factor) * self.running_var
                # self.running_var = exponential_average_factor * var * n / (n - 1)\
                # + (1 - exponential_average_factor) * self.running_var
            input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps)).detach()
        else:
            mean = self.running_mean
            var = self.running_var
            input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))

        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG_DetachClipVar(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_DetachClipVar, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           GradBatchNorm(x),
                           nn.ReLU(inplace=False)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
