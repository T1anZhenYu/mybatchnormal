'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import numpy as np

class Detach_mean(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(Detach_mean, self).__init__(
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
            mean = (input.mean(dim=(0, 2, 3), keepdim=True)).detach()
            # print('mean size:', mean.size())
            # use biased var in train
            var = (input-mean).pow(2).mean(dim=(0,2, 3), keepdim=True)
            mean = mean.squeeze()
            var = var.squeeze()

            n = input.numel() / (input.size(1))
            self.total = self.total + 1
            if n==4 and self.total %300 == 1 :
                print("saving")
                dic = {}
                dic['var']=var.cpu().detach().numpy()
                dic['mean']=mean.cpu().detach().numpy()
                np.savez("./npz/"+str(self.total)+"tempiter",**dic)
            # print("n:",n)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * (var) * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
                # for i in range(var.size(0)):
                #     self.running_var = exponential_average_factor * var[i] * n / (n - 1)\
                #         + (1 - exponential_average_factor) * self.running_var
                # self.running_var = exponential_average_factor * var * n / (n - 1)\
                # + (1 - exponential_average_factor) * self.running_var
            input = (input - (mean[None, :, None, None])) / (torch.sqrt(var[None, :, None, None] + self.eps))
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


class VGG_Detach_mean(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_Detach_mean, self).__init__()
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
                           Detach_mean(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
