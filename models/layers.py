from torch import nn
import torch
import sys

Pool = nn.MaxPool2d

def batchnorm(x):
    return nn.BatchNorm2d(x.size()[1])(x)


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, bn = False, relu = True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x



#### Stacked Hourglass Network ####

class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        # Paper figure 4 left: input is added to the output within a residual module (residual connection)
        # if channel_input_size = channel_output_size we simply use the identity of x, else we use a 1x1 conv layer to
        # get the required channel number
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class Hourglass(nn.Module):
    def __init__(self, n, f, bn=None, increase=0):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Residual(f, f)
        # Lower branch
        self.pool1 = Pool(2, 2)
        self.low1 = Residual(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n-1, nf, bn=bn)
        else:
            self.low2 = Residual(nf, nf)
        self.low3 = Residual(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1  = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return up1 + up2





############ Soft Gated Skip Connections ##############


class SoftGatedSkipConnection(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(SoftGatedSkipConnection, self).__init__()
        if inp_dim != out_dim:
            self.need_skip_layer = True
            self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        else:
            self.need_skip_layer = False

        # for each channel we have one alpha parameter
        # as in paper alphas are initialized with zero
        self.alpha = nn.Parameter(torch.zeros((1,out_dim, 1, 1)))

    def forward(self, x):
        if self.need_skip_layer:
            x = self.skip_layer(x)
        x = x * self.alpha
        return x

class SoftGatedBlock(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(SoftGatedBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.relu1 = nn.ReLU()
        self.conv1 = Conv(inp_dim, int(out_dim/2), 3, relu=False)

        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.relu2 = nn.ReLU()
        self.conv2 = Conv(int(out_dim/2), int(out_dim/4), 3, relu=False)

        self.bn3 = nn.BatchNorm2d(int(out_dim/4))
        self.relu3 = nn.ReLU()
        self.conv3 = Conv(int(out_dim/4), int(out_dim/4), 3, relu=False)
        self.skip_connection = SoftGatedSkipConnection(inp_dim, out_dim)



    def forward(self, x):
        skip = self.skip_connection(x)

        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)

        x2 = self.bn2(x1)
        x2 = self.relu2(x2)
        x2 = self.conv2(x2)

        x3 = self.bn3(x2)
        x3 = self.relu3(x3)
        x3 = self.conv3(x3)

        stacked = torch.cat([x1, x2, x3], dim=1)

        out = skip + stacked

        return out



class SoftGatedHourglass(nn.Module):
    def __init__(self, n, f, bn=None, increase=0):
        super(SoftGatedHourglass, self).__init__()
        nf = f + increase
        self.up1 = SoftGatedBlock(f, f)

        # Lower branch
        self.pool1 = Pool(2, 2)
        self.low1 = SoftGatedBlock(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = SoftGatedHourglass(n-1, nf, bn=bn)
        else:
            self.low2 = SoftGatedBlock(nf, nf)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = Conv(2*nf, nf, 3)
        self.low3 = SoftGatedBlock(nf, f)


    def forward(self, x):
        up1  = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)

        low2 = self.low2(low1)
        up2  = self.up2(low2)

        stacked = torch.cat([up1, up2], dim=1)
        conv = self.conv(stacked)
        low3 = self.low3(conv)
        return low3
