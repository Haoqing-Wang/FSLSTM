import torch.nn as nn
import torch.nn.functional as F

class Conv4(nn.Module):
    def __init__(self, avg_pool=False):
        super(Conv4, self).__init__()
        # set size
        self.hidden = 128
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=self.hidden, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=self.hidden, out_channels=int(self.hidden*1.5), kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
                                    nn.BatchNorm2d(num_features=int(self.hidden*1.5)),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=int(self.hidden*1.5), out_channels=self.hidden*2, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 2),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.4))
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=self.hidden*2, out_channels=self.hidden*4, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 4),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.5))
        self.keep_avg_pool = avg_pool
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        if self.keep_avg_pool:
            out = F.avg_pool2d(out, kernel_size=(2, 2), padding=1)
        out = out.view(x.size(0), -1)
        return out