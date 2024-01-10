import torch
import torch.nn as nn


class ConvEncoder(nn.Module):

    class ResBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()

            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.relu = nn.ReLU(inplace=True)

            self.downsample = None

            num_groups = out_channels // 8
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                    self.norm3
                )

        def forward(self, x):
            y = x
            y = self.relu(self.norm1(self.conv1(y)))
            y = self.relu(self.norm2(self.conv2(y)))

            if self.downsample is not None:
                x = self.downsample(x)

            return self.relu(x+y)

    def __init__(self, in_channels, out_channels=128, dropout=0.0):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = nn.BatchNorm2d(32)

        self.block1 = nn.Sequential(
            self.ResBlock(32, 64, stride=2),
            self.ResBlock(64, 64, stride=1)
        )
        self.block2 = nn.Sequential(
            self.ResBlock(64, 128, stride=2),
            self.ResBlock(128, 128, stride=1)
        )

        self.conv2 = nn.Conv2d(128, out_channels, kernel_size=1)

        self.dropout = None
        if dropout > 0.0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.conv2(x2)

        if self.training and self.dropout is not None:
            x3 = self.dropout(x3)

        return x3, [x, x1]


class ConvDecoder(nn.Module):

    class Block(nn.Module):
        def __init__(self, in_channels, mid_channels, out_channels):
            super().__init__()

            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
            self.relu = nn.ReLU(inplace=True)

            num_groups = out_channels // 8
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        def forward(self, x1, x2):
            x1 = self.relu(self.norm1(self.up(x1)))
            x1 = torch.cat((x1, x2), dim=1)
            x1 = self.relu(self.norm2(self.conv(x1)))
            return x1

    def __init__(self, in_channels, out_channels=32, dropout=0.0):
        super().__init__()

        self.decoder2 = self.Block(in_channels, 64+64, 64)
        self.decoder1 = self.Block(64, 32+32, 32)

        self.conv = nn.Conv2d(32, out_channels, kernel_size=1)

        self.dropout = None
        if dropout > 0.0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, x_list):
        y1_ = self.decoder2(x, x_list[-1])
        y_ = self.decoder1(y1_, x_list[-2])
        y = self.conv(y_)

        if self.training and self.dropout is not None:
            y = self.dropout(y)

        return y


class ConvFusionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        self.encoder = ConvEncoder(in_channels, 128, dropout=dropout)
        self.decoder = ConvDecoder(128, out_channels, dropout=dropout)

    def forward(self, image, mask):
        x = torch.cat((image, mask), dim=1)
        
        x_down, x_list = self.encoder(x)
        y = self.decoder(x_down, x_list)

        return y
    

class ConvPromptLayer(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, out_channels, kernel_size=1),
        )
        
        self.conv0 = None
        if in_channels != out_channels:
            self.conv0 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        y = self.conv(x)
        if self.conv0 is not None:
            x1 = self.conv0(x)
            y = y + x1
        
        return y


if __name__ == '__main__':
    x = torch.ones((1, 3, 256, 256))
    x1 = torch.zeros((1, 1, 256, 256))

    net = ConvFusionLayer(4, 8)
    y = net(x, x1)
    print(y.shape)
