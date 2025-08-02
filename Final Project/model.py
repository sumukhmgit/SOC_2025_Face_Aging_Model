import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Residual Block with two conv layers for Generator.
    """
    def __init__(self, features):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, kernel_size=3),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, kernel_size=3),
            nn.InstanceNorm2d(features)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    """
    Generator network for CycleGAN. Uses 9 residual blocks for 256x256 images.
    Input and output channels default to 3 (RGB).
    """
    def __init__(self, ngf=64, n_residual_blocks=9, input_nc=3, output_nc=3):
        super().__init__()
        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]
        # Downsampling
        in_features = ngf
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_features, output_nc, kernel_size=7),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    """
    Discriminator (PatchGAN) for CycleGAN.
    """
    def __init__(self, ndf=64, input_nc=3):
        super().__init__()
        layers = []
        # Initial conv layer
        layers.append(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        # Pyramid of conv layers
        in_features = ndf
        out_features = in_features * 2
        for mult in [2, 4, 8]:
            stride = 2 if mult < 8 else 1
            layers.append(nn.Conv2d(in_features, out_features, kernel_size=4, stride=stride, padding=1))
            layers.append(nn.InstanceNorm2d(out_features))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_features = out_features
            out_features = in_features * 2
        # Output 1-channel patch matrix
        layers.append(nn.Conv2d(in_features, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
