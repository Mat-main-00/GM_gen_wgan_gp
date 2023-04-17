"""
Part of this code was created based on the code in the following repositories.
    https://github.com/ayukat1016/gan_sample.git
"""
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class GroundMotionDatasets(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.data_path = df["file_name"]

        # This label is fake and not actually used
        self.label = df.iloc[:, 1].values

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        path = self.data_path[index]
        mat = np.load(path, allow_pickle=True)

        out_tensor = torch.from_numpy(mat.astype(np.float32)).clone()
        out_tensor = out_tensor.reshape(1, 1, -1)

        out_label = self.label[index]

        return out_tensor, out_label


class Generator(nn.Module):
    def __init__(self, nch):
        super(Generator, self).__init__()

        self.layers = nn.ModuleDict(
            {
                "layer0": nn.Sequential(
                    nn.ConvTranspose2d(
                        nch, 4096, kernel_size=(1, 4), stride=(1, 1), padding=(0, 0)
                    ),
                    nn.BatchNorm2d(4096),
                    nn.ReLU(),
                ),
                "layer1": nn.Sequential(
                    nn.ConvTranspose2d(
                        4096, 2048, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
                    ),
                    nn.BatchNorm2d(2048),
                    nn.ReLU(),
                ),
                "layer2": nn.Sequential(
                    nn.ConvTranspose2d(
                        2048, 1024, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
                    ),
                    nn.BatchNorm2d(1024),
                    nn.ReLU(),
                ),
                "layer3": nn.Sequential(
                    nn.ConvTranspose2d(
                        1024, 512, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
                    ),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                ),
                "layer4": nn.Sequential(
                    nn.ConvTranspose2d(
                        512, 256, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
                    ),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                ),
                "layer5": nn.Sequential(
                    nn.ConvTranspose2d(
                        256, 128, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
                    ),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                ),
                "layer6": nn.Sequential(
                    nn.ConvTranspose2d(
                        128, 64, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
                    ),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                ),
                "layer7": nn.Sequential(
                    nn.ConvTranspose2d(
                        64, 32, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
                    ),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                ),
                "layer8": nn.Sequential(
                    nn.ConvTranspose2d(
                        32, 16, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
                    ),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                ),
                "layer9": nn.Sequential(
                    nn.ConvTranspose2d(
                        16, 8, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
                    ),
                    nn.BatchNorm2d(8),
                    nn.ReLU(),
                ),
                "layer10": nn.Sequential(
                    nn.ConvTranspose2d(
                        8, 1, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
                    ),
                    # nn.BatchNorm2d(1024),
                    nn.Tanh(),
                ),
            }
        )

    def forward(self, z):
        for layer in self.layers.values():
            z = layer(z)
        return z


class Discriminator(nn.Module):
    def __init__(self, nch=1):
        super(Discriminator, self).__init__()

        self.layers = nn.ModuleDict(
            {
                "layer0": nn.Sequential(
                    nn.Conv2d(
                        nch, 4, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
                    ),
                    nn.LeakyReLU(negative_slope=0.2),
                ),
                "layer1": nn.Sequential(
                    nn.Conv2d(4, 8, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
                    nn.BatchNorm2d(8),
                    nn.LeakyReLU(negative_slope=0.2),
                ),
                "layer2": nn.Sequential(
                    nn.Conv2d(8, 16, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
                    nn.BatchNorm2d(16),
                    nn.LeakyReLU(negative_slope=0.2),
                ),
                "layer3": nn.Sequential(
                    nn.Conv2d(
                        16, 32, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
                    ),
                    nn.BatchNorm2d(32),
                    nn.LeakyReLU(negative_slope=0.2),
                ),
                "layer4": nn.Sequential(
                    nn.Conv2d(
                        32, 64, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
                    ),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(negative_slope=0.2),
                ),
                "layer5": nn.Sequential(
                    nn.Conv2d(
                        64, 128, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
                    ),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(negative_slope=0.2),
                ),
                "layer6": nn.Sequential(
                    nn.Conv2d(
                        128, 256, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
                    ),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(negative_slope=0.2),
                ),
                "layer7": nn.Sequential(
                    nn.Conv2d(
                        256, 512, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
                    ),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(negative_slope=0.2),
                ),
                "layer8": nn.Sequential(
                    nn.Conv2d(
                        512, 1024, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
                    ),
                    nn.BatchNorm2d(1024),
                    nn.LeakyReLU(negative_slope=0.2),
                ),
                "layer9": nn.Sequential(
                    nn.Conv2d(
                        1024, 2048, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
                    ),
                    nn.BatchNorm2d(2048),
                    nn.LeakyReLU(negative_slope=0.2),
                ),
                "layer10": nn.Sequential(
                    nn.Conv2d(
                        2048, 1, kernel_size=(1, 4), stride=(1, 1), padding=(0, 0)
                    )
                ),
            }
        )

    def forward(self, wave):
        for layer in self.layers.values():
            wave = layer(wave)
        return wave.squeeze()


def weight_init(m):
    class_name = m.__class__.__name__

    if class_name.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif class_name.find("Linear") != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif class_name.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def gradient_penalty(model, real_data, fake_data, device, wei=10):
    alpha_size = tuple((len(real_data), *(1,) * (real_data.dim() - 1)))
    alpha_t = torch.Tensor
    alpha = alpha_t(*alpha_size).to(device).uniform_()

    x_hat = (
        real_data.detach() * alpha + fake_data.detach() * (1 - alpha)
    ).requires_grad_()

    def eps_norm(_x):
        _x = _x.view(len(_x), -1)
        return (_x * _x + 1e-15).sum().sqrt()

    def bi_penalty(_x):
        return (_x - 1) ** 2

    disc_out_x_hat = model(x_hat)
    grad_x_hat = torch.autograd.grad(
        disc_out_x_hat,
        x_hat,
        grad_outputs=torch.ones(disc_out_x_hat.size()).to(device),
        create_graph=True,
        only_inputs=True,
    )[0]
    penalty = wei * bi_penalty(eps_norm(grad_x_hat)).mean()
    return penalty
