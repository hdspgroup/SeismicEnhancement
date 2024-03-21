# -----------------------------------------------------------------------------
# Original Source: Auto-Encoding Variational Bayes
# GitHub Repository: AntixK/PyTorch-VAE
# Link: https://github.com/AntixK/PyTorch-VAE
# -----------------------------------------------------------------------------
# Description:
# A Collection of Variational Autoencoders (VAE) in PyTorch.
# -----------------------------------------------------------------------------

import torch
from .base import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class AttentionBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.channels = channels

        self.group_norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.mhsa = nn.MultiheadAttention(embed_dim=self.channels, num_heads=4, batch_first=True)

    def forward(self, x):
        B, _, H, W = x.shape
        h = self.group_norm(x)
        h = h.reshape(B, self.channels, H * W).swapaxes(1, 2)  # [B, C, H, W] --> [B, C, H * W] --> [B, H*W, C]
        h, _ = self.mhsa(h, h, h)  # [B, H*W, C]
        h = h.swapaxes(2, 1).view(B, self.channels, H, W)  # [B, C, H*W] --> [B, C, H, W]
        return x + h

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels, dropout_rate=0.1, time_emb_dims=512, apply_attention=False, context_embd_dims=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.act_fn = nn.SiLU()
        # Group 1

        if self.in_channels > 1:
            # self.normlize_1 = nn.GroupNorm(
            #     num_groups=8, num_channels=self.in_channels)
            self.normlize_1 = nn.BatchNorm2d(num_features=self.in_channels)
        else:
            self.normlize_1 = nn.Identity()
        
        self.conv_1 = nn.Conv2d(in_channels=self.in_channels,
                                out_channels=self.out_channels, kernel_size=3, stride=1, padding="same")

        # Group 2
        # self.normlize_2 = nn.GroupNorm(
        #     num_groups=8, num_channels=self.out_channels)
        self.normlize_2 = nn.BatchNorm2d(num_features=self.out_channels)
        
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv_2 = nn.Conv2d(in_channels=self.out_channels,
                                out_channels=self.out_channels, kernel_size=3, stride=1, padding="same")

        if self.in_channels != self.out_channels:
            self.match_input = nn.Conv2d(
                in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1)
        else:
            self.match_input = nn.Identity()

        if apply_attention:
            self.attention = AttentionBlock(channels=self.out_channels)
        else:
            self.attention = nn.Identity()

    def forward(self, x, ):
        # group 1
        h = self.act_fn(self.normlize_1(x))
        h = self.conv_1(h)

        # group 2
        h = self.act_fn(self.normlize_2(h))
        h = self.dropout(h)
        h = self.conv_2(h)

        # Residual and attention
        h = h + self.match_input(x)
        h = self.attention(h)

        return h


class VanillaVAE(BaseVAE):
    def __init__(
        self, in_channels: int, latent_dim: int, hidden_dims: List = None, **kwargs
    ) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            # hidden_dims = [16, 32, 64, 128, 256, 512]
            hidden_dims = [16, 32, 64, 64, 128]

        self.last_hidden_dim = hidden_dims[-1]
        self.final_size = 128 // (2 ** len(hidden_dims))
        # Build Encoder
        current_dim = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    # nn.Conv2d(
                    #     current_dim,
                    #     out_channels=h_dim,
                    #     kernel_size=3,
                    #     stride=1,
                    #     padding=1,
                    #     bias=False,
                    # ),
                    # nn.BatchNorm2d(h_dim),
                    ResnetBlock(in_channels=current_dim, out_channels=h_dim, apply_attention=False),
                    nn.LeakyReLU(),
                    nn.MaxPool2d(2, 2),
                )
            )
            current_dim = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(
            hidden_dims[-1] * (self.final_size**2), latent_dim)
        self.fc_var = nn.Linear(
            hidden_dims[-1] * (self.final_size**2), latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(
            latent_dim, hidden_dims[-1] * (self.final_size**2)
        )

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=(2, 2)),
                    # nn.Conv2d(
                    #     hidden_dims[i],
                    #     hidden_dims[i + 1],
                    #     kernel_size=3,
                    #     stride=1,
                    #     padding=1,
                    #     bias=False,
                    # ),
                    # nn.BatchNorm2d(hidden_dims[i + 1]),
                    ResnetBlock(in_channels=hidden_dims[i], out_channels=hidden_dims[i + 1], apply_attention=False),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2)),
            # nn.Conv2d(
            #     hidden_dims[-1],
            #     hidden_dims[-1],
            #     kernel_size=3,
            #     stride=1,
            #     padding=1,
            #     bias=False,
            # ),
            # nn.BatchNorm2d(hidden_dims[-1]),
            ResnetBlock(in_channels=hidden_dims[-1], out_channels=hidden_dims[-1], apply_attention=False),
            nn.LeakyReLU(),
            nn.Conv2d(
                hidden_dims[-1], out_channels=in_channels, kernel_size=1, padding=0
            ),
            # nn.Tanh()
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.last_hidden_dim,
                             self.final_size, self.final_size)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        # Account for the minibatch samples from the dataset
        kld_weight = kwargs["M_N"]
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        loss = recons_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": -kld_loss.detach(),
        }

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
