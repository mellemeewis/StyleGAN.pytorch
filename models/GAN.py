"""
-------------------------------------------------
   File Name:    GAN.py
   Author:       Zhonghao Huang
   Date:         2019/10/17
   Description:  Modified from:
                 https://github.com/akanimax/pro_gan_pytorch
                 https://github.com/lernapparat/lernapparat
                 https://github.com/NVlabs/stylegan
-------------------------------------------------
"""

import sys, os
import datetime
import time
import timeit
import copy
import random
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate
from torch.autograd import Variable


import models.Losses as Losses
from data import get_data_loader
from models import update_average
from models.Blocks import DiscriminatorTop, DiscriminatorBlock, InputBlock, GSynthesisBlock
from models.CustomLayers import EqualizedConv2d, PixelNormLayer, EqualizedLinear, Truncation
from models.StyleEncoder import StyleEncoder

import slack_util


class GMapping(nn.Module):

    def __init__(self, latent_size=512, dlatent_size=512, dlatent_broadcast=None, 
                 mapping_layers=8, mapping_fmaps=512, mapping_lrmul=0.01, mapping_nonlinearity='lrelu',
                 use_wscale=True, normalize_latents=True, **kwargs):
        """
        Mapping network used in the StyleGAN paper.

        :param latent_size: Latent vector(Z) dimensionality.
        # :param label_size: Label dimensionality, 0 if no labels.
        :param dlatent_size: Disentangled latent (W) dimensionality.
        :param dlatent_broadcast: Output disentangled latent (W) as [minibatch, dlatent_size]
                                  or [minibatch, dlatent_broadcast, dlatent_size].
        :param mapping_layers: Number of mapping layers.
        :param mapping_fmaps: Number of activations in the mapping layers.
        :param mapping_lrmul: Learning rate multiplier for the mapping layers.
        :param mapping_nonlinearity: Activation function: 'relu', 'lrelu'.
        :param use_wscale: Enable equalized learning rate?
        :param normalize_latents: Normalize latent vectors (Z) before feeding them to the mapping layers?
        :param kwargs: Ignore unrecognized keyword args.
        """

        super().__init__()

        self.latent_size = latent_size
        self.mapping_fmaps = mapping_fmaps
        self.dlatent_size = dlatent_size
        self.dlatent_broadcast = dlatent_broadcast

        # Activation function.
        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[mapping_nonlinearity]

        # Embed labels and concatenate them with latents.
        # TODO

        layers = []
        # Normalize latents.
        if normalize_latents:
            layers.append(('pixel_norm', PixelNormLayer()))

        # Mapping layers. (apply_bias?)
        layers.append(('dense0', EqualizedLinear(self.latent_size, self.mapping_fmaps,
                                                 gain=gain, lrmul=mapping_lrmul, use_wscale=use_wscale)))
        layers.append(('dense0_act', act))
        for layer_idx in range(1, mapping_layers):
            fmaps_in = self.mapping_fmaps
            fmaps_out = self.dlatent_size if layer_idx == mapping_layers - 1 else self.mapping_fmaps
            layers.append(
                ('dense{:d}'.format(layer_idx),
                 EqualizedLinear(fmaps_in, fmaps_out, gain=gain, lrmul=mapping_lrmul, use_wscale=use_wscale)))
            layers.append(('dense{:d}_act'.format(layer_idx), act))

        # Output.
        self.map = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        # First input: Latent vectors (Z) [mini_batch, latent_size].
        x = self.map(x)

        # Broadcast -> batch_size * dlatent_broadcast * dlatent_size
        if self.dlatent_broadcast is not None:
            x = x.unsqueeze(1).expand(-1, self.dlatent_broadcast, -1)
        return x


class GSynthesis(nn.Module):

    def __init__(self, dlatent_size=512, num_channels=3, resolution=1024,
                 fmap_base=8192, fmap_decay=1.0, fmap_max=512,
                 use_styles=True, const_input_layer=True, use_noise=True, nonlinearity='lrelu',
                 use_wscale=True, use_pixel_norm=False, use_instance_norm=True, blur_filter=None,
                 structure='linear', **kwargs):
        """
        Synthesis network used in the StyleGAN paper.

        :param dlatent_size: Disentangled latent (W) dimensionality.
        :param num_channels: Number of output color channels.
        :param resolution: Output resolution.
        :param fmap_base: Overall multiplier for the number of feature maps.
        :param fmap_decay: log2 feature map reduction when doubling the resolution.
        :param fmap_max: Maximum number of feature maps in any layer.
        :param use_styles: Enable style inputs?
        :param const_input_layer: First layer is a learned constant?
        :param use_noise: Enable noise inputs?
        # :param randomize_noise: True = randomize noise inputs every time (non-deterministic),
                                  False = read noise inputs from variables.
        :param nonlinearity: Activation function: 'relu', 'lrelu'
        :param use_wscale: Enable equalized learning rate?
        :param use_pixel_norm: Enable pixel_wise feature vector normalization?
        :param use_instance_norm: Enable instance normalization?
        :param blur_filter: Low-pass filter to apply when resampling activations. None = no filtering.
        :param structure: 'fixed' = no progressive growing, 'linear' = human-readable
        :param kwargs: Ignore unrecognized keyword args.
        """

        super().__init__()

        # if blur_filter is None:
        #     blur_filter = [1, 2, 1]

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.structure = structure

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        self.depth = resolution_log2 - 1

        self.num_layers = resolution_log2 * 2 - 2
        self.num_styles = self.num_layers if use_styles else 1

        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]

        # Early layers.
        self.init_block = InputBlock(nf(1), dlatent_size, const_input_layer, gain, use_wscale,
                                     use_noise, use_pixel_norm, use_instance_norm, use_styles, act)
        # create the ToRGB layers for various outputs
        rgb_converters = [EqualizedConv2d(nf(1), num_channels*2, 1, gain=1, use_wscale=use_wscale)]

        # Building blocks for remaining layers.
        blocks = []
        for res in range(3, resolution_log2 + 1):
            last_channels = nf(res - 2)
            channels = nf(res - 1)
            # name = '{s}x{s}'.format(s=2 ** res)
            blocks.append(GSynthesisBlock(last_channels, channels, blur_filter, dlatent_size, gain, use_wscale,
                                          use_noise, use_pixel_norm, use_instance_norm, use_styles, act))
            rgb_converters.append(EqualizedConv2d(channels, num_channels*2, 1, gain=1, use_wscale=use_wscale))

        self.blocks = nn.ModuleList(blocks)
        self.to_rgb = nn.ModuleList(rgb_converters)

        # register the temporary upsampler
        self.temporaryUpsampler = lambda x: interpolate(x, scale_factor=2)

    def forward(self, dlatents_in, noise, depth=0, alpha=0., labels_in=None):
        """
            forward pass of the Generator
            :param dlatents_in: Input: Disentangled latents (W) [mini_batch, num_layers, dlatent_size].
            :param labels_in:
            :param depth: current depth from where output is required
            :param alpha: value of alpha for fade-in effect
            :return: y => output
        """

        # print("\nGSynthesis\n", noise)

        assert depth < self.depth, "Requested output depth cannot be produced"
        # print(len(noise), len(self.blocks))
        # assert len(noise) == len(self.blocks) + 1, "Number of noise tensors does not correspond with state of model."

        if self.structure == 'fixed':
            x = self.init_block(dlatents_in[:, 0:2], noise[0])
            for i, block in enumerate(self.blocks):
                # print(f"\n\nBLOCK {i}:\n", noise[i+1])
                x = block(x, dlatents_in[:, 2 * (i + 1):2 * (i + 2)], noise[i+1])
            images_out = self.to_rgb[-1](x)
        elif self.structure == 'linear':
            x = self.init_block(dlatents_in[:, 0:2], noise[0])

            if depth > 0:
                for i, block in enumerate(self.blocks[:depth - 1]):
                    # print(f"\n\nBLOCK {i}:\n", noise[i+1])
                    x = block(x, dlatents_in[:, 2 * (i + 1):2 * (i + 2)], noise[i+1])

                residual = self.to_rgb[depth - 1](self.temporaryUpsampler(x))
                straight = self.to_rgb[depth](self.blocks[depth - 1](x, dlatents_in[:, 2 * depth:2 * (depth + 1)], noise[-1]))

                images_out = (alpha * straight) + ((1 - alpha) * residual)
            else:
                images_out = self.to_rgb[0](x)
        else:
            raise KeyError("Unknown structure: ", self.structure)

        return images_out


class Generator(nn.Module):

    def __init__(self, resolution, latent_size=512, dlatent_size=512,
                 truncation_psi=0.7, truncation_cutoff=8, dlatent_avg_beta=0.995,
                 style_mixing_prob=0.9, **kwargs):
        """
        # Style-based generator used in the StyleGAN paper.
        # Composed of two sub-networks (G_mapping and G_synthesis).

        :param resolution:
        :param latent_size:
        :param dlatent_size:
        :param truncation_psi: Style strength multiplier for the truncation trick. None = disable.
        :param truncation_cutoff: Number of layers for which to apply the truncation trick. None = disable.
        :param dlatent_avg_beta: Decay for tracking the moving average of W during training. None = disable.
        :param style_mixing_prob: Probability of mixing styles during training. None = disable.
        :param kwargs: Arguments for sub-networks (G_mapping and G_synthesis).
        """

        super(Generator, self).__init__()

        self.style_mixing_prob = style_mixing_prob
        print(self.style_mixing_prob)

        # Setup components.
        self.num_layers = (int(np.log2(resolution)) - 1) * 2
        self.g_mapping = GMapping(latent_size, dlatent_size, dlatent_broadcast=self.num_layers, **kwargs)
        self.g_synthesis = GSynthesis(resolution=resolution, **kwargs)

        if truncation_psi > 0:
            self.truncation = Truncation(avg_latent=torch.zeros(dlatent_size),
                                         max_layer=truncation_cutoff,
                                         threshold=truncation_psi,
                                         beta=dlatent_avg_beta)
        else:
            self.truncation = None

    def forward(self, latents_in, noise, depth, alpha, labels_in=None, mode='reconstruction'):
        """
        :param latents_in: First input: Latent vectors (Z) [mini_batch, latent_size].
        :param depth: current depth from where output is required
        :param alpha: value of alpha for fade-in effect
        :param labels_in: Second input: Conditioning labels [mini_batch, label_size].
        :return:
        """
        # print('GENERATOR\n', noise)

        dlatents_in = self.g_mapping(latents_in)

        if self.training:
            # Update moving average of W(dlatent).
            # TODO
            if self.truncation is not None:
                self.truncation.update(dlatents_in[0, 0].detach())

            # Perform style mixing regularization.
            if mode == 'style_mixing' and self.style_mixing_prob is not None and self.style_mixing_prob > 0:
                latents2 = torch.randn(latents_in.shape).to(latents_in.device)
                dlatents2 = self.g_mapping(latents2)
                layer_idx = torch.from_numpy(np.arange(self.num_layers)[np.newaxis, :, np.newaxis]).to(
                    latents_in.device)
                cur_layers = 2 * (depth + 1)
                mixing_cutoff = random.randint(1,
                                               cur_layers) if random.random() < self.style_mixing_prob else cur_layers
                dlatents_in = torch.where(layer_idx < mixing_cutoff, dlatents_in, dlatents2)

            # Apply truncation trick.
            if self.truncation is not None:
                dlatents_in = self.truncation(dlatents_in)

        fake_images = self.g_synthesis(dlatents_in, noise, depth, alpha)

        return fake_images


class Discriminator(nn.Module):

    def __init__(self, resolution, num_channels=3, fmap_base=8192, fmap_decay=1.0, fmap_max=512,
                 nonlinearity='lrelu', use_wscale=True, mbstd_group_size=4, mbstd_num_features=1,
                 blur_filter=None, structure='linear', **kwargs):
        """
        Discriminator used in the StyleGAN paper.

        :param num_channels: Number of input color channels. Overridden based on dataset.
        :param resolution: Input resolution. Overridden based on dataset.
        # label_size=0,  # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
        :param fmap_base: Overall multiplier for the number of feature maps.
        :param fmap_decay: log2 feature map reduction when doubling the resolution.
        :param fmap_max: Maximum number of feature maps in any layer.
        :param nonlinearity: Activation function: 'relu', 'lrelu'
        :param use_wscale: Enable equalized learning rate?
        :param mbstd_group_size: Group size for the mini_batch standard deviation layer, 0 = disable.
        :param mbstd_num_features: Number of features for the mini_batch standard deviation layer.
        :param blur_filter: Low-pass filter to apply when resampling activations. None = no filtering.
        :param structure: 'fixed' = no progressive growing, 'linear' = human-readable
        :param kwargs: Ignore unrecognized keyword args.
        """
        super(Discriminator, self).__init__()

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.mbstd_num_features = mbstd_num_features
        self.mbstd_group_size = mbstd_group_size
        self.structure = structure
        # if blur_filter is None:
        #     blur_filter = [1, 2, 1]

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        self.depth = resolution_log2 - 1

        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]

        # create the remaining layers
        blocks = []
        from_rgb = []
        for res in range(resolution_log2, 2, -1):
            # name = '{s}x{s}'.format(s=2 ** res)
            blocks.append(DiscriminatorBlock(nf(res - 1), nf(res - 2),
                                             gain=gain, use_wscale=use_wscale, activation_layer=act,
                                             blur_kernel=blur_filter))
            # create the fromRGB layers for various inputs:
            from_rgb.append(EqualizedConv2d(num_channels, nf(res - 1), kernel_size=1,
                                            gain=gain, use_wscale=use_wscale))
        self.blocks = nn.ModuleList(blocks)

        # Building the final block.
        self.final_block = DiscriminatorTop(self.mbstd_group_size, self.mbstd_num_features,
                                            in_channels=nf(2), intermediate_channels=nf(2),
                                            gain=gain, use_wscale=use_wscale, activation_layer=act)
        from_rgb.append(EqualizedConv2d(num_channels, nf(2), kernel_size=1,
                                        gain=gain, use_wscale=use_wscale))
        self.from_rgb = nn.ModuleList(from_rgb)

        # register the temporary downSampler
        self.temporaryDownsampler = nn.AvgPool2d(2)

    def forward(self, images_in, depth, alpha=1., labels_in=None):
        """
        :param images_in: First input: Images [mini_batch, channel, height, width].
        :param labels_in: Second input: Labels [mini_batch, label_size].
        :param depth: current height of operation (Progressive GAN)
        :param alpha: current value of alpha for fade-in
        :return:
        """

        assert depth < self.depth, "Requested output depth cannot be produced"

        if self.structure == 'fixed':
            x = self.from_rgb[0](images_in)
            for i, block in enumerate(self.blocks):
                x = block(x)
            scores_out = self.final_block(x)
        elif self.structure == 'linear':
            if depth > 0:
                residual = self.from_rgb[self.depth - depth](self.temporaryDownsampler(images_in))
                straight = self.blocks[self.depth - depth - 1](self.from_rgb[self.depth - depth - 1](images_in))
                x = (alpha * straight) + ((1 - alpha) * residual)

                for block in self.blocks[(self.depth - depth):]:
                    x = block(x)
            else:
                x = self.from_rgb[-1](images_in)

            scores_out = self.final_block(x)
        else:
            raise KeyError("Unknown structure: ", self.structure)

        return scores_out


class StyleGAN:

    def __init__(self, structure, resolution, num_channels, latent_size, update_encoder_as_discriminator, use_sleep, use_adverserial, use_vae,
                 g_args, d_args, e_args, g_opt_args, d_opt_args, e_opt_args, loss="relativistic-hinge", recon_loss='siglaplace', drift=0.001,
                 d_repeats=1, use_ema=False, ema_decay=0.999, noise_channel_dropout=0.25, betas=[0.001,0.1,0.001,0.001,0.0005,0.0005,0.0005,5,1], device=torch.device("cpu")):
        """
        Wrapper around the Generator and the Discriminator.

        :param structure: 'fixed' = no progressive growing, 'linear' = human-readable
        :param resolution: Input resolution. Overridden based on dataset.
        :param num_channels: Number of input color channels. Overridden based on dataset.
        :param latent_size: Latent size of the manifold used by the GAN
        :param g_args: Options for generator network.
        :param d_args: Options for discriminator network.
        :param g_opt_args: Options for generator optimizer.
        :param d_opt_args: Options for discriminator optimizer.
        :param loss: the loss function to be used
                     Can either be a string =>
                          ["wgan-gp", "wgan", "lsgan", "lsgan-with-sigmoid",
                          "hinge", "standard-gan" or "relativistic-hinge"]
                     Or an instance of GANLoss
        :param drift: drift penalty for the
                      (Used only if loss is wgan or wgan-gp)
        :param d_repeats: How many times the discriminator is trained per G iteration.
        :param use_ema: boolean for whether to use exponential moving averages
        :param ema_decay: value of mu for ema
        :param device: device to run the GAN on (GPU / CPU)
        """

        # state of the object
        assert structure in ['fixed', 'linear']
        self.output_resolution = resolution
        self.structure = structure
        self.depth = int(np.log2(resolution)) - 1
        self.latent_size = latent_size
        self.device = device
        self.d_repeats = d_repeats
        self.update_encoder_as_discriminator = update_encoder_as_discriminator
        self.use_sleep = use_sleep
        self.use_vae = use_vae
        self.use_adverserial = use_adverserial
        self.noise_channel_dropout = nn.Dropout2d(p=noise_channel_dropout, inplace=False) if noise_channel_dropout>0 else None
        print(self.noise_channel_dropout)
        self.num_channels = num_channels
        # b = betas[0]         
        # self.betas = [b/32, b*(0.25**0), b*(0.25**1), b*(0.25**2), b*(0.25**3), b*(0.25**4), b*(0.25**5), betas[7], betas[8]]
        self.betas = betas
        self.__update_betas(kl_loss=None, noise=None)
        # self.betas = [b*(0.25**5)*32, b*(0.25**5), b*(0.25**4), b*(0.25**3), b*(0.25**2), b*(0.25**1), b*(0.25**0), betas[7], betas[8]]

        print(self.betas)

        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # Create the Generator and the Discriminator
        self.gen = Generator(num_channels=num_channels, resolution=resolution, structure=self.structure, **g_args).to(self.device)
        self.encoder = StyleEncoder((num_channels, resolution, resolution), **e_args).to(self.device)

        # if code is to be run on GPU, we can use DataParallel:
        # TODO

        # define the optimizers for the discriminator and generator
        self.__setup_gen_optim(**g_opt_args)
        self.__setup_encoder_optim(**e_opt_args)

        # if self.use_discriminator:
        #     self.dis = Discriminator(num_channels=num_channels, resolution=resolution, structure=self.structure, **d_args).to(self.device)
        #     self.__setup_dis_optim(**d_opt_args)

        # define the loss function used for training the GAN
        self.drift = drift
        self.recon_loss = recon_loss
        print(self.recon_loss)
        self.loss = self.__setup_loss(loss)

        # Use of ema
        if self.use_ema:
            # create a shadow copy of the generator
            self.gen_shadow = copy.deepcopy(self.gen)
            # updater function:
            self.ema_updater = update_average
            # initialize the gen_shadow weights equal to the weights of gen
            self.ema_updater(self.gen_shadow, self.gen, beta=0)

    def __update_betas(self, kl_loss=None, noise=None):

        if self.use_vae == False:
            return
        with torch.no_grad():
            # start = [i for i in self.betas]
            kl_betas = [i for i in  self.betas[:7]]

            if kl_loss and noise:
                relative_kl = []
                for kl, n in zip(kl_loss, noise):
                    size = np.prod(list(n.size()[1:]))
                    relative_kl.append(kl/size)
                max_index = np.argmax(relative_kl)
                # print(max_index, kl_betas)
                kl_betas[max_index] += 0.00001
                # print(kl_betas)
                # print(kl_betas, '\n\n')

            ## SOFTMAX
            kl_betas = np.exp(kl_betas) / np.exp(kl_betas).sum()

            self.betas[:7] = kl_betas
            # if kl_loss and noise:

            #     if self.betas==start:
            #         print('NO CHANGE', relative_kl)
            #     else:
            #         print("CHANGE ", relative_kl)

    def __setup_gen_optim(self, learning_rate, beta_1, beta_2, eps):
        self.gen_optim = torch.optim.Adam(self.gen.parameters(), lr=learning_rate, betas=(beta_1, beta_2), eps=eps)

    def __setup_dis_optim(self, learning_rate, beta_1, beta_2, eps):
        self.dis_optim = torch.optim.Adam(self.dis.parameters(), lr=learning_rate, betas=(beta_1, beta_2), eps=eps)

    def __setup_encoder_optim(self, learning_rate, beta_1, beta_2, eps):
        self.encoder_optim = torch.optim.Adam(self.encoder.parameters(), lr=learning_rate, betas=(beta_1, beta_2), eps=eps)


    def __setup_loss(self, loss):
        if isinstance(loss, str):
            loss = loss.lower()  # lowercase the string
            if loss == 'vae':
                loss = Losses.GANLoss(None, self.recon_loss)
            elif loss == "standard-gan":
                loss = Losses.StandardGAN(self.dis, self.recon_loss)
            elif loss == "hinge":
                loss = Losses.HingeGAN(self.dis, self.recon_loss)
            elif loss == "relativistic-hinge":
                loss = Losses.RelativisticAverageHingeGAN(self.dis, self.recon_loss)
            elif loss == "logistic":
                loss = Losses.LogisticGAN(self.dis, self.recon_loss)
            else:
                raise ValueError("Unknown loss function requested")

        elif not isinstance(loss, Losses.GANLoss):
            raise ValueError("loss is neither an instance of GANLoss nor a string")

        return loss

    def sample_latent_and_noise_from_encoder_output(self, z, noise):
        ## Sample z
        b, z_length = z.size()[0], z.size()[1]//2
        zmean, zlsig = z[:, :z_length], z[:, z_length:]
        eps = torch.randn(b, z_length).to(zmean.device)
        eps = Variable(eps)
        zsample = zmean + eps * (zlsig * 0.5).exp()

        if noise == None:
            return zsample
        ## Sample noise
        sample_noise = []
        for n in noise:
            if n is not None:
                b, c, h, w = n.size()
                mean = n[:, :c//2, :, :].view(b, -1)
                sig = n[:, c//2:, :, :].view(b, -1)
                eps = torch.randn(b, c//2, h, w).view(b, -1).to(zmean.device)
                eps = Variable(eps)
                sample_n = mean + eps * (sig * 0.5).exp()
                sample_noise.append(sample_n.view(b, c//2, h, w))

        return zsample, sample_noise

    def sample_latent(self, b, depth):
        """
        Samples latents from the normal distribution.
        :param b:
        :param zsize:
        :param outsize:
        :param depth:
        :param zchannels:
        :param dev:
        :return:
        """

        h, w =self.output_resolution, self.output_resolution
        zc0, zc1, zc2, zc3, zc4, zc5 = self.encoder.zchannels
        zsize = self.latent_size


        n = [None] * 6

        z = torch.randn(b, zsize, device=self.device)

        n[0] = torch.randn(b, zc0, h, w, device=self.device)

        if depth >=1:
            n[1] = torch.randn(b, zc1, h // 2, w // 2, device=self.device)

        if depth >= 2:
            n[2] = torch.randn(b, zc2, h // 4, w // 4, device=self.device)

        if depth >= 3:
            n[3] = torch.randn(b, zc3, h // 8, w // 8, device=self.device)

        if depth >= 4:
            n[4] = torch.randn(b, zc4, h // 16, w // 16, device=self.device)

        if depth >= 5:
            n[5] = torch.randn(b, zc5, h // 32, w // 32, device=self.device)

        return z, n


    def sample_images(self, image_distr, distribution, n=1, eps=None):

        if image_distr is None:
            return None
        b, c, h, w = image_distr.size()

        if distribution == 'siglaplace':

            loc = image_distr[:, :c//2, :, :].view(b, -1)
            scale = image_distr[:, c//2:, :, :].clamp(min=0.001).view(b, -1)

            distribution = torch.distributions.laplace.Laplace(loc, scale, validate_args=None)
            sample = distribution.rsample(sample_shape=(n,))
            sample = torch.sigmoid(sample)

        return sample.view(b, c//2, h, w)

    def __progressive_down_sampling(self, real_batch, depth, alpha):
        """
        private helper for down_sampling the original images in order to facilitate the
        progressive growing of the layers.

        :param real_batch: batch of real samples
        :param depth: depth at which training is going on
        :param alpha: current value of the fade-in alpha
        :return: real_samples => modified real batch of samples
        """

        from torch.nn import AvgPool2d
        from torch.nn.functional import interpolate

        if self.structure == 'fixed':
            return real_batch

        # down_sample the real_batch for the given depth
        down_sample_factor = int(np.power(2, self.depth - depth - 1))
        prior_down_sample_factor = max(int(np.power(2, self.depth - depth)), 0)

        ds_real_samples = AvgPool2d(down_sample_factor)(real_batch)

        if depth > 0:
            prior_ds_real_samples = interpolate(AvgPool2d(prior_down_sample_factor)(real_batch), scale_factor=2)
        else:
            prior_ds_real_samples = ds_real_samples

        # real samples are a combination of ds_real_samples and prior_ds_real_samples
        real_samples = (alpha * ds_real_samples) + ((1 - alpha) * prior_ds_real_samples)

        # return the so computed real_samples
        return real_samples

    def update_enc_as_discriminator(self, real_batch, depth, alpha, print_=False):
        """
        performs one step of weight update on discriminator using the batch of data

        :param noise: input noise of sample generation
        :param real_batch: real samples batch
        :param depth: current depth of optimization
        :param alpha: current alpha for fade-in
        :return: current loss (Wasserstein loss)
        """

        real_samples = self.__progressive_down_sampling(real_batch, depth, alpha)
        b = real_samples.size()[0]
        # generate a batch of samples

        for i in range(4): #tesst 2
            sample_z, sample_n = self.sample_latent(b, depth)
            gen_out = self.gen(sample_z, sample_n[::-1], depth, alpha, mode='reconstruction').detach()
            fake_samples = self.sample_images(gen_out, self.recon_loss).detach()


            z_recon_real = self.encoder(real_samples, depth)#z_recon_real, noise_recon_real = self.encoder(real_samples, depth)
            z_recon_fake = self.encoder(fake_samples, depth)#z_recon_fake, noise_recon_fake = self.encoder(fake_samples, depth)


            real_loss = self.loss.kl_discriminator(z_recon_real, None, print_=print_)#real_loss = self.loss.kl_discriminator(z_recon_real, noise_recon_real, print_=print_)
            fake_loss = self.loss.enc_as_dis_loss(z_recon_fake, None, sample_z, sample_n, print_=print_)#fake_loss = self.loss.enc_as_dis_loss(z_recon_fake, noise_recon_fake, sample_z, sample_n, print_=print_)

            real_total = real_loss[0]# + real_loss[1] + real_loss[2] + real_loss[3] + real_loss[4] + real_loss[5] + real_loss[6] 
            fake_total = fake_loss[0]# + 0.5*fake_loss[1] + 0.5*fake_loss[2] + 0.5*fake_loss[3] + 0.5*fake_loss[4] + 0.5*fake_loss[5] + 0.5*fake_loss[6] 
            dis_loss = real_total + fake_total

            # optimize discriminator
            self.encoder_optim.zero_grad()
            dis_loss.backward()
            nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.)

            self.encoder_optim.step()
        return dis_loss

    def vae_phase(self, z_distr, noise_distr, z, noise, images, depth, alpha):
        """
        performs one step of weight update on generator for the given batch_size

        :param noise: input random noise required for generating samples
        :param real_batch: batch of real samples
        :param depth: depth of the network at which optimization is done
        :param alpha: value of alpha for fade-in effect
        :return: current loss (Wasserstein estimate)
        """
        # betas = self.betas
        betas=self.betas
        b = images.size()[0]
        recon_target = self.__progressive_down_sampling(images, depth, alpha)

        z_distr, noise_distr = self.encoder(images, current_depth)
        zsample, noise_sample = self.sample_latent_and_noise_from_encoder_output(z_distr, noise_distr)
        if self.noise_channel_dropout:
            noise_sample = [self.noise_channel_dropout(n) for n in noise_sample]

        # generate reconstruction:
        reconstruction = self.gen(zsample, noise_sample[::-1], depth, alpha, mode='reconstruction')         

        # Change this implementation for making it compatible for relativisticGAN
        recon_loss = self.loss.reconstruction_loss(reconstruction, recon_target)
        kl_loss = self.loss.kl_loss(z_distr, noise_distr)

        # loss = recon_loss + kl_loss + adverserial_loss if self.use_discriminator else recon_loss + kl_loss
        while len(kl_loss) < 7:
            kl_loss.append(torch.tensor([0]).to(self.device))
        kl_total = kl_loss[0] * betas[0] + kl_loss[1] * betas[1] + kl_loss[2] * betas[2] + kl_loss[3] * betas[3] + kl_loss[4] * betas[4] + kl_loss[5] * betas[5] + kl_loss[6] * betas[6]
        loss = betas[7] * recon_loss + kl_total

        # optimize the generator and encoder
        self.gen_optim.zero_grad()
        self.encoder_optim.zero_grad()

        loss.backward()
        # Gradient Clipping
        nn.utils.clip_grad_norm_(self.gen.parameters(), max_norm=1.)
        nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.)

        self.gen_optim.step()
        self.encoder_optim.step()

        # if use_ema is true, apply ema to the generator parameters
        if self.use_ema:
            self.ema_updater(self.gen_shadow, self.gen, self.ema_decay)

        return [round(k.item(),5) for k in kl_loss], recon_loss.item()

    def sleep_phase(self, b, depth, alpha):
        sample_z, sample_n = self.sample_latent(b, depth)
        with torch.no_grad():
            gen_out = self.gen(sample_z, sample_n[::-1], depth, alpha, mode='reconstruction')   
            images = self.sample_images(gen_out, self.recon_loss)

        z_recon, noise_recon = self.encoder(images, depth)

        sleep_loss = self.loss.sleep_loss(z_recon, noise_recon, sample_z, sample_n)
        sleep_total = sleep_loss[0] * 1 + sleep_loss[1] * 1 + sleep_loss[2] * 1 + sleep_loss[3] * 1 + sleep_loss[4] * 1 + sleep_loss[5] * 1 + sleep_loss[6] * 1
        
        self.encoder_optim.zero_grad()
        self.gen_optim.zero_grad()

        sleep_total.backward()
        nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.)
        self.encoder_optim.step()
        return round(sleep_total.item(), 3)

    def adverserial_phase(self, b, depth, alpha):
        betas = self.betas
        sample_z, sample_n = self.sample_latent(b, depth)
        
        gen_out = self.gen(sample_z, sample_n[::-1], depth, alpha, mode='reconstruction')   
        images = self.sample_images(gen_out, self.recon_loss)

        z_recon = self.encoder(images, depth)#z_recon, noise_recon = self.encoder(images, depth)

        adverserial_loss = self.loss.kl_discriminator(z_recon, None)# adverserial_loss = self.loss.kl_discriminator(z_recon, noise_recon)        


        adverserial_total = adverserial_loss[0] * betas[0] #+ adverserial_loss[1] * betas[1] + adverserial_loss[2] * betas[2] + adverserial_loss[3] * betas[3] + adverserial_loss[4] * betas[4] + adverserial_loss[5] * betas[5] + adverserial_loss[6] * betas[6]


        self.encoder_optim.zero_grad()
        self.gen_optim.zero_grad()

        adverserial_total.backward()

        nn.utils.clip_grad_norm_(self.gen.parameters(), max_norm=1.)
        self.gen_optim.step()
        return round(adverserial_total.item(), 3)

        # noise_detached = [n.detach() for n in noise]
        # reconstruction_style_mixing = self.gen(z.detach(), noise_detached, depth, alpha, mode='style_mixing')
        # adverserial_loss = self.loss.gen_loss(real_samples, reconstruction_style_mixing, depth, alpha)





    @staticmethod
    def create_grid(samples, scale_factor, img_file):
        """
        utility function to create a grid of GAN samples

        :param samples: generated samples for storing
        :param scale_factor: factor for upscaling the image
        :param img_file: name of file to write
        :return: None (saves a file)
        """
        from torchvision.utils import save_image
        from torch.nn.functional import interpolate

        # upsample the image
        if scale_factor > 1:
            samples = interpolate(samples, scale_factor=scale_factor)

        # save the images:
        # save_image(samples, img_file, nrow=int(np.sqrt(len(samples))),
        #            normalize=True, scale_each=True, pad_value=128, padding=1)
        save_image(samples, img_file, nrow=8,
                   normalize=True, scale_each=True, pad_value=128, padding=1)

    def train(self, dataset, num_workers, epochs, batch_sizes, fade_in_percentage, logger, output,
              num_samples=36, start_depth=0, feedback_factor=100, checkpoint_factor=1):
        """
        Utility method for training the GAN. Note that you don't have to necessarily use this
        you can use the optimize_generator and optimize_discriminator for your own training routine.

        :param dataset: object of the dataset used for training.
                        Note that this is not the data loader (we create data loader in this method
                        since the batch_sizes for resolutions can be different)
        :param num_workers: number of workers for reading the data. def=3
        :param epochs: list of number of epochs to train the network for every resolution
        :param batch_sizes: list of batch_sizes for every resolution
        :param fade_in_percentage: list of percentages of epochs per resolution used for fading in the new layer
                                   not used for first resolution, but dummy value still needed.
        :param logger:
        :param output: Output dir for samples,models,and log.
        :param num_samples: number of samples generated in sample_sheet. def=36
        :param start_depth: start training from this depth. def=0
        :param feedback_factor: number of logs per epoch. def=100
        :param checkpoint_factor:
        :return: None (Writes multiple files to disk)
        """

        assert self.depth <= len(epochs), "epochs not compatible with depth"
        assert self.depth <= len(batch_sizes), "batch_sizes not compatible with depth"
        assert self.depth <= len(fade_in_percentage), "fade_in_percentage not compatible with depth"

        # turn the generator and discriminator into train mode
        self.gen.train()
        self.encoder.train()

        if self.use_ema:
            self.gen_shadow.train()

        # create a global time counter
        global_time = time.time()

        # create fixed_input for debugging
        fixed_latent = torch.randn(num_samples, self.latent_size).to(self.device)
        fixed_noise = [torch.randn(num_samples, 1, self.output_resolution//32, self.output_resolution//32).to(self.device),
                                torch.randn(num_samples, 1, self.output_resolution//16, self.output_resolution//16).to(self.device),
                                torch.randn(num_samples, 1, self.output_resolution//8, self.output_resolution//8).to(self.device),
                                torch.randn(num_samples, 1, self.output_resolution//4, self.output_resolution//4).to(self.device),
                                torch.randn(num_samples, 1, self.output_resolution//2, self.output_resolution//2).to(self.device),
                                torch.randn(num_samples, 1, self.output_resolution, self.output_resolution).to(self.device)]

        # config depend on structure
        logger.info("Starting the training process ... \n")
        if self.structure == 'fixed':
            start_depth = self.depth - 1
        step = 1  # counter for number of iterations
        for current_depth in range(start_depth, self.depth):
            current_res = np.power(2, current_depth + 2)
            logger.info("Currently working on depth: %d", current_depth + 1)
            logger.info("Current resolution: %d x %d" % (current_res, current_res))

            ticker = 1

            # Choose training parameters and configure training ops.
            # TODO
            data = get_data_loader(dataset, batch_sizes[current_depth], num_workers)

            for epoch in range(1, epochs[current_depth] + 1):
                print_=True
                self.loss.update_simp(epoch-1, epochs[current_depth])

                start = timeit.default_timer()  # record time at the start of epoch

                logger.info("Epoch: [%d]" % epoch)
                # total_batches = len(iter(data))
                total_batches = len(data)

                fade_point = int((fade_in_percentage[current_depth] / 100)
                                 * epochs[current_depth] * total_batches)

                for (i, (batch,_)) in enumerate(data, 1):
                    # calculate the alpha for fading in the layers
                    alpha = ticker / fade_point if ticker <= fade_point else 1

                    # extract current batch of data for training
                    if torch.cuda.is_available():
                        images = batch.cuda()

                    # optimize the discriminator:
                    dis_loss = self.update_enc_as_discriminator(images, current_depth, alpha, print_=print_) if self.update_encoder_as_discriminator else 0
                    print_=False
                    # optimize the generator:
                    kl_loss, recon_loss = self.vae_phase(images, current_depth, alpha) if self.use_vae else 0, 0

                    sleep_loss = self.sleep_phase(batch_sizes[current_depth], current_depth, alpha) if self.use_sleep else 0
                    adv_loss = self.adverserial_phase(batch_sizes[current_depth], current_depth, alpha) if self.use_adverserial else 0

                    self.__update_betas(kl_loss, [fixed_latent] + fixed_noise)
                    # provide a loss feedback
                    if i % int(total_batches / feedback_factor + 1) == 0 or i == 1:
                        print_=True
                        elapsed = time.time() - global_time
                        elapsed = str(datetime.timedelta(seconds=elapsed)).split('.')[0]
                        logger.info(
                            "Elapsed: [%s] Step: %d  Batch: %d  Sleep_Loss: %f  Diss_Loss: %f  AD_Loss: %f, KL_Loss: %s, ReconLoss: %f, Betas: %s"
                            % (elapsed, step, i, sleep_loss, dis_loss, adv_loss, kl_loss, recon_loss, self.betas))

                        # logger.info(
                        #     "Elapsed: [%s] Step: %d  Batch: %d  D_Loss: %f  AD_Loss: %f, KL_Loss: %f, ReconLoss: %f"
                        #     % (elapsed, step, i, dis_loss, adv_loss, kl_loss[0], recon_loss))
                        # create a grid of samples and save it
                        os.makedirs(os.path.join(output, 'samples'), exist_ok=True)
                        gen_img_file = os.path.join(output, 'samples', "gen_" + str(current_depth)
                                                    + "_" + str(epoch) + "_" + str(i) + ".png")

                        with torch.no_grad():

                            z = self.encoder(images, current_depth); _, noise_sample = self.sample_latent(images.size()[0], current_depth) #z, noise = self.encoder(images, current_depth)
                            zsample = self.sample_latent_and_noise_from_encoder_output(z, None) #zsample, noise_sample = self.sample_latent_and_noise_from_encoder_output(z, noise)      
                            reconstruction = self.gen(zsample, noise_sample[::-1], current_depth, alpha)[:,:self.num_channels,:,:].detach() if not self.use_ema else self.gen_shadow(zsample, noise_sample[::-1], current_depth, alpha)[:,:self.num_channels,:,:].detach()
                            mix_fixed_noise = self.gen(zsample, fixed_noise[-current_depth-1:], current_depth, alpha)[:,:self.num_channels,:,:].detach() if not self.use_ema else self.gen_shadow(zsample, fixed_noise[:current_depth+1], current_depth, alpha)[:,:self.num_channels,:,:].detach()
                            fixed_reconstruction = self.gen(fixed_latent, fixed_noise[-current_depth-1:], current_depth, alpha)[:,:self.num_channels,:,:].detach() if not self.use_ema else self.gen_shadow(fixed_latent, fixed_noise[:current_depth+1], current_depth, alpha)[:,:self.num_channels,:,:].detach()

                            self.create_grid(
                                samples=torch.cat([images, torch.sigmoid(reconstruction), torch.sigmoid(mix_fixed_noise), torch.sigmoid(fixed_reconstruction)]),
                                scale_factor=int(
                                    np.power(2, self.depth - current_depth - 1)) if self.structure == 'linear' else 1,
                                img_file=gen_img_file,
                            )


                            try:
                                s = "Elapsed: [%s] Step: %d  Batch: %d  Sleep_Loss: %f  Diss_Loss: %f  AD_Loss: %f, KL_Loss: %s, ReconLoss: %f, Betas: %s" % (elapsed, step, i, sleep_loss, dis_loss, adv_loss, kl_loss, recon_loss, self.betas)
                                slack_util.send_image(gen_img_file, s)
                            except Exception as e:
                                print("Sending image failed.")
                                print(e)
                    # increment the alpha ticker and the step
                    ticker += 1
                    step += 1

                elapsed = timeit.default_timer() - start
                elapsed = str(datetime.timedelta(seconds=elapsed)).split('.')[0]
                logger.info("Time taken for epoch: %s\n" % elapsed)



                if checkpoint_factor > 0:
                    if epoch % checkpoint_factor == 0 or epoch == 1 or epoch == epochs[current_depth]:
                        save_dir = os.path.join(output, 'models')
                        os.makedirs(save_dir, exist_ok=True)
                        gen_save_file = os.path.join(save_dir, "GAN_GEN_" + str(current_depth) + "_" + str(epoch) + ".pth")
                        # if self.use_discriminator:
                        #     dis_save_file = os.path.join(save_dir, "GAN_DIS_" + str(current_depth) + "_" + str(epoch) + ".pth")
                        enc_save_file = os.path.join(save_dir, "GAN_ENC_" + str(current_depth) + "_" + str(epoch) + ".pth")

                        gen_optim_save_file = os.path.join(
                            save_dir, "GAN_GEN_OPTIM_" + str(current_depth) + "_" + str(epoch) + ".pth")
                        # if self.use_discriminator:
                        #     dis_optim_save_file = os.path.join(
                        #         save_dir, "GAN_DIS_OPTIM_" + str(current_depth) + "_" + str(epoch) + ".pth")
                        enc_optim_save_file = os.path.join(
                            save_dir, "GAN_ENC_OPTIM_" + str(current_depth) + "_" + str(epoch) + ".pth")


                        logger.info("Saving the model to: %s\n" % gen_save_file)
                        torch.save(self.gen.state_dict(), gen_save_file)
                        # if self.use_discriminator:
                        #     torch.save(self.dis.state_dict(), dis_save_file)
                        torch.save(self.encoder.state_dict(), dis_save_file)

                        torch.save(self.gen_optim.state_dict(), gen_optim_save_file)
                        # if self.use_discriminator:
                        #     torch.save(self.dis_optim.state_dict(), dis_optim_save_file)
                        torch.save(self.encoder_optim.state_dict(), dis_optim_save_file)

                        # also save the shadow generator if use_ema is True
                        if self.use_ema:
                            gen_shadow_save_file = os.path.join(
                                save_dir, "GAN_GEN_SHADOW_" + str(current_depth) + "_" + str(epoch) + ".pth")
                            torch.save(self.gen_shadow.state_dict(), gen_shadow_save_file)
                            logger.info("Saving the model to: %s\n" % gen_shadow_save_file)

        logger.info('Training completed.\n')


if __name__ == '__main__':
    print('Done.')
