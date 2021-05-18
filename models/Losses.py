"""
-------------------------------------------------
   File Name:    Losses.py
   Author:       Zhonghao Huang
   Date:         2019/10/21
   Description:  Module implementing various loss functions
                 Copy from: https://github.com/akanimax/pro_gan_pytorch
-------------------------------------------------
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================
# Interface for the losses
# =============================================================

class GANLoss:
    """ Base class for all losses

        @args:
        dis: Discriminator used for calculating the loss
             Note this must be a part of the GAN framework
    """

    def __init__(self, dis, recon_loss):
        self.dis = dis
        self.recon_loss = recon_loss
        self.simp = 0

    def update_simp(self, current_epoch, total_epochs):
        epochs = total_epochs / 2
        grow = 1/ epochs
        self.simp = min(1, self.simp + grow)
        print(self.simp)

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        """
        calculate the discriminator loss using the following data
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :param height: current height at which training is going on
        :param alpha: current value of the fader alpha
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps, height, alpha):
        """
        calculate the generator loss
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :param height: current height at which training is going on
        :param alpha: current value of the fader alpha
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("gen_loss method has not been implemented")

    def kl_loss(self, latent, noise):

        b, l = latent.size()
        zmean, zlsig = latent[:, :l//2], latent[:, l//2:]
        kl = 0.5 * torch.sum(zlsig.exp() - zlsig + zmean.pow(2) - 1, dim=1)

        kl = torch.clamp(kl, min=0.01)
        kl_list = [kl]
        for i, n in enumerate(noise):
            if n is None:
                continue

            b, c, h, w = n.size()
            mean = n[:, :c//2, :, :].view(b, -1)
            sig = n[:, c//2:, :, :].view(b, -1)
            kl_list.append(torch.clamp(0.5 * torch.sum(sig.exp() - sig + mean.pow(2) - 1, dim=1), min=0.01))
        return [k.mean() for k in kl_list]

    def sleep_loss(self, z_recon, noise_recon, target_z, target_noise):

        b,l = z_recon.size()
        loc, scale = z_recon[:,:l//2], z_recon[:, l//2:].clamp(min=0.0001)
        distribution = torch.distributions.normal.Normal(loc, scale, validate_args=None)
        sleep_loss = [-distribution.log_prob(target_z)]

        for i, n in enumerate(noise_recon):
            if n is None:
                continue
            b, c, h, w = n.size()
            loc, scale = n[:,:c//2:,:], n[:, c//2:,:,:].clamp(min=0.0001)
            distribution = torch.distributions.normal.Normal(loc, scale, validate_args=None)
            sleep_loss.append(-distribution.log_prob(target_noise[i]))

        return [s.mean() for s in sleep_loss]

    def enc_as_dis_loss(self, z_recon, noise_recon, target_z, target_noise, eps=1e-5):
        b,l = z_recon.size()
        zmean, zsig = z_recon[:, :l//2], z_recon[:, l//2:]
        zvar = zsig.exp() # variance
        diss_loss = [zsig + self.simp * (1.0 / (2.0 * zvar.pow(2.0) + eps)) * (target_z - zmean).pow(2.0)]

        for i, n in enumerate(noise_recon):
            if n is None:
                continue
            b,c,h,w = n.size()
            zmean, zsig = n[:,:c//2:,:], n[:, c//2:,:,:]
            zvar = zsig.exp()
            diss_loss.append(zsig + (1.0 - self.simp) * (1.0 / (2.0 * zvar.pow(2.0) + eps)) * (target_noise[i] - zmean).pow(2.0))

        return [s.mean() for d in diss_loss]


        # return kl.mean().to(latent.device)

    def reconstruction_loss(self, output, target):
        assert self.recon_loss in ['siglaplace', 'bce'], f'Loss {self.recon_loss} not recognized, pick siglaplace or bce'
        b, c, w, h = output.size()

        if self.recon_loss == 'siglaplace':
            mus = output[:,:c//2,:,:]
            VARMULT = 1e-5
            EPS = 1e-5

            sgs, lsgs  = torch.exp(output[:,c//2:,:,:] * VARMULT), output[:,c//2:,:,:] * VARMULT
            lny = torch.log(target + EPS)
            ln1y = torch.log(1 - target + EPS)
            x = lny - ln1y
            rec = lny + ln1y + lsgs + math.log(2.0) + (x - mus).abs() / sgs

        elif self.recon_loss == 'bce':

            WEIGHT = 0.1
            EPS = 1e-5

            rloss = F.binary_cross_entropy_with_logits(output[:, :c//2, :, :], target, reduction='none')

            za = output[:, :c//2, :, :].abs()
            eza = (-za).exp()

            logpart = - (za + EPS).log() + (-eza + EPS).log1p() - (eza + EPS).log1p()
            rec = rloss + WEIGHT * logpart

            # rec = F.binary_cross_entropy_with_logits(output, target, reduction='none')

        return rec.mean().to(output.device)

class ConditionalGANLoss:
    """ Base class for all conditional losses """

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps, labels, height, alpha):
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps, labels, height, alpha):
        raise NotImplementedError("gen_loss method has not been implemented")


# =============================================================
# Normal versions of the Losses:
# =============================================================

class StandardGAN(GANLoss):

    def __init__(self, dis):
        from torch.nn import BCEWithLogitsLoss

        super().__init__(dis)

        # define the criterion and activation used for object
        self.criterion = BCEWithLogitsLoss()

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        # small assertion:
        assert real_samps.device == fake_samps.device, \
            "Real and Fake samples are not on the same device"

        # device for computations:
        device = fake_samps.device

        # predictions for real images and fake images separately :
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        # calculate the real loss:
        real_loss = self.criterion(
            torch.squeeze(r_preds),
            torch.ones(real_samps.shape[0]).to(device))

        # calculate the fake loss:
        fake_loss = self.criterion(
            torch.squeeze(f_preds),
            torch.zeros(fake_samps.shape[0]).to(device))

        # return final losses
        return (real_loss + fake_loss) / 2

    def gen_loss(self, _, fake_samps, height, alpha):
        preds, _, _ = self.dis(fake_samps, height, alpha)
        return self.criterion(torch.squeeze(preds),
                              torch.ones(fake_samps.shape[0]).to(fake_samps.device))


class HingeGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        loss = (torch.mean(nn.ReLU()(1 - r_preds)) +
                torch.mean(nn.ReLU()(1 + f_preds)))

        return loss

    def gen_loss(self, _, fake_samps, height, alpha):
        return -torch.mean(self.dis(fake_samps, height, alpha))


class RelativisticAverageHingeGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        # Obtain predictions
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        # return the loss
        loss = (torch.mean(nn.ReLU()(1 - r_f_diff))
                + torch.mean(nn.ReLU()(1 + f_r_diff)))

        return loss

    def gen_loss(self, real_samps, fake_samps, height, alpha):
        # Obtain predictions
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        # return the loss
        return (torch.mean(nn.ReLU()(1 + r_f_diff))
                + torch.mean(nn.ReLU()(1 - f_r_diff)))


class LogisticGAN(GANLoss):
    def __init__(self, dis):
        super().__init__(dis)

    # gradient penalty
    def R1Penalty(self, real_img, height, alpha):

        # TODO: use_loss_scaling, for fp16
        apply_loss_scaling = lambda x: x * torch.exp(x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))
        undo_loss_scaling = lambda x: x * torch.exp(-x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))

        real_img = torch.autograd.Variable(real_img, requires_grad=True)
        real_logit = self.dis(real_img, height, alpha)
        # real_logit = apply_loss_scaling(torch.sum(real_logit))
        real_grads = torch.autograd.grad(outputs=real_logit, inputs=real_img,
                                         grad_outputs=torch.ones(real_logit.size()).to(real_img.device),
                                         create_graph=True, retain_graph=True)[0].view(real_img.size(0), -1)
        # real_grads = undo_loss_scaling(real_grads)
        r1_penalty = torch.sum(torch.mul(real_grads, real_grads))
        return r1_penalty

    def dis_loss(self, real_samps, fake_samps, height, alpha, r1_gamma=10.0):
        # Obtain predictions
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        loss = torch.mean(nn.Softplus()(f_preds)) + torch.mean(nn.Softplus()(-r_preds))

        if r1_gamma != 0.0:
            r1_penalty = self.R1Penalty(real_samps.detach(), height, alpha) * (r1_gamma * 0.5)
            loss += r1_penalty

        return loss

    def gen_loss(self, _, fake_samps, height, alpha):
        f_preds = self.dis(fake_samps, height, alpha)

        return torch.mean(nn.Softplus()(-f_preds))
