import torch
from torch import nn
from collections import OrderedDict
from abc import ABC, abstractmethod
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


###############################################################################
# Helper Functions
###############################################################################
def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.InstanceNorm2d
        else:
            use_bias = norm_layer != nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class FocalDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    
    def dice_loss(self, pred, target, smooth=1.0):
        """Dice loss for binary segmentation"""
        pred = torch.sigmoid(pred)   # convert logits → probabilities
        intersection = (pred * target).sum(dim=(2,3))
        dice = (2. * intersection + smooth) / (pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + smooth)
        return 1 - dice.mean()
    
    def focal_loss(self,pred, target, alpha=0.25, gamma=2.0):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt = torch.exp(-bce)  # probability of correct class
        focal = alpha * (1-pt)**gamma * bce
        return focal.mean()

    def forward(self, pred, target):
        f_loss = self.focal_loss(pred, target)
        d_loss = self.dice_loss(pred, target)
        return 0.5 * f_loss + 0.5 * d_loss

class BCEDiceLoss(nn.Module):
    '''
    In case of pos_weight, it becomes weighted BCE loss.
    '''
    def __init__(self, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def dice_loss(self, pred, target, smooth=1.0):
        """Dice loss for binary segmentation"""
        pred = torch.sigmoid(pred)   # convert logits → probabilities
        intersection = (pred * target).sum(dim=(2,3))
        dice = (2. * intersection + smooth) / (pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + smooth)
        return 1 - dice.mean()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        d_loss = self.dice_loss(pred, target)
        return 0.5 * bce_loss + 0.5 * d_loss
    

# ---------------------------
# 2. Simple U-Net Model
# ---------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, n_classes=1):
        super(UNet, self).__init__()
        self.n_classes = n_classes
        self.dconv_down1 = DoubleConv(3, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.ConvTranspose2d(512, 512, 2, stride=2)

        self.dconv_up3 = DoubleConv(256 + 512, 256)
        self.dconv_up2 = DoubleConv(128 + 256, 128)
        self.dconv_up1 = DoubleConv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, self.n_classes, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)
        return out 
    
    def info(self):
        title = "U-Net"
        class_type = __class__.__name__
        num_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        text = (f"Model: {title} ({class_type})\n"
                f"Total parameters: {num_params}\n"
                f"Trainable parameters: {trainable_params}\n")
        return text
        

class ConvEncoder(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: tuple[int], kernel_size: int=3, stride: int=1, padding=3, dropout: float=0.2):
        super(ConvEncoder, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding 
        self.dropout_rate = dropout
        self.n_layers = len(hidden_channels)
        
        self.set_layers()
    
    def set_layers(self):
        self.input_layer = nn.Conv2d(self.input_channels, self.hidden_channels[0], 
                                 kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.activation_in = nn.ReLU()
        self.pool_in = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.stride)
        for i in range(self.n_layers-1):
            setattr(self, f'layer_{i}', nn.Conv2d(self.hidden_channels[i], self.hidden_channels[i+1], 
                                                  kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
            setattr(self, f'activation_{i}', nn.ReLU())
            setattr(self, f'max_pool{i}', nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.stride))
        self.dropout = nn.Dropout(self.dropout_rate)
        # self.pool_out = nn.MaxPool1d(kernel_size=self.kernel_size, stride=self.stride)
        # self.output_layer = nn.Linear(self.hidden_channels[-2], self.hidden_channels[-1])


    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation_in(x)
        x = self.pool_in(x)
        for i in range(self.n_layers-1):
            layer_i = getattr(self, f'layer_{i}')
            activation_i = getattr(self, f'activation_{i}')
            x = activation_i(layer_i(x))
            x = self.dropout(x)
            x = getattr(self, f'max_pool{i}')(x)
        return x
    
    def predict(self, model_input):
        return self.forward(model_input)
    
class ConvDecoder(nn.Module):
    def __init__(self, encoder, n_classes: int = 1, kernel_size: int=3, stride: int=1, padding=1):
        super(ConvDecoder, self).__init__()
        self.encoder = encoder
        self.hidden_channels = encoder.hidden_channels[::-1]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.n_layers = len(self.hidden_channels) - 1
        
        self.set_layers(n_classes)

    def set_layers(self, n_classes):
        for i in range(self.n_layers):
            setattr(
                self,
                f'layer_{i}',
                nn.ConvTranspose2d(
                    in_channels=self.hidden_channels[i],
                    out_channels=self.hidden_channels[i+1],
                    kernel_size=self.kernel_size,
                    stride=2,  # upsampling
                    padding=self.padding,
                    output_padding=1
                )
            )
            setattr(self, f'activation_{i}', nn.ReLU())
        
        # Final segmentation output
        self.output_layer = nn.ConvTranspose2d(
            in_channels=self.hidden_channels[-1],
            out_channels=n_classes,
            kernel_size=1
        )

    def forward(self, x):
        for i in range(self.n_layers):
            layer_i = getattr(self, f'layer_{i}')
            activation_i = getattr(self, f'activation_{i}')
            x = activation_i(layer_i(x))
        x = self.output_layer(x)
        return x
    
class ConvAutoEncoder(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: tuple[int], n_classes: int=1, kernel_size: int=3, stride: int=1, padding=9, dropout: float=0.2):
        super(ConvAutoEncoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_channels = input_channels
        self.n_classes = n_classes
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout = dropout
        
        self.set_encoder()
        self.set_decoder()

    def set_encoder(self):
        self.encoder = ConvEncoder(self.input_channels, self.hidden_channels, self.kernel_size, self.stride, self.padding, self.dropout)

    def set_decoder(self):
        self.decoder = ConvDecoder(self.encoder, n_classes=self.n_classes, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def center_crop(self, output, target_shape):
        _, _, h, w = output.shape
        _, _, th, tw = target_shape
        start_h = (h - th) // 2
        start_w = (w - tw) // 2
        return output[:, :, start_h:start_h+th, start_w:start_w+tw]


    def forward(self, x):
        input_size = x.shape
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        if self.padding > 0:
            decoded = self.center_crop(decoded, input_size)
        return decoded
    
    def predict(self, model_input):
        return self.forward(model_input)
    
    def info(self):
        title = "ConvAutoEncoder"
        class_type = __class__.__name__
        num_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        text = (f"Model: {title} ({class_type})\n"
                f"Total parameters: {num_params}\n"
                f"Trainable parameters: {trainable_params}\n")
        return text

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class UNetPlusPlus(nn.Module):
    """UNet++ (Nested U-Net) with optional deep supervision.
    Paper: https://arxiv.org/abs/1807.10165 (for reference)
    This is an encoder-decoder with dense skip connections between same-resolution nodes.
    """
    def __init__(self, in_channels=3, n_classes=1, deep_supervision: bool=False, filters=(64,128,256,512,1024)):
        super().__init__()
        self.n_classes = n_classes
        self.deep_supervision = deep_supervision

        # Encoder blocks (downsampling)
        self.conv00 = DoubleConv(in_channels, filters[0])
        self.pool0 = nn.MaxPool2d(2)

        self.conv10 = DoubleConv(filters[0], filters[1])
        self.pool1 = nn.MaxPool2d(2)

        self.conv20 = DoubleConv(filters[1], filters[2])
        self.pool2 = nn.MaxPool2d(2)

        self.conv30 = DoubleConv(filters[2], filters[3])
        self.pool3 = nn.MaxPool2d(2)

        self.conv40 = DoubleConv(filters[3], filters[4])

        # Up-convolutions
        self.up01 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.up11 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.up21 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        self.up31 = nn.ConvTranspose2d(filters[4], filters[3], 2, stride=2)

        self.up02 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.up12 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.up22 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)

        self.up03 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.up13 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)

        self.up04 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)

        # Decoder nested convs (dense skip concatenations)
        self.conv01 = DoubleConv(filters[0] + filters[0], filters[0])
        self.conv11 = DoubleConv(filters[1] + filters[1], filters[1])
        self.conv21 = DoubleConv(filters[2] + filters[2], filters[2])
        self.conv31 = DoubleConv(filters[3] + filters[3], filters[3])

        self.conv02 = DoubleConv(filters[0]*2 + filters[0], filters[0])
        self.conv12 = DoubleConv(filters[1]*2 + filters[1], filters[1])
        self.conv22 = DoubleConv(filters[2]*2 + filters[2], filters[2])

        self.conv03 = DoubleConv(filters[0]*3 + filters[0], filters[0])
        self.conv13 = DoubleConv(filters[1]*3 + filters[1], filters[1])

        self.conv04 = DoubleConv(filters[0]*4 + filters[0], filters[0])

        # Final classifiers
        if deep_supervision:
            self.final1 = nn.Conv2d(filters[0], self.n_classes, kernel_size=1)
            self.final2 = nn.Conv2d(filters[0], self.n_classes, kernel_size=1)
            self.final3 = nn.Conv2d(filters[0], self.n_classes, kernel_size=1)
            self.final4 = nn.Conv2d(filters[0], self.n_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(filters[0], self.n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool0(x00))
        x20 = self.conv20(self.pool1(x10))
        x30 = self.conv30(self.pool2(x20))
        x40 = self.conv40(self.pool3(x30))

        # Decoder with nested dense skip connections
        x01 = self.conv01(torch.cat([x00, self.up01(x10)], dim=1))
        x11 = self.conv11(torch.cat([x10, self.up11(x20)], dim=1))
        x21 = self.conv21(torch.cat([x20, self.up21(x30)], dim=1))
        x31 = self.conv31(torch.cat([x30, self.up31(x40)], dim=1))

        x02 = self.conv02(torch.cat([x00, x01, self.up02(x11)], dim=1))
        x12 = self.conv12(torch.cat([x10, x11, self.up12(x21)], dim=1))
        x22 = self.conv22(torch.cat([x20, x21, self.up22(x31)], dim=1))

        x03 = self.conv03(torch.cat([x00, x01, x02, self.up03(x12)], dim=1))
        x13 = self.conv13(torch.cat([x10, x11, x12, self.up13(x22)], dim=1))

        x04 = self.conv04(torch.cat([x00, x01, x02, x03, self.up04(x13)], dim=1))

        if self.deep_supervision:
            out1 = self.final1(x01)
            out2 = self.final2(x02)
            out3 = self.final3(x03)
            out4 = self.final4(x04)
            return [out1, out2, out3, out4]
        else:
            return self.final(x04)
        
    def info(self):
        title = "ConvAutoEncoder"
        class_type = __class__.__name__
        num_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        text = (f"Model: {title} ({class_type})\n"
                f"Total parameters: {num_params}\n"
                f"Trainable parameters: {trainable_params}\n")
        return text
    
def Conv3X3(in_, out):
    return torch.nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = Conv3X3(in_, out)
        self.activation = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class Down(nn.Module):

    def __init__(self, nn):
        super(Down,self).__init__()
        self.nn = nn
        self.maxpool_with_argmax = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self,inputs):
        down = self.nn(inputs)
        unpooled_shape = down.size()
        outputs, indices = self.maxpool_with_argmax(down)
        return outputs, down, indices, unpooled_shape

class Up(nn.Module):

    def __init__(self, nn):
        super().__init__()
        self.nn = nn
        self.unpool=torch.nn.MaxUnpool2d(2,2)

    def forward(self,inputs,indices,output_shape):
        outputs = self.unpool(inputs, indices=indices, output_size=output_shape)
        outputs = self.nn(outputs)
        return outputs

class Fuse(nn.Module):

    def __init__(self, nn, scale):
        super().__init__()
        self.nn = nn
        self.scale = scale
        self.conv = Conv3X3(64,1)

    def forward(self,down_inp,up_inp):
        outputs = torch.cat([down_inp, up_inp], 1)
        outputs = F.interpolate(outputs, scale_factor=self.scale, mode='bilinear')
        outputs = self.nn(outputs)

        return self.conv(outputs)



class DeepCrack(nn.Module):

    def __init__(self, num_classes=1000):
        super(DeepCrack, self).__init__()

        self.down1 = Down(torch.nn.Sequential(
            ConvRelu(3,64),
            ConvRelu(64,64),
        ))

        self.down2 = Down(torch.nn.Sequential(
            ConvRelu(64,128),
            ConvRelu(128,128),
        ))

        self.down3 = Down(torch.nn.Sequential(
            ConvRelu(128,256),
            ConvRelu(256,256),
            ConvRelu(256,256),
        ))

        self.down4 = Down(torch.nn.Sequential(
            ConvRelu(256, 512),
            ConvRelu(512, 512),
            ConvRelu(512, 512),
        ))

        self.down5 = Down(torch.nn.Sequential(
            ConvRelu(512, 512),
            ConvRelu(512, 512),
            ConvRelu(512, 512),
        ))

        self.up1 = Up(torch.nn.Sequential(
            ConvRelu(64, 64),
            ConvRelu(64, 64),
        ))

        self.up2 = Up(torch.nn.Sequential(
            ConvRelu(128, 128),
            ConvRelu(128, 64),
        ))

        self.up3 = Up(torch.nn.Sequential(
            ConvRelu(256, 256),
            ConvRelu(256, 256),
            ConvRelu(256, 128),
        ))

        self.up4 = Up(torch.nn.Sequential(
            ConvRelu(512, 512),
            ConvRelu(512, 512),
            ConvRelu(512, 256),
        ))

        self.up5 = Up(torch.nn.Sequential(
            ConvRelu(512, 512),
            ConvRelu(512, 512),
            ConvRelu(512, 512),
        ))

        self.fuse5 = Fuse(ConvRelu(512 + 512, 64), scale=16)
        self.fuse4 = Fuse(ConvRelu(512 + 256, 64), scale=8)
        self.fuse3 = Fuse(ConvRelu(256 + 128, 64), scale=4)
        self.fuse2 = Fuse(ConvRelu(128 + 64, 64), scale=2)
        self.fuse1 = Fuse(ConvRelu(64 + 64, 64), scale=1)

        self.final = Conv3X3(5,1)

    def forward(self,inputs):

        # encoder part
        out, down1, indices_1, unpool_shape1 = self.down1(inputs)
        out, down2, indices_2, unpool_shape2 = self.down2(out)
        out, down3, indices_3, unpool_shape3 = self.down3(out)
        out, down4, indices_4, unpool_shape4 = self.down4(out)
        out, down5, indices_5, unpool_shape5 = self.down5(out)

        # decoder part
        up5 = self.up5(out, indices=indices_5, output_shape=unpool_shape5)
        up4 = self.up4(up5, indices=indices_4, output_shape=unpool_shape4)
        up3 = self.up3(up4, indices=indices_3, output_shape=unpool_shape3)
        up2 = self.up2(up3, indices=indices_2, output_shape=unpool_shape2)
        up1 = self.up1(up2, indices=indices_1, output_shape=unpool_shape1)

        fuse5 = self.fuse5(down_inp=down5,up_inp=up5)
        fuse4 = self.fuse4(down_inp=down4, up_inp=up4)
        fuse3 = self.fuse3(down_inp=down3, up_inp=up3)
        fuse2 = self.fuse2(down_inp=down2, up_inp=up2)
        fuse1 = self.fuse1(down_inp=down1, up_inp=up1)

        output = self.final(torch.cat([fuse5,fuse4,fuse3,fuse2,fuse1],1))

        return output


class DeepCrackNet(nn.Module):
    def __init__(self, in_nc, num_classes, ngf, norm='batch'):
        super(DeepCrackNet, self).__init__()

        norm_layer = get_norm_layer(norm_type=norm)
        self.conv1 = nn.Sequential(*self._conv_block(in_nc, ngf, norm_layer, num_block=2))
        self.side_conv1 = nn.Conv2d(ngf, num_classes, kernel_size=1, stride=1, bias=False)

        self.conv2 = nn.Sequential(*self._conv_block(ngf, ngf*2, norm_layer, num_block=2))
        self.side_conv2 = nn.Conv2d(ngf*2, num_classes, kernel_size=1, stride=1, bias=False)

        self.conv3 = nn.Sequential(*self._conv_block(ngf*2, ngf*4, norm_layer, num_block=3))
        self.side_conv3 = nn.Conv2d(ngf*4, num_classes, kernel_size=1, stride=1, bias=False)

        self.conv4 = nn.Sequential(*self._conv_block(ngf*4, ngf*8, norm_layer, num_block=3))
        self.side_conv4 = nn.Conv2d(ngf*8, num_classes, kernel_size=1, stride=1, bias=False)

        self.conv5 = nn.Sequential(*self._conv_block(ngf*8, ngf*8, norm_layer, num_block=3))
        self.side_conv5 = nn.Conv2d(ngf*8, num_classes, kernel_size=1, stride=1, bias=False)

        self.fuse_conv = nn.Conv2d(num_classes*5, num_classes, kernel_size=1, stride=1, bias=False)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        #self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        #self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        #self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

    def _conv_block(self, in_nc, out_nc, norm_layer, num_block=2, kernel_size=3, 
        stride=1, padding=1, bias=False):
        conv = []
        for i in range(num_block):
            cur_in_nc = in_nc if i == 0 else out_nc
            conv += [nn.Conv2d(cur_in_nc, out_nc, kernel_size=kernel_size, stride=stride, 
                               padding=padding, bias=bias),
                     norm_layer(out_nc),
                     nn.ReLU(True)]
        return conv

    def forward(self, x):
        h,w = x.size()[2:]
        # main stream features
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.maxpool(conv1))
        conv3 = self.conv3(self.maxpool(conv2))
        conv4 = self.conv4(self.maxpool(conv3))
        conv5 = self.conv5(self.maxpool(conv4))
        # side output features
        side_output1 = self.side_conv1(conv1)
        side_output2 = self.side_conv2(conv2)
        side_output3 = self.side_conv3(conv3)
        side_output4 = self.side_conv4(conv4)
        side_output5 = self.side_conv5(conv5)
        # upsampling side output features
        side_output2 = F.interpolate(side_output2, size=(h, w), mode='bilinear', align_corners=True) #self.up2(side_output2)
        side_output3 = F.interpolate(side_output3, size=(h, w), mode='bilinear', align_corners=True) #self.up4(side_output3)
        side_output4 = F.interpolate(side_output4, size=(h, w), mode='bilinear', align_corners=True) #self.up8(side_output4)
        side_output5 = F.interpolate(side_output5, size=(h, w), mode='bilinear', align_corners=True) #self.up16(side_output5)

        fused = self.fuse_conv(torch.cat([side_output1, 
                                          side_output2, 
                                          side_output3,
                                          side_output4,
                                          side_output5], dim=1))
        return side_output1, side_output2, side_output3, side_output4, side_output5, fused

def define_deepcrack(in_nc, 
                     num_classes, 
                     ngf, 
                     norm='batch',
                     init_type='xavier', 
                     init_gain=0.02, 
                     gpu_ids=[]):
    net = DeepCrackNet(in_nc, num_classes, ngf, norm)
    return init_net(net, init_type, init_gain, gpu_ids)


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, size_average=True):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.size_average = size_average
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.criterion(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.size_average:
            return F_loss.mean()
        else:
            return F_loss.sum()
class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
class DeepCrackModel(BaseModel):
    """
    This class implements the DeepCrack model.
    DeepCrack paper: https://www.sciencedirect.com/science/article/pii/S0925231219300566
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options."""
        parser.add_argument('--lambda_side', type=float, default=1.0, help='weight for side output loss')
        parser.add_argument('--lambda_fused', type=float, default=1.0, help='weight for fused loss')
        return parser

    def __init__(self, opt):
        """Initialize the DeepCrack class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['side', 'fused', 'total']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.display_sides = opt.display_sides
        self.visual_names = ['image', 'label_viz', 'fused']
        if self.display_sides:
            self.visual_names += ['side1', 'side2', 'side3', 'side4', 'side5']
        # specify the models you want to save to the disk. 
        self.model_names = ['G']

        # define networks 
        self.netG = define_deepcrack(opt.input_nc, 
                                     opt.num_classes, 
                                     opt.ngf, 
                                     opt.norm,
                                     opt.init_type, 
                                     opt.init_gain, 
                                     self.gpu_ids)

        self.softmax = torch.nn.Softmax(dim=1)

        if self.isTrain:
            # define loss functions
            #self.weight = torch.from_numpy(np.array([0.0300, 1.0000], dtype='float32')).float().to(self.device)
            #self.criterionSeg = torch.nn.CrossEntropyLoss(weight=self.weight)
            if self.opt.loss_mode == 'focal':
                self.criterionSeg = BinaryFocalLoss()
            else: 
                self.criterionSeg = nn.BCEWithLogitsLoss(size_average=True, reduce=True, 
                    pos_weight=torch.tensor(1.0/3e-2).to(self.device))
            self.weight_side = [0.5, 0.75, 1.0, 0.75, 0.5]

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.SGD(self.netG.parameters(), lr=opt.lr, momentum=0.9, weight_decay=2e-4)
            self.optimizers.append(self.optimizer)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.image = input['image'].to(self.device)
        self.label = input['label'].to(self.device)
        #self.label3d = self.label.squeeze(1)
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.outputs = self.netG(self.image)

        # for visualization
        self.label_viz = (self.label.float()-0.5)/0.5
        #self.fused = (self.softmax(self.outputs[-1])[:,1].detach().unsqueeze(1)-0.5)/0.5
        #if self.display_sides:
        #    self.side1 = (self.softmax(self.outputs[0])[:,1].detach().unsqueeze(1)-0.5)/0.5
        #    self.side2 = (self.softmax(self.outputs[1])[:,1].detach().unsqueeze(1)-0.5)/0.5
        #    self.side3 = (self.softmax(self.outputs[2])[:,1].detach().unsqueeze(1)-0.5)/0.5
        #    self.side4 = (self.softmax(self.outputs[3])[:,1].detach().unsqueeze(1)-0.5)/0.5
        #    self.side5 = (self.softmax(self.outputs[4])[:,1].detach().unsqueeze(1)-0.5)/0.5
        self.fused = (torch.sigmoid(self.outputs[-1])-0.5)/0.5
        if self.display_sides:
            self.side1 = (torch.sigmoid(self.outputs[0])-0.5)/0.5
            self.side2 = (torch.sigmoid(self.outputs[1])-0.5)/0.5
            self.side3 = (torch.sigmoid(self.outputs[2])-0.5)/0.5
            self.side4 = (torch.sigmoid(self.outputs[3])-0.5)/0.5
            self.side5 = (torch.sigmoid(self.outputs[4])-0.5)/0.5

    def backward(self):
        """Calculate the loss"""
        lambda_side = self.opt.lambda_side
        lambda_fused = self.opt.lambda_fused

        self.loss_side = 0.0
        for out, w in zip(self.outputs[:-1], self.weight_side):
            #self.loss_side += self.criterionSeg(out, self.label3d) * w
            self.loss_side += self.criterionSeg(out, self.label) * w

        #self.loss_fused = self.criterionSeg(self.outputs[-1], self.label3d)
        self.loss_fused = self.criterionSeg(self.outputs[-1], self.label)
        self.loss_total = self.loss_side * lambda_side + self.loss_fused * lambda_fused
        self.loss_total.backward()

    def optimize_parameters(self, epoch=None):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute predictions.
        self.optimizer.zero_grad()  # set G's gradients to zero
        self.backward()             # calculate gradients for G
        self.optimizer.step()       # update G's weights

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math


class DWConv(nn.Module):
    def __init__(self, dim=768,group_num=4):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim//group_num)

    def forward(self, x):
        x = self.dwconv(x)
        return x


def Conv1X1(in_, out):
    return torch.nn.Conv2d(in_, out, 1, padding=0)


def Conv3X3(in_, out):
    return torch.nn.Conv2d(in_, out, 3, padding=1)


class Mlp(nn.Module):
    def __init__(self, in_features, out_features, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = out_features // 4
        self.fc1 = Conv1X1(in_features, hidden_features)
        self.gn1=nn.GroupNorm(hidden_features//4,hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.gn2 = nn.GroupNorm(hidden_features // 4, hidden_features)
        self.act = act_layer()
        self.fc2 = Conv1X1(hidden_features, out_features)
        self.gn3=nn.GroupNorm(out_features//4,out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x=self.gn1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x)
        x=self.gn2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x=self.gn3(x)
        x = self.drop(x)
        return x


class LocalSABlock(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, k=16, u=1, m=7):
        super(LocalSABlock, self).__init__()
        self.kk, self.uu, self.vv, self.mm, self.heads = k, u, out_channels // heads, m, heads
        self.padding = (m - 1) // 2

        self.queries = nn.Sequential(
            nn.Conv2d(in_channels, k * heads, kernel_size=1, bias=False),
            nn.GroupNorm(k*heads//4,k*heads)
        )
        self.keys = nn.Sequential(
            nn.Conv2d(in_channels, k * u, kernel_size=1, bias=False),
            nn.GroupNorm(k*u//4,k*u)
        )
        self.values = nn.Sequential(
            nn.Conv2d(in_channels, self.vv * u, kernel_size=1, bias=False),
            nn.GroupNorm(self.vv*u//4,self.vv*u)
        )

        self.softmax = nn.Softmax(dim=-1)

        self.embedding = nn.Parameter(torch.randn([self.kk, self.uu, 1, m, m]), requires_grad=True)

    def forward(self, x):
        n_batch, C, w, h = x.size()
        queries = self.queries(x).view(n_batch, self.heads, self.kk, w * h)  # b, heads, k , w * h
        softmax = self.softmax(self.keys(x).view(n_batch, self.kk, self.uu, w * h))  # b, k, uu, w * h
        values = self.values(x).view(n_batch, self.vv, self.uu, w * h)  # b, v, uu, w * h
        content = torch.einsum('bkum,bvum->bkv', (softmax, values))
        content = torch.einsum('bhkn,bkv->bhvn', (queries, content))
        values = values.view(n_batch, self.uu, -1, w, h)
        context = F.conv3d(values, self.embedding, padding=(0, self.padding, self.padding))
        context = context.view(n_batch, self.kk, self.vv, w * h)
        context = torch.einsum('bhkn,bkvn->bhvn', (queries, context))

        out = content + context
        out = out.contiguous().view(n_batch, -1, w, h)

        return out


class TFBlock(nn.Module):

    def __init__(self, in_chnnels, out_chnnels, mlp_ratio=2., drop=0.3,
                 drop_path=0., act_layer=nn.GELU, linear=False):
        super(TFBlock, self).__init__()
        self.in_chnnels = in_chnnels
        self.out_chnnels = out_chnnels
        self.attn = LocalSABlock(
            in_channels=in_chnnels, out_channels=out_chnnels
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=in_chnnels, out_features=out_chnnels, act_layer=act_layer, drop=drop, linear=linear)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        hidden_planes = max(planes,in_planes) // self.expansion
        self.conv1 = nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(hidden_planes //4,
                                hidden_planes)  
        self.conv2 = nn.ModuleList([TFBlock(hidden_planes, hidden_planes)])
        self.bn2 = nn.GroupNorm(hidden_planes // 4,
                                hidden_planes)  
        self.conv2.append(nn.GELU()) 
        self.conv2 = nn.Sequential(*self.conv2)
        self.conv3 = nn.Conv2d(hidden_planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(planes // 4, planes)  
        self.GELU=nn.GELU()
        self.shortcut = nn.Sequential()
        if in_planes!=planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                nn.GroupNorm(planes//4,planes)
            )
    def forward(self, x):
        out = self.GELU(self.bn1(self.conv1(x)))  
        out = self.conv2(out)
        out = self.GELU(self.bn3(self.conv3(out))) 
        out += self.shortcut(x)
        return out


class Trans_EB(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = Bottleneck(in_, out)
        self.activation=torch.nn.GELU()
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = Conv3X3(in_, out)
        self.activation = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class LABlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(LABlock, self).__init__()
        self.W_1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(output_channels//4,output_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(output_channels//4,output_channels),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        self.gelu=nn.GELU()
    def forward(self, inputs):
        sum = 0
        for input in inputs:
            sum += input
        sum=self.gelu(sum)
        out = self.W_1(sum)
        psi = self.psi(out)  # Mask
        return psi


class Fuse(nn.Module):
    def __init__(self, nn, scale):
        super().__init__()
        self.nn = nn
        self.scale = scale
        self.conv = Conv3X3(64, 1)

    def forward(self, down_inp, up_inp, size=None, attention=None):
        outputs = torch.cat([down_inp, up_inp], 1)
        outputs = self.nn(outputs)

        if attention is not None:
            outputs = attention * outputs

        outputs = self.conv(outputs)

        # Prefer explicit output size if provided (more robust than scale_factor)
        if size is not None:
            outputs = F.interpolate(outputs, size=size, mode="bilinear", align_corners=False)
        else:
            outputs = F.interpolate(outputs, scale_factor=self.scale, mode="bilinear", align_corners=False)

        return outputs


class Down1(nn.Module):

    def __init__(self):
        super(Down1, self).__init__()
        self.nn1 = ConvRelu(3, 64)
        self.nn2 = Trans_EB(64, 64)
        self.patch_embed = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, inputs):
        scale1_1 = self.nn1(inputs)
        scale1_2 = self.nn2(scale1_1)
        unpooled_shape = scale1_2.size()
        outputs, indices = self.patch_embed(scale1_2)
        return outputs, indices, unpooled_shape, scale1_1, scale1_2


class Down2(nn.Module):

    def __init__(self):
        super(Down2, self).__init__()
        self.nn1 = Trans_EB(64, 128) 
        self.nn2 = Trans_EB(128, 128)
        self.patch_embed = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, inputs):
        scale2_1 = self.nn1(inputs)
        scale2_2 = self.nn2(scale2_1)
        unpooled_shape = scale2_2.size()
        outputs, indices = self.patch_embed(scale2_2)
        return outputs, indices, unpooled_shape, scale2_1, scale2_2


class Down3(nn.Module):

    def __init__(self):
        super(Down3, self).__init__()

        self.nn1 = Trans_EB(128, 256) 
        self.nn2 = Trans_EB(256, 256)
        self.nn3 = Trans_EB(256, 256)
        self.patch_embed = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, inputs):
        scale3_1 = self.nn1(inputs)
        scale3_2 = self.nn2(scale3_1)
        scale3_3 = self.nn2(scale3_2)
        unpooled_shape = scale3_3.size()
        outputs, indices = self.patch_embed(scale3_3)
        return outputs, indices, unpooled_shape, scale3_1, scale3_2, scale3_3


class Down4(nn.Module):

    def __init__(self):
        super(Down4, self).__init__()

        self.nn1 = Trans_EB(256, 512)
        self.nn2 = Trans_EB(512, 512)
        self.nn3 = Trans_EB(512, 512)
        self.patch_embed = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, inputs):
        scale4_1 = self.nn1(inputs)
        scale4_2 = self.nn2(scale4_1)
        scale4_3 = self.nn2(scale4_2)
        unpooled_shape = scale4_3.size()
        outputs, indices = self.patch_embed(scale4_3)
        return outputs, indices, unpooled_shape, scale4_1, scale4_2, scale4_3


class Down5(nn.Module):

    def __init__(self):
        super(Down5, self).__init__()

        self.nn1 = Trans_EB(512, 512)
        self.nn2 = Trans_EB(512, 512)
        self.nn3 = Trans_EB(512, 512)
        self.patch_embed = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, inputs):
        scale5_1 = self.nn1(inputs)
        scale5_2 = self.nn2(scale5_1)
        scale5_3 = self.nn2(scale5_2)
        unpooled_shape = scale5_3.size()
        outputs, indices = self.patch_embed(scale5_3)
        return outputs, indices, unpooled_shape, scale5_1, scale5_2, scale5_3


class Up1(nn.Module):

    def __init__(self):
        super().__init__()
        self.nn1 = Trans_EB(64, 64)
        self.nn2 = Trans_EB(64, 64)
        self.inv_patch_embed = torch.nn.MaxUnpool2d(2, 2)

    def forward(self, inputs, indices, output_shape):
        outputs = self.inv_patch_embed(inputs, indices=indices, output_size=output_shape)
        scale1_3 = self.nn1(outputs)
        scale1_4 = self.nn2(scale1_3)
        return scale1_3, scale1_4


class Up2(nn.Module):

    def __init__(self):
        super().__init__()
        self.nn1 = Trans_EB(128, 128)
        self.nn2 = Trans_EB(128, 64)
        self.inv_patch_embed = torch.nn.MaxUnpool2d(2, 2)

    def forward(self, inputs, indices, output_shape):
        outputs = self.inv_patch_embed(inputs, indices=indices, output_size=output_shape)
        scale2_3 = self.nn1(outputs)
        scale2_4 = self.nn2(scale2_3)
        return scale2_3, scale2_4


class Up3(nn.Module):

    def __init__(self):
        super().__init__()
        self.nn1 = Trans_EB(256, 256)
        self.nn2 = Trans_EB(256, 256)
        self.nn3 = Trans_EB(256, 128)
        self.inv_patch_embed = torch.nn.MaxUnpool2d(2, 2)

    def forward(self, inputs, indices, output_shape):
        outputs = self.inv_patch_embed(inputs, indices=indices, output_size=output_shape)
        scale3_4 = self.nn1(outputs)
        scale3_5 = self.nn2(scale3_4)
        scale3_6 = self.nn3(scale3_5)
        return scale3_4, scale3_5, scale3_6


class Up4(nn.Module):

    def __init__(self):
        super().__init__()
        self.nn1 = Trans_EB(512, 512)
        self.nn2 = Trans_EB(512, 512)
        self.nn3 = Trans_EB(512, 256)
        self.inv_patch_embed = torch.nn.MaxUnpool2d(2, 2)

    def forward(self, inputs, indices, output_shape):
        outputs = self.inv_patch_embed(inputs, indices=indices, output_size=output_shape)
        scale4_4 = self.nn1(outputs)
        scale4_5 = self.nn2(scale4_4)
        scale4_6 = self.nn3(scale4_5)
        return scale4_4, scale4_5, scale4_6


class Up5(nn.Module):

    def __init__(self):
        super().__init__()
        self.nn1 = Trans_EB(512, 512)
        self.nn2 = Trans_EB(512, 512)
        self.nn3 = Trans_EB(512, 512)
        self.inv_patch_embed = torch.nn.MaxUnpool2d(2, 2)

    def forward(self, inputs, indices, output_shape):
        outputs = self.inv_patch_embed(inputs, indices=indices, output_size=output_shape)
        scale5_4 = self.nn1(outputs)
        scale5_5 = self.nn2(scale5_4)
        scale5_6 = self.nn3(scale5_5)
        return scale5_4, scale5_5, scale5_6


class crackformer(nn.Module):

    def __init__(self):
        super(crackformer, self).__init__()

        self.down1 = Down1()
        self.down2 = Down2()
        self.down3 = Down3()
        self.down4 = Down4()
        self.down5 = Down5()

        self.up1 = Up1()
        self.up2 = Up2()
        self.up3 = Up3()
        self.up4 = Up4()
        self.up5 = Up5()

        self.fuse5 = Fuse(ConvRelu(512 + 512, 64), scale=16)
        self.fuse4 = Fuse(ConvRelu(512 + 256, 64), scale=8)
        self.fuse3 = Fuse(ConvRelu(256 + 128, 64), scale=4)
        self.fuse2 = Fuse(ConvRelu(128 + 64, 64), scale=2)
        self.fuse1 = Fuse(ConvRelu(64 + 64, 64), scale=1)

        self.final = Conv1X1(5, 1)

        self.LABlock_1 = LABlock(64, 64)
        self.LABlock_2 = LABlock(128, 64)
        self.LABlock_3 = LABlock(256, 64)
        self.LABlock_4 = LABlock(512, 64)
        self.LABlock_5 = LABlock(512, 64)
    def forward(self, inputs):
        # encoder part
        out, indices_1, unpool_shape1, scale1_1, scale1_2 = self.down1(inputs)
        out, indices_2, unpool_shape2, scale2_1, scale2_2 = self.down2(out)
        out, indices_3, unpool_shape3, scale3_1, scale3_2, scale3_3 = self.down3(out)
        out, indices_4, unpool_shape4, scale4_1, scale4_2, scale4_3 = self.down4(out)
        out, indices_5, unpool_shape5, scale5_1, scale5_2, scale5_3 = self.down5(out)
        # decoder part
        scale5_4, scale5_5, up5 = self.up5(out, indices=indices_5, output_shape=unpool_shape5)
        scale4_4, scale4_5, up4 = self.up4(up5, indices=indices_4, output_shape=unpool_shape4)
        scale3_4, scale3_5, up3 = self.up3(up4, indices=indices_3, output_shape=unpool_shape3)
        scale2_3, up2 = self.up2(up3, indices=indices_2, output_shape=unpool_shape2)
        scale1_3, up1 = self.up1(up2, indices=indices_1, output_shape=unpool_shape1)
        # attention part
        attention1 = self.LABlock_1([scale1_1, scale1_3])
        attention2 = self.LABlock_2([scale2_1, scale2_3])
        attention3 = self.LABlock_3([scale3_1, scale3_2, scale3_4, scale3_5])
        attention4 = self.LABlock_4([scale4_1, scale4_2, scale4_4, scale4_5])
        attention5 = self.LABlock_5([scale5_1, scale5_2, scale5_4, scale5_5])
        # fuse part
        fuse5 = self.fuse5(down_inp=scale5_3, up_inp=up5, size=[inputs.shape[2], inputs.shape[3]], attention=attention5)
        fuse4 = self.fuse4(down_inp=scale4_3, up_inp=up4, size=[inputs.shape[2], inputs.shape[3]], attention=attention4)
        fuse3 = self.fuse3(down_inp=scale3_3, up_inp=up3, size=[inputs.shape[2], inputs.shape[3]], attention=attention3)
        fuse2 = self.fuse2(down_inp=scale2_2, up_inp=up2, size=[inputs.shape[2], inputs.shape[3]], attention=attention2)
        fuse1 = self.fuse1(down_inp=scale1_2, up_inp=up1, size=[inputs.shape[2], inputs.shape[3]], attention=attention1)

        output = self.final(torch.cat([fuse5, fuse4, fuse3, fuse2, fuse1], 1))

        return output
    
##