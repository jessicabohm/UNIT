from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

# Adapted from UNIT paper - just updating to work on 3D rather than 2D


##################################################################################
# Encoder and Decoders
##################################################################################

# HARDCORE FOR NOW
two_d = False # NOTE: 2D vs 3D
n_downsample = 2
dim=8
img_x = 64 #128
img_y = 64 #120
img_z = 64 #120
latent_dim = 2000


if two_d:
    flattened_dim = dim*2**n_downsample*img_x/2**n_downsample*img_y/2**n_downsample
else:
    flattened_dim = dim*2**n_downsample*img_x/2**n_downsample*img_y/2**n_downsample*img_z/2**n_downsample

flattened_dim = int(flattened_dim)



# They had a Style and Content encoder - this content encoder just had resnet blocks instead
# of global average pooling??
class Encoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(Encoder, self).__init__()
        self.model = []
        if two_d:
            self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        else:
            self.model += [Conv3dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]

        # downsampling blocks
        for i in range(n_downsample):
            if two_d:
                self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            else:
                self.model += [Conv3dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='in', activ='relu', pad_type='zero'): # NOTE: updated normalization to in to not have to compute weight and bias externally
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks # NOTE: changed!!
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            if two_d:
                self.model += [nn.Upsample(scale_factor=2),
                            Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='in', activation=activ, pad_type=pad_type)] # NOTE: could update to instance norm since only a batch size of 2 -> don't want to normalize over full layer??
            else:
                self.model += [nn.Upsample(scale_factor=2),
                    Conv3dBlock(dim, dim // 2, 5, 1, 2, norm='in', activation=activ, pad_type=pad_type)] # NOTE: could update to instance norm since only a batch size of 2 -> don't want to normalize over full layer??
            
            dim //= 2
        # use reflection padding in the last conv layer
        if two_d:
            self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='none', pad_type=pad_type)] 
        else:
            self.model += [Conv3dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='none', pad_type=pad_type)] 

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)
    
### Updated to be VAE style
class Encoder_VAE(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(Encoder_VAE, self).__init__()
        self.model = []
        if two_d:
            self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        else:
            self.model += [Conv3dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]

        # downsampling blocks
        for i in range(n_downsample):
            if two_d:
                self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            else:
                self.model += [Conv3dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2

        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]

        # NOTE: extra to map down to lower dim
        self.model += [nn.Flatten(), nn.Linear(flattened_dim, latent_dim), nn.LayerNorm(latent_dim), nn.ReLU()]

        self.output_dim = dim

        # NOTE: dim might be incorrect here??
        #self.inplace = nn.Linear(latent_dim, latent_dim)
        #self.inplace = nn.Linear(flattened_dim, flattened_dim)
        
        self.fc_mu = nn.Linear(latent_dim, latent_dim)       # outputs means
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)   # outputs log variances
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        out = self.model(x)
        means = self.fc_mu(out) # self.inplace(out)
        log_vars = self.fc_logvar(out) #self.inplace(out) # why was it called log vars? Probs cuz of how it's used in KL divergence
        return means, log_vars
    
class Reshape(nn.Module):
    def __init__(self):
        super().__init__()
        #self.shape = (16, 16, 15, 15)  # e.g., (-1, 512) or (batch_size, channels, height, width)
        self.shape = (dim*2**n_downsample, int(img_x/2**n_downsample), int(img_y/2**n_downsample), int(img_z/2**n_downsample))  # e.g., (-1, 512) or (batch_size, channels, height, width)

    def forward(self, x):
        return x.reshape(x.size(0), *self.shape)  # keeps batch dim intact

class Decoder_VAE(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='in', activ='relu', pad_type='zero'): # NOTE: updated normalization to in to not have to compute weight and bias externally
        super(Decoder_VAE, self).__init__()

        self.model = []

        self.model += [nn.Linear(latent_dim, flattened_dim), nn.LayerNorm(flattened_dim), nn.ReLU(), Reshape()]
 
        # AdaIN residual blocks # NOTE: changed!!
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]

        # upsampling blocks
        for i in range(n_upsample):
            if two_d:
                self.model += [nn.Upsample(scale_factor=2),
                            Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='in', activation=activ, pad_type=pad_type)] # NOTE: could update to instance norm since only a batch size of 2 -> don't want to normalize over full layer??
            else:
                self.model += [nn.Upsample(scale_factor=2),
                    Conv3dBlock(dim, dim // 2, 5, 1, 2, norm='in', activation=activ, pad_type=pad_type)] # NOTE: could update to instance norm since only a batch size of 2 -> don't want to normalize over full layer??
            
            dim //= 2
        # use reflection padding in the last conv layer
        if two_d:
            self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='none', pad_type=pad_type)] 
        else:
            self.model += [Conv3dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='none', pad_type=pad_type)] 

        # use reflection padding in the last conv layer
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)
    



##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model) # just unpacks the layers and makes them torch nn layers

    def forward(self, x):
        return self.model(x)
    

##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []

        if two_d:
            model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        else:
            model += [Conv3dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Conv3dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class Conv3dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv3dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad3d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad3d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ConstantPad3d(padding, value=0)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm3d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm3d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm3d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm3d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv3d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias) # NOTE: just updated this to be 3d?? is that ok??

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
    
class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
    

##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

# NOTE: why was it 2d???
class AdaptiveInstanceNorm3d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm3d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
