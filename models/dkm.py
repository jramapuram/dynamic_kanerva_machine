from copy import deepcopy

import torch
import numpy as np
import torch.nn as nn

import helpers.layers as layers
from models.memory import KanervaMemory
from models.vae.abstract_vae import AbstractVAE
# from models.vae.reparameterizers import get_reparameterizer


class DynamicKanervaMachine(AbstractVAE):
    def __init__(self, input_shape, **kwargs):
        """ Implements a standard simple VAE.

        :param input_shape: the input shape
        :returns: an object of AbstractVAE
        :rtype: AbstractVAE

        """
        super(DynamicKanervaMachine, self).__init__(input_shape, **kwargs)
        self.reparameterizer = None  # Stub reparameterizer
        # self.z_reparameterizer = get_reparameterizer(self.config['reparam_type'])(config=self.config)

        # Build the encoder and decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        # Build the Kanerva memory
        self.memory = KanervaMemory(code_size=self.config['code_size'],
                                    memory_size=self.config['memory_size'])

        # Projections from embedding to write and reads
        self.write_projector = self._build_dense(input_size=self.config['latent_size'],
                                                 output_size=self.config['code_size'],
                                                 name='write')
        self.read_projector = self._build_dense(input_size=self.config['latent_size'],
                                                output_size=self.config['code_size'],
                                                name='read')

    def _build_dense(self, input_size, output_size, name):
        """Simple helper to build a dense network"""
        key_config = deepcopy(self.config)
        key_config['encoder_layer_type'] = 'dense'
        key_config['layer_modifier'] = self.config['encoder_layer_modifier']
        key_config['input_shape'] = [input_size]
        episode_length = self.config['episode_length']
        return nn.Sequential(
            layers.View([-1, input_size]),
            layers.get_dense_encoder(**key_config, name=name)(output_size=output_size),
            layers.View([-1, episode_length, output_size])
        )

    def build_encoder(self):
        """Helper to build the encoder type. Differs from VAE in that it projects to latent_size.

        :returns: an encoder
        :rtype: nn.Module

        """
        # Following for standard 2d image decoders:
        # enc_conf = deepcopy(self.config)
        # enc_conf['encoder_layer_type'] = 'resnet50'
        # return nn.Sequential(
        #     layers.View([-1, *self.input_shape]),                                         # fold in: [T*B, C, H, W]
        #     # layers.get_encoder(**self.config)(
        #     #     output_size=self.config['latent_size']
        #     # ),
        #     layers.get_torchvision_encoder(**enc_conf)(
        #         output_size=self.config['latent_size'],
        #     ),
        #     layers.View([-1, self.config['episode_length'], self.config['latent_size']])  # un-fold episode to [T, B, F]
        # )

        # encoder = layers.S3DEncoder(
        #     output_size=self.config['latent_size'],
        #     latent_size=self.config['latent_size'],
        #     activation_str=self.config['encoder_activation'],
        #     conv_normalization_str='none', # self.config['conv_normalization'],
        #     dense_normalization_str='none', # self.config['dense_normalization'],
        #     norm_first_layer=False,
        #     norm_last_layer=False,
        #     pretrained=False
        # )

        import torchvision
        encoder = layers.TSMResnetEncoder(
            pretrained_output_size=512,
            output_size=self.config['latent_size'],
            latent_size=self.config['latent_size'],
            activation_str=self.config['encoder_activation'],
            conv_normalization_str=self.config['conv_normalization'],
            dense_normalization_str=self.config['dense_normalization'],
            norm_first_layer=True,
            norm_last_layer=False,
            layer_fn=torchvision.models.resnet18,
            pretrained=False,
            num_segments=1,  # self.config['episode_length'],
            shift_div=self.config['shift_div'],
            temporal_pool=False
        )
        if self.config['encoder_layer_modifier'] == 'sine':
            layers.convert_layer(encoder.model,                 # remove BN
                                 from_layer=nn.BatchNorm2d,
                                 to_layer=layers.Identity,
                                 set_from_layer_kwargs=False)
            layers.convert_layer(encoder.model,                 # remove ReLU
                                 from_layer=nn.ReLU,
                                 to_layer=layers.Identity,
                                 set_from_layer_kwargs=False)
            layers.convert_layer(encoder.model,                 # replace Conv2d --> SineConv2d
                                 from_layer=nn.Conv2d,
                                 to_layer=layers.SineConv2d,
                                 set_from_layer_kwargs=True)

        return encoder

    def build_decoder(self, reupsample=True):
        """ helper function to build convolutional or dense decoder

        :returns: a decoder
        :rtype: nn.Module

        """
        dec_conf = deepcopy(self.config)
        if dec_conf['nll_type'] == 'pixel_wise':
            dec_conf['input_shape'][0] *= 256

        episode_length = self.config['episode_length']
        decoder = nn.Sequential(
            layers.View([-1, self.config['code_size']]),
            layers.get_decoder(output_shape=dec_conf['input_shape'], **dec_conf)(
                input_size=self.config['code_size']
            ),
            layers.View([-1, episode_length, *self.input_shape]),
        )

        # append the variance as necessary
        decoder = self._append_variance_projection(decoder)
        return torch.jit.script(decoder) if self.config['jit'] else decoder

    def generate_synthetic_samples(self, batch_size, memory_state, **kwargs):
        """ Generates samples with VAE.

        :param batch_size: the number of samples to generate.
        :param memory: an updated memory posterior
        :returns: decoded logits
        :rtype: torch.Tensor

        """
        episode_length = self.config['episode_length']
        w_samples = self.memory.sample_prior_w(seq_length=episode_length,
                                               batch_size=batch_size)
        z_samples = self.memory.get_w_to_z_mean(w_samples, memory_state.M_mean)
        z_read, _ = self.memory.read_with_z(z=z_samples, memory_state=memory_state)

        # Decode the read samples & return
        generated = self.nll_activation(self.decode(z_read))
        return {
            # Fold in episode into batch and return for visualization.
            'generated_imgs': generated.view([-1, *generated.shape[-3:]])
        }

    def encode(self, x):
        """ Encodes a tensor x to a set of logits.

        :param x: the input tensor
        :returns: logits
        :rtype: torch.Tensor

        """
        # encoded = self.encoder(x)                              # Standard (non-temporal) encoder
        # encoded = self.encoder(x, reduction='mean').squeeze()  # S3D with pooling
        encoded = self.encoder(x, reduction='none')              # TSM (no pooling)
        if encoded.dim() < 2:
            return encoded.unsqueeze(-1)

        return encoded

    def posterior(self, x, labels=None, force=False):
        """ get a reparameterized Q(z|x) for a given x

        :param x: input tensor
        :param labels: (optional) labels
        :param force:  force reparameterization
        :returns: reparam dict
        :rtype: torch.Tensor

        """
        inputs = torch.cat([i.unsqueeze(1) for i in x], 1)              # expand to [B, T, C, W, H]
        batch_size = inputs.shape[0]
        z_episode = self.encode(inputs)                                 # [B, T, F] base logits.

        # Project the episode logits to write and read logits
        read_logits = self.read_projector(z_episode).transpose(0, 1)    # [T, B, code_size] for DKM
        write_logits = self.write_projector(z_episode).transpose(0, 1)  # [T, B, code_size] for DKM

        # Grab the prior and update it to the posterior given the episode
        prior_memory = self.memory.get_prior_state(batch_size)
        posterior_memory, _, dkl_w, _ = self.memory.update_state(write_logits,
                                                                 prior_memory)  # dkl_w: [T, B]

        # Read from the memory & compute the total KL penalty
        read_z, dkl_r = self.memory.read_with_z(read_logits,                    # read_z: [T, B, code_size]
                                                posterior_memory)               # dkl_r: [T, B]
        dkl_M = self.memory.get_dkl_total(posterior_memory)                     # dkl_M: [T, B]

        return read_z, {
            'memory_kl': dkl_M,
            'read_kl': dkl_r,
            'write_kl': dkl_w,

            # We need an actual updated memory to generate samples.
            'memory_state': posterior_memory,
        }

    def kld(self, dist_a):
        """ KL-Divergence of the distribution dict and the prior of that distribution.

        :param dist_a: the distribution dict.
        :returns: tensor that is of dimension batch_size
        :rtype: torch.Tensor

        """
        # reduce over the episode dimension for the KLD
        return torch.mean(dist_a['memory_kl'] + dist_a['read_kl'] + dist_a['write_kl'], 0)

    def loss_function(self, recon_x, x, params, K=1):
        """ Loss function for Kanerva ++

        :param recon_x: the reconstruction logits
        :param x: the original samples
        :param params: the reparameterized parameter dict containing reparam
        :param K: number of monte-carlo samples to use.
        :returns: loss dict
        :rtype: dict

        """
        x = torch.cat([xi.unsqueeze(1) for xi in x], 1).contiguous()  # [B, T, C, H, W]
        loss_dict = super(DynamicKanervaMachine, self).loss_function(
            recon_x=recon_x, x=x, params=params, K=K)

        # Add extra tracking metrics
        loss_dict['bits_per_dim_recon_mean'] = (loss_dict['nll_mean'] / np.prod(self.input_shape)) / np.log(2)
        loss_dict['bits_per_dim_elbo_mean'] = (loss_dict['elbo_mean'] / np.prod(self.input_shape)) / np.log(2)

        # Return full loss dict
        return loss_dict

    def get_images_from_reparam(self, reparam_dict):
        """ returns a dictionary of images from the reparam map

        :param reparam_maps_list: a list of reparam dicts
        :returns: a dictionary of images
        :rtype: dict

        """
        memory = reparam_dict['memory_state'].M_mean
        return {
            'memory_imgs': memory.view(-1, 1, *memory.shape[-2:]),  # Flatten mem channels
        }

    def get_activated_reconstructions(self, reconstr_container):
        """ Returns activated reconstruction

        :param reconstr: unactivated reconstr logits list
        :returns: activated reconstr
        :rtype: dict

        """
        activated_recon = self.nll_activation(reconstr_container)
        shape = activated_recon.shape
        return {
            'reconstruction_imgs': activated_recon.view(-1, *shape[-3:])
        }
