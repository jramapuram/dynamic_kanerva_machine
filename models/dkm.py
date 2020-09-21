from copy import deepcopy

import tree
import torch
import numpy as np
import torch.nn as nn


import helpers.layers as layers
import helpers.distributions as dist
from models.memory import KanervaMemory, MemoryState
from models.vae.abstract_vae import AbstractVAE
import helpers.distributions as distributions


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
                                    memory_size=self.config['memory_size'],
                                    num_opt_iters=self.config['num_opt_iters'],
                                    sample_w=self.config['sample_w'],
                                    sample_M=self.config['sample_memory'])

        # Projections from embedding to write and reads
        self.write_projector = self._build_dense(input_size=self.config['code_size'],  # self.config['latent_size'],
                                                 output_size=self.config['code_size'],
                                                 name='write')
        self.read_projector = self._build_dense(input_size=self.config['code_size'],  # self.config['latent_size'],
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
        is_3d_model = 'tsm' in self.config['encoder_layer_type'] \
            or 's3d' in self.config['encoder_layer_type']
        import torchvision
        encoder = nn.Sequential(
            # fold in if we have a 3d model: [T*B, C, H, W]
            layers.View([-1, *self.input_shape]) if not is_3d_model else layers.Identity(),
            layers.get_encoder(
                norm_first_layer=True, norm_last_layer=False,
                layer_fn=torchvision.models.resnet18,
                pretrained=False, num_segments=1,
                temporal_pool=False, **self.config)
            (
                output_size=self.config['code_size']
            ),
            layers.View([-1, self.config['episode_length'], self.config['code_size']])  # un-fold episode to [T, B, F]
        )
        is_torchvision_encoder = isinstance(encoder[1], (layers.TSMResnetEncoder,
                                                         layers.S3DEncoder,
                                                         layers.TorchvisionEncoder))
        if self.config['encoder_layer_modifier'] == 'sine' and is_torchvision_encoder:
            layers.convert_to_sine_module(encoder[1].model)

        if is_torchvision_encoder and self.config['encoder_activation'] != 'relu':
            layers.convert_layer(encoder[1].model, nn.ReLU,
                                 layers.str_to_activ_module(self.config['encoder_activation']),
                                 set_from_layer_kwargs=False)

        if is_torchvision_encoder and self.config['conv_normalization'] == 'groupnorm':
            layers.convert_batchnorm_to_groupnorm(encoder[1].model)

        if is_torchvision_encoder and self.config['conv_normalization'] == 'evonorms0':
            layers.convert_batchnorm_to_evonorms0(encoder[1].model)

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
            )
        )

        # append the variance as necessary
        decoder = self._append_variance_projection(decoder)
        output_shape = dec_conf['input_shape']
        output_shape[0] *= 2 if dist.nll_has_variance(self.config['nll_type']) else 1
        decoder = nn.Sequential(decoder, layers.View([-1, episode_length, *output_shape]))
        return torch.jit.script(decoder) if self.config['jit'] else decoder

    def _expand_memory(self, memory, batch_size):
        """Internal helper to expand the memory view for the ST

        :param key: the indexing key: [B*T*NR, 3]
        :returns: an expanded view of the memory
        :rtype: torch.Tensor

        """
        def _expand_mean_or_cov(memory):
            if memory.shape[0] >= batch_size:
                # Base case where we don't need to expand.
                # Useful for generations of less than batch_size.
                return memory[0:batch_size]

            expand_count = int(np.ceil(batch_size / float(memory.shape[0])))  # expand by NR
            expanded_memory_shape = [-1] + list(memory.shape[1:])
            expanded_memory = memory.unsqueeze(1).expand(  # [B, MS, WS] -> [B, NR, MS, WS]
                -1, expand_count, -1, -1).contiguous().view(expanded_memory_shape)  # [-1, MS, WS]
            return expanded_memory[0:batch_size]

        return MemoryState(M_mean=_expand_mean_or_cov(memory.M_mean),
                           M_cov=_expand_mean_or_cov(memory.M_cov))

    def _attractor_loop(self, inputs, posterior_memory):
        """Runs the encode -> decode loop without doing a memory posterior update.

        :param inputs: the inputs [B, T, C, W, H]
        :param posterior_memory:
        :returns:
        :rtype:

        """
        z_episode = self.encode(inputs)                                 # [B, T, F] base logits.
        read_z, _ = self.memory.read_with_z(z_episode.transpose(0, 1),  # read_z: [T, B, code_size]
                                            posterior_memory)
        return self.nll_activation(self.decode(read_z))

    def generate_synthetic_samples(self, batch_size, memory_state, **kwargs):
        """ Generates samples with VAE.

        :param batch_size: the number of samples to generate.
        :param memory: an updated memory posterior
        :returns: decoded logits
        :rtype: torch.Tensor

        """
        with torch.no_grad():
            # expand the memory if needed, simply tiles to get to batch_size memories
            memory_state = self._expand_memory(memory_state, batch_size)

            episode_length = self.config['episode_length']
            w_samples = self.memory.sample_prior_w(seq_length=episode_length,
                                                   batch_size=batch_size)
            z_samples = self.memory.get_w_to_z_mean(w_samples, memory_state.M_mean)
            z_read, _ = self.memory.read_with_z(z=z_samples, memory_state=memory_state)

            # Decode the read samples & return
            generations = {'generated0_imgs': self.nll_activation(self.decode(z_read))}

            # do the attractor cleanup iterations
            for idx in range(1, self.config['attractor_cleanup_interations'] + 1):
                previous_generations = generations['generated{}_imgs'.format(idx - 1)]
                generations['generated{}_imgs'.format(idx)] = self._attractor_loop(
                    previous_generations, memory_state)

            # Fold in episode into batch and return for visualization.
            return tree.map_structure(lambda t: t.view([-1, *t.shape[-3:]]), generations)

    def encode(self, x):
        """ Encodes a tensor x to a set of logits.

        :param x: the input tensor
        :returns: logits
        :rtype: torch.Tensor

        """
        encoded = self.encoder(x)
        if encoded.dim() < 2:
            return encoded.unsqueeze(-1)

        return encoded

    def preprocess_minibatch_and_labels(self, minibatch, labels):
        """Simple helper to push the minibatch to the correct device and shape."""
        minibatch = minibatch.cuda(non_blocking=True) if self.config['cuda'] else minibatch
        labels = labels.cuda(non_blocking=True) if self.config['cuda'] else labels

        if minibatch.dim() == 4 or minibatch.dim() == 2:
            minibatch = minibatch.view([-1, self.config['episode_length'], *minibatch.shape[1:]])
            labels = labels.view([-1, self.config['episode_length'], *labels.shape[1:]])

        return minibatch, labels

    def likelihood(self, loader, K=1000):
        """ Likelihood by integrating ELBO.

        :param loader: the data loader to iterate over.
        :param K: number of importance samples.
        :returns: likelihood produced by monte-carlo integration of elbo.
        :rtype: float32

        """
        with torch.no_grad():
            likelihood = []

            for num_minibatches, (minibatch, labels) in enumerate(loader):
                minibatch, labels = self.preprocess_minibatch_and_labels(minibatch, labels)
                batch_size = minibatch.shape[0]

                for idx in range(batch_size):
                    minibatch_i = minibatch[idx].expand_as(minibatch).contiguous()
                    labels_i = labels[idx].expand_as(labels).contiguous()

                    elbo = []
                    for count in range(K // batch_size):
                        z, params = self.posterior(minibatch_i, labels=labels_i)
                        decoded_logits = self.decode(z)
                        loss_t = self.loss_function(decoded_logits, minibatch_i, params)
                        elbo.append(loss_t['elbo'])

                    # compute the log-sum-exp of the elbo of the single sample taken over K replications
                    multi_sample_elbo = torch.cat([e.unsqueeze(0) for e in elbo], 0).view([-1])
                    likelihood.append(torch.logsumexp(multi_sample_elbo, dim=0) - np.log(count + 1))

            return torch.mean(torch.cat([l.unsqueeze(0) for l in likelihood], 0))

    def posterior(self, inputs, labels=None, force=False):
        """ get a reparameterized Q(z|x) for a given x

        :param x: input tensor
        :param labels: (optional) labels
        :param force:  force reparameterization
        :returns: reparam dict
        :rtype: torch.Tensor

        """
        if isinstance(inputs, (list, tuple)):
            inputs = torch.cat([i.unsqueeze(1) for i in inputs], 1)     # expand to [B, T, C, W, H]

        batch_size, episode_size = [inputs.shape[0], inputs.shape[1]]
        z_episode = self.encode(inputs)                                 # [B, T, F] base logits.

        # Project the episode logits to write and read logits
        # read_logits = self.read_projector(z_episode).transpose(0, 1)    # [T, B, code_size] for DKM
        # write_logits = self.write_projector(z_episode).transpose(0, 1)  # [T, B, code_size] for DKM
        read_logits = write_logits = z_episode.transpose(0, 1)

        # Grab the prior and update it to the posterior given the episode
        prior_memory = self.memory.get_prior_state(batch_size)
        posterior_memory, _, dkl_w, dkl_M = self.memory.update_state(write_logits,
                                                                     prior_memory)  # dkl_w: [T, B]

        # Read from the memory & compute the total KL penalty
        read_z, dkl_r = self.memory.read_with_z(read_logits,              # read_z: [T, B, code_size]
                                                posterior_memory)         # dkl_r: [T, B]
        # dkl_M = self.memory.get_dkl_total(posterior_memory)             # dkl_M: [T, B]

        return read_z.transpose(0, 1), {                                  # read_z.T: [B, T, code_size]
            'memory_kl': dkl_M / episode_size,
            'read_kl': dkl_r / episode_size,
            'write_kl': dkl_w / episode_size,

            # We need an actual updated memory to generate samples.
            'memory_state': posterior_memory,

            # add the auto-encoded projection (specified in paper)
            'ae_decode': self.decode(z_episode),
        }

    def kld(self, dist_a):
        """ KL-Divergence of the distribution dict and the prior of that distribution.

        :param dist_a: the distribution dict.
        :returns: tensor that is of dimension batch_size
        :rtype: torch.Tensor

        """
        # reduce over the episode dimension for the KLD
        return torch.mean(dist_a['memory_kl'] + dist_a['read_kl'] + dist_a['write_kl'], 0)

    def nll(self, x, recon_x, nll_type):
        """ Grab the negative log-likelihood for a specific NLL type

        :param x: the true tensor
        :param recon_x: the reconstruction tensor
        :param nll_type: the NLL type (str)
        :returns: [B] dimensional tensor
        :rtype: torch.Tensor

        """
        episode_len = x.shape[1]
        return distributions.nll(x, recon_x, nll_type) / episode_len

    def loss_function(self, recon_x, x, params, K=1):
        """ Loss function for Kanerva ++

        :param recon_x: the reconstruction logits
        :param x: the original samples
        :param params: the reparameterized parameter dict containing reparam
        :param K: number of monte-carlo samples to use.
        :returns: loss dict
        :rtype: dict

        """
        if isinstance(x, (list, tuple)):
            x = torch.cat([xi.unsqueeze(1) for xi in x], 1).contiguous()  # [B, T, C, H, W]

        # compute the AE loss as specified in the paper
        ae_loss = distributions.nll(x=x, recon_x=params['ae_decode'],
                                    nll_type=self.config['nll_type'])

        loss_dict = super(DynamicKanervaMachine, self).loss_function(
            recon_x=recon_x, x=x, params=params, K=K, ae_loss=ae_loss)

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
