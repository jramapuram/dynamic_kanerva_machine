#!/usr/bin/env python3

"""Memory module for Kanerva Machines; ported to pytorch from:
   https://github.com/deepmind/dynamic-kanerva-machines

  Functions of the module always take inputs with shape:
  [seq_length, batch_size, ...]

  Examples:

    import torch

    # Set example variables
    episode_length, batch_size, code_size, memory_size = [8, 5, 100, 32]

    # Initialisation
    memory = KanervaMemory(code_size=code_size, memory_size=memory_size)
    prior_memory = memory.get_prior_state(batch_size)

    # Update memory posterior
    z_episode = torch.randn([episode_length, batch_size, code_size])
    posterior_memory, _, _, _ = memory.update_state(z_episode, prior_memory)

    # Read from the memory using cues z_q
    z_q = torch.randn([episode_length, batch_size, code_size])
    read_z, dkl_w = memory.read_with_z(z_q, posterior_memory)

    # Compute the KL-divergence between posterior and prior memory
    dkl_M = memory.get_dkl_total(posterior_memory)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist


MemoryState = collections.namedtuple(
    'MemoryState',
    # Mean of memory slots, [batch_size, memory_size, word_size]
    # Covariance of memory slots, [batch_size, memory_size, memory_size]
    ('M_mean', 'M_cov'))

EPSILON = 1e-6


def lstsq(A, Y, lamb=0.0):
    """Differentiable, regularized least squares. Supports hacky batched ops.
       Sourced from: https://github.com/pytorch/pytorch/issues/27036

    :param A: [B, M, N] or [M, N]
    :param Y: [B, N, 1] or [N, 1]
    :param lamb: scalar for l2 regularizer
    :returns: solution to regularized least squares
    :rtype: torch.Tensor

    """
    def _lstsq(A, Y, lamb=0.0):
        """Single batch regularized least squares"""
        # Assuming A to be full column rank
        cols = A.shape[1]
        if cols == torch.matrix_rank(A):
            q, r = torch.qr(A)
            x = torch.inverse(r) @ q.T @ Y
        else:
            A_dash = A.permute(1, 0) @ A + lamb * torch.eye(cols, device=A.device)
            Y_dash = A.permute(1, 0) @ Y
            x = lstsq(A_dash, Y_dash)

        return x

    if A.dim() == 3:  # TODO: no batched least squares currently
        batch_size = A.shape[0]
        soln = torch.stack([_lstsq(A[i], Y[i]) for i in range(batch_size)], dim=0)
    elif A.dim() == 2:
        soln = _lstsq(A, Y)
    else:
        raise NotImplementedError("unknown number of dimensions requested for regularized least squares")

    # if A.dim() == 3:  # TODO: no batched least squares currently
    #     batch_size = A.shape[0]
    #     ls_outputs = [torch.lstsq(input=Y[i], A=A[i]) for i in range(batch_size)]
    #     return torch.stack([l[0] for l in ls_outputs], dim=0)
    # elif A.dim() == 2:
    #     soln, _ = torch.lstsq(input=Y, A=A)

    return soln


def batch_fill_diag(matrix, vector):
    """Does a batch fill of vector into matrix. TODO: vectorize"""

    def _fill_diag(matrix, vector):
        """single matrix - vector solution"""
        diag = torch.diagonal(matrix)
        eye = torch.eye(diag.shape[-1], device=matrix.device)
        return (matrix - diag * eye) + vector * eye

    if matrix.dim() == 3:
        soln = torch.stack([_fill_diag(m, v) for m, v in zip(matrix, vector)], dim=0)
    elif matrix.dim() == 2:
        soln = _fill_diag(matrix, vector)
    else:
        raise NotImplementedError("unknown number of dimensions requested for batch_fill_diag.")

    return soln


# disable lint warnings for cleaner algebraic expressions
# pylint: disable=invalid-name
class KanervaMemory(nn.Module):
    """A memory-based generative model."""

    def __init__(self,
                 code_size,
                 memory_size,
                 num_opt_iters=1,
                 w_prior_stddev=1.0,
                 obs_noise_stddev=1.0,
                 sample_w=False,
                 sample_M=False):
        """Initialise the memory module.

        batch_prior_mean = torch.Size([5, 32, 100]) | cov = torch.Size([5, 32, 32])
        M = torch.Size([5, 32, 100]) | new_z_mean = torch.Size([1, 5, 100]) | w_matrix = torch.Size([5, 32, 32])
        w_mean =  torch.Size([5, 32, 1])
        posterior_cov = torch.Size([5, 32, 32]) | mean = torch.Size([5, 32, 100])

         Args:
             code_size: Integer specifying the size of each encoded input.
             memory_size: Integer specifying the total number of rows in the memory.
             num_opt_iters: Integer specifying the number of optimisation iterations.
             w_prior_stddev: Float specifying the standard deviation of w's prior.
             obs_noise_stddev: Float specifying the standard deviation of the
               observational noise.
             sample_w: Boolean specifying whether to sample w or simply take its mean.
             sample_M: Boolean specifying whether to sample M or simply take its mean.
        """
        super(KanervaMemory, self).__init__()
        self._memory_size = memory_size
        self._code_size = code_size
        self._num_opt_iters = num_opt_iters
        self._sample_w = sample_w
        self._sample_M = sample_M
        self._w_prior_stddev = w_prior_stddev

        # trainable std-deviation
        self._log_w_stddev = nn.Parameter(torch.zeros(1) + np.log(0.3),
                                          requires_grad=True)
        if obs_noise_stddev > 0.0:
            self._obs_noise_stddev = nn.Parameter(torch.zeros(1) + np.log(obs_noise_stddev),
                                                  requires_grad=False)  # constant
        else:  # learned parameter
            self._obs_noise_stddev = nn.Parameter(torch.zeros(1), requires_grad=True)

        self._prior_log_var = nn.Parameter(torch.zeros(1), requires_grad=True)
        self._prior_mean = nn.Parameter(nn.init.trunc_normal_(
            torch.zeros([self._memory_size, self._code_size])), requires_grad=True)

    def _get_w_dist(self, mu_w):
        w_stddev = torch.exp(self._log_w_stddev)
        cov = w_stddev * torch.eye(self._memory_size,
                                   dtype=mu_w.dtype,
                                   device=mu_w.device)
        return dist.MultivariateNormal(
          loc=mu_w, covariance_matrix=cov)

    def sample_prior_w(self, seq_length, batch_size):
        """Sample w from its prior.

        Args:
          seq_length: length of sequence
          batch_size: batch size of samples
        Returns:
          w: [batch_size, memory_size]
        """
        w_stddev = torch.exp(self._log_w_stddev)
        eye = torch.eye(self._memory_size,
                        dtype=w_stddev.dtype,
                        device=w_stddev.device)
        cov = self._w_prior_stddev * eye
        loc = torch.zeros([self._memory_size],
                          dtype=w_stddev.dtype,
                          device=w_stddev.device)
        return dist.MultivariateNormal(
          loc=loc, covariance_matrix=cov).rsample([seq_length, batch_size])

    def read_with_z(self, z, memory_state):
        """Query from memory (specified by memory_state) using embedding z.

        Args:
          z: Tensor with dimensions [episode_length, batch_size, code_size]
            containing an embedded input.
          memory_state: Instance of `MemoryState`.

        Returns:
          A tuple of tensors containing the mean of read embedding and the
            KL-divergence between the w used in reading and its prior.
        """
        M = self.sample_M(memory_state)
        w_mean = self._solve_w_mean(z, M)
        w_samples = self.sample_w(w_mean)
        dkl_w = self.get_dkl_w(w_mean)
        z_mean = self.get_w_to_z_mean(w_samples, M)
        return z_mean, dkl_w

    def wrap_z_dist(self, z_mean):
        """Wrap the mean of z as an observation (Gaussian) distribution."""
        cov = torch.exp(self._obs_noise_stddev) * torch.eye(self._memory_size,
                                                            dtype=z_mean.dtype,
                                                            device=z_mean.device)
        return dist.MultivariateNormal(
          loc=z_mean, covariance_matrix=cov)

    def sample_w(self, w_mean):
        """Sample w from its posterior distribution."""
        if self._sample_w:
            return self._get_w_dist(w_mean).rsample()
        else:
            return w_mean

    def sample_M(self, memory_state):
        """Sample the memory from its distribution specified by memory_state."""
        if self._sample_M:
            loc = torch.zeros(memory_state.M_cov.shape[-1], dtype=memory_state.M_cov.dtype)
            noise_dist = dist.MultivariateNormal(loc=loc, covariance_matrix=memory_state.M_cov)

            # C, B, M
            noise = torch.transpose(noise_dist.rsample(self._code_size),
                                    [1, 2, 0])
            return memory_state.M_mean + noise
        else:
            return memory_state.M_mean

    def get_w_to_z_mean(self, w_p, R):
        """Return the mean of z by reading from memory using weights w_p."""
        return torch.einsum('sbm,bmc->sbc', w_p, R)  # Rw

    def _read_cov(self, w_samples, memory_state):
        episode_size, batch_size = w_samples.shape[:2]
        _, U = memory_state  # cov: [B, M, M]
        wU = torch.einsum('sbm,bmn->sbn', w_samples, U)
        wUw = torch.einsum('sbm,sbm->sb', wU, w_samples)
        assert wUw.shape == (episode_size, batch_size)
        return wU, wUw

    def get_dkl_total(self, memory_state):
        """Compute the KL-divergence between a memory distribution and its prior."""
        R, U = memory_state
        B, K, _ = R.shape
        # print("U = {} | R = {} ".format(U.shape, R.shape))
        assert U.shape == (B, K, K)
        R_prior, U_prior = self.get_prior_state(B)
        p_diag = U_prior.diagonal(0, dim1=-2, dim2=-1)
        q_diag = U.diagonal(0, dim1=-2, dim2=-1)  # B, K
        t1 = self._code_size * torch.sum(q_diag / p_diag, -1)
        t2 = torch.sum((R - R_prior)**2 / p_diag.unsqueeze(-1), [-2, -1])
        t3 = -self._code_size * self._memory_size
        t4 = self._code_size * torch.sum(torch.log(p_diag) - torch.log(q_diag), -1)
        return t1 + t2 + t3 + t4

    def _get_dkl_update(self, memory_state, w_samples, new_z_mean, new_z_var):
        """Compute memory_kl after updating prior_state."""
        B, K, C = memory_state.M_mean.shape
        S = w_samples.shape[0]

        # check shapes
        assert w_samples.shape == (S, B, K)
        assert new_z_mean.shape == (S, B, C)

        delta = new_z_mean - self.get_w_to_z_mean(w_samples, memory_state.M_mean)
        _, wUw = self._read_cov(w_samples, memory_state)
        var_z = wUw + new_z_var + torch.exp(self._obs_noise_stddev)**2
        beta = wUw / var_z

        dkl_M = -0.5 * (self._code_size * beta
                        - torch.sum((beta / var_z).unsqueeze(-1)
                                    * delta**2, -1)
                        + self._code_size * torch.log(1 - beta))
        assert dkl_M.shape == (S, B)
        return dkl_M

    def _get_prior_params(self):
        self._prior_var = torch.ones([self._memory_size],
                                     dtype=self._prior_log_var.dtype,
                                     device=self._prior_log_var.device) * torch.exp(self._prior_log_var) + EPSILON
        prior_cov = torch.diag(self._prior_var)
        return self._prior_mean, prior_cov

    @property
    def prior_avg_var(self):
        """return the average of prior memory variance."""
        return torch.mean(self._prior_var)

    def _solve_w_mean(self, new_z_mean, M):
        """Minimise the conditional KL-divergence between z wrt w."""
        w_matrix = torch.matmul(M, M.permute(0, 2, 1))
        # M = torch.Size([160, 100])
        # new_z_mean = torch.Size([1, 5, 100])
        # w_matrix = torch.Size([160, 160])
        # print('M = {} | new_z_mean = {} | w_matrix = {}'.format(M.shape, new_z_mean.shape, w_matrix.shape))
        w_rhs = torch.einsum('bmc,sbc->bms', M, new_z_mean)
        w_mean = lstsq(w_matrix, w_rhs, torch.exp(self._obs_noise_stddev)**2 / self._w_prior_stddev**2)
        # print('w_mean = ', w_mean.shape)
        w_mean = torch.einsum('bms->sbm', w_mean)
        return w_mean

    def get_prior_state(self, batch_size):
        """Return the prior distribution of memory as a MemoryState."""
        prior_mean, prior_cov = self._get_prior_params()
        batch_prior_mean = torch.cat([prior_mean.unsqueeze(0)] * batch_size)
        batch_prior_cov = torch.cat([prior_cov.unsqueeze(0)] * batch_size)
        # print('batch_prior_mean = {} | cov = {}'.format(batch_prior_mean.shape, batch_prior_cov.shape))
        return MemoryState(M_mean=batch_prior_mean,
                           M_cov=batch_prior_cov)

    def update_state(self, z, memory_state):
        """Update the memory state using Bayes' rule.

        Args:
          z: A tensor with dimensions [episode_length, batch_size, code_size]
            containing a sequence of embeddings to write into memory.
          memory_state: A `MemoryState` namedtuple containing the memory state to
            be written to.

        Returns:
          A tuple containing the following elements:
          final_memory: A `MemoryState` namedtuple containing the new memory state
            after the update.
          w_mean_episode: The mean of w for the written episode.
          dkl_w_episode: The KL-divergence of w for the written episode.
          dkl_M_episode: The KL-divergence between the memory states before and
            after the update.
        """

        episode_size, batch_size = z.shape[:2]
        w_array = torch.zeros([episode_size, batch_size, self._memory_size],
                              dtype=z.dtype, device=z.device)
        dkl_w_array = torch.zeros([episode_size, batch_size], dtype=z.dtype, device=z.device)
        dkl_M_array = torch.zeros([episode_size, batch_size], dtype=z.dtype, device=z.device)

        def loop_body(i, old_memory, w_array, dkl_w_array, dkl_M_array):
            """Update memory step-by-step."""
            z_step = z[i].unsqueeze(0)
            new_memory = old_memory
            for _ in range(self._num_opt_iters):
                w_step_mean = self._solve_w_mean(z_step, self.sample_M(new_memory))
                w_step_sample = self.sample_w(w_step_mean)
                new_memory = self._update_memory(old_memory,
                                                 w_step_mean,
                                                 z_step, 0)
            dkl_w_step = self.get_dkl_w(w_step_mean)
            dkl_M_step = self._get_dkl_update(old_memory,
                                              w_step_sample,
                                              z_step, 0)
            # Update the i-th row
            w_array[i] = w_step_sample
            dkl_w_array[i] = dkl_w_step
            dkl_M_array[i] = dkl_M_step

            return (new_memory,
                    w_array,
                    dkl_w_array,
                    dkl_M_array)

        # Run the loop in pytorch
        final_memory, w_mean, dkl_w, dkl_M = memory_state, w_array, dkl_w_array, dkl_M_array
        for i in range(episode_size):
            final_memory, w_mean, dkl_w, dkl_M = loop_body(i, final_memory, w_mean, dkl_w, dkl_M)

        w_mean_episode = w_mean
        dkl_w_episode = dkl_w
        dkl_M_episode = dkl_M
        assert dkl_M_episode.shape == (episode_size, batch_size)

        return final_memory, w_mean_episode, dkl_w_episode, dkl_M_episode

    def _update_memory(self, old_memory, w_samples, new_z_mean, new_z_var):
        """Setting new_z_var=0 for sample based update."""
        old_mean, old_cov = old_memory
        wR = self.get_w_to_z_mean(w_samples, old_memory.M_mean)
        wU, wUw = self._read_cov(w_samples, old_memory)
        sigma_z = wUw + new_z_var + torch.exp(self._obs_noise_stddev)**2  # [S, B]
        delta = new_z_mean - wR  # [S, B, C]
        c_z = wU / sigma_z.unsqueeze(-1)  # [S, B, M]
        posterior_mean = old_mean + torch.einsum('sbm,sbc->bmc', c_z, delta)
        posterior_cov = old_cov - torch.einsum('sbm,sbn->bmn', c_z, wU)

        # Clip diagonal elements for numerical stability
        # print('posterior_cov = {} | mean = {}'.format(posterior_cov.shape, posterior_mean.shape))
        posterior_cov = batch_fill_diag(
            posterior_cov,
            torch.clamp(posterior_cov.diagonal(0, dim1=-2, dim2=-1), EPSILON, 1e10)
        )
        # posterior_cov.fill_diagonal_(
        #     torch.clamp(torch.diag(posterior_cov), EPSILON, 1e10))
        new_memory = MemoryState(M_mean=posterior_mean, M_cov=posterior_cov)
        return new_memory

    def get_dkl_w(self, w_mean):
        """Return the KL-divergence between posterior and prior weights w."""
        posterior_dist = self._get_w_dist(w_mean)

        # Build the prior distribution
        prior_loc = torch.zeros([self._memory_size],
                                dtype=w_mean.dtype,
                                device=w_mean.device)
        eye = torch.eye(self._memory_size,
                        dtype=w_mean.dtype,
                        device=w_mean.device)
        prior_cov = self._w_prior_stddev * eye
        w_prior_dist = dist.MultivariateNormal(loc=prior_loc, covariance_matrix=prior_cov)

        # use torch distributions to take the KLD
        dkl_w = dist.kl_divergence(posterior_dist, w_prior_dist)
        assert dkl_w.shape == w_mean.shape[:-1]
        return dkl_w


def _test_memory():
    import torch

    # Set example variables
    episode_length, batch_size, code_size, memory_size = [8, 5, 100, 32]

    # Initialisation
    memory = KanervaMemory(code_size=code_size, memory_size=memory_size)
    prior_memory = memory.get_prior_state(batch_size)

    # Update memory posterior
    z_episode = torch.randn([episode_length, batch_size, code_size])
    posterior_memory, _, _, _ = memory.update_state(z_episode, prior_memory)

    # Read from the memory using cues z_q
    z_q = torch.randn([episode_length, batch_size, code_size])
    read_z, dkl_w = memory.read_with_z(z_q, posterior_memory)

    # Compute the KL-divergence between posterior and prior memory
    dkl_M = memory.get_dkl_total(posterior_memory)
    assert dkl_M.shape[0] == batch_size

    # test grads
    print("\nPRE .backward() call:")
    print("posterior_memory has grads? ", posterior_memory.M_mean.grad is not None)
    print("self._w_stddev has grads?", memory._w_stddev.grad is not None)
    print("\nrunning backward to propagate gradients..\n")
    dkl_M.mean().backward()
    print("posterior_memory has grads? ", posterior_memory.M_mean.grad is not None)
    print("self._w_stddev has grads?", memory._w_stddev.grad is not None)
