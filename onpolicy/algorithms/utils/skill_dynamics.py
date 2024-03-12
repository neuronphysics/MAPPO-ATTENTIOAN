import torch
import torch.nn as nn
from torch.distributions import Beta
import numpy as np
import torch.distributions as D
import torch.nn.functional as F
import abc
from functools import reduce
from onpolicy.algorithms.utils.cnn import Flatten
from typing import Any, Dict, List, Tuple, Union, Optional
from onpolicy.algorithms.utils.transporter import CNNKeyPointsBase as CNNBase

TensorType = Any


def get_activation_fn(name: Optional[str] = None):
    """Returns a framework specific activation function, given a name string.

    Args:
        name (Optional[str]): One of "relu" (default), "tanh", "swish", or
            "linear" or None.
        framework (str): One of "jax", "tf|tfe|tf2" or "torch".

    Returns:
        A framework-specific activtion function. e.g. tf.nn.tanh or
            torch.nn.ReLU. None if name in ["linear", None].

    Raises:
        ValueError: If name is an unknown activation function.
    """
    # Already a callable, return as-is.
    if callable(name):
        return name

    # Infer the correct activation function from the string specifier.

    if name == "relu":
        return nn.ReLU
    elif name == "tanh":
        return nn.Tanh
    else:
        raise ValueError("Unknown activation ({}) for pytorch!".format(
            name))


class SlimFC(nn.Module):
    """Simple PyTorch version of `linear` function"""

    def __init__(self,
                 in_size: int,
                 out_size: int,
                 initializer: Any = None,
                 use_batch_normalization: Any = None,
                 activation_fn: Any = None,
                 use_bias: bool = True,
                 bias_init: float = 0.0,
                 use_skip_connection: bool = False,
                 apply_spectral_norm: bool = False):
        """Creates a standard FC layer, similar to torch.nn.Linear

        Args:
            in_size(int): Input size for FC Layer
            out_size (int): Output size for FC Layer
            initializer (Any): Initializer function for FC layer weights
            activation_fn (Any): Activation function at the end of layer
            use_bias (bool): Whether to add bias weights or not
            bias_init (float): Initalize bias weights to bias_init const
            apply_spectral_norm (bool): Apply spectral normalization to the linear layer
        """
        super(SlimFC, self).__init__()
        layers = []
        self._use_skip_connection = use_skip_connection
        # Actual nn.Linear layer (including correct initialization logic).
        if self._use_skip_connection:
            self.skip_connection = nn.Identity()
        linear = nn.Linear(in_size, out_size, bias=use_bias)
        if initializer:
            initializer(linear.weight)
        if use_bias is True:
            nn.init.constant_(linear.bias, bias_init)
        if apply_spectral_norm:
            linear = nn.utils.spectral_norm(linear)

        layers.append(linear)
        if use_batch_normalization is not None:
            layers.append(nn.BatchNorm1d(out_size))
        # Activation function (if any; default=None (linear)).
        if isinstance(activation_fn, str):
            activation_fn = get_activation_fn(activation_fn, "torch")
        if activation_fn is not None:
            layers.append(activation_fn)
        # Put everything in sequence.
        self._model = nn.Sequential(*layers)

    def forward(self, x: TensorType) -> TensorType:
        if self._use_skip_connection:
            identity = self.skip_connection(x)
        out = self._model(x)
        if self._use_skip_connection:
            out += identity
        return out


class SkillDynamics(nn.Module):
    """The default skill dynamics model for DADS"""

    def __init__(self,
                 args,
                 obs_shape,
                 z_range=(-1, 1),
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 z_type='cont',
                 prior_samples=100,
                 use_batch_normalization=True,
                 dynamics_reg_hiddens=False,
                 dynamics_orth_reg=True,
                 dynamics_l2_reg=False,
                 dynamics_spectral_norm=False,  # dynamics spectral norm
                 fix_variance=False,
                 base_kwargs=None):
        super().__init__()

        if base_kwargs is None:
            base_kwargs = {}
        self.z_dim = args.skill_dim
        self.max_num_experts = args.skill_max_num_experts
        self.share_policy = args.share_policy
        self.num_reps = prior_samples
        self.z_range = z_range
        self.z_type = z_type
        self.z_dist = D.Uniform(z_range[0], z_range[1])
        self.fix_variance = fix_variance
        self._dynamics_reg_hiddens = dynamics_reg_hiddens
        self._dynamics_orth_reg = dynamics_orth_reg
        self._dynamics_spectral_norm = dynamics_spectral_norm
        self._dynamics_l2_reg = dynamics_l2_reg
        self.device = device

        # Assuming obs_shape is something like (batch_size, dim1, dim2, ...)
        self.obs_dim = reduce(lambda x, y: x * y, obs_shape[0:])
        self.hiddens = CNNBase(obs_shape, **base_kwargs)

        input_dim = self.z_dim
        print(f"size of observational skill dynamics input {self.obs_dim}")
        print(f'dynamics input dim: {input_dim}')
        self.bn_in = nn.BatchNorm2d(obs_shape[0])
        self.bn_target = nn.BatchNorm2d(obs_shape[0], affine=False)
        self.flatten = Flatten()

        self.logits = SlimFC(self.hiddens.output_size + input_dim, self.max_num_experts,
                             initializer=lambda w: nn.init.xavier_uniform_(w, 1.0),
                             apply_spectral_norm=self._dynamics_spectral_norm)  # nn.Linear(hidden_dim, self.num_experts)

        self.means = SlimFC(self.hiddens.output_size + input_dim, self.max_num_experts * self.obs_dim,
                            initializer=lambda w: nn.init.xavier_uniform_(w, 1.0),
                            apply_spectral_norm=self._dynamics_spectral_norm)  # nn.Linear(hidden_dim, self.num_experts * self.obs_dim)
        if not fix_variance:
            self.stddev = SlimFC(self.hiddens.output_size + input_dim, self.max_num_experts * self.obs_dim,
                                 initializer=lambda w: nn.init.xavier_uniform_(w, 1.0),
                                 apply_spectral_norm=self._dynamics_spectral_norm)
        # print(self.hiddens._modules)
        # print(self.hiddens._modules['0']._model._modules['0'])

        self.to(self.device)

        self._dynamics_lr = args.dynamics_lr  # Assuming dynamics learning rate is passed through 'args'
        self.dynamics_opt = torch.optim.Adam(self.parameters(), lr=self._dynamics_lr, eps=args.opti_eps,
                                             weight_decay=args.weight_decay)

    def orthogonal_regularization(self):
        reg = 1e-4
        orth_loss = torch.zeros(1).to(self.device)
        layers = [self.logits, self.means]
        if self._dynamics_reg_hiddens:
            layers.append(self.hiddens)
        for layer in layers:
            for name, param in layer.named_parameters():
                if 'bias' not in name:
                    param_flat = param.view(param.shape[0], -1)
                    sym = torch.mm(param_flat, torch.t(param_flat))
                    sym -= torch.eye(param_flat.shape[0], device=self.device)
                    orth_loss = orth_loss + (reg * sym.abs().sum())
        return torch.sum(orth_loss)

    def l2_regularization(self):
        reg = 1e-4
        l2_loss = torch.zeros(1).to(self.device)
        for name, param in self.hiddens.named_parameters():
            if 'bias' not in name:
                l2_loss = l2_loss + (0.5 * reg * torch.sum(torch.pow(param, 2)))
        return torch.sum(l2_loss)

    def forward(self, obs, z, rnn_hxs, masks, output_mask, output_feat, training=False):

        # obs = batch_norm(obs)
        # Debugging: Print shapes to verify dimensions
        obs = self.process_obs(obs)
        self.bn_in.train(mode=training)
        norm_obs = self.bn_in(obs)
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).to(self.device)
        else:
            z = z.to(self.device)
        # inp = torch.cat([norm_obs, z], dim=-1)
        inp = norm_obs.permute(0, 3, 1, 2)
        critic_lin, x, rnn = self.hiddens(inp, rnn_hxs, masks, output_mask=output_mask, output_feat=output_feat)
        combined_input = torch.cat([z, x], dim=-1)  # skip connection
        # https://luiarthur.github.io/TuringBnpBenchmarks/dpsbgmm
        eta = self.logits(combined_input)  # Adjusted for skip connection [batch,num_experts]
        means = self.means(combined_input)
        means = means.reshape(obs.shape[0], self.max_num_experts, self.obs_dim)
        if not self.fix_variance:
            stddevs = nn.Softplus()(self.stddev(combined_input))
            stddevs = stddevs.reshape(obs.shape[0], self.max_num_experts, self.obs_dim)
        else:
            stddevs = torch.ones_like(means)  # (num_components, obs_size)
        return eta, means, stddevs

    def get_distribution(self, obs, z, rnn_hxs, masks, output_mask=False, output_feat=False, training=False):
        eta, means, diags = self.forward(obs, z, rnn_hxs, masks, output_mask, output_feat, training)
        eta = torch.softmax(eta, dim=-1)  # Apply softmax to get probabilities
        # Ensure that self.variance is positive

        mix = D.Categorical(eta)
        comp = D.Independent(D.Normal(means, diags), 1)
        gmm = D.MixtureSameFamily(mix, comp)
        return gmm

    def get_log_prob(self, obs, z, next_obs, rnn_hxs, masks, output_mask, output_feat, training=False):
        gmm = self.get_distribution(obs, z, rnn_hxs, masks, output_mask, output_feat, training)
        self.bn_target.train(mode=training)

        next_obs = self.process_obs(next_obs)
        norm_next_obs = self.bn_target(next_obs)
        return gmm.log_prob(self.flatten(norm_next_obs))

    def _calculate_intrinsic_rewards(self, obs, z, next_obs, rnn_hxs, masks, output_mask=False, output_feat=False,
                                     weights=None):
        num_reps = self.num_reps if self.z_type == 'cont' else self.z_dim
        if self.z_type == 'cont':
            # alt_obs = obs.repeat(num_reps, 1)  # [Batch_size*num_reps, obs_dim]
            # alt_next_obs = next_obs.repeat(num_reps, 1)
            alt_obs = obs.copy()
            alt_next_obs = next_obs.copy()
            # continuous uniform
            alt_skill = np.random.uniform(self.z_range[0], self.z_range[1],
                                          size=[alt_obs.shape[0] * (1 if not self.share_policy else alt_obs.shape[1]),
                                                self.z_dim]).astype(
                np.float32)
        elif self.z_type == 'discrete':
            # alt_obs = obs.repeat(self.z_dim, 1)
            # alt_next_obs = next_obs.repeat(self.z_dim, 1)
            alt_skill = np.tile(np.eye(self.z_dim), [obs.shape[0], 1]).astype(np.float32)
            alt_obs = obs.copy()
            alt_next_obs = next_obs.copy()
        # implement https://github.com/google-research/dads/blob/abc37f532c26658e41ae309b646e8963bd7a8676/unsupervised_skill_learning/skill_discriminator.py#L108C1-L114C60
        alt_skill = torch.from_numpy(alt_skill)

        log_prob = self.get_log_prob(obs, z, next_obs, rnn_hxs, masks, output_mask, output_feat,
                                     training=False)  # [Batch_size]
        log_prob = log_prob.reshape(obs.shape[0] * (1 if not self.share_policy else obs.shape[1]), 1)
        alt_log_prob = self.get_log_prob(alt_obs, alt_skill, alt_next_obs, rnn_hxs, masks, output_mask, output_feat,
                                         training=False)  # [Batch_size*num_reps]

        if weights is not None:
            diff = (alt_log_prob - log_prob) * weights
        else:
            diff = (alt_log_prob - log_prob)

        reward = - torch.log(1 + torch.exp(torch.clamp(diff, -50, 50)).sum(dim=-1))
        # print(reward.shape)
        return reward, {'log_prob': log_prob, 'alt_log_prob': alt_log_prob,
                        'num_higher_prob': ((-diff) >= 0).sum().item()}

    def compute_loss(self, next_obs, obs, z, rnn_hxs, masks, output_mask=False, output_feat=False):
        next_dynamics_obs = next_obs - obs
        log_prob = self.get_log_prob(obs, z, next_dynamics_obs, rnn_hxs, masks, output_mask=output_mask,
                                     output_feat=output_feat, training=True)
        dynamics_loss = -torch.mean(log_prob)
        orth_loss = self.orthogonal_regularization()
        l2_loss = self.l2_regularization()
        if self._dynamics_orth_reg:
            dynamics_loss += orth_loss
        if self._dynamics_l2_reg and not self._dynamics_spectral_norm:
            dynamics_loss += l2_loss
        return dynamics_loss

    def process_obs(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self.device)

        if self.share_policy:
            if len(obs.shape) == 5:
                thread_num, agent_num, h, w, c = obs.shape
                obs = torch.reshape(obs, (thread_num * agent_num, h, w, c))

        return obs


class SkillDiscriminator(nn.Module):
    """Skill Discriminator model in PyTorch, compatible with the provided SkillDynamics class"""

    def __init__(self,
                 args,
                 obs_shape,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 discriminator_spectral_norm=False,
                 base_kwargs=None):
        super(SkillDiscriminator, self).__init__()

        if base_kwargs is None:
            base_kwargs = {}
        self.obs_dim = reduce(lambda x, y: x * y, obs_shape[0:])
        self.skill_dim = args.skill_dim
        self.device = device
        self.bn_in = nn.BatchNorm2d(obs_shape[0])

        self.hiddens = CNNBase(obs_shape, **base_kwargs)

        self.std = SlimFC(self.hiddens.output_size, self.skill_dim,
                          initializer=lambda w: nn.init.xavier_uniform_(w, 1.0),
                          apply_spectral_norm=discriminator_spectral_norm)

        self.means = SlimFC(self.hiddens.output_size, self.skill_dim,
                            initializer=lambda w: nn.init.xavier_uniform_(w, 1.0),
                            apply_spectral_norm=discriminator_spectral_norm)

        self.to(self.device)

        # Assuming discriminator learning rate is passed through 'args'
        self._discriminator_lr = args.skill_discriminator_lr
        self.discriminator_opt = torch.optim.Adam(self.parameters(), lr=self._discriminator_lr, eps=args.opti_eps,
                                                  weight_decay=args.weight_decay)

    def forward(self, obs, rnn_hxs, masks, output_mask, output_feat, training):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self.device)

        self.bn_in.train(mode=training)
        norm_obs = self.bn_in(obs)
        norm_obs = norm_obs.permute(0, 3, 1, 2)
        critic_lin, x, rnn = self.hiddens(norm_obs, rnn_hxs, masks, output_mask=output_mask, output_feat=output_feat)
        eta = self.std(x)  # [batch,num_experts]
        mean = self.means(x)
        return eta, mean

    def get_distribution(self, obs, rnn_hxs, masks, output_mask=False, output_feat=False, training=False):
        stddev, mean = self.forward(obs, rnn_hxs, masks, output_mask=output_mask, output_feat=output_feat,
                                    training=training)
        # Ensure that stddev and mean are tensors
        assert torch.is_tensor(stddev), "stddev must be a torch.Tensor"
        assert torch.is_tensor(mean), "mean must be a torch.Tensor"

        # Apply Softplus to stddev and ensure no negative or zero values
        positive_stddev = nn.Softplus()(stddev) + 1e-20
        assert torch.all(positive_stddev > 0), "stddev must be positive after Softplus"

        return D.Independent(D.Normal(mean, positive_stddev), 1)

    def get_log_prob(self, obs, z, rnn_hxs, masks, output_mask=False, output_feat=False, training=False):
        dist = self.get_distribution(obs, rnn_hxs, masks, output_mask=output_mask, output_feat=output_feat,
                                     training=training)
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).to(self.device)
        log_prob = dist.log_prob(z)
        return log_prob

    def compute_loss(self, obs, z, rnn_hxs, masks, output_mask=False, output_feat=False):
        log_prob = self.get_log_prob(obs, z, rnn_hxs, masks, output_mask, output_feat, training=True)
        loss = -torch.mean(log_prob)
        return loss
