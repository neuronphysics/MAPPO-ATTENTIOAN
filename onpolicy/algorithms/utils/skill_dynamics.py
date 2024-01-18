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
TensorType = Any




def stickbreak(v):
    """Stick break function in PyTorch"""
    batch_ndims = len(v.shape) - 1
    cumprod_one_minus_v = torch.exp(torch.log1p(-v).cumsum(-1))
    one_v = nn.functional.pad(v,  pad=[0, 1]*batch_ndims + [0, 1], value=1)
    c_one = nn.functional.pad(cumprod_one_minus_v, pad=[0, 0]*batch_ndims + [1, 0], value=1)
    return one_v * c_one

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
                 activation_fn: Any = None,
                 use_bias: bool = True,
                 bias_init: float = 0.0,
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
        # Actual nn.Linear layer (including correct initialization logic).
        linear = nn.Linear(in_size, out_size, bias=use_bias)
        if initializer:
            initializer(linear.weight)
        if use_bias is True:
            nn.init.constant_(linear.bias, bias_init)
        if apply_spectral_norm:
            linear = nn.utils.spectral_norm(linear)

        layers.append(linear)
        # Activation function (if any; default=None (linear)).
        if isinstance(activation_fn, str):
            activation_fn = get_activation_fn(activation_fn, "torch")
        if activation_fn is not None:
            layers.append(activation_fn())
        # Put everything in sequence.
        self._model = nn.Sequential(*layers)

    def forward(self, x: TensorType) -> TensorType:
        return self._model(x)

class SkillDynamics(nn.Module):
    """The default skill dynamics model for DADS"""

    def __init__(self, 
                 args, 
                 obs_shape, 
                 hidden_dim,
                 num_hiddens = 2, 
                 z_range = (-1,1), 
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') ,
                 z_type = 'discrete', 
                 prior_samples = 100, 
                 dynamics_reg_hiddens = False, 
                 dynamics_orth_reg = True,
                 dynamics_l2_reg = False,
                 dynamics_spectral_norm = False,#dynamics spectral norm
                 variance = 1):
        super().__init__()

        
        
        self.z_dim = args.skill_dim
        self.max_num_experts = args.skill_max_num_experts
        num_hiddens = num_hiddens
        self.num_reps = prior_samples
        self.z_range = z_range
        self.z_type = z_type
        self.z_dist = D.Uniform(z_range[0],z_range[1])
        self.variance = variance
        self._dynamics_reg_hiddens = dynamics_reg_hiddens
        self._dynamics_orth_reg = dynamics_orth_reg
        self._dynamics_spectral_norm = dynamics_spectral_norm
        self._dynamics_l2_reg = dynamics_l2_reg
        self.device = device
        self.flat_tool = nn.Flatten()

        # Assuming obs_shape is something like (batch_size, dim1, dim2, ...)
        self.obs_dim = reduce(lambda x, y: x * y, obs_shape[0:])
        
        input_dim = self.obs_dim + self.z_dim
        print(f"size of observational skill dynamics input {self.obs_dim}")
        print(f'dynamics input dim: {input_dim}')

        self.bn_in = nn.BatchNorm1d(self.obs_dim)
        self.bn_target = nn.BatchNorm1d(self.obs_dim, affine=False)
        activation_fn = nn.ELU

        hiddens = [SlimFC(input_dim, hidden_dim, activation_fn=activation_fn,
                          initializer=lambda w: nn.init.xavier_uniform_(w,1.0),
                          apply_spectral_norm=self._dynamics_spectral_norm),
                    nn.LayerNorm(hidden_dim)]

        for _ in range(num_hiddens-1):
            hiddens.append(SlimFC(hidden_dim,hidden_dim,activation_fn=activation_fn,
                                  initializer=lambda w: nn.init.xavier_uniform_(w,1.0),
                                  apply_spectral_norm=self._dynamics_spectral_norm))
        self.hiddens = nn.Sequential(*hiddens)

        self.logits = SlimFC(hidden_dim + self.z_dim, self.max_num_experts,
                            initializer=lambda w: nn.init.xavier_uniform_(w,1.0),
                            apply_spectral_norm=self._dynamics_spectral_norm) # nn.Linear(hidden_dim, self.num_experts)
        
        self.means = SlimFC(hidden_dim + self.z_dim, self.max_num_experts * self.obs_dim,
                            initializer=lambda w: nn.init.xavier_uniform_(w,1.0),
                            apply_spectral_norm=self._dynamics_spectral_norm) # nn.Linear(hidden_dim, self.num_experts * self.obs_dim)
        # print(self.hiddens._modules)
        # print(self.hiddens._modules['0']._model._modules['0'])

        self.to(self.device)
        
        self._dynamics_lr = args.dynamics_lr  # Assuming dynamics learning rate is passed through 'args'
        self.dynamics_opt = torch.optim.Adam(self.parameters(), lr = self._dynamics_lr, eps=args.opti_eps, weight_decay=args.weight_decay)

    def orthogonal_regularization(self):
        reg = 1e-4
        orth_loss = torch.zeros(1)
        layers = [self.logits,self.means]
        if self._dynamics_reg_hiddens:
            layers.append(self.hiddens)
        for layer in layers:
            for name, param in layer.named_parameters():
                if 'bias' not in name:
                    param_flat = param.view(param.shape[0], -1)
                    sym = torch.mm(param_flat, torch.t(param_flat))
                    sym -= torch.eye(param_flat.shape[0])
                    orth_loss = orth_loss + (reg * sym.abs().sum())
        return torch.sum(orth_loss)

    def l2_regularization(self):
        reg = 1e-4
        l2_loss = torch.zeros(1)
        for name, param in self.hiddens.named_parameters():
            if 'bias' not in name:
                l2_loss = l2_loss + (0.5 * reg * torch.sum(torch.pow(param, 2)))
        return torch.sum(l2_loss)


    def forward(self, obs, z, training=False):
        
        # obs = batch_norm(obs)
        # Debugging: Print shapes to verify dimensions
        print(f"Shape of obs in skill Dynamics: {obs.shape}")  # The shape of the incoming observations
        print("Expected input dimension for bn_in:", self.obs_dim)  # Expected dimension
        
        obs = self.flat_tool(obs)
        print(f"skill size {self.z_dim} new shape of obs {obs.shape}")
        self.bn_in.train(mode=training)
        norm_obs = self.bn_in(obs)
        z = torch.tensor(z, dtype=torch.float32, device=self.device)
        inp = torch.cat([ norm_obs, z ], dim=-1)
        x = self.hiddens(inp)
        
        #https://luiarthur.github.io/TuringBnpBenchmarks/dpsbgmm
        eta = self.logits(torch.cat([x, z], dim=-1))# [batch,num_experts]
        means = self.means(torch.cat([x,z], dim=-1))
        means = means.reshape(obs.shape[0], self.max_num_experts, self.obs_dim)
        return eta, means

    def get_distribution(self, obs, z, training=False):
        print(f"latent in skill dynamics :{z} and observation {obs}")
        eta, means = self.forward(obs,z,training)
        eta = torch.softmax(eta, dim=-1) # Apply softmax to get probabilities
        diags = torch.ones_like(means) * self.variance # (num_components, obs_size)
        mix = D.Categorical(eta) 
        comp = D.Independent(D.Normal(means, diags),1)
        gmm = D.MixtureSameFamily(mix, comp)
        print(f"mixture skills {gmm}")
        return gmm
    
    
    def get_log_prob(self, obs, z, next_obs, training=False):
        gmm = self.get_distribution(obs,z,training)
        self.bn_target.train(mode=training)
        norm_next_obs = self.bn_target(next_obs)
        return gmm.log_prob(norm_next_obs)
    

    def _calculate_intrinsic_rewards(self, obs, z, next_obs, weights=None):
        num_reps = self.num_reps if self.z_type=='cont' else self.z_dim
        if self.z_type=='cont':
            alt_obs = obs.repeat(num_reps,1) # [Batch_size*num_reps, obs_dim]
            alt_next_obs = next_obs.repeat(num_reps,1)
            # continuous uniform
            alt_skill = np.random.uniform(self.z_range[0],self.z_range[1],size=[alt_obs.shape[0],self.z_dim]).astype(np.float32)
        elif self.z_type=='discrete':
            alt_obs = obs.repeat(self.z_dim,1)
            alt_next_obs = next_obs.repeat(self.z_dim,1)
            alt_skill = np.tile(np.eye(self.z_dim),[obs.shape[0],1]).astype(np.float32)
        #implement https://github.com/google-research/dads/blob/abc37f532c26658e41ae309b646e8963bd7a8676/unsupervised_skill_learning/skill_discriminator.py#L108C1-L114C60
        alt_skill = torch.from_numpy(alt_skill)

        log_prob = self.get_log_prob(obs,z,next_obs,training=False) # [Batch_size]
        log_prob = log_prob.reshape(obs.shape[0],1)
        alt_log_prob = self.get_log_prob(alt_obs,alt_skill,alt_next_obs,training=False) # [Batch_size*num_reps]
        # alt_log_prob = torch.cat(torch.split(alt_log_prob, num_reps,dim=0),dim=0) # [Batch_size, num_reps]
        alt_log_prob = alt_log_prob.reshape(obs.shape[0],num_reps) # [Batch_size, num_reps]
        # print(log_prob.shape)
        # print(alt_log_prob.shape)
        if weights is not None:
           diff =(alt_log_prob - log_prob)*weights
        else:
           diff = (alt_log_prob - log_prob)
           

        reward = np.log(num_reps+1) - np.log(1 + np.exp( torch.clamp(
                diff, -50, 50)).sum(axis=-1))
        # print(reward.shape)
        return reward, {'log_prob':log_prob,'alt_log_prob':alt_log_prob, 'num_higher_prob':((-diff)>=0).sum().item()}
    
    def compute_loss(self, next_obs, obs, z):
        next_dynamics_obs = next_obs - obs
        log_prob = self.get_log_prob(obs, z, next_dynamics_obs,training=True)
        dynamics_loss = -torch.mean(log_prob )
        orth_loss = self.orthogonal_regularization()
        l2_loss = self.l2_regularization()
        if self._dynamics_orth_reg:
            dynamics_loss += orth_loss
        if self._dynamics_l2_reg and not self._dynamics_spectral_norm:
            dynamics_loss += l2_loss
        return dynamics_loss
    


class SkillDiscriminator(nn.Module):
    """Skill Discriminator model in PyTorch, compatible with the provided SkillDynamics class"""

    def __init__(self, 
                 args, 
                 obs_shape, 
                 hidden_dim, 
                 skill_dim, 
                 num_hiddens=2,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 discriminator_spectral_norm=False):
        super(SkillDiscriminator, self).__init__()
        self.obs_dim = obs_shape[0]  # Assuming obs_shape is something like (dim1, dim2, ...)
        self.skill_dim = skill_dim
        self.device = device
        self.flat_tool = nn.Flatten()
        
        input_dim = self.obs_dim
        self.bn_in = nn.BatchNorm1d(self.obs_dim)
        activation_fn = nn.ELU

        hiddens = [SlimFC(input_dim, hidden_dim, activation_fn=activation_fn,
                          initializer=lambda w: nn.init.xavier_uniform_(w,1.0),
                          apply_spectral_norm=discriminator_spectral_norm),
                   nn.LayerNorm(hidden_dim)]

        for _ in range(num_hiddens-1):
            hiddens.append(SlimFC(hidden_dim,hidden_dim,activation_fn=activation_fn,
                                  initializer=lambda w: nn.init.xavier_uniform_(w,1.0),
                                  apply_spectral_norm=discriminator_spectral_norm))
        self.hiddens = nn.Sequential(*hiddens)

        self.logits = SlimFC(hidden_dim, skill_dim,
                             initializer=lambda w: nn.init.xavier_uniform_(w,1.0),
                             apply_spectral_norm=discriminator_spectral_norm)
        
        self.to(self.device)
        
        self._discriminator_lr = args.skill_discriminator_lr  # Assuming discriminator learning rate is passed through 'args'
        self.discriminator_opt = torch.optim.Adam(self.parameters(), lr=self._discriminator_lr, eps=args.opti_eps, weight_decay=args.weight_decay)

    def forward(self, obs, training=False):
        obs = self.flat_tool(obs)
        self.bn_in.train(mode=training)
        norm_obs = self.bn_in(obs)
        x = self.hiddens(norm_obs)
        logits = self.logits(x)
        return logits

    def get_distribution(self, obs, training=False):
        logits = self.forward(obs, training)
        probs = F.softmax(logits, dim=-1)
        dist = D.Categorical(probs)
        return dist

    def get_log_prob(self, obs, z, training=False):
        dist = self.get_distribution(obs, training)
        log_prob = dist.log_prob(z)
        return log_prob

    def compute_loss(self, obs, z):
        log_prob = self.get_log_prob(obs, z, training=True)
        loss = -torch.mean(log_prob)
        return loss
