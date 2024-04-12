import torch
import torch.nn as nn
from onpolicy.algorithms.utils.util import init, check, calculate_conv_params
from onpolicy.algorithms.utils.cnn import CNNBase, Encoder
from onpolicy.algorithms.utils.modularity import RIM, SCOFF
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space
import numpy as np

from absl import logging


#NEW SAF

from perceiver.model.core import InputAdapter
import math
from einops import rearrange
from perceiver.model.core import PerceiverEncoder, CrossAttention
from torch.distributions.categorical import Categorical
from onpolicy.algorithms.utils.SAF.mlp import MLP





#PARAMETERS TO BE INITIALIZED
#FOR ACTOR:
#self.input_shape , ok
#self.hidden_dim, ok
#self.n_layers, ok
#self.action_shape, ok
#self.activation, ok

#FOR CRITIC
#self.input_critic
#self.hidden_dim
#self.n_layers
#self.activation
#self.n_agents


class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(R_Actor, self).__init__()
        
        #new parameters
        self.drop_out= args.drop_out
        self.rnn_attention_module= args.rnn_attention_module
        self.use_bidirectional= args.use_bidirectional
        
        print("new parameters", self.drop_out, self.rnn_attention_module, self.use_bidirectional)
        
        #SAF
        self.hidden_size = args.hidden_size
        self.hidden_dim = args.hidden_size
        self.n_layers= args.layer_N
        self.action_shape= action_space.shape
        self.activation='tanh'

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._recurrent_N = args.recurrent_N
        self._use_version_scoff = args.use_version_scoff
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        obs_shape = get_shape_from_obs_space(obs_space)
        
        #print("agent vision space: ", obs_space.shape)
        ##Zahra added
        self.use_attention = args.use_attention
        #self.use_attention = args.use_attention
        self._attention_module = args.attention_module
        #print(f"value of use attention is {self.use_attention} ")

        self._obs_shape = obs_shape
        #print(f"actor network observation shape {obs_shape} {len(self._obs_shape)}")
        
        if self.use_attention==True:
            self._use_naive_recurrent_policy = False
            self._use_recurrent_policy = False
        else:
            self._use_naive_recurrent_policy = True
            self._use_recurrent_policy = True



        #ENCODER
        
        #print(f"we are using both CNN and attention module.... {obs_shape} {len(self._obs_shape)}")
        if obs_shape[0]==3:
            input_channel = obs_shape[0]
            input_width = obs_shape[1]
            input_height = obs_shape[2]
        elif obs_shape[2]==3:
            input_channel = obs_shape[2]
            input_width = obs_shape[0]
            input_height = obs_shape[1]     

        self.input_shape= obs_shape

        #making parametrs of encoder for CNN compatible with different image sizes
        #print(f"input channel and input image width in actor network c: {input_channel}, w: {input_width}, h: {input_height}")
        kernel, stride, padding = calculate_conv_params((input_width,input_height,input_channel))
        self.base = Encoder(input_channel, input_height, input_width, self.hidden_size, device, max_filters=256, num_layers=3, kernel_size= kernel, stride_size=stride, padding_size=padding)

        if self._attention_module== "SAF":
            logging.info('Using attention module %s: input width: %d', self._attention_module, obs_shape[1]) 
            self.rnn =  MLP(
                    np.array(self.input_shape).prod(), 
                    [self.hidden_dim]*self.n_layers, 
                    np.array(self.action_shape).prod(), 
                    std=0.01,
                    activation=self.activation)
            self.SAF = Communication_and_policy(input_dim=np.array(self.input_shape).prod(),
                                    key_dim=np.array(self.input_shape).prod(),
                                    N_SK_slots=self.N_SK_slots,
                                    n_agents=self.n_agents, n_policy=self.n_policy,
                                    hidden_dim=self.hidden_dim, n_layers=self.n_layers,
                                    activation=self.activation, latent_kl=self.latent_kl,
                                    latent_dim=self.latent_dim)
            
        
        else:

            if self.use_attention and len(self._obs_shape) >= 3:
            #print(f"value of use attention is {self.use_attention} ")
                logging.info('Using attention module %s: input width: %d', self._attention_module, obs_shape[1]) 
            
            if self._attention_module == "RIM": 
                    print("We are using RIM...")
                    self.rnn =  RIM(device, self.hidden_size, self.hidden_size, args.rim_num_units, args.rim_topk, rnn_cell = self.rnn_attention_module , n_layers = 1, bidirectional = self.use_bidirectional ,dropout=self.drop_out , batch_first = True)
            elif self._attention_module == "SCOFF":
                    print("We are using SCOFF...")
                    self.rnn =  SCOFF(device, self.hidden_size, self.hidden_size, args.scoff_num_units, args.scoff_topk, num_templates = 2,  rnn_cell = self.rnn_attention_module , n_layers = 1, bidirectional =self.use_bidirectional ,dropout=self.drop_out, version= self._use_version_scoff)
                                                
            elif not self.use_attention:
                #print(f"value of use attention is {self.use_attention} ")
                #base = CNNBase if len(obs_shape) >= 3 else MLPBase
                
                logging.info("observation space %s number of dimensions of observation space is %d", str(obs_space.shape), len(obs_shape))
                if len(obs_shape) == 3: 
                    logging.info('Not using any attention module, input width: %d ', obs_shape[1]) 
                #self.base = base(args, obs_shape)
                
                
                if self._use_naive_recurrent_policy or self._use_recurrent_policy:
                    print("We are using LSTM...") 
                    self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            
            
        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)
        self.algo = args.algorithm_name

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        
        #SAF
        KL=0
        bs = obs.shape[0]
        n_ags = obs.shape[1]
        actor_features = self.base(obs)

        if self._attention_module== "SAF":
            x_saf=self.SAF(obs)
            state = x_saf.reshape(bs, n_ags * self.input_shape[0])
            state = state.unsqueeze(1).repeat(1, n_ags, 1)
            actor_features , rnn_states=self.rnn(actor_features, rnn_states)
            probs = Categorical(logits=actor_features)
            actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
            action_log_probs=probs.log_prob(actions)

            out_actions = torch.stack(actions, dim=1)
            logprobs = torch.stack(action_log_probs, dim=1)
            out_actions = torch.cat(out_actions, 2)
            logprobs = torch.cat(logprobs, dim=2)

            attention_score = self.SAF.Run_policy_attention(obs)  # (bsz,n_agents,n_policy)
            actions = torch.einsum('bij,bij->bi', out_actions.float(), attention_score).long()  # bsz x N_agents
            action_log_probs = torch.einsum('bij,bij->bi', logprobs, attention_score)  # bsz x N_agents



        else:
            obs = check(obs).to(**self.tpdv)
            #print(f"agent vision cut : {obs.shape}")
            rnn_states = check(rnn_states).to(**self.tpdv)
            masks = check(masks).to(**self.tpdv)
            if available_actions is not None:
                available_actions = check(available_actions).to(**self.tpdv)
                
            if self.use_attention and len(self._obs_shape) >= 3:
                #print(f"actor features shape of shared observation before cnn.... {obs.shape} {masks.shape}")
                #print(f"actor features shape before rnn.... {actor_features.shape} {rnn_states.shape}")#torch.Size([9, 64]) torch.Size([9, 1, 64])
                actor_features, rnn_states = self.rnn(actor_features, rnn_states)
                #print(f"actor features shape after normal RNN in an actor network (attention).... {actor_features[0].shape} {rnn_states[0].shape}")
                if self._attention_module == "RIM":
                    rnn_states= tuple( t.permute(1,0,2) for t in rnn_states )
            else:

                #print(f"actor features shape base CNN and RNN.... {actor_features.shape} {rnn_states.shape}")
                if self._use_naive_recurrent_policy or self._use_recurrent_policy:
                    actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
                    #print(f"actor features shape after normal RNN in an actor network (lstm).... {actor_features.shape} {rnn_states.shape}")
                    rnn_states =rnn_states.permute(1,0,2)
            actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
            
        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        #print(f"actor obs shape in evaluate actions before change{obs.shape}")
        obs = check(obs).to(**self.tpdv)
        #print(f"actor obs shape in evaluate actions after change{obs.shape}")
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

            #print("INSIDE EVALUAT EACTIONS -RNN")
        actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self.algo == "hatrpo":
            action_log_probs, dist_entropy ,action_mu, action_std, all_probs= self.act.evaluate_actions_trpo(actor_features,
                                                                    action, available_actions,
                                                                    active_masks=
                                                                    active_masks if self._use_policy_active_masks
                                                                    else None)

            #print("dist_entropy is: ", dist_entropy)
            return action_log_probs, dist_entropy, action_mu, action_std, all_probs
        else:    
            if self.use_attention:
                #print("INSIDE EVALUAT EACTIONS -ATTENTION")
                available_actions=available_actions.unsqueeze(0)
                action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)
            else:
                action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)

        return action_log_probs, dist_entropy


class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(R_Critic, self).__init__()
        
        #new parameters
        self.drop_out= args.drop_out
        self.rnn_attention_module= args.rnn_attention_module
        self.use_bidirectional= args.use_bidirectional
        
        
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        
        ## Zahra added
        self._use_version_scoff = args.use_version_scoff
        self.use_attention = args.use_attention
        #self.use_attention = args.use_attention
        self._attention_module = args.attention_module
        
        self._obs_shape = cent_obs_shape
        #print(f"critic network observation shape {cent_obs_shape} {len(self._obs_shape)}")
        if self.use_attention==True:
            self._use_naive_recurrent_policy = False
            self._use_recurrent_policy = False
        else:
            self._use_naive_recurrent_policy = True
            self._use_recurrent_policy = True

        #SAF
        self.hidden_dim = args.hidden_size
        self.n_layers= args.layer_N
        self.activation='tanh'
        self.n_agents=args.num_agents
        self.input_critic=np.array(self.cent_obs_shape).prod()

        #ENCODER

        if self._obs_shape[0]==3:
            input_channel = cent_obs_shape[0]
            input_width   = cent_obs_shape[1]
            input_height  = cent_obs_shape[2]
        elif self._obs_shape[2]==3:
            input_channel = cent_obs_shape[2]
            input_width = cent_obs_shape[0]
            input_height = cent_obs_shape[1]          
        #making parametrs of encoder for CNN compatible with different image sizes
        #print(f"input channel and input image width in critic network c:{input_channel}, w: {input_width}, h: {input_height} observation: {cent_obs_shape}")
        kernel, stride, padding = calculate_conv_params((input_width,input_height,input_channel))
        self.base = Encoder(input_channel, input_height, input_width, self.hidden_size, device, max_filters=256, num_layers=3, kernel_size= kernel, stride_size=stride, padding_size=padding)  
        

        if self._attention_module== "SAF":
            print("we are using SAF")
            self.rnn =  nn.ModuleList([MLP(self.input_critic,[self.hidden_dim]*self.n_layers,  1, std=1.0, activation=self.activation) for _ in range(self.n_agents)])
            self.SAF = Communication_and_policy(input_dim=np.array(self.input_shape).prod(),
                                    key_dim=np.array(self.input_shape).prod(),
                                    N_SK_slots=self.N_SK_slots,
                                    n_agents=self.n_agents, n_policy=self.n_policy,
                                    hidden_dim=self.hidden_dim, n_layers=self.n_layers,
                                    activation=self.activation, latent_kl=self.latent_kl,
                                    latent_dim=self.latent_dim)
            
        else: 
           
            if self.use_attention and len(self._obs_shape) >= 3:
                                
                if self._attention_module == "RIM": 
                    
                    self.rnn = RIM(device, self.hidden_size, self.hidden_size, args.rim_num_units, args.rim_topk,  rnn_cell = self.rnn_attention_module , n_layers = 1, bidirectional = self.use_bidirectional ,dropout=self.drop_out, batch_first = True)
                                                    
                elif self._attention_module == "SCOFF":

                    self.rnn = SCOFF(device,  self.hidden_size, self.hidden_size, args.scoff_num_units, args.scoff_topk, num_templates = 2,  rnn_cell = self.rnn_attention_module , n_layers = 1, bidirectional = self.use_bidirectional ,dropout=self.drop_out, version = self._use_version_scoff)
            
            elif not self.use_attention:

                if self._use_naive_recurrent_policy or self._use_recurrent_policy:
    
                    print("We are using LSTM...")
                    self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)


        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        #print(f"critic obs shape in forward {cent_obs.shape}")
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if self.use_attention and len(self._obs_shape) == 3:
            #print(f"critic features shape of shared observation before cnn.... {cent_obs.shape} {masks.shape}")
            critic_features = self.base(cent_obs)
            #print(f"critic features shape before rnn.... {critic_features.shape} {rnn_states.shape}")
            critic_features, rnn_states = self.rnn(critic_features, rnn_states)
            #print(f"critic features shape after rnn using attention.... {critic_features.shape} {rnn_states[0].shape}") # torch.Size([1, rollout,hidden_size]) torch.Size([1, rollout, hidden_size])
            if self._attention_module == "RIM":
               rnn_states = tuple( t.permute(1,0,2) for t in rnn_states )
        else:
           if self._attention_module== "SAF":
                critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
           else:
                critic_features = self.base(cent_obs)
                #print(f"critic features shape before rnn using (normal rnn).... {critic_features.shape} {rnn_states.shape}")
                if self._use_naive_recurrent_policy or self._use_recurrent_policy:
                    critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
                    #print(f"critic features shape after rnn using (normal rnn).... {critic_features.shape} {rnn_states.shape}") #torch.Size([rollout,hidden_size]) torch.Size([rollout, 1, hidden_size])
                    critic_features = critic_features.unsqueeze(0)
                    rnn_states = rnn_states.permute(1,0,2)

        values = self.v_out(critic_features)
        
        return values, rnn_states




# Input adapater for perceiver
class agent_input_adapter(InputAdapter):
    def __init__(self, max_seq_len: int, num_input_channels: int):
        super().__init__(num_input_channels=num_input_channels)

        self.pos_encoding = nn.Parameter(
            torch.empty(max_seq_len, num_input_channels))
        self.scale = math.sqrt(num_input_channels)
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.pos_encoding.uniform_(-0.5, 0.5)

    def forward(self, x):
        b, l, dim = x.shape  # noqa: E741
        p_enc = rearrange(self.pos_encoding[:l], "... -> () ...")
        return x * self.scale + p_enc




class Communication_and_policy(nn.Module):
    def __init__(self, input_dim, key_dim, N_SK_slots, n_agents, n_policy, hidden_dim, n_layers, activation, latent_kl, latent_dim):
        super(Communication_and_policy, self).__init__()
        self.N_SK_slots = N_SK_slots

        self.n_agents = n_agents

        self.n_policy = n_policy
        self.latent_kl = latent_kl
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.key_dim = key_dim
        self.n_agents = n_agents
        self.n_policy = n_policy

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_keys = torch.nn.Parameter(
            torch.randn(self.n_policy, 1, key_dim)).to(self.device)
        self.policy_attn = nn.MultiheadAttention(
            embed_dim=key_dim, num_heads=1, batch_first=False)

        self.query_projector_s1 = MLP(input_dim,
                                      [self.hidden_dim]*self.n_layers,
                                      key_dim,
                                      std=1.0,
                                      activation=self.activation)  # for sending out message to sk

        self.original_state_projector = MLP(input_dim,
                                            [self.hidden_dim]*self.n_layers,
                                            key_dim,
                                            std=1.0,
                                            activation=self.activation)  # original agent's own state
        self.policy_query_projector = MLP(input_dim,
                                          [self.hidden_dim]*self.n_layers,
                                          key_dim,
                                          std=1.0,
                                          activation=self.activation)  # for query-key attention pick policy form pool

        self.combined_state_projector = MLP(2*key_dim,
                                            [self.hidden_dim]*self.n_layers,
                                            key_dim,
                                            std=1.0,
                                            activation=self.activation).to(self.device)  # responsible for independence of the agent
        # shared knowledge(workspace)

        input_adapter = agent_input_adapter(num_input_channels=key_dim, max_seq_len=n_agents).to(
            self.device)  # position encoding included as well, so we know which agent is which

        self.PerceiverEncoder = PerceiverEncoder(
            input_adapter=input_adapter,
            num_latents=N_SK_slots,  # N
            num_latent_channels=key_dim,  # D
            num_cross_attention_qk_channels=key_dim,  # C
            num_cross_attention_heads=1,
            num_self_attention_heads=1,  # small because observational space is small
            num_self_attention_layers_per_block=self.n_layers,
            num_self_attention_blocks=self.n_layers,
            dropout=0.0,
        ).to(self.device)
        self.SK_attention_read = CrossAttention(
            num_heads=1,
            num_q_input_channels=key_dim,
            num_kv_input_channels=key_dim,
            num_qk_channels=key_dim,
            num_v_channels=key_dim,
            dropout=0.0,
        ).to(self.device)

        if self.latent_kl:
            self.encoder = nn.Sequential(
                nn.Linear(int(2*key_dim), 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, int(2*self.latent_dim)),
            ).to(self.device)

            self.encoder_prior = nn.Sequential(
                nn.Linear(key_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, int(2*self.latent_dim)),
            ).to(self.device)

        self.previous_state = torch.randn(5, 1, key_dim).to(self.device)

    def forward(self, state):
        # sate has shape (bsz,N_agents,embsz)
        # communicate among agents using perceiver
        state = state.to(self.device).permute(1, 0, 2)
        N_agents, bsz, Embsz = state.shape
        state = state.permute(1, 0, 2)
        # message (bsz,N_agent,dim), for communication
        message_to_send = self.query_projector_s1(state)
        state_encoded = self.original_state_projector(
            state)  # state_encoded, for agent's internal uses

        # use perceiver arttecture to collect information from all agents by attention

        SK_slots = self.PerceiverEncoder(message_to_send)
        message = self.SK_attention_read(message_to_send, SK_slots)

        # message plus original state
        # shape (bsz,N_agents,2*dim)
        state_with_message = torch.cat([state_encoded, message], 2)

        state_with_message = state_with_message.permute(
            1, 0, 2)  # (N_agents,bsz,2*dim)

        state_with_message = self.combined_state_projector(
            state_with_message)  # (N_agents,bsz,dim)

        state_with_message = state_with_message.permute(
            1, 0, 2)  # (bsz,N_agents,dim)

        # print(state_with_message.shape)
        return state_with_message

    def forward_NoCommunication(self, state):
        # jsut encoder the original state without communication
        state = state.to(self.device)
        N_agents, bsz, Embsz = state.shape
        state = state.permute(1, 0, 2)
        state_encoded = self.original_state_projector(
            state)  # state_encoded, for agent's internal uses

        state_without_message = torch.cat([state_encoded, torch.zeros(
            state_encoded.shape).to(self.device)], 2)  # without information from other agents

        state_without_message = state_without_message.permute(
            1, 0, 2)  # (N_agents,bsz,2*dim)

        state_without_message = self.combined_state_projector(
            state_without_message)  # (N_agents,bsz,dim)

        return state_without_message

    def Run_policy_attention(self, state):
        '''
        state hasshape (bsz,N_agents,embsz)
        '''
        state = state.permute(1, 0, 2)  # (N_agents,bsz,embsz)
        state = state.to(self.device)
        # how to pick rules and if they are shared across agents
        query = self.policy_query_projector(state)
        N_agents, bsz, Embsz = query.shape

        keys = self.policy_keys.repeat(1, bsz, 1)  # n_ploicies,bsz,Embsz,

        _, attention_score = self.policy_attn(query, keys, keys)

        attention_score = nn.functional.gumbel_softmax(
            attention_score, tau=1, hard=True, dim=2)  # (Bz, N_agents , N_behavior)

        return attention_score

    def information_bottleneck(self, state_with_message, state_without_message, s_agent_previous_t):


        z_ = self.encoder(
            torch.cat((state_with_message, state_without_message), dim=2))
        mu, sigma = z_.chunk(2, dim=2)
        z = (mu + sigma * torch.randn_like(sigma)).reshape(z_.shape[0], -1)
        z_prior = self.encoder_prior(s_agent_previous_t)
        mu_prior, sigma_prior = z_prior.chunk(2, dim=2)
        KL = 0.5 * torch.sum(((mu - mu_prior) ** 2 + sigma ** 2)/(sigma_prior ** 2 + 1e-8) + torch.log(1e-8 + (sigma_prior ** 2)/(
            sigma ** 2 + 1e-8)) - 1) / np.prod(torch.cat((state_with_message, state_without_message), dim=2).shape)

        return z, KL
    