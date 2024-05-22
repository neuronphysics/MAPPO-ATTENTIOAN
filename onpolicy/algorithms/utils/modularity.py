import torch.nn as nn
import torch
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from SCOFF.rnn_models_scoff import RNNModel as RNNModelScoff


class Identity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input * 1.0

    def backward(ctx, grad_output):
        return grad_output * 1.0


class SCOFF(nn.Module):
    def __init__(self,
                 device,
                 input_size,
                 hidden_size,
                 num_units,
                 k,
                 args
                 ):
        super().__init__()
        """
        - Wrappper for SCOFF.
        - Mirrors nn.LSTM or nn.GRU
        - supports bidirection and multiple layers
        - Option to specify num_rules and rule_time_steps.

        Parameters:
            device: 'cuda' or 'cpu'
            input_size
            hidden_size
            num_units: Number of RIMs
            k: topk
            rnn_cell: 'LSTM' or 'GRU' (default = LSTM)
            n_layers: num layers (default = 1)
            bidirectional: True or False (default = False)
            num_rules: number of rules (default = 0)
            rule_time_steps: Number of times to apply rules per time step (default = 0)
        """
        if input_size % num_units != 0:
            print('ERROR: input_size should be evenly divisible by num_units')
            exit()
        if device == 'cuda':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.n_layers = 1
        self.num_templates = args.scoff_num_schemas
        self.rnn_cell = args.rnn_attention_module
        self.num_units = num_units
        self.hidden_size = hidden_size // num_units
        self.batch_first = False
        self.do_rel = args.scoff_do_relational_memory
        self.version = args.use_version_scoff
        self.drop_out = args.drop_out
        self.attention_out = 85

        self.scoff_cell = RNNModelScoff(self.rnn_cell, input_size, hidden_size, hidden_size, nlayers=1,
                                        n_templates=self.num_templates, tie_weights=False, num_blocks=num_units,
                                        update_topk=k, dropout=self.drop_out,
                                        attention_out=self.attention_out,
                                        use_cudnn_version=False, use_adaptive_softmax=False, discrete_input=False,
                                        use_gru=self.rnn_cell == 'GRU', version=self.version,
                                        do_rel=self.do_rel, args=args).to(self.device)

    def forward(self, x, h, masks=None):
        """
        Input: x (seq_len, batch_size, feature_size
               hidden (num_layers * num_directions, batch_size, hidden_size * num_units)
        Output: outputs (batch_size, seqlen, hidden_size * num_units * num-directions)
                h(and c) (num_layer * num_directions, batch_size, hidden_size* num_units)
        """

        batch_num = h.size(0)
        if x.size(0) == h.size(0):
            ep_len = 1
            x = x.unsqueeze(0)
        else:
            # x is a (episode_len, batch_num, -1) tensor that has been flattened to (episode_len * batch_num, -1)
            ep_len = int(x.size(0) / batch_num)
            x = x.view(ep_len, batch_num, x.size(1))

        # Same deal with masks
        masks = masks.view(ep_len, batch_num)

        h = h.transpose(0, 1)

        x_fw, hs, _, _, _ = self.scoff_cell(x, h.unsqueeze(0), masks=masks)

        # x should be (seq, batch, input_dim)
        x_fw = x_fw.reshape(ep_len * batch_num, self.num_units * self.hidden_size)
        return x_fw, hs
