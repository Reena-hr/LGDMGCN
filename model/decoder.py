import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import  utils
from torch.nn import init


class DiffusionGraphConv(nn.Module):
    def __init__(self, supports_len, input_dim, hid_dim, num_nodes,
                 max_diffusion_step, output_dim, filter_type, bias_start=0.0):
        super(DiffusionGraphConv, self).__init__()
        self.num_matrices = supports_len*max_diffusion_step + 1  # add itself, i.e. x0
        input_size = input_dim + hid_dim
        self._num_nodes = num_nodes
        self.output_dim = output_dim
        self._max_diffusion_step = max_diffusion_step
        self.weight = nn.Parameter(torch.FloatTensor(size=(input_size*self.num_matrices, output_dim)))
        self.biases = nn.Parameter(torch.FloatTensor(size=(output_dim,)))
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.constant_(self.biases.data, val=bias_start)
        self.filter_type = filter_type

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def forward(self, supports, inputs, state, bias_start=0.0):
       
        batch_size = inputs.shape[0]
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.shape[2]

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, dim=0)
        x0_ori = x0

        if self._max_diffusion_step == 0:
            pass
        else:
           
            for support in supports:
                x1 = torch.sparse.mm(support, x0)
                x = self._concat(x, x1)
                for k in range(2, self._max_diffusion_step + 1):
                    x2 = torch.sparse.mm(support, x1)
                    x = self._concat(x, x2)
                    x1 = x2

        x = torch.reshape(x, shape=[self.num_matrices, self._num_nodes, input_size, batch_size])
        x = torch.transpose(x, dim0=0, dim1=3)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * self.num_matrices])

        x = torch.matmul(x, self.weight)  # (batch_size * self._num_nodes, output_size)
        x = torch.add(x, self.biases)
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes, self.output_dim])

class TemporalAttention(nn.Module):
    def __init__(self, input_dim, max_diffusion_step, num_nodes, dec_hid_dim, output_dim, filter_type):
        super().__init__()

        self.external_flag = False  
        self.decoding_cell = GRUCell(input_dim=input_dim + dec_hid_dim, num_units=dec_hid_dim,
                                       max_diffusion_step=max_diffusion_step,
                                       num_nodes=num_nodes, num_proj=output_dim, filter_type=filter_type)

    def forward(self, supports,  decoder_inputs,external_inputs,hidden_state, encoder_outputs, attention_type, output_size=64):

        encoder_outputs = encoder_outputs.transpose(1, 0)
        top_states = [e.contiguous().view(-1, 1, output_size) for e in encoder_outputs]
        attention_states = torch.cat(top_states, 1)
        batch_size = decoder_inputs[0].data.size(0)
        attn_length = attention_states.size(1)
        attn_size = attention_states.size(2)
        hidden = attention_states.view(-1, attn_size, attn_length, 1) 
        attention_vec_size = attn_size
        w_conv = nn.Conv2d(attn_size, attention_vec_size, (1, 1), (1, 1))
        w_conv.to('cuda:0')
        hidden_features = w_conv(hidden)
        node_num = int(hidden_features.size(0) / batch_size)      
        hidden_features = hidden_features.reshape(batch_size, node_num, attn_size, attn_length, 1)
        v = nn.Parameter(torch.FloatTensor(attention_vec_size))
        init.normal(v)
        v = v.to('cuda:0')

        def attention(query):
            # linear map
            y = utils.Linear(query, attention_vec_size, True)
            y = y.view(-1, node_num,  1, 1, attention_vec_size)            
            s = torch.sum(v * torch.tanh(hidden_features + y), dim=[2, 4])            
            a = F.softmax(s)
            d = torch.sum(a.view(-1, 1, attn_length, 1) * hidden, dim=[2, 3])

            return d.view(-1, node_num, attn_size)
      
        attn = nn.Parameter(torch.FloatTensor(batch_size,  node_num, attn_size))
        init.xavier_uniform(attn)
        attn = attn.to('cuda:0')
        i = 0
        outputs = []
        for (inp, ext_inp) in zip(decoder_inputs, external_inputs):
               
            input_size = inp.data.size(2)
            # print(i, input_size)
            
            if self.external_flag:                
                x = utils.Linear([inp.float()] + [ext_inp.float()] + [attn.float()], input_size, True)
            else:                
                x = utils.Linear([inp.float()] + [attn.float()], 65, True)
                
            cell_output, state = self.decoding_cell(supports,  x,  hidden_state, attention_type)
            # print(state.size())
            attn = attention([state])            
            output = utils.Linear([cell_output] + [attn], output_size, True)
            outputs.append(output)
            i += 1


        return outputs, state

class MultiplicativeAttention(nn.Module):
    def __init__(self, num_nodes, enc_hid_dim, dec_hid_dim, attn_dim):
        super().__init__()

        # Perform linear transform.
        self.W_q = nn.Linear(dec_hid_dim, attn_dim)
        self.W_k = nn.Linear(enc_hid_dim, attn_dim)
        # self.W_v = nn.Linear(enc_hid_dim, enc_hid_dim)
        # self.scale = torch.sqrt(torch.FloatTensor([num_nodes*attn_dim]))
        self.scale = math.sqrt(num_nodes*attn_dim)

    def forward(self, dec_hidden, encoder_outputs):
        
        q = self.W_q(dec_hidden)
        k = self.W_k(encoder_outputs)
        # v = self.W_v(encoder_outputs)
        # q = dec_hidden
        # k = encoder_outputs
        batch_size, time_step, num_nodes, _ = encoder_outputs.shape

        q = torch.reshape(q, [batch_size, -1]).unsqueeze(1)  # reshape to (64, 1, 207*32)
        k = v = torch.transpose(torch.reshape(k, [batch_size, time_step, -1]), 1, 2)
        v = torch.reshape(v, [batch_size, time_step, -1])
        
        energy = torch.matmul(q, k) / self.scale

        attention = F.softmax(energy, dim=-1)
        # attention = [batch_size, trg_len=1, src_len]
        weighted = torch.bmm(attention, v)
        # [batch_size, trg_len=1, src_len] * [batch_size, src_len, n_route*hidden_size]

        return attention, weighted


class GRUCell(nn.Module):
    
    def __init__(self, input_dim, num_units, max_diffusion_step, num_nodes,
                 num_proj=None, activation=torch.tanh, filter_type='dual_random_walk'):
        
        super(GRUCell, self).__init__()
        self._activation = activation
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._num_proj = num_proj
        self._supports = []
        if filter_type == "dual_random_walk":
            supports_len = 2
        elif filter_type == "identity":
            supports_len = 1
        else:
            raise ValueError("Unknown filter type...")

        self.dconv_gate = DiffusionGraphConv(supports_len=supports_len, input_dim=input_dim,
                                             hid_dim=num_units, num_nodes=num_nodes,
                                             max_diffusion_step=max_diffusion_step,
                                             output_dim=num_units*2, filter_type=filter_type)
        self.dconv_candidate = DiffusionGraphConv(supports_len=supports_len, input_dim=input_dim,
                                                  hid_dim=num_units, num_nodes=num_nodes,
                                                  max_diffusion_step=max_diffusion_step,
                                                  output_dim=num_units, filter_type=filter_type)
        if num_proj is not None:
            self.project1 = nn.Linear(self._num_units, self._num_units)
            self.project2 = nn.Linear(self._num_units, self._num_proj)
            
    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        if self._num_proj is not None:
            output_size = self._num_nodes * self._num_proj
        return output_size

    def forward(self, supports, inputs, state, attention_type):
        
        output_size = 2 * self._num_units
        # we start with bias 1.0 to not reset and not update
        value = torch.sigmoid(self.dconv_gate(supports, inputs, state, bias_start=1.0)) 
        r, u = torch.split(value, split_size_or_sections=int(output_size/2), dim=-1)
        c = self.dconv_candidate(supports, inputs, r * state)  # batch_size, self._num_nodes, output_size
        if self._activation is not None:
            c = self._activation(c)
        output = new_state = u * state + (1 - u) * c
        if self._num_proj is not None:
            # apply linear projection to state
            batch_size = inputs.shape[0]
            output = torch.reshape(new_state, shape=(-1, self._num_units))  # (batch*num_nodes, num_units)
            if attention_type == "multiplicative":
                output = torch.reshape(self.project2(torch.relu(self.project1(output))),
                                       shape=(batch_size, self.output_size))  # (64, 207*1)
            elif attention_type == "temporal":
                output = torch.reshape(torch.relu(self.project1(output)),
                                       shape=(batch_size, self.output_size, -1))
        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)


class Decoder(nn.Module):
    def __init__(self, input_dim, max_diffusion_step, num_nodes,
                 enc_hid_dim, dec_hid_dim, output_dim, attention_type, filter_type):
        super().__init__()
        self.dec_hid_dim = dec_hid_dim
        self.enc_hid_dim = enc_hid_dim
        self._num_nodes = num_nodes  # 207
        self._output_dim = output_dim  # should be 1
        self.attention_type = attention_type

        self.decoding_cell = GRUCell(input_dim=input_dim+dec_hid_dim, num_units=dec_hid_dim,
                                       max_diffusion_step=max_diffusion_step,
                                       num_nodes=num_nodes, num_proj=output_dim, filter_type=filter_type)

        if attention_type == "multiplicative":
            self.attention = MultiplicativeAttention(num_nodes=num_nodes, enc_hid_dim=enc_hid_dim,
                                                     dec_hid_dim=dec_hid_dim, attn_dim=dec_hid_dim)
            self.decoding_cell = GRUCell(input_dim=input_dim + dec_hid_dim, num_units=dec_hid_dim,
                                           max_diffusion_step=max_diffusion_step,
                                           num_nodes=num_nodes, num_proj=output_dim, filter_type=filter_type)
        elif attention_type == "temporal":
            self.attention = TemporalAttention(input_dim, max_diffusion_step, num_nodes, dec_hid_dim, output_dim, filter_type)
            self.w_out = nn.Parameter(torch.FloatTensor(self.dec_hid_dim,  self._output_dim))
            self.b_out = nn.Parameter(torch.FloatTensor(self._output_dim))
            self.decoder_cell = nn.LSTMCell(self._output_dim, 64, bias=True)

        elif attention_type == "no_attention":
            self.attention = None
            self.decoding_cell = GRUCell(input_dim=input_dim, num_units=dec_hid_dim,
                                           max_diffusion_step=max_diffusion_step,
                                           num_nodes=num_nodes, num_proj=output_dim, filter_type=filter_type)
        else:
            raise ValueError("Unknown attention type")

    def forward(self, inputs, external_inputs, attention_states,external_flag, output_size=64):
        decoder_inputs = inputs[:-1]
        # Needed for reshaping.
        batch_size = decoder_inputs[0].data.size(0)
        attn_length = attention_states.data.size(1)
        n_node = attention_states.data.size(2)
        attn_size = attention_states.data.size(3)

        # Calculate W_d * h_o by a 1-by-1 convolution        
        hidden = attention_states.view(-1, attn_size, attn_length, 1)  
        # Size of query vectors for attention.
        attention_vec_size = attn_size
        w_conv = nn.Conv2d(attn_size, attention_vec_size, (1, 1), (1, 1))
        w_conv = w_conv.to('cuda')
        hidden_features = w_conv(hidden)       
        v = nn.Parameter(torch.FloatTensor(attention_vec_size))
        init.normal(v)
        v = v.to('cuda:0')

        def attention(query):
            # linear map
            y = utils.Linear(query, attention_vec_size, True)
            y = y.view(-1, 1, 1, attention_vec_size)
            # Attention mask is a softmax of v_d^{\top} * tanh(...).
            s = torch.sum(v * torch.tanh(hidden_features + y), dim=[1, 3])
            # Now calculate the attention-weighted vector
            a = F.softmax(s)           
            d = torch.sum(a.view(-1, 1, attn_length, 1) * hidden, dim=[2, 3])

            return d.view(-1, attn_size)

        # attn = Variable(torch.zeros(batch_size, attn_size))
        attn = nn.Parameter(torch.FloatTensor(batch_size, n_node,attn_size))
        init.xavier_uniform(attn)
        attn = attn.to('cuda:0')

        i = 0
        outputs = []
        prev = None

        for (inp, ext_inp) in zip(decoder_inputs, external_inputs):        
            # Merge input and previous attentions into one vector of the right size
            input_size = inp.data.size(2)
            # print(i, input_size)       
            
            if external_flag:
                # print(inp.data.size(1),ext_inp.data.size(1),attn.data.size(1))
                x = utils.Linear([inp.float()] + [ext_inp.float()] + [attn.float()], input_size, True)
            else:
                x =utils. Linear([inp.float()] + [attn.float()], input_size, True)
            # Run the RNN.
            # print(x.size())
            x = x.view(-1,1)
            cell_output, state = self.decoder_cell(x)
            cell_output = cell_output.view(batch_size, n_node, -1)
            state = state.view(batch_size, n_node, -1)
            # Run the attention mechanism.
            # print(state.size())
            attn = attention([state])
            attn = attn.view(batch_size, n_node,-1)

            # Attention output projection
            # print(cell_output.size(), attn.size())
            output = utils.Linear([cell_output] + [attn], output_size, True)
            outputs.append(output)
            i += 1

            preds = [torch.matmul(i, self.w_out) + self.b_out for i in outputs]

        return outputs, preds
    

class GRUDecoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_nodes):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.decoding_cell = nn.GRUCell(input_size=input_dim+hidden_dim, hidden_size=hidden_dim, bias=True)
        self.proj1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.proj2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.attention = MultiplicativeAttention(num_nodes=num_nodes, enc_hid_dim=hidden_dim,
                                                 dec_hid_dim=hidden_dim, attn_dim=hidden_dim)

    def forward(self, inputs, initial_hidden_state, encoder_outputs, teacher_forcing_ratio=0.5):
       
        _, enc_seq_length, _, _ = encoder_outputs.shape
        seq_length, batch_size, num_nodes, _ = inputs.shape

        # tensor to store decoder outputs
        outputs = torch.zeros(seq_length, batch_size, num_nodes * self.output_dim) 

        # Alleviate information compression
        initial_hidden_state = torch.reshape(initial_hidden_state, (batch_size * num_nodes, -1))

        current_input = inputs[0]  # start
        hidden_state = initial_hidden_state  # [batch_size, num_nodes, hidden_dim]
        for t in range(1, seq_length):
            # GRUCell input should be (batch_size, input_size), hidden should be (batch_size, hidden_size)
            current_input = torch.reshape(current_input, (batch_size * num_nodes, -1))
            a, weighted = self.attention(hidden_state.reshape((batch_size, num_nodes, -1)), encoder_outputs)            
            weighted = torch.reshape(weighted.squeeze(1), [batch_size*num_nodes, -1])
            rnn_input = torch.cat((current_input, weighted), dim=-1)
            hidden_state = self.decoding_cell(rnn_input, hidden_state)  
            output = self.proj2(torch.relu(self.proj1(hidden_state)))  # [batch_size*num_nodes, output_dim]
            outputs[t] = torch.reshape(output.squeeze(), (batch_size, num_nodes*self.output_dim))
            teacher_force = random.random() < teacher_forcing_ratio  # a bool value
            current_input = (inputs[t] if teacher_force else output)
        return outputs[1:, ...]
