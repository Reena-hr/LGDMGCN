import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EncGraphConv(nn.Module):
    def __init__(self, supports_len, batch_size, num_nodes, his_len, d_in,
                 max_diffusion_step, d_out, filter_type, bias_start=0.0, adpt_type="pam"):
        super().__init__()
        if adpt_type == "no_adpt":
            self.num_matrices = supports_len * max_diffusion_step + 1
        else: 
            self.num_matrices = supports_len*max_diffusion_step + 2
        self.batch_size = batch_size
        self.his_len = his_len
        self.batch_size_t = batch_size * his_len
        self.d_in = d_in
        self.d_out = d_out
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self.weight = nn.Parameter(torch.FloatTensor(size=(d_in*self.num_matrices, d_out)))
        self.biases = nn.Parameter(torch.FloatTensor(size=(d_out,)))
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.constant_(self.biases.data, val=bias_start)
        self.filter_type = filter_type
        self.adpt_type = adpt_type

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def forward(self, supports, x, adpt_adj=None):
        """
        :param supports:
        :param x: tensor, [batch_size * time_step * d_in, num_nodes]. treat batch_size*time_step as batch_size
        :param x: tensor, [batch_size, time_step, n_route, d_in]
        :return: tensor, [batch_size, num_nodes, d_out].
        """
        
        if self.adpt_type == "random_embedding":
            # adpt_adj (num_nodes, num_nodes)
            x_add = torch.transpose(x, dim0=0, dim1=1)  # [num_nodes, batch_size * time_step * d_in]
            x_add = torch.matmul(adpt_adj, x_add)
        elif self.adpt_type == "dynamic":
            x_add = torch.reshape(x, [self.batch_size, self.his_len, self.d_in, self._num_nodes]).permute(0, 3, 1, 2)
            x_add = torch.reshape(x_add, [self.batch_size, self._num_nodes, -1])
            if x_add.dtype != adpt_adj.dtype:
                # x_add = x_add.double()
                adpt_adj = adpt_adj.float()

            if  adpt_adj.shape[1] != x_add.shape[1]: 
                aa = x_add.shape[1]
                adpt_adj = adpt_adj[:, :aa, :aa]
            x_add = torch.reshape(torch.bmm(adpt_adj, x_add), [self.batch_size, self._num_nodes, self.his_len, -1])           
            x_add = torch.reshape(torch.transpose(x_add, 0, 1),
                                  [self._num_nodes, self.batch_size * self.his_len * self.d_in])

        else:
                x_add = None

        x0 = torch.transpose(x, dim0=0, dim1=1)  # [num_nodes, batch_size * time_step * d_in]
        x = torch.unsqueeze(x0, dim=0)
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
        if x_add is not None:       
            x = self._concat(x, x_add)
            # x_add= torch.unsqueeze(x_add, 0)
            # x = torch.cat([x, x_add], dim=0)

        x = torch.reshape(x, shape=[self.num_matrices, self._num_nodes, self.batch_size_t, self.d_in])
        x = torch.transpose(x, dim0=0, dim1=2)  # (batch_size, num_nodes, order, input_dim)
        x = torch.reshape(x, shape=[self.batch_size_t*self._num_nodes, self.num_matrices*self.d_in])        
        x = torch.matmul(x, self.weight)  # (batch_size * self._num_nodes, d_out)
        x = torch.add(x, self.biases)
        return torch.reshape(x, [self.batch_size_t, self._num_nodes, self.d_out])


class TemporalConvLayer(nn.Module):
    def __init__(self, kt, d_in, d_out):
       
        super().__init__()
        self.kt = kt
        self.d_in = d_in
        self.d_out = d_out
        if d_in > d_out:
            self.input_conv = nn.Conv2d(d_in, d_out, (1, 1))
        self.conv1 = nn.Conv2d(d_in, 2*d_out, (kt, 1), padding=(int((kt-1)/2), 0))
        # kernel size: [kernel_height, kernel_width]
        self.scale = math.sqrt(0.5)

    def forward(self, x):        
              
        x = x.permute(0, 3, 1, 2).contiguous()  # [batch_size, d_in, P, num_nodes]
        _, _, P, n = x.shape
        if self.d_in > self.d_out:
            x_input = self.input_conv(x)
        elif self.d_in < self.d_out:            
            x_input = torch.cat([x, torch.zeros([x.shape[0], self.d_out-self.d_in, P, n], device=x.device)], dim=1)
        else:
            x_input = x

        # gated liner unit
        x_conv = self.conv1(x)
        out = (F.glu(x_conv, dim=1) + x_input) * self.scale
        out = out.permute(0, 2, 3, 1)
        return out


class SpatialConvLayer(nn.Module):
    def __init__(self, batch_size, num_nodes, d_in, d_out, max_diffusion_step,
                 filter_type, adpt_type="pam"):
        
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        if d_in > d_out:
            self.input_conv = nn.Conv2d(d_in, d_out, (1, 1))
        if filter_type == "dual_random_walk":
            supports_len = 2
        elif filter_type == "identity":
            supports_len = 1
        else:
            raise ValueError("unknown filter type")
        self.spatial_conv = EncGraphConv(supports_len=supports_len, batch_size=batch_size, num_nodes=num_nodes,
                                         his_len=12, d_in=d_in, max_diffusion_step=max_diffusion_step,
                                         d_out=d_out, filter_type=filter_type, bias_start=0.0, adpt_type=adpt_type)

    def forward(self, supports, x, adpt_adj=None):
       
        batch_size, seq_len, n, _ = x.shape

        x = x.permute(0, 3, 1, 2)  # [batch_size, d_in, seq_len, n_route]
        if self.d_in > self.d_out:
            x_input = self.input_conv(x)
        elif self.d_in < self.d_out:           
            x_input = torch.cat([x, torch.zeros([x.shape[0], self.d_out - self.d_in, seq_len, n],
                                                device=x.device)], dim=1)
        else:
            x_input = x

        x = torch.transpose(x, dim0=1, dim1=2)  # [batch_size, seq_len, d_in, n_route]
        x_conv = self.spatial_conv(supports, torch.reshape(x, (-1, n)), adpt_adj)  # [batch_size, n_route, d_out]
        x_conv = torch.reshape(x_conv, [batch_size, seq_len, n, self.d_out])
        x_input = x_input.permute(0, 2, 3, 1)

        return torch.relu(x_conv[:, :, :, 0:self.d_out] + x_input)


class STConvBlock(nn.Module):
    def __init__(self, batch_size, num_nodes, kt, channels, max_diffusion_step, filter_type, adpt_type="adpt"):
        
        super().__init__()
        self.kt = kt
        self.c_si, self.c_t, self.c_oo = channels
        self.temporal_conv_layer = TemporalConvLayer(kt, self.c_si, self.c_t)
        self.spatial_conv_layer = SpatialConvLayer(batch_size=batch_size,
                                                   num_nodes=num_nodes, d_in=self.c_t, d_out=self.c_oo,
                                                   max_diffusion_step=max_diffusion_step,
                                                   filter_type=filter_type, adpt_type=adpt_type)
        self.layer_norm = nn.LayerNorm([num_nodes, self.c_oo])

    def forward(self, supports, x, adpt_adj=None):
       
        x_s = self.temporal_conv_layer(x)
        x_t_g = self.spatial_conv_layer(supports, x_s, adpt_adj[1])  # global
        x_t_l = self.spatial_conv_layer(supports, x_s, adpt_adj[0])  # local
        x_t = 0.2 * x_t_g + 0.8 * x_t_l  # TODO: global & local比例（需调整看结果）
        x_ln = self.layer_norm(x_t)
        return x_ln


class ProjLayer(nn.Module):
    def __init__(self, time_step, channel, dec_hid_dim):
       
        super(ProjLayer, self).__init__()
        self.channel = channel
        self.dec_hid_dim = dec_hid_dim
        self.temporal_conv = nn.Conv2d(channel, dec_hid_dim, (time_step, 1))

    def forward(self, x):
       
        batch_size, _, _, channel = x.shape
        assert channel == self.channel
        # maps multi-steps to one.
        x = x.permute(0, 3, 1, 2)
        x = self.temporal_conv(x)  # (batch_size, _, 1, 207)
        x = torch.transpose(x.squeeze(), 1, 2)
        return torch.relu(x)


def DynamicAdj(inputs, dy_supports):

    times = inputs[..., 1]
    nows = times[:, 11, :]
    # 【30min】
    # days = torch.ceil((nows + 1) / 48)
    # 【2h】
    n_cal = 12
    days = torch.floor(nows / n_cal) + 1

    days = days[:, 0]
    dy_mats_list = []
    for j in range(2):
        dy_mats = []
        for day_id in days:
            day_id = int(day_id)
            if day_id < 168:
                dy_mat = dy_supports[j][0]
            elif (day_id >= 168) & (day_id < 198):
                dy_mat = dy_supports[j][1]
            elif (day_id >= 198) & (day_id < 205):
                dy_mat = dy_supports[j][2]
            elif (day_id >= 205) & (day_id < 211):
                dy_mat = dy_supports[j][3]
            elif (day_id >= 211) & (day_id < 214):
                dy_mat = dy_supports[j][4]
            elif (day_id >= 214) & (day_id < 226):
                dy_mat = dy_supports[j][5]
            elif (day_id >= 226) & (day_id < 227):
                dy_mat = dy_supports[j][6]
            elif (day_id >= 227) & (day_id < 232):
                dy_mat = dy_supports[j][7]
            elif (day_id >= 232) & (day_id < 245):
                dy_mat = dy_supports[j][8]
            elif (day_id >= 245) & (day_id < 252):
                dy_mat = dy_supports[j][9]
            elif (day_id >= 252) & (day_id < 254):
                dy_mat = dy_supports[j][10]
            elif (day_id >= 254) & (day_id < 290):
                dy_mat = dy_supports[j][11]
            elif (day_id >= 290) & (day_id < 293):
                dy_mat = dy_supports[j][12]
            elif (day_id >= 293) & (day_id < 294):
                dy_mat = dy_supports[j][13]
            elif (day_id >= 294) & (day_id < 306):
                dy_mat = dy_supports[j][14]
            elif day_id >= 306:
                dy_mat = dy_supports[j][15]

            dy_mat = torch.from_numpy(dy_mat.todense())
            dy_mats.append(dy_mat)

        dy_mats_tensor = torch.tensor(np.array(([item.numpy() for item in dy_mats])))
        dy_mats_list.append(dy_mats_tensor)

    dy_mats = torch.tensor(np.array(([item.numpy() for item in dy_mats])))
    dy_mats_list = torch.tensor(np.array(([item.numpy() for item in dy_mats_list])))



    return dy_mats_list.to(device='cuda:0')




class Encoder(nn.Module):
    def __init__(self, batch_size, num_nodes, his_len, max_diffusion_step, kt, blocks, dec_hid_dim,
                 filter_type, adpt_type):
   
        super().__init__()
        self.his_len = his_len
        self.kt = kt
        self.blocks = blocks
        self.adpt_type = adpt_type
        if adpt_type == "random_embedding":
            self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 64), requires_grad=True)
            self.nodevec2 = nn.Parameter(torch.randn(64, num_nodes), requires_grad=True)
        elif adpt_type == "dynamic":
            # self.getdyadj = DynamicAdj()
            pass
        else:
            raise ValueError("Wrong adpt_type...")
        st_conv_block_list = list()
        for channels in blocks:
            st_conv_block_list.append(STConvBlock(batch_size=batch_size,
                                                  num_nodes=num_nodes, kt=kt, channels=channels,
                                                  max_diffusion_step=max_diffusion_step,
                                                  filter_type=filter_type,
                                                  adpt_type=adpt_type))

        self.st_conv_blocks = nn.ModuleList(st_conv_block_list)
        self.output_layer = ProjLayer(time_step=self.his_len, channel=blocks[-1][-1], dec_hid_dim=dec_hid_dim)

    def forward(self, supports, inputs, batch_idx, batch_size, dy_supports, tinterval, order2time_map):
       
        outputs = inputs
        
        if self.adpt_type == "random_embedding":
            adpt_adj = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        elif self.adpt_type == "no_adpt":
            adpt_adj = None
        elif self.adpt_type == "dynamic":
            adpt_adj = DynamicAdj(inputs, dy_supports)
        else:
            raise ValueError("adpt_adj should be one of {0}, {1} and {2}".format("pam", "random_embedding", "no_adpt"))
        for i, _ in enumerate(self.blocks):
            outputs = self.st_conv_blocks[i](supports, outputs, adpt_adj)
        
        hidden = self.output_layer(outputs)
        return outputs, hidden
