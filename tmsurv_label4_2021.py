import os
import sys
import copy
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.data import Data
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.nn import GlobalAttention
from torch_geometric.nn import SAGEConv,LayerNorm
from mae_utils import get_sinusoid_encoding_table,Block
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


class my_GlobalAttention(torch.nn.Module):
    def __init__(self, gate_nn, out_channels, nn=None):
        super(my_GlobalAttention, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn
        self.out_channels = out_channels
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)

    def forward(self, x, batch, size=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, self.out_channels)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)
        gate = softmax(gate, batch, num_nodes=size)

        if self.nn is not None:
            out = gate * x
        else:
            out = scatter_add(gate * x, batch, dim=0, dim_size=size)  # [1, 256]
        return out, gate

    def __repr__(self):
        return '{}(gate_nn={}, nn={})'.format(self.__class__.__name__,
                                              self.gate_nn, self.nn)
    
    
def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def Mix_mlp(dim1):
    
    return nn.Sequential(
            nn.Linear(dim1, dim1),
            nn.GELU(),
            nn.Linear(dim1, dim1))

class MixerBlock(nn.Module):
    def __init__(self,dim1,dim2):
        super(MixerBlock,self).__init__() 
        
        self.norm = LayerNorm(dim1)
        self.mix_mip_1 = Mix_mlp(dim1)
        self.mix_mip_2 = Mix_mlp(dim2)
        
    def forward(self,x): 
        
        y = self.norm(x)
        y = y.transpose(0,1)
        y = self.mix_mip_1(y)
        y = y.transpose(0,1)
        x = x + y
        y = self.norm(x)
        x = x + self.mix_mip_2(y)

        return x

def MLP_Block(dim1, dim2, dropout=0.3):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)
    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Dropout(p=dropout))


def GNN_relu_Block(dim2, dropout=0.3):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)
    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    return nn.Sequential(
            nn.ReLU(),
            LayerNorm(dim2),
            nn.Dropout(p=dropout))


class Mlp(nn.Module):
    def __init__(self, hidden_size=8, mlp_dim=32):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.act_fn = nn.functional.gelu
        self.dropout = nn.Dropout(0.3)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class GCN(nn.Module):
    def __init__(self, num_state, num_node):
        super(GCN, self).__init__()
        self.num_state = num_state  # num_state=1
        self.num_node = num_node  # num_node=81
        self.conv1 = nn.Conv1d(num_state, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=1)

    def forward(self, seg, aj):
        n, h, w = seg.size()
        seg = seg.view(n, self.num_state, -1).contiguous()
        seg = self.relu(self.conv1(seg))
        seg_similar = torch.bmm(seg, aj)
        out = self.relu(self.conv2(seg_similar))
        output = out + seg
        return output


class MLP(torch.nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim, bias=False),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim, bias=False),
            torch.nn.Dropout(dropout)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        return self.net(x)
    

class GraphMixerBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4., dropout=0.1, drop_path=0.):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.token_mix = GATv2Conv(dim, dim, dropout=0.1)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.channel_mix = MLP(dim=dim, hidden_dim=mlp_hidden_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, edge_index):
        x = x + self.token_mix(self.norm1(x), edge_index)
        x = x + self.channel_mix(x)

        return x

class QKVAttention(nn.Module):
    def __init__(self):
        super(QKVAttention, self).__init__()

    def forward(self, q, k, v):
        dk = q.size()[-1]
        scores = q.matmul(k.transpose(-2, -1)) / math.sqrt(dk)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(v), scores
    

class MBGCN(nn.Module):
    def __init__(self, in_feats, num_layers, dropout=0.3):
        super(MBGCN, self).__init__()
        self.num_layers = num_layers

        self.gcn1 = SAGEConv(in_channels=256, out_channels=128)  # SEUNet的in_channels是256, DMCTNet的是192
        self.gcn2 = SAGEConv(in_channels=128, out_channels=64)
        self.gcn3 = SAGEConv(in_channels=64, out_channels=64)
#         self.gcn4 = SAGEConv(32, 1)
        self.relu1 = GNN_relu_Block(128)
        self.relu2 = GNN_relu_Block(64)
        self.relu3 = GNN_relu_Block(64)

        self.embedding = nn.Sequential(*[nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.25)])
        self.graphmixer_blocks = nn.ModuleList([])
        for level in range(1, self.num_layers+1):
            self.graphmixer_blocks.append(GraphMixerBlock(256, level))

    def forward(self, all_thing):
        x = all_thing.mbgcn_data
        data_id = all_thing.data_id
        edge_index = all_thing.mbgcn_edge_index

        x = self.gcn1(x, edge_index)
        x = self.relu1(x)
        x = self.gcn2(x, edge_index)
        x = self.relu2(x)
        x = self.gcn3(x, edge_index)
        x = self.relu3(x)

        return x


class DBGCN(nn.Module):
    def __init__(self, in_feats, n_hidden, out_classes, num_layers, dropout=0.3):
        super(DBGCN, self).__init__()
        self.num_layers = num_layers
        self.gnn = SAGEConv(in_channels=in_feats, out_channels=out_classes)
        self.relu_2 = GNN_relu_Block(out_classes) 
        att_net_ct = nn.Sequential(nn.Linear(out_classes, out_classes//2), nn.ReLU(), nn.Linear(out_classes//2, out_classes//4))
        x_project = nn.Sequential(nn.Linear(out_classes, out_classes//4), nn.ReLU())
        self.mpool_ct = my_GlobalAttention(att_net_ct, out_classes//4, x_project)

        self.lin1_db = torch.nn.Linear(out_classes+720, out_classes//4)  # out_class=256, out_class//4=64
        self.lin2_db = torch.nn.Linear(out_classes//4, 32)
        
        self.norm1_db = LayerNorm(out_classes//4)
        self.norm2_db = LayerNorm(32)
        self.relu = torch.nn.ReLU() 
        self.dropout=nn.Dropout(p=dropout)
        self.lin3_out = torch.nn.Linear(72, 4)
        self.sigmoid = nn.Sigmoid()

        self.embedding = nn.Sequential(*[nn.Linear(1000, 1000), nn.ReLU(), nn.Dropout(0.25)])
        self.graphmixer_blocks = nn.ModuleList([])
        for level in range(1, self.num_layers + 1):
            self.graphmixer_blocks.append(GraphMixerBlock(1000, level))

    def forward(self, all_thing):
        x = all_thing.dbgcn_data
        edge_index = all_thing.dbgcn_edge_index

        x = self.gnn(x, edge_index)
        x = self.relu_2(x)
        batch = torch.zeros(len(x), dtype=torch.long).to(device)
        pool_x, att_2 = self.mpool_ct(x, batch)
        x = pool_x
        x = F.normalize(x, dim=1)
        return x
    
class CLNGCN(nn.Module):
    def __init__(self, ):
        super(CLNGCN, self).__init__()
        self.conv1 = nn.Conv1d(1, 1, 1)
        self.mlp1 = Mlp()
        self.mlp2 = Mlp()
        self.softmax = nn.Softmax(dim=1)
        
        self.conv2 = nn.Conv1d(1, 1, 1)
        self.gcn = GCN(1, 64)

    def forward(self, cli):
        cli_conv1 = self.conv1(cli)
        sigma = cli_conv1
        sigma_T = cli_conv1.view(1, 8, 1)
        sigma_out = torch.bmm(sigma_T, sigma)
        x = cli.squeeze(0)
        cli_ss = self.mlp1(x).unsqueeze(0).view(1, 1, 8)
        cli_mm = self.mlp2(x).unsqueeze(0).view(1, 8, 1)
        diag_att = torch.bmm(cli_mm, cli_ss) * sigma_out

        cli_conv2 = self.conv2(cli)
        alpha = cli_conv2
        alpha_T = cli_conv2.view(1, 8, 1)
        diag_cha = torch.bmm(cli_mm, cli_ss)
        diag_cha_alpha_T = torch.bmm(diag_cha, alpha_T)
        similarity_c = torch.bmm(diag_cha_alpha_T, alpha)
        similarity = similarity_c + diag_att
        similarity = self.softmax(similarity)
        x_gcn = self.gcn(cli, similarity).squeeze(1)
        return x_gcn


class TMSurv(nn.Module):
    def __init__(self, in_feats, n_hidden, out_classes, dropout=0.3):
        super(TMSurv, self).__init__()
        self.mbgcn = MBGCN(in_feats, 4)
        self.dbgcn = DBGCN(in_feats, n_hidden, out_classes, 4, dropout)
        self.clngcn = CLNGCN()

        self.coattn_3to2 = QKVAttention()
        self.coattn_2to3 = QKVAttention()
        self.coattn_cli2img = QKVAttention()

        mb_attn_net_ct = nn.Sequential(nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 1))
        self.mb_mpool_ct = my_GlobalAttention(mb_attn_net_ct, 1)

        db_attn_net_ct = nn.Sequential(nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 1))
        self.db_mpool_ct = my_GlobalAttention(db_attn_net_ct, 1)
        
        self.cli_fc1 = nn.Linear(8, 64)
        self.cli_norm = LayerNorm(64)
        self.cli_fc2 = nn.Linear(64, 128)

        self.conv1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.norm = LayerNorm(64)
        self.dropout = nn.Dropout(p=dropout)

        self.conv2 = nn.Linear(64, 4)

    def forward(self, graph):
        mbgcn_vector = self.mbgcn(graph)
        cli_fea = graph.cli_fea.unsqueeze(0).unsqueeze(0)
        cli_vector = self.clngcn(cli_fea)
        dbgcn_vector = self.dbgcn(graph, mbgcn_vector, cli_vector)

        db_vector, db_scores = self.coattn_2to3(dbgcn_vector, mbgcn_vector, mbgcn_vector)
        mb_vector, mb_scores = self.coattn_3to2(mbgcn_vector, dbgcn_vector, dbgcn_vector)
        mb_batch = torch.zeros(len(mb_vector), dtype=torch.long).to(device)
        mb_pool_x, mb_attn2 = self.mb_mpool_ct(mb_vector, mb_batch)

        db_batch = torch.zeros(len(db_vector), dtype=torch.long).to(device)
        db_pool_x, db_attn2 = self.db_mpool_ct(db_vector, db_batch)

        img_vector = torch.cat([mb_pool_x, db_pool_x], dim=-1)
        cli_vector = self.cli_fc2(self.cli_norm(self.cli_fc1(cli_vector)))
        cli_vector = cli_vector.transpose(-2, -1)
        img_vector = img_vector.transpose(-2, -1)
        cli_img_vector, cli_img_score = self.coattn_cli2img(cli_vector, img_vector, img_vector)

        cli_img_vector = cli_img_vector.transpose(-2, -1)

        logits = self.conv2(self.dropout(self.relu(self.norm(self.conv1(cli_img_vector)))))

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1-hazards, dim=1)
        
        return hazards, S, Y_hat

