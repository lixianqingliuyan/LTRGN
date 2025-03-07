from helper import *
from model.compgcn_conv import CompGCNConv
from model.compgcn_conv_basis import CompGCNConvBasis
from torch import nn

from model.trans import TransConv


class BaseModel(torch.nn.Module):
    def __init__(self, params):
        super(BaseModel, self).__init__()

        self.p = params
        self.act = torch.tanh
        self.bceloss = torch.nn.BCELoss()

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)


class CompGCNBase(BaseModel):
    def __init__(self, edge_index, edge_type, num_rel, params=None):
        super(CompGCNBase, self).__init__(params)

        self.use_trans = self.p.use_trans
        self.trans_weight = self.p.trans_weight
        self.edge_index = edge_index
        self.edge_type = edge_type

        self.trans_dropout = self.p.trans_dropout
        self.trans_use_bn = self.p.trans_use_bn
        self.trans_use_residual = self.p.trans_use_residual
        self.trans_use_weight = self.p.trans_use_weight
        self.trans_use_act = self.p.trans_use_act
        self.trans_layer = self.p.trans_layer
        self.p.gcn_dim = self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
        self.init_embed = get_param((self.p.num_ent, self.p.init_dim))
        self.device = self.edge_index.device
        self.num_heads = self.p.num_heads

        if self.p.num_bases > 0:
            self.init_rel = get_param((self.p.num_bases, self.p.init_dim))
            if self.p.score_func == 'transe':
                self.init_rel = get_param((num_rel, self.p.init_dim))
            else:
                self.init_rel = get_param((num_rel * 2, self.p.init_dim))

        else:
            if self.p.score_func == 'transe':
                self.init_rel = get_param((num_rel, self.p.init_dim))
            else:
                self.init_rel = get_param((num_rel * 2, self.p.init_dim))

        if self.p.num_bases > 0:
            self.conv1 = CompGCNConvBasis(self.p.init_dim, self.p.gcn_dim, num_rel, self.p.num_bases, act=self.act,
                                          params=self.p)
            self.conv2 = CompGCNConv(self.p.gcn_dim, self.p.embed_dim, num_rel, act=self.act,
                                     params=self.p) if self.p.gcn_layer == 2 else None
        else:
            self.conv1 = CompGCNConv(self.p.init_dim, self.p.gcn_dim, num_rel, act=self.act, params=self.p)
            self.conv2 = CompGCNConv(self.p.gcn_dim, self.p.embed_dim, num_rel, act=self.act,
                                     params=self.p) if self.p.gcn_layer == 2 else None

        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

        self.trans_conv = TransConv(self.p.init_dim, self.p.embed_dim, self.p.trans_layer, self.num_heads,
                                    self.trans_dropout,
                                    self.trans_use_bn, self.trans_use_residual, self.trans_use_weight,
                                    self.trans_use_act)

        self.positional_embedding = get_param((self.p.num_ent, self.p.init_dim))
        self.Lin_pos = nn.Linear(self.p.init_dim, self.p.init_dim)

    def forward_base(self, sub, rel, drop1, drop2):
        pos = self.Lin_pos(self.positional_embedding)
        init_embed = self.init_embed + pos

        r = self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
        x1, r = self.conv1(init_embed, self.edge_index, self.edge_type, rel_embed=r)
        x1 = drop1(x1)
        x1, r = self.conv2(x1, self.edge_index, self.edge_type, rel_embed=r) if self.p.gcn_layer == 2 else (x1, r)
        x1 = drop2(x1) if self.p.gcn_layer == 2 else x1
        if self.use_trans == 'True':
            x2 = self.trans_conv(init_embed)
            if self.p.fusion == 'weight_sum':
                x = self.trans_weight * x2 + (1 - self.trans_weight) * x1
            elif self.p.fusion == 'add':
                x = x2 + x1
            elif self.p.fusion == 'mult':
                x = x2 * x1
            else:
                raise NotImplementedError
        else:
            x = x1

        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)

        return sub_emb, rel_emb, x


"""class CompGCN_TransE(CompGCNBase):
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
        self.drop = torch.nn.Dropout(self.p.hid_drop)

    def forward(self, sub, rel):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.drop, self.drop)

        obj_emb = sub_emb + rel_emb

        x = torch.mm(obj_emb, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)

        return score


class CompGCN_DistMult(CompGCNBase):
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
        self.drop = torch.nn.Dropout(self.p.hid_drop)

    def forward(self, sub, rel):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.drop, self.drop)
        obj_emb = sub_emb * rel_emb


        x = torch.mm(obj_emb, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score"""


class CompGCN_ConvE(CompGCNBase):
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.p.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)

        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.p.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.p.feat_drop)
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz),
                                       stride=1, padding=0, bias=self.p.bias)

        flat_sz_h = int(2 * self.p.k_w) - self.p.ker_sz + 1
        flat_sz_w = self.p.k_h - self.p.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)

    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.p.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.p.embed_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.p.k_w, self.p.k_h))
        return stack_inp

    def forward(self, sub, rel):
        sub_emb, rel_emb, all_ent= self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)


        stk_inp = self.concat(sub_emb, rel_emb)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)
        score = torch.sigmoid(x)

        return score


"""class CompGCN_MLP(CompGCNBase):
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
        self.drop = torch.nn.Dropout(self.p.hid_drop)
        self.mlp = nn.Sequential(nn.LayerNorm(self.p.embed_dim * 2), nn.ReLU(),
                                 nn.Linear(self.p.embed_dim * 2, self.p.embed_dim), nn.LayerNorm(self.p.embed_dim)
                                 , nn.ReLU(),
                                 nn.Linear(self.p.embed_dim, self.p.embed_dim), nn.LayerNorm(self.p.embed_dim))

    def forward(self, sub, rel):
        sub_emb, rel_emb, all_ent= self.forward_base(sub, rel, self.drop, self.drop)

        x = torch.cat((sub_emb, rel_emb), dim=1)
        x = self.mlp(x)

        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)
        score = torch.sigmoid(x)

        return score

class CompGCN_InteractE(CompGCNBase):
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)

        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.feature_drop = torch.nn.Dropout(self.p.feat_drop)

        flat_sz_h = self.p.ik_h
        flat_sz_w = 2 * self.p.ik_w
        self.padding = 0

        self.flat_sz = flat_sz_h * flat_sz_w * self.p.inum_filt * self.p.iperm
        self.chequer_perm = self.get_chequer_perm()
        self.bn0 = torch.nn.BatchNorm2d(self.p.iperm)
        self.bn1 = torch.nn.BatchNorm2d(self.p.inum_filt * self.p.iperm)
        self.feature_map_drop = torch.nn.Dropout2d(self.p.ifeat_drop)
        self.ihidden_drop = torch.nn.Dropout(self.p.ihid_drop)
        self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)
        self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)
        self.register_parameter('conv_filt',
                                Parameter(torch.zeros(self.p.inum_filt, 1, self.p.iker_sz, self.p.iker_sz)))
        torch.nn.init.xavier_uniform_(self.conv_filt)

    def circular_padding_chw(self, batch, padding):
        upper_pad = batch[..., -padding:, :]
        lower_pad = batch[..., :padding, :]
        temp = torch.cat([upper_pad, batch, lower_pad], dim=2)

        left_pad = temp[..., -padding:]
        right_pad = temp[..., :padding]
        padded = torch.cat([left_pad, temp, right_pad], dim=3)
        return padded

    def forward(self, sub, rel):
        sub_emb, rel_emb, all_ent= self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)



        comb_emb = torch.cat([sub_emb, rel_emb], dim=1)
        chequer_perm = comb_emb[:, self.chequer_perm]
        stack_inp = chequer_perm.reshape((-1, self.p.iperm, 2 * self.p.ik_w, self.p.ik_h))
        stack_inp = self.bn0(stack_inp)
        x = stack_inp
        x = self.circular_padding_chw(x, self.p.iker_sz // 2)
        x = F.conv2d(x, self.conv_filt.repeat(self.p.iperm, 1, 1, 1), padding=self.padding, groups=self.p.iperm)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.ihidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)
        score = torch.sigmoid(x)

        return score

    def get_chequer_perm(self):

        ent_perm = np.int32([np.random.permutation(self.p.embed_dim) for _ in range(self.p.iperm)])
        rel_perm = np.int32([np.random.permutation(self.p.embed_dim) for _ in range(self.p.iperm)])

        comb_idx = []
        for k in range(self.p.iperm):
            temp = []
            ent_idx, rel_idx = 0, 0

            for i in range(self.p.ik_h):
                for j in range(self.p.ik_w):
                    if k % 2 == 0:
                        if i % 2 == 0:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1
                        else:
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                    else:
                        if i % 2 == 0:
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                        else:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1

            comb_idx.append(temp)

        chequer_perm = torch.LongTensor(np.int32(comb_idx)).to(self.device)
        return chequer_perm
"""

