import numpy as np
import torch
from torch.nn.init import xavier_normal_
from torch import nn
import torch.nn.functional as F
from load_data import Data
from torch.autograd import  Variable
class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x,adj_mask_matrix,com_mask_matrix,lattice_rel):
        out = self.attention(x,adj_mask_matrix,com_mask_matrix,lattice_rel)

        out = self.feed_forward(out)

        return out


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x,adj_mask_matrix,com_mask_matrix,lattice_rel):
        batch_size = x.size(0)

        neighbors = x[:, 0, :].unsqueeze(0)
        neighbors = neighbors.repeat(1, x.size(1), 1)
        # print("ff",neighbors.size())
        Q = self.fc_Q(x)
        # Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5
        context = self.attention(Q, K, V,adj_mask_matrix,com_mask_matrix,lattice_rel,scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  #
        out = self.layer_norm(out)
        return out




class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout):
        super(Positional_Encoding, self).__init__()
        # self.device = device
        self.pe = torch.tensor(
            [[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).cuda()
        out = self.dropout(out)
        return out



class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''

    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V,adj_mask_matrix,com_mask_matrix,lattice_rel,scale=None):

        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale:
        Return:
            self-attention      '''
        len_Q=Q.size(1)


        adj_relation=torch.tensor(lattice_rel[0])
        adj_relation=Variable(adj_relation.repeat(1,len_Q,1)).cuda()
        Adj_Q=adj_relation+Q
        Adj_K=adj_relation+K

        com_relation = torch.tensor(lattice_rel[1])
        com_relation = Variable(com_relation.repeat(1, len_Q, 1)).cuda()
        com_Q = com_relation + Q
        com_K = com_relation + K

        attention_adj = torch.matmul(Adj_Q, Adj_K.permute(0, 2, 1))# (1,6,6)
        attention_com = torch.matmul(com_Q, com_K.permute(0, 2, 1))  # (1,6,6)

        if scale:
            attention_adj = attention_adj * scale
            attention_com=attention_com*scale
        # if mask:  # TODO change this


        mask_attention1 = torch.mul(F.softmax(attention_adj, dim=-1), adj_mask_matrix)

        mask_attention2 = torch.mul(F.softmax(attention_com, dim=-1), com_mask_matrix)

        mask_attention=mask_attention1+mask_attention2
        # print("vvvv",mask_attention.size())

        context = torch.matmul(mask_attention, V)

        return context




class TuckER(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(TuckER, self).__init__()  # d1=d2=200

        self.E = torch.nn.Embedding(len(d.entities), d1)
        # self.seen_R = torch.nn.Embedding(len(d.seen_relations), d2)

        self.R = torch.nn.Embedding(len(d.byte_dic), d2)
        # self.unseen_R = torch.nn.Embedding(len(d.unseen_relations), d2)

        # self.seen_R.weight.data.copy_(torch.from_numpy(d.seen_relations_vector))
        # self.seen_R.weight.requires_grad = False
        #
        # self.unseen_R.weight.data.copy_(torch.from_numpy(d.unseen_relations_vector))
        # self.unseen_R.weight.requires_grad = False

        # self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)),
        #                             dtype=torch.float, device="cuda", requires_grad=True))
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)),
                                                 dtype=torch.float, requires_grad=True))
        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)

        dim_model = 200
        num_head = 1
        hidden = 128
        dropout = 0.5
        num_encoder = 6
        embed=200
        pad_size=90 # the length of sequence

        self.postion_embedding = Positional_Encoding(embed, pad_size, dropout)
        self.encoder = Encoder(dim_model, num_head, hidden, dropout)

    def init(self):
        xavier_normal_(self.E.weight.data)
        # xavier_normal_(self.R.weight.data)


    def mean_pooling(self, model_output):
        token_embeddings = model_output  # First element of model_output contains all token embeddings
        sum_embeddings = torch.mean(token_embeddings, 1)
        return sum_embeddings

    def forward(self, e1_idx, r_all_ngram_idx,adj_matrix,com_matrix,lattice_rel):
        # print("e1_idx",e1_idx)
        e1 = self.E(e1_idx)
        # print("e1",e1.size())
        x = self.bn0(e1)
        head = self.input_dropout(x)

        r_all_ngram_idx=torch.tensor(r_all_ngram_idx)

        neighbor_relation = torch.zeros([r_all_ngram_idx.size(0), 200]).cuda()
        idx = -1
        for j in range(len(r_all_ngram_idx)):  # B=[[1,2,3,4,5],[4,5,6,7,8],[7,8,9,06,7],....]  len(B)  batch size
            idx = idx + 1

            # ----------relations--
            ids_r = torch.tensor(r_all_ngram_idx[j]).cuda()
            # print("ids_r",ids_r)
            relations_feather = self.R(ids_r)

            adj_matrix=torch.tensor(adj_matrix).to(torch.float32)
            adj_mask_matrix = adj_matrix[j].cuda()

            com_matrix = torch.tensor(com_matrix).to(torch.float32)
            com_mask_matrix = com_matrix[j].cuda()

            relations_feather = relations_feather.unsqueeze(0)

            relations_feather=self.postion_embedding(relations_feather)
            relations_feather = self.encoder(relations_feather,adj_mask_matrix,com_mask_matrix,lattice_rel)
            relations_feather = self.mean_pooling(relations_feather)
            neighbor_relation[idx, :] = relations_feather
        r=neighbor_relation
        # print("r.size()",r.size())
        relation = self.hidden_dropout1(r)
        x = head + relation

        # x = torch.bmm(x, W_mat)
        # x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        # print("x--------",x.size()) torch.Size([7, 139])
        # print("x---x-----", self.E.weight.transpose(1,0).size()) #torch.Size([13,139])
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        pred = torch.sigmoid(x)
        # print("pred",pred.size())
        return pred

