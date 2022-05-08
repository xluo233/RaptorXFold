import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from .Common import Residual, FeedForward
from .Common import activation_func, normalization_func
from .AxialTransformer import AxialTransformerLayer
from .AttentionModules import MHSelfAttention
from .AttentionModules import PointwiseAttention
from .ResNet1D import ResNet1DBlock


class PositionalEncoding2D(nn.Module): 
    def __init__(self, n_embedding, p_drop=0.1):
        super(PositionalEncoding2D, self).__init__()
        self.drop = nn.Dropout(p_drop, inplace=True)

        n_embedding_half = n_embedding // 2
        div_term = torch.exp(torch.arange(0., n_embedding_half, 2) *
                             -(math.log(10000.0) / n_embedding_half))
        self.register_buffer('div_term', div_term)

    def forward(self, x, idx_s):
        B, L, _, K = x.shape
        K_half = K//2
        pe = torch.zeros_like(x)
        i_batch = -1
        for idx in idx_s:
            i_batch += 1
            sin_inp = idx.unsqueeze(1) * self.div_term
            # (L, K//2)
            emb = torch.cat((sin_inp.sin(), sin_inp.cos()), dim=-1)
            pe[i_batch, :, :, :K_half] = emb.unsqueeze(1)
            pe[i_batch, :, :, K_half:] = emb.unsqueeze(0)
        x = x + torch.autograd.Variable(pe, requires_grad=False)
        return self.drop(x)


class Seq_to_Pair(nn.Module):
    '''
    Generate 2D pairwise template feature from 1D sequential template feature.
    '''
    def __init__(self, featsize_1d, n_emb_1D,
                 n_ResNet1D_block=4,
                 activation='elu', normalization='instance', bias=False):
        super().__init__()
        self.proj_1D = nn.Conv1d(
            featsize_1d, n_emb_1D, kernel_size=1, bias=True)
        self.ResNet1D_blocks = nn.ModuleList(
            [
                ResNet1DBlock(
                    n_emb_1D, n_emb_1D, kernel_size=3, dilation=1, dropout=0,
                    activation=activation, normalization=normalization,
                    bias=bias) for _ in range(n_ResNet1D_block)
            ]
        )


    def forward(self, t1d):
        B, T, L, _ = t1d.shape
        # (B, T, L, feat_1d) -> (B*T, feat_1d, L)
        # 1D project -> (B*T, n_emb_1D, L)
        t1d = t1d.permute(0, 1, 3, 2).reshape(B*T, -1, L)
        t1d = self.proj_1D(t1d)
        for block in self.ResNet1D_blocks:
            t1d = block(t1d)
        t1d = t1d.reshape(B, T, -1, L).permute(0, 1, 3, 2)
        left = t1d.unsqueeze(3).expand(-1, -1, -1, L, -1)
        right = t1d.unsqueeze(2).expand(-1, -1, L, -1, -1)
        pair_feat = torch.cat((left, right), -1)

        return pair_feat


class TemplateEmbedding(nn.Module): 
    def __init__(
            self, featsize_1d, featsize_2d, n_inner=64, n_emb_1D=32, p_drop=0):
        super().__init__()
        self.proj = nn.Linear(featsize_2d+2*n_emb_1D+1,
                              n_inner)
        self.pos = PositionalEncoding2D(n_inner, p_drop=p_drop)
        self.seq_to_pair = Seq_to_Pair(featsize_1d=featsize_1d, n_emb_1D=n_emb_1D)


    def forward(self, t1d, t2d):
        # Input
        #   - t1d: 1D template info (B, T, L, d_t1d)
        #   - t2d: 2D template info (B, T, L, L, d_2d)
        # n_inner = d_t1d * 2 + d_t2d + 1
        # Output:
        # Template_emb: (B, L, L, d_templ)
        B, T, L, _ = t1d.shape
        pair_feat = self.seq_to_pair(t1d)
        idx = torch.arange(L).view(1, L).expand(B, -1).to(pair_feat.device)
        seqsep = torch.abs(idx[:, :, None]-idx[:, None, :]) + 1
        seqsep = torch.log(seqsep.float()).view(
            B, L, L, 1).unsqueeze(1).expand(-1, T, -1, -1, -1)
        pair_feat = torch.cat((t2d, pair_feat, seqsep), -1)
        pair_feat = self.proj(pair_feat).reshape(B*T, L, L, -1)
        # (B*T, L, L, d_templ)
        pair_feat = self.pos(pair_feat, idx)   # add positional embedding
        return pair_feat


class TemplateAxialTransformer(nn.Module):
    def __init__(self, n_t1d: int, n_t2d: int, n_q1d: int, n_q2d: int,
                 n_inner: int, n_output: int, n_layer: int,
                 n_head: int, n_ff_hidden: int, head_by_head: bool, 
                 attn_dropout: float, ff_dropout: float, activation: str
                 ):
        super().__init__()
        self.template_emb = TemplateEmbedding(
            featsize_1d=n_t1d, featsize_2d=n_t2d )
        self.norm_templ = nn.LayerNorm(n_inner)


        self.layers = nn.ModuleList(
            [AxialTransformerLayer(emb_dim=n_inner, n_head=n_head, n_ff_hidden=n_ff_hidden, head_by_head=head_by_head,
                                    attn_dropout=attn_dropout, ff_dropout=ff_dropout, activation=activation)
             for i in range(n_layer)])
        # attn cross template
        self.col_attn_layer = nn.ModuleList(
            [Residual(MHSelfAttention(emb_dim=n_inner, n_head = n_head, attn_dropout=attn_dropout, attn_seq_weight=False, mean_attn=False),
                        n_inner, n_inner)
                for i in range(n_layer)])

        self.template_to_pair = Template2Pair(
            n_input=n_inner, n_output=n_inner, n_q1d=n_q1d, n_q2d=n_q2d,)
        self.projection = nn.Linear(n_inner, n_output)

        # merge the output
        self.merge_linear = nn.Linear(n_q2d + n_output, n_q2d)
        self.norm_out = nn.LayerNorm(n_q2d)


    # input:
    #   seq_encoding: (B, L, n_q1d)
    #   pair_feat: (B, L, L, n_q2d)
    #   t1d: template feature 1D (B, T, L, n_t1d)
    #   t2d: template feature 2D (B, T, L, L, n_t2d)
    #   res_mask: (B, L)
    #   templ_mask: (B, T)
    # output:
    #   template pairwise representation: (B，L，L, d_model)
    def forward(self, seq_encoding, pair_feat, t1d, t2d,
                res_mask=None, templ_mask=None):
        B, T, L, _ = t1d.shape

        # TemplateEmbedding, (B * T, L, L, n_inner)
        templ_emb = self.template_emb(t1d, t2d)
        templ_emb = self.norm_templ(templ_emb)

        # AxialTransformer, (B * T, L, L, n_inner)
        for i, layer in enumerate(self.layers):

            templ_emb = layer(
                templ_emb, templ_mask.reshape(B * T, 1))
            # (B * T, L, L, n_inner) -> (B, L * L, T, n_inner)
            templ_emb = templ_emb.reshape(B, T, L, L, -1).permute(
                0, 2, 3, 1, 4).contiguous().reshape(B, L*L, T, -1)
            templ_emb = self.col_attn_layer[i](
                templ_emb, seq_mask=templ_mask)
            templ_emb = templ_emb.reshape(B, L, L, T, -1).permute(
                0, 3, 1, 2, 4).contiguous().reshape(B*T, L, L, -1)

        templ_emb = templ_emb.reshape(B, T, L, L, -1)

        # Template to Pair
        template_feat = self.template_to_pair(
            seq_encoding, pair_feat, templ_emb, templ_mask)
        template_feat = self.projection(template_feat)    # (B, L, L, d_model)

        template_feat = self.merge_linear(
            torch.cat((template_feat, pair_feat), dim=-1))

        template_feat = self.norm_out(template_feat)
        return template_feat


class Template2Pair(nn.Module):
    def __init__(self, n_input, n_output, n_q1d, n_q2d,
                 ):
        super().__init__()


        self.templ_attn = TemplatePointwiseAttention(
            n_input=n_q1d, n_pair_input=n_q2d, n_templ_input=n_input,)

        self.norm_out = nn.LayerNorm(n_output)

    def forward(self, seq_encoding, pair_feat, templ_feat, templ_mask):
        # templ_emb: (B, T, L, L, n_inner)
        templ_feat = self.templ_attn(templ_feat, seq_encoding, pair_feat,
                                        templ_mask)

        templ_feat = self.norm_out(templ_feat)

        return templ_feat



class TemplatePointwiseAttention(nn.Module):
    def __init__(
            self, n_input: int, n_pair_input: int,
            n_templ_input: int, n_inner_emb: int=64,
            key_dim: int=128,
            activation: str = 'relu'):
        super().__init__()
        self.emb_fc_left = nn.Sequential(
            nn.Linear(n_input, n_inner_emb // 2, bias=False),
            normalization_func('1D', 'layer', n_inner_emb // 2),
            activation_func(activation),
        )
        self.emb_fc_right = nn.Sequential(
            nn.Linear(n_input, n_inner_emb // 2, bias=False),
            normalization_func('1D', 'layer', n_inner_emb // 2),
            activation_func(activation),
        )
        self.templ_feat_fc = nn.Sequential(
            nn.Linear(n_templ_input, n_inner_emb, bias=False),
            normalization_func('1D', 'layer', n_inner_emb),
            activation_func(activation),
        )
        self.pair_feat_fc = nn.Sequential(
            nn.Linear(n_pair_input, n_inner_emb, bias=False),
            normalization_func('1D', 'layer', n_inner_emb),
            activation_func(activation),
        )
        self.norm_templ = nn.LayerNorm(key_dim)
        self.templ_attn = PointwiseAttention(key_dim=key_dim)

    # seq_encoding: (B, L, n_input)
    # pair_feat: (B, L, L, n_pair_input)
    # templ_feat: (B, T, L, L, n_templ_input)
    # template_mask: (B, T)
    def forward(self, templ_feat, seq_encoding, pair_feat, mask):
        B, L, _ = seq_encoding.shape
        _, T, _, _, _ = templ_feat.shape
        seq_left = self.emb_fc_left(seq_encoding)
        seq_right = self.emb_fc_right(seq_encoding)

        seq_left = seq_left.unsqueeze(2).expand(-1, -1, L, -1)
        seq_right = seq_right.unsqueeze(1).expand(-1, L, -1, -1)
        pair_feat = self.pair_feat_fc(pair_feat)
        # seq_encoding: (B, L, L, n_inner_emb + n_inner_emb)
        seq_encoding = torch.cat((seq_left, seq_right, pair_feat), dim=-1)
        seq_encoding = self.norm_templ(seq_encoding)

        # templ_feat: (B * T, L, L, n_inner_emb)
        templ_feat = templ_feat.reshape(B * T, L, L, -1)
        templ_feat = self.templ_feat_fc(templ_feat)

        # seq_encoding: (B, L, L, C1) -> (B*L*L, 1, C1)
        seq_encoding = seq_encoding.reshape(B*L*L, 1, -1)
        # templ_feat: (B*T, L, L, C2) -> (B*L*L, T, C1)
        templ_feat = templ_feat.reshape(B, T, L, L, -1).permute(0, 2, 3, 1, 4)
        templ_feat = templ_feat.reshape(B*L*L, T, -1)
        # bias: (B, T) -> (B * L * L, T) -> (B * L * L, :, :, T)
        mask = mask.unsqueeze(1).expand(-1, L*L, -1).reshape(B*L*L, T)

        # out: (B, L * L, 1, n_output)
        output = self.templ_attn(seq_encoding, templ_feat, mask)
        output = output.reshape(B, L, L, -1)

        return output
