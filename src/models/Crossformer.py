import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from src.models.layers.Crossformer_EncDec import scale_block, Encoder, Decoder, DecoderLayer
from src.models.layers.Embed import PatchEmbedding
from src.models.layers.SelfAttention_Family import AttentionLayer, FullAttention, TwoStageAttentionLayer
from src.models.PatchTST import FlattenHead


from math import ceil


class Model(nn.Module):
    def __init__(self, 
                 enc_in=1,
                 seq_len=48,
                 pred_len=48,
                 d_model=64,
                 e_layers=2,
                 n_heads=8,
                 d_ff=128,
                 dropout=0.1,
                 factor=4,
                 output_attention=False,
                 ):
        super(Model, self).__init__()
        self.enc_in =  enc_in  
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.seg_len = 12
        self.win_size = 2 
        self.d_model = d_model
        self.e_layers = e_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.factor = factor
        self.output_attention = output_attention
  

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * self.seq_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * self.pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = ceil(self.in_seg_num / (self.win_size ** (self.e_layers - 1)))
        self.head_nf = self.d_model * self.out_seg_num

        # Embedding
        self.enc_value_embedding = PatchEmbedding(self.d_model, self.seg_len, self.seg_len, self.pad_in_len - self.seq_len, 0)
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, self.enc_in, self.in_seg_num, self.d_model))
        self.pre_norm = nn.LayerNorm(self.d_model)

        # Encoder
        self.encoder = Encoder(
            [
                scale_block(1 if l is 0 else self.win_size, self.d_model, self.n_heads, self.d_ff,
                            1, self.dropout,
                            self.in_seg_num if l is 0 else ceil(self.in_seg_num / self.win_size ** l), self.factor, self.output_attention
                            ) for l in range(self.e_layers)
            ]
        )
        # Decoder
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, self.enc_in, (self.pad_out_len // self.seg_len), self.d_model))

        self.decoder = Decoder(
            [
                DecoderLayer(
                    TwoStageAttentionLayer((self.pad_out_len // self.seg_len), self.factor, self.d_model, self.n_heads,
                                           self.d_ff, self.dropout, self.output_attention),
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=False),
                        self.d_model, self.n_heads),
                    self.seg_len,
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    # activation=self.activation,
                )
                for l in range(self.e_layers + 1)
            ],
        )


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d = n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=x_enc.shape[0])
        dec_out = self.decoder(dec_in, enc_out)
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]