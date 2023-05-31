import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers.Transformer_EncDec import Encoder, EncoderLayer
from src.models.layers.SelfAttention_Family import ReformerLayer
from src.models.layers.Embed import DataEmbedding


class Model(nn.Module):
    """
    Reformer with O(LlogL) complexity
    Paper link: https://openreview.net/forum?id=rkgNKkHtvB
    """

    def __init__(self,
                 pred_len=48,
                 seq_len=48,
                 enc_in=1,
                 d_model=64,
                 embed=64,
                 freq='t',
                 dropout=0.1,
                 e_layers=2,
                 d_ff=128,
                 n_heads=8,
                 factor=4,
                 activation='gelu',
                 c_out=1,
                 bucket_size=4, 
                 n_hashes=4, ):
        """
        bucket_size: int, 
        n_hashes: int, 
        """
        super(Model, self).__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.embed = embed
        self.freq = freq
        self.dropout = dropout
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.factor = factor
        self.activation = activation
        self.c_out = c_out

        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, self.embed, self.freq,
                                           self.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ReformerLayer(None, self.d_model, self.n_heads,
                                  bucket_size=bucket_size, n_hashes=n_hashes),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )

        self.projection = nn.Linear(
            self.d_model, self.c_out, bias=True)

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # add placeholder
        x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        if x_mark_enc is not None:
            x_mark_enc = torch.cat(
                [x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)

        return dec_out  # [B, L, D]
    

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]