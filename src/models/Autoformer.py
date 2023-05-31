import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from src.models.layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from src.models.layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


class Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    """

    def __init__(self, 
                 seq_len=48, 
                 label_len=24,
                 pred_len=48, 
                 moving_avg=12, 
                 output_attention=False,
                 enc_in=1,
                 dec_in=1,
                 d_model=64,
                 embed=64,
                 freq='t',
                 dropout=0.1,
                 e_layers=2,
                 d_layers=2,
                 d_ff=128,
                 n_heads=8,
                 factor=4,
                 activation='gelu',
                 c_out=1,
                 ):
        super(Autoformer, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.moving_avg = moving_avg
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.d_model = d_model
        self.embed = embed
        self.freq = freq
        self.dropout = dropout
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.factor = factor
        self.activation = activation
        self.c_out = c_out


        # Decomp
        kernel_size = moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(self.enc_in, self.d_model, self.embed, self.freq,
                                                  self.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, self.factor, attention_dropout=self.dropout,
                                        output_attention=self.output_attention),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    moving_avg=self.moving_avg,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=my_Layernorm(self.d_model)
        )
        # Decoder
        self.dec_embedding = DataEmbedding_wo_pos(self.dec_in, self.d_model, self.embed, self.freq,
                                                    self.dropout)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, self.factor, attention_dropout=self.dropout,
                                        output_attention=False),
                        self.d_model, self.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, self.factor, attention_dropout=self.dropout,
                                        output_attention=False),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.c_out,
                    self.d_ff,
                    moving_avg=self.moving_avg,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.d_layers)
            ],
            norm_layer=my_Layernorm(self.d_model),
            projection=nn.Linear(self.d_model, self.c_out, bias=True)
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(
            1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len,
                             x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat(
            [trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

