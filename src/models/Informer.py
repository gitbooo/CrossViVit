import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from src.models.layers.SelfAttention_Family import ProbAttention, AttentionLayer
from src.models.layers.Embed import DataEmbedding


class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132
    """
    def __init__(self, 
                 enc_in=1,
                 dec_in=1,
                 c_out=1,
                 label_len=24,
                 pred_len=48,
                 d_model=64,
                 e_layers=2,
                 d_layers=1, 
                 n_heads=8,
                 embed=64,
                 freq='t',
                 d_ff=128,
                 dropout=0.1,
                 factor=4,
                 output_attention=False,
                 activation='gelu',
                 distil=True):
        super(Model, self).__init__()
        self.pred_len = pred_len
        self.label_len = label_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = dropout
        self.factor = factor
        self.output_attention = output_attention
        self.embed = embed
        self.freq = freq
        self.dec_in = dec_in
        self.activation = activation
        self.c_out = c_out
        self.distil = distil

        # Embedding
        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, self.embed, self.freq,
                                           self.dropout)
        self.dec_embedding = DataEmbedding(self.dec_in, self.d_model, self.embed, self.freq,
                                           self.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=self.output_attention),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            [
                ConvLayer(
                    self.d_model
                ) for l in range(self.e_layers - 1)
            ] if self.distil else None,
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, self.factor, attention_dropout=self.dropout, output_attention=False),
                        self.d_model, self.n_heads),
                    AttentionLayer(
                        ProbAttention(False, self.factor, attention_dropout=self.dropout, output_attention=False),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model),
            projection=nn.Linear(self.d_model, self.c_out, bias=True)
        )

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out  # [B, L, D]
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]