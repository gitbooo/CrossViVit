import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers.Embed import DataEmbedding
from src.models.layers.AutoCorrelation import AutoCorrelationLayer
from src.models.layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from src.models.layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from src.models.layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class Model(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    Paper link: https://proceedings.mlr.press/v162/zhou22g.html
    """

    def __init__(self,
                 seq_len=48,
                 label_len=24,
                 pred_len=48,
                 moving_avg=12,
                 enc_in=1,
                 dec_in=1,
                 d_model=64,
                 embed='fixed',
                 freq='t',
                 dropout=0.1,
                 e_layers=2,
                 d_layers=2,
                 d_ff=128,
                 n_heads=8,
                 factor=4,
                 activation='gelu',
                 c_out=1,
                 version='fourier', 
                 mode_select='random', 
                 modes=32):
        """
        version: str, for FEDformer, there are two versions to choose, options: [Fourier, Wavelets].
        mode_select: str, for FEDformer, there are two mode selection method, options: [random, low].
        modes: int, modes to be selected.
        """
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
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
        self.version = version
        self.mode_select = mode_select
        self.modes = modes

        # Decomp
        self.decomp = series_decomp(self.moving_avg)
        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, self.embed, self.freq,
                                           self.dropout)
        self.dec_embedding = DataEmbedding(self.dec_in, self.d_model, self.embed, self.freq,
                                           self.dropout)

        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=self.d_model, L=1, base='legendre')
            decoder_self_att = MultiWaveletTransform(ich=self.d_model, L=1, base='legendre')
            decoder_cross_att = MultiWaveletCross(in_channels=self.d_model,
                                                  out_channels=self.d_model,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=self.modes,
                                                  ich=self.d_model,
                                                  base='legendre',
                                                  activation='tanh')
        else:
            encoder_self_att = FourierBlock(in_channels=self.d_model,
                                            out_channels=self.d_model,
                                            seq_len=self.seq_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
            decoder_self_att = FourierBlock(in_channels=self.d_model,
                                            out_channels=self.d_model,
                                            seq_len=self.seq_len // 2 + self.pred_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
            decoder_cross_att = FourierCrossAttention(in_channels=self.d_model,
                                                      out_channels=self.d_model,
                                                      seq_len_q=self.seq_len // 2 + self.pred_len,
                                                      seq_len_kv=self.seq_len,
                                                      modes=self.modes,
                                                      mode_select_method=self.mode_select)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,  # instead of multi-head attention in transformer
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
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        self.d_model, self.n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
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
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc)  # x - moving_avg, moving_avg
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
