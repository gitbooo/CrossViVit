import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionDistil(nn.Module):
    def __init__(self, c_in):
        super(SelfAttentionDistil, self).__init__()
        self.conv = nn.Conv1d(c_in, c_in, kernel_size=3, padding=2, padding_mode="circular")
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.max_pool(x)
        x = torch.transpose(x, 1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attention_mask=None):
        new_x, attention = self.attention(x, x, x, attention_mask=attention_mask)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attention


class Encoder(nn.Module):
    def __init__(self, attention_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attention_layers = nn.ModuleList(attention_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attention_mask=None):
        attentions = []
        if self.conv_layers is not None:
            for attention_layer, conv_layer in zip(self.attention_layers, self.conv_layers):
                x, attention = attention_layer(x, attention_mask=attention_mask)
                x = conv_layer(x)
                attentions.append(attention)
            x, attention = self.attention_layers[-1](x)
            attentions.append(attention)
        else:
            for attention_layer in self.attention_layers:
                x, attention = attention_layer(x, attention_mask=attention_mask)
                attentions.append(attention)
        if self.norm is not None:
            x = self.norm(x)
        return x, attentions


class EncoderStack(nn.Module):
    def __init__(self, encoders):
        super(EncoderStack).__init__()
        self.encoders = nn.ModuleList(encoders)

    def forward(self, x, attention_mask=None):
        inp_len = x.size(1)
        x_stack = []
        attentions = []
        for encoder in self.encoders:
            if encoder is None:
                inp_len //= 2
                continue
            x, attention = encoder(x[:, -inp_len:, :])
            x_stack.append(x)
            attentions.append(attention)
            inp_len //= 2
        x_stack = torch.cat(x_stack, -2)
        return x_stack, attentions
