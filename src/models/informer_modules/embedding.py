import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_length=5000):
        super(PositionalEmbedding, self).__init__()
        embedding = torch.zeros(max_length, d_model)

        position = torch.arange(0, max_length).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        embedding[:, 0::2] = torch.sin(position * div_term)
        embedding[:, 1::2] = torch.cos(position * div_term)

        embedding = embedding.unsqueeze(0)
        self.register_buffer("embedding", embedding)

    def forward(self, x):
        return self.embedding[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.token_conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
        )
        nn.init.kaiming_normal_(self.token_conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        return self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()
        weight = torch.zeros(c_in, d_model)

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        weight[:, 0::2] = torch.sin(position * div_term)
        weight[:, 1::2] = torch.cos(position * div_term)

        self.embedding = nn.Embedding(c_in, d_model)
        self.embedding.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        return self.embedding(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embedding_type="fixed"):
        super(TemporalEmbedding, self).__init__()

        MINUTE_SIZE = 2
        HOUR_SIZE = 24
        WEEKDAY_SIZE = 7
        DAY_SIZE = 32
        MONTH_SIZE = 13

        Embedding = FixedEmbedding if embedding_type == "fixed" else nn.Embedding
        self.minute_embedding = Embedding(MINUTE_SIZE, d_model)
        self.hour_embedding = Embedding(HOUR_SIZE, d_model)
        self.weekday_embedding = Embedding(WEEKDAY_SIZE, d_model)
        self.day_embedding = Embedding(DAY_SIZE, d_model)
        self.month_embedding = Embedding(MONTH_SIZE, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embedding(x[:, :, 4]) if hasattr(self, "minute_embedding") else 0.0
        hour_x = self.hour_embedding(x[:, :, 3])
        weekday_x = self.weekday_embedding(x[:, :, 2])
        day_x = self.day_embedding(x[:, :, 1])
        month_x = self.month_embedding(x[:, :, 0])

        return minute_x + hour_x + weekday_x + day_x + month_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TimeFeatureEmbedding, self).__init__()

        FREQUENCY_MAP = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_input = FREQUENCY_MAP["t"]
        self.embedding = nn.Linear(d_input, d_model)

    def forward(self, x):
        return self.embedding(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embedding_type="fixed", dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        if embedding_type != "timefeature":
            self.temporal_embedding = TemporalEmbedding(d_model, embedding_type=embedding_type)
        else:
            self.temporal_embedding = TimeFeatureEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)
