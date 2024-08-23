import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import torch
from torch import nn
from torch.nn import init


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMAttention(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        b, c, _, _ = x.size()
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder

        self.projection = nn.Linear(configs.d_model, 128, bias=True)
        self.fc = nn.Sequential(
            nn.Linear(128, 256, bias=True),
            nn.Mish(),
            nn.Linear(256, 92, bias=True),
            nn.Mish(),
            nn.Linear(92, configs.pred_len, bias=True)
        )

    def forecast(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        dec_out = self.projection(enc_out)
        dec_out = self.fc(dec_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, C]


class CNNLSTMDecoder_1(nn.Module):
    def __init__(self, d_model, pred_len, dropout, kernel_size, num_lstm_layers, d_ff):
        super(CNNLSTMDecoder_1, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1)
        )
        self.lstm = nn.LSTM(input_size=d_ff, hidden_size=d_model, num_layers=num_lstm_layers, batch_first=True,
                            dropout=dropout)
        self.fc = nn.Linear(d_model, 128, bias=True)
        self.projection = nn.Sequential(
            nn.Linear(128, 256, bias=True),
            nn.Mish(),
            nn.Linear(256, 92, bias=True),
            nn.Mish(),
            nn.Linear(92, pred_len, bias=True)
        )

    def forward(self, x):
        # x: [batch_size, input_size,seq_len]
        # CNN
        # 对输入进行转置，将特征维度和时间维度对换，利用cnn在时间维度上进行卷积，在转成[b,i_p,s_q],，用lstm提取时序特征
        cnn_output = self.cnn(x.transpose(1,
                                          2))  # Change from [batch_size, input_size,seq_len] to [batch_size, seq_len,input_size] for Conv1d
        cnn_output = cnn_output.transpose(1, 2)  # Change back to [batch_size, input_size,seq_len]
        # print(cnn_output.shape)

        # GRU层
        # gru_output, _ = self.gru(cnn_output)

        # LSTM
        lstm_output, _ = self.lstm(cnn_output)  # [batch_size,input_size, seq_len ]
        # print(lstm_output.shape)  #(bc,24,64)

        # Projection
        out_put = self.fc(lstm_output)
        projection_output = self.projection(out_put)
        return projection_output


class Model_att(nn.Module):
    def __init__(self, configs):
        super(Model_att, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)
        # attention
        self.era_attn = CBAMAttention(9, 1, 3)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.decoder = CNNLSTMDecoder_1(configs.d_model, configs.pred_len, configs.dropout, kernel_size=3,
                                        num_lstm_layers=1, d_ff=configs.d_ff)

    def forecast(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        # 注意力机制
        enc_out = enc_out.permute(0, 2, 1)
        era_in, temp_wind = enc_out[:, :, :6 * 9], enc_out[:, :, -2:]
        # print(era_in.shape)
        era_in = era_in.reshape(-1, 64, 6, 9).permute(0, 3, 2, 1)
        era_in = self.era_attn(era_in).permute(0, 3, 1, 2).reshape(-1, 64, 6 * 9)
        enc_out = torch.cat([era_in, temp_wind], axis=-1).permute(0, 2, 1)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # 使用新的 CNN-LSTM 解码器
        # print(enc_out.shape)
        dec_out = self.decoder(enc_out).permute(0, 2, 1)[:, :, :N]
        # print(dec_out.shape)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, C]


class CNNLSTMDecoder(nn.Module):
    def __init__(self, d_model, pred_len, dropout, kernel_size, num_lstm_layers, d_ff):
        super(CNNLSTMDecoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=128, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1)
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=d_model, num_layers=num_lstm_layers, batch_first=True,
                            dropout=dropout)
        self.fc = nn.Linear(d_model, d_ff, bias=True)
        self.projection = nn.Sequential(
            nn.Linear(d_ff, 128, bias=True),
            nn.Mish(),
            nn.Linear(128, 64, bias=True),
            nn.Mish(),
            nn.Linear(64, pred_len, bias=True)
        )

    def forward(self, x):
        # x: [batch_size, input_size,seq_len]
        # CNN
        # 对输入进行转置，将特征维度和时间维度对换，利用cnn在时间维度上进行卷积，在转成[b,i_p,s_q],，用lstm提取时序特征
        cnn_output = self.cnn(x.transpose(1,
                                          2))  # Change from [batch_size, input_size,seq_len] to [batch_size, seq_len,input_size] for Conv1d
        cnn_output = cnn_output.transpose(1, 2)  # Change back to [batch_size, input_size,seq_len]
        # print(cnn_output.shape)

        # GRU层
        # gru_output, _ = self.gru(cnn_output)

        # LSTM
        lstm_output, _ = self.lstm(cnn_output)  # [batch_size,input_size, seq_len ]
        # print(lstm_output.shape)  #(bc,24,64)

        # Projection
        out_put = self.fc(lstm_output)
        projection_output = self.projection(out_put)
        return projection_output


class Model1(nn.Module):
    def __init__(self, configs):
        super(Model1, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)
        # Encoder
        self.mlp = nn.Sequential(
            nn.Linear(configs.d_model, 128, bias=True),
            nn.Mish(),
            nn.Linear(128, 92, bias=True),
            nn.Mish(),
            nn.Linear(92, configs.d_model, bias=True)
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.decoder = CNNLSTMDecoder(configs.d_model, configs.pred_len, configs.dropout, kernel_size=3,
                                      num_lstm_layers=1, d_ff=configs.d_ff)

    def forecast(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out = self.mlp(enc_out)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # 使用新的 CNN-LSTM 解码器
        # print(enc_out.shape)
        dec_out = self.decoder(enc_out).permute(0, 2, 1)[:, :, :N]
        # print(dec_out.shape)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, C]