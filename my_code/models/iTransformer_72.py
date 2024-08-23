import torch
import torch.nn as nn
from my_code.layers.Transformer_EncDec import Encoder, EncoderLayer
from my_code.layers.SelfAttention_Family import FullAttention, AttentionLayer
from my_code.layers.Embed import DataEmbedding_inverted


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

        self.projection = nn.Linear(configs.d_model+168,configs.d_ff, bias=True)
        self.fc = nn.Sequential(
            nn.Linear(configs.d_ff, 256, bias=True),
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
        enc_out = torch.concat((x_enc.permute(0, 2, 1), enc_out), dim=-1)
        dec_out = self.projection(enc_out)
        dec_out = self.fc(dec_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, C]


class CNNLSTMDecoder(nn.Module):
    def __init__(self, d_model, pred_len, dropout, kernel_size, num_lstm_layers, d_ff):
        super(CNNLSTMDecoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=64, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1)
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=d_model, num_layers=num_lstm_layers, batch_first=True,
                            dropout=dropout,bidirectional=True)
        self.fc = nn.Linear(d_model*2, d_ff, bias=True)
        self.projection = nn.Sequential(
            nn.Linear(d_ff, 128, bias=True),
            nn.Mish(),
            nn.Linear(128, 64, bias=True),
            nn.Mish(),
            nn.Linear(64, pred_len, bias=True)
        )
    def forward(self, x):
        # x: [batch_size, input_size,seq_len]
        cnn_output = self.cnn(x.transpose(1,2))  # Change from [batch_size, input_size,seq_len] to [batch_size, seq_len,input_size] for Conv1d
        cnn_output = cnn_output.transpose(1, 2)
        # LSTM
        lstm_output, _ = self.lstm(cnn_output)  # [batch_size,input_size, seq_len ]
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

        self.decoder = CNNLSTMDecoder(configs.d_model, configs.pred_len, configs.dropout, kernel_size=3,num_lstm_layers=1, d_ff=configs.d_ff)

    def forecast(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        # enc_out = self.cnn(enc_out)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # enc_out = torch.concat((x_enc.permute(0, 2, 1), enc_out), dim=-1)
        dec_out = self.decoder(enc_out).permute(0, 2, 1)[:, :, :N]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, C]