import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class itransformer_lstm(nn.Module):
    def __init__(self, seq_len, pred_len, output_attention, dropout, d_model, n_heads, d_ff, activation, e_layers):
        super(itransformer_lstm, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        # Embedding
        self.enc_embedding_1 = DataEmbedding_inverted(seq_len, int(d_model / 2), dropout)
        self.enc_embedding_2 = DataEmbedding_inverted(seq_len, d_model, dropout)
        # Decoder

        self.encoder_1 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=dropout,
                                      output_attention=output_attention), int(d_model / 2), n_heads),
                    int(d_model / 2),
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(int(d_model / 2))
        )
        self.encoder_2 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.lstm = nn.LSTM(input_size=int(d_model / 2) + d_model, hidden_size=d_ff, num_layers=1, batch_first=True,
                            dropout=dropout, bidirectional=False)

        self.fc = nn.Sequential(
            nn.Linear(d_ff, 256, bias=True),
            nn.Mish(),
            nn.Linear(256, 92, bias=True),
            nn.Mish(),
            nn.Linear(92, pred_len, bias=True)
        )

    def forecast(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out_1 = self.enc_embedding_1(x_enc, None)
        enc_out_2 = self.enc_embedding_2(x_enc, None)

        enc_out_1, att1 = self.encoder_1(enc_out_1)
        enc_out_2, att2 = self.encoder_2(enc_out_2)

        # encode_fusion
        # bs time dmodel
        enc_out = torch.concat((enc_out_1, enc_out_2), dim=-1)
        #
        enc_out, _ = self.lstm(enc_out)
        dec_out = self.fc(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, C]