import torch
import torch.nn as nn
import torch.nn.functional as F
from my_code.layers.Embed import DataEmbedding_inverted,Deform_Temporal_Embedding,DataEmbedding
from my_code.layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer
from my_code.layers.SelfAttention_Family import CBAMAttention,AttentionLayer,FullAttention,FlowAttention
import numpy as np

class DAIN_Layer(nn.Module):
    def __init__(self, mode='adaptive_avg', mean_lr=0.0001, gate_lr=0.001, scale_lr=0.0001, input_dim=120):
        super(DAIN_Layer, self).__init__()
        self.mode = mode
        self.mean_lr = mean_lr
        self.gate_lr = gate_lr
        self.scale_lr = scale_lr

        # Parameters for adaptive average
        self.mean_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.mean_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))

        # Parameters for adaptive std
        self.scaling_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.scaling_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))
        # Parameters for adaptive scaling
        self.gating_layer = nn.Linear(input_dim, input_dim)
        self.eps = 1e-8

    def forward(self, x):
        # Expecting  (n_samples, dim,  n_feature_vectors)
        # Nothing to normalize
        if self.mode == None:
            pass
        # Do simple average normalization
        elif self.mode == 'avg':
            avg = torch.mean(x, 2)
            avg = avg.resize(avg.size(0), avg.size(1), 1)
            x = x - avg

        # Perform only the first step (adaptive averaging)
        elif self.mode == 'adaptive_avg':
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.resize(adaptive_avg.size(0), adaptive_avg.size(1), 1)
            x = x - adaptive_avg
        # Perform the first + second step (adaptive averaging + adaptive scaling )
        elif self.mode == 'adaptive_scale':
            # Step 1:
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.resize(adaptive_avg.size(0), adaptive_avg.size(1), 1)
            x = x - adaptive_avg

            # Step 2:
            std = torch.mean(x ** 2, 2)
            std = torch.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            adaptive_std[adaptive_std <= self.eps] = 1
            adaptive_std = adaptive_std.resize(adaptive_std.size(0), adaptive_std.size(1), 1)
            x = x / (adaptive_std)

        elif self.mode == 'full':
            # Step 1:
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.resize(adaptive_avg.size(0), adaptive_avg.size(1), 1)
            x = x - adaptive_avg
            # # Step 2:
            std = torch.mean(x ** 2, 2)
            std = torch.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            adaptive_std[adaptive_std <= self.eps] = 1
            adaptive_std = adaptive_std.resize(adaptive_std.size(0), adaptive_std.size(1), 1)
            x = x / adaptive_std
            # Step 3: 
            avg = torch.mean(x, 2)
            gate = F.sigmoid(self.gating_layer(avg))
            gate = gate.resize(gate.size(0), gate.size(1), 1)
            x = x * gate
        else:
            assert False
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x*self.sigmoid(out)

class AdaptiveLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(AdaptiveLayerNorm, self).__init__()
        self.eps = eps
        self.gamma_net = nn.Sequential(
            nn.Linear(normalized_shape, normalized_shape),
            nn.ReLU(),
            nn.Linear(normalized_shape, normalized_shape)
        )
        self.beta_net = nn.Sequential(
            nn.Linear(normalized_shape, normalized_shape),
            nn.ReLU(),
            nn.Linear(normalized_shape, normalized_shape)
        )
 
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        gamma = self.gamma_net(x)
        beta = self.beta_net(x)
        x_normalized = (x - mean) / (std + self.eps)
        return gamma * x_normalized + beta

class STAR(nn.Module):
    def __init__(self, d_series, d_core):
        super(STAR, self).__init__()
        """
        STar Aggregate-Redistribute Module
        """
        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, input, *args, **kwargs):
        batch_size, channels, d_series = input.shape
        # set FFN
        combined_mean = F.gelu(self.gen1(input))
        combined_mean = self.gen2(combined_mean)
        # stochastic pooling
        # if self.training:
        #     ratio = F.softmax(combined_mean, dim=1)
        #     ratio = ratio.permute(0, 2, 1)
        #     ratio = ratio.reshape(-1, channels)
        #     indices = torch.multinomial(ratio, 1)
        #     indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)
        #     combined_mean = torch.gather(combined_mean, 1, indices)
        #     combined_mean = combined_mean.repeat(1, channels, 1)
        # else:
        weight = F.softmax(combined_mean, dim=1)
        combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True).repeat(1, channels, 1)
        # mlp fusion
        combined_mean_cat = torch.cat([input, combined_mean], -1)
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat))
        combined_mean_cat = self.gen4(combined_mean_cat)
        output = combined_mean_cat
        return output, None

class SOFTS_ATTN(nn.Module):
    def __init__(self,seq_len,enc_in,enc_d_model,enc_d_core,pred_len,d_model,dropout,use_norm,d_core,d_ff,activation,e_layers,n_heads=1):
        super(SOFTS_ATTN, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        # Embedding
        self.time_dain = DAIN_Layer(input_dim=self.seq_len)
        self.feature_dain = DAIN_Layer(input_dim=enc_in)
        self.time_attn = CBAMAttention(d_model,5,7)
        self.enc_embedding = DataEmbedding_inverted(seq_len,d_model,dropout)
        self.feature_embedding = DataEmbedding(enc_in,enc_d_model,dropout)
        self.use_norm = use_norm
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    STAR(d_model,d_core),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                ) for l in range(e_layers)
            ],
            norm_layer=AdaptiveLayerNorm(d_model)
        )
        self.feature_encoder =Encoder(
            [
                EncoderLayer(
                    STAR(enc_d_model,enc_d_core),
                    enc_d_model,
                    enc_d_model*2,
                    dropout=dropout,
                    activation=activation,
                ) for l in range(e_layers)
            ],
            norm_layer=AdaptiveLayerNorm(enc_d_model)
        )
        # Decoder
        #self.dct_layer=dct_channel_block(28)
        #self.dct_norm = nn.LayerNorm([28], eps=1e-6)
        self.projection = nn.Linear(d_model,pred_len,bias=True)
        self.fc_prejection = nn.Linear(enc_d_model,enc_in, bias=True)

    def forecast(self, x_enc):
        # Normalization from Non-stationary Transformer
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        _, _, N = x_enc.shape
        #split
        x_enc = self.time_dain(x_enc)
        x_enc = self.feature_dain(x_enc.permute(0,2,1)).permute(0,2,1)
        enc_out = self.enc_embedding(x_enc, None)
        enc_out = enc_out.permute(0,2,1)
        enc_out = self.feature_embedding(enc_out)
        # mid  = self.dct_layer(enc_out)
        # enc_out = enc_out+mid
        # enc_out = self.dct_norm(enc_out)
        enc_out = enc_out.permute(0,2,1)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = enc_out.permute(0,2,1)
        enc_out, attns =self.feature_encoder(enc_out,attn_mask=None)
        enc_out = enc_out.unsqueeze(-1)
        enc_out = self.time_attn(enc_out).squeeze(-1).permute(0,2,1)
        dec_out = self.projection(enc_out).permute(0,2,1)
        dec_out = self.fc_prejection(dec_out) 
        # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc,mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]