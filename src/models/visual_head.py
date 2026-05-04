import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MaskedNorm(nn.Module):
    def __init__(self, num_features, norm_type='batch'):
        super().__init__()
        self.norm_type = norm_type
        if norm_type == 'batch':
            self.norm = nn.BatchNorm1d(num_features)
        else:
            self.norm = nn.LayerNorm(num_features)

    def forward(self, x, mask=None):
        if self.norm_type == 'batch':
            x = x.transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.norm(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, input_size, ff_size, dropout=0.1, kernel_size=[3,3], skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection
        k1, k2 = kernel_size[0], kernel_size[1]
        self.conv1 = nn.Conv1d(input_size, ff_size, k1, padding=k1//2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(ff_size, input_size, k2, padding=k2//2)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        h = self.conv1(x_t)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = h.transpose(1, 2)
        if self.skip_connection:
            return x + h
        return h


class VisualHead(nn.Module):
    def __init__(self, cls_num, input_size=512, hidden_size=512, ff_size=2048, pe=True,
                 ff_kernelsize=[3,3], pretrained_ckpt=None, is_empty=False):
        super().__init__()
        self.is_empty = is_empty
        if not is_empty:
            self.hidden_size = hidden_size
            if input_size is None:
                self.fc1 = nn.Identity()
            else:
                self.fc1 = nn.Linear(input_size, hidden_size)
            self.bn1 = MaskedNorm(num_features=hidden_size, norm_type='batch')
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(p=0.1)
            if pe:
                self.pe = PositionalEncoding(hidden_size)
            else:
                self.pe = nn.Identity()
            self.feedforward = PositionwiseFeedForward(
                input_size=hidden_size, ff_size=ff_size,
                dropout=0.1, kernel_size=ff_kernelsize, skip_connection=True
            )
            self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
            self.gloss_output_layer = nn.Linear(hidden_size, cls_num)
            nn.init.constant_(self.gloss_output_layer.bias, 0.0)
        else:
            self.gloss_output_layer = nn.Linear(input_size, cls_num)

    def forward(self, x, mask=None, valid_len_in=None):
        B, Tin, D = x.shape
        if not self.is_empty:
            x = self.fc1(x)
            x = self.bn1(x, mask)
            x = self.relu1(x)
            x = self.dropout1(x)
            x = self.pe(x)
            x = self.feedforward(x)
            x = self.layer_norm(x)
        logits = self.gloss_output_layer(x)
        gloss_probabilities_log = logits.log_softmax(2)
        gloss_probabilities = logits.softmax(2)
        return {
            'gloss_feature': x,
            'gloss_logits': logits,
            'gloss_probabilities_log': gloss_probabilities_log,
            'gloss_probabilities': gloss_probabilities,
            'valid_len_out': valid_len_in
        }
