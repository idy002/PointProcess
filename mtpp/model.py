import torch
import numpy as np
from functools import partial
from scipy.integrate import quad
import json
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=12, n_classes=7):
        super(MLP, self).__init__()
        self.linear = nn.Linear(2 * input_dim, hidden_dim)
        self.event_linear = nn.Linear(hidden_dim, n_classes)
        self.time_linear = nn.Linear(hidden_dim, 1)

    def forward(self, input):
        time = input[0]  # batch x input_dim
        event = input[1].float()  # batch x input_dim
        concated = torch.cat([time, event], dim=1)
        hidden = self.linear(concated)
        time_output = self.time_linear(hidden)
        event_output = self.event_linear(hidden)
        time_output = torch.squeeze(time_output)
        return time_output, event_output

    def loss(self, input, target):
        time_output, event_output = self.forward(input)
        time_target, event_target = target
        time_loss = F.mse_loss(time_output, time_target)
        event_loss = F.cross_entropy(event_output, event_target)
        merged_loss = time_loss + event_loss
        return time_loss, event_loss, merged_loss

    def inference(self, input):
        time_output, event_output = self.forward(input)
        time_output = time_output.detach()
        event_output = event_output.detach()
        time_output = torch.squeeze(time_output)
        event_choice = torch.argmax(event_output, dim=1)
        return time_output, event_choice


class RMTPP(nn.Module):
    def __init__(self, event_classes=7, event_embed_dim=12, lstm_hidden_dim=32, hidden_dim=16, loss_alpha=0.3):
        super(RMTPP, self).__init__()
        self.event_embedding = nn.Embedding(event_classes, event_embed_dim)
        self.event_embedding_dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(event_embed_dim + 1, lstm_hidden_dim)
        self.hidden_linear = nn.Linear(lstm_hidden_dim, hidden_dim)
        self.time_linear = nn.Linear(hidden_dim, 1)
        self.event_linear = nn.Linear(hidden_dim, event_classes)
        self.loss_alpha = loss_alpha
        self.w = nn.Parameter(torch.full((1,), 0.1))
        self.b = nn.Parameter(torch.full((1,), 0.1))

    def forward(self, input):
        time = input[0]
        event = input[1]
        event_embedding = self.event_embedding(event)
        time = time[:, :, None]
        time_event = torch.cat([time, event_embedding], dim=2)
        hidden, _ = self.lstm(time_event)  # seq_len x batch x hidden_dim
        hidden = hidden[:, -1, :]
        hidden = self.hidden_linear(hidden)
        event_output = self.event_linear(hidden)
        time_output = torch.squeeze(self.time_linear(hidden))
        return time_output, event_output

    def loss(self, input, target):
        time_output, event_output = self.forward(input)
        time_target, event_target = target
        event_loss = F.cross_entropy(event_output, event_target)

        def time_nll(vh, w, b, t):
            return -(vh + w * t + b + (torch.exp(vh + b) - torch.exp(vh + w * t + b)) / w)

        time_loss = torch.mean(time_nll(time_output, self.w, self.b, time_target))
        merged_loss = self.loss_alpha * time_loss + event_loss
        return time_loss, event_loss, merged_loss

    def inference(self, input):
        time_output, event_output = self.forward(input)
        event_output = event_output.detach()
        time_output = time_output.detach()
        event_choice = torch.argmax(event_output, dim=1)

        last_time = input[2].cpu().numpy()
        time_output = time_output.cpu().numpy()

        w = self.w.detach().cpu().item()
        b = self.b.detach().cpu().item()

        time_predicted = torch.tensor([
            tj + quad(lambda t: t * np.exp(vh + w * t + b + (np.exp(vh + b) - np.exp(vh + w * t + b)) / w), a=0.0, b=10.0)[0]
            for vh, tj
            in zip(time_output, last_time)
        ])

        return time_predicted, event_choice


class ERPP(nn.Module):
    def __init__(self, event_classes=7, event_embed_dim=12, lstm_hidden_dim=32, hidden_dim=16, loss_alpha=0.05,
                 event_count=None, use_gaussian_penalty=True):
        super(ERPP, self).__init__()
        self.event_embedding = nn.Embedding(event_classes, event_embed_dim)
        self.event_embedding_dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(event_embed_dim + 1, lstm_hidden_dim)
        self.hidden_linear = nn.Linear(lstm_hidden_dim, hidden_dim)
        self.event_linear = nn.Linear(hidden_dim, event_classes)
        self.time_linear = nn.Linear(hidden_dim, 1)
        self.loss_alpha = loss_alpha
        self.event_loss_weight = torch.zeros(event_classes)
        for si, c in event_count.items():
            self.event_loss_weight[int(si)] += float(c)
        self.event_loss_weight = nn.Parameter(torch.sum(self.event_loss_weight) / self.event_loss_weight, requires_grad=False)
        self.use_gaussian_penalty = use_gaussian_penalty

    def forward(self, input):
        time = input[0]
        event = input[1]
        event_embedding = self.event_embedding(event)
        time = time[:, :, None]
        time_event = torch.cat([time, event_embedding], dim=2)
        hidden, _ = self.lstm(time_event)  # seq_len x batch x hidden_dim
        hidden = hidden[:, -1, :]
        hidden = self.hidden_linear(hidden)
        event_output = self.event_linear(hidden)
        time_output = torch.squeeze(self.time_linear(hidden))
        return time_output, event_output

    def loss(self, input, target):
        time_output, event_output = self.forward(input)
        time_target, event_target = target
        if self.use_gaussian_penalty:
            time_loss = F.mse_loss(time_output, time_target)  # -log(f) is equivalent to l2_loss here
        else:
            time_loss = F.l1_loss(time_output, time_target)
        event_loss = F.cross_entropy(event_output, event_target, self.event_loss_weight)
        merged_loss = self.loss_alpha * time_loss + event_loss
        return self.loss_alpha * time_loss, event_loss, merged_loss

    def inference(self, input):
        time_output, event_output = self.forward(input)
        event_choice = torch.argmax(event_output, dim=1)
        return time_output.detach(), event_choice


def load_model(name, args):
    name = name.lower()
    if name == 'mlp':
        return MLP()
    elif name == 'rmtpp':
        return RMTPP()
    elif name == 'erpp':
        event_count = json.load(open('data/event_count.json', 'r'))
        return ERPP(event_count=event_count)
    else:
        raise ValueError()
