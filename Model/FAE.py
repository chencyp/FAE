import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim import lr_scheduler
from center_loss_fcm import CenterLoss
import torch.optim as optim


class FAE(nn.Module):
    """
    Fuzzy Autoencoder with center loss and fuzzy weighting.
    """
    def __init__(self, input_dim: int = None, classes: int = None, device=None, **kwargs):
        super(FAE, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device

        self.cfg = OmegaConf.create({
            "classes": None,
            "input_dim": None,
            "k_selected": 50,
            "neighbors_k": 10,
            "decoder_lr": 1e-3,
            "selector_lr": 1e-3,
            "center_lr": 1e-3,
            "min_lr": 1e-5,
            "weight_decay": 0,
            "batch_size": 64,
            "hidden_dim": 128,
            "model": 'FAE',
            "start_temp": 10,
            "min_temp": 1e-2,
            "rec_lambda": 1,
            "fr_penalty": 0,
            "num_epochs": 500,
            "verbose": True
        })

        self.cfg.input_dim = input_dim
        self.cfg.classes = classes
        self.cfg.update((key, kwargs['kwargs'][key]) for key in self.cfg.keys() if key in kwargs['kwargs'].keys())

        self.model = self.cfg.model

        self.selector = SelectLayer(self.cfg).to(self.device)
        self.decoder = Decoder(self.cfg).to(self.device)
        self.center = CenterLoss(self.cfg.classes, self.cfg.k_selected).to(self.device)

        self.optim_center = optim.Adam(self.center.parameters(), lr=self.cfg.center_lr, betas=(0.5, 0.999))

        self.optim = optim.Adam([
            {'params': self.decoder.parameters(), 'lr': self.cfg.decoder_lr},
            {'params': self.selector.parameters(), 'lr': self.cfg.selector_lr}
        ], lr=self.cfg.decoder_lr, betas=(0.5, 0.999), weight_decay=self.cfg.weight_decay)

        self.scheduler = lr_scheduler.LambdaLR(self.optim, lr_lambda=self.lambda_rule)

    def forward(self, x, epoch=None):
        selected_feature, weights = self.selector(x, epoch)
        output, hidden = self.decoder(selected_feature)
        return weights, hidden, output

    def get_selected_feats(self):
        return self.selector.get_selected_feats().detach().cpu().numpy()

    def get_selection_probs(self, epoch):
        return self.selector.get_weights(epoch=epoch).detach().cpu().numpy()

    @staticmethod
    def lambda_rule(i) -> float:
        lr_decay_factor = .1
        decay_step_size = 100
        exponent = int(np.floor((i + 1) / decay_step_size))
        return np.power(lr_decay_factor, exponent)

    def update_lr(self):
        self.scheduler.step()
        lr = self.optim.param_groups[0]['lr']
        if lr < self.cfg.min_lr:
            self.optim.param_groups[0]['lr'] = self.cfg.min_lr

    def compute_ratio(self, X, y):
        distances = torch.cdist(X, X)
        distances.fill_diagonal_(float('inf'))
        _, indices = torch.topk(distances, self.cfg.neighbors_k, largest=False, dim=1)
        same_label_count = torch.sum(y[indices] == y.unsqueeze(1), dim=1).float()
        ratio = (self.cfg.neighbors_k - same_label_count) / self.cfg.neighbors_k
        return ratio

    def compute_membership(self, X, y):
        m = 2
        N, d = X.size()
        centers = torch.zeros(self.cfg.classes, d, dtype=X.dtype, device=X.device)
        for i in range(self.cfg.classes):
            X_class = X[y == i]
            centers[i] = torch.mean(X_class, dim=0) if len(X_class) > 0 else torch.full((d,), float('inf'), device=X.device)

        distances = torch.cdist(X, centers)
        inv_distances = 1.0 / (distances + 1e-8)
        memberships = inv_distances ** (2.0 / (m - 1))
        memberships = memberships / torch.sum(memberships, dim=1, keepdim=True)

        proportions = self.compute_ratio(X, y)
        weights = memberships[torch.arange(memberships.size(0)), y]
        weights_n = (1 - weights) * proportions

        membership = torch.where(weights_n == 0, weights,
                                 torch.where(weights <= weights_n, torch.tensor(0.0, device=X.device),
                                             (1 - weights_n) / (2 - weights_n - weights)))
        return membership

    def train_step(self, x, y, epoch):
        weights, hidden, output = self(x, epoch)
        center_loss = self.center(y, hidden)
        recon_criterion = nn.MSELoss(reduction='none').to(self.device)
        loss_recon = recon_criterion(output, x).mean(dim=1)
        weights = self.compute_membership(x, y)
        recon_loss = (loss_recon * weights).mean()

        if self.model == 'cae':
            loss = recon_loss
        elif self.model == 'center':
            loss = center_loss
        elif self.model == 'FAE':
            loss = recon_loss + self.cfg.rec_lambda * center_loss

        fr_penalty = self.selector.regularization(epoch) * self.cfg.fr_penalty
        loss += fr_penalty

        self.optim_center.zero_grad()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.optim_center.step()

        return loss.item(), center_loss.item(), recon_loss.item()

    def select_features(self, dataloader):
        for epoch in range(self.cfg.num_epochs):
            losses, center_losses, recon_losses = [], [], []
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                loss, c_loss, r_loss = self.train_step(data, target, epoch)
                losses.append(loss)
                center_losses.append(c_loss)
                recon_losses.append(r_loss)
            self.update_lr()

        print('Finished training')
        selected_features = sorted(set(self.get_selected_feats()))
        print('Selected features:', selected_features)
        return selected_features


class SelectLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_features = cfg.input_dim
        self.output_features = cfg.k_selected
        self.num_epochs = cfg.num_epochs
        self.start_temp = cfg.start_temp
        self.min_temp = torch.tensor(cfg.min_temp)
        self.min_thr = torch.tensor(1 / 50)
        self.start_thr = torch.tensor(1.0)
        self.logits = torch.nn.Parameter(torch.zeros(self.input_features, self.output_features), requires_grad=True)

    def current_temp(self, epoch, sched_type='exponential'):
        schedules = {
            'exponential': torch.max(self.min_temp, self.start_temp * ((self.min_temp / self.start_temp) ** (epoch / self.num_epochs))),
            'linear': torch.max(self.min_temp, self.start_temp - (self.start_temp - self.min_temp) * (epoch / self.num_epochs)),
            'cosine': self.min_temp + 0.5 * (self.start_temp - self.min_temp) * (1. + np.cos(epoch * math.pi / self.num_epochs))
        }
        return schedules[sched_type]

    def current_threshold(self, epoch, sched_type='exponential'):
        schedules = {
            'exponential': torch.max(self.min_thr, self.start_thr * ((self.min_thr / self.start_thr) ** (epoch / self.num_epochs))),
            'linear': torch.max(self.min_thr, self.start_thr - (self.start_thr - self.min_thr) * (epoch / self.num_epochs)),
            'cosine': self.min_thr + 0.5 * (self.start_thr - self.min_thr) * (1. + np.cos(epoch * math.pi / self.num_epochs))
        }
        return schedules[sched_type]

    def forward(self, x, epoch=None):
        from torch.distributions.uniform import Uniform
        uniform_pdfs = Uniform(low=1e-6, high=1.).sample(self.logits.size()).to(x.device)
        gumbel = -torch.log(-torch.log(uniform_pdfs))

        if self.training:
            temp = self.current_temp(epoch)
            noisy_logits = (self.logits + gumbel) / temp
            weights = F.softmax(noisy_logits / temp, dim=0)
            x = x @ weights
        else:
            weights = F.one_hot(torch.argmax(self.logits, dim=0), self.input_features).float()
            x = x @ weights.T
        return x, weights

    def regularization(self, epoch=None):
        eps = 1e-10
        threshold = self.current_threshold(epoch)
        p = torch.clamp(torch.softmax(self.logits, dim=0), eps, 1)
        ds = torch.sum(F.relu(torch.norm(p, 1, dim=1) - threshold))
        return ds

    def get_weights(self, epoch):
        temp = self.current_temp(epoch)
        return F.softmax(self.logits / temp, dim=0)

    def get_selected_feats(self):
        return torch.argmax(self.logits, dim=0)


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.input_features = cfg.k_selected
        self.hidden_dim = cfg.input_dim
        self.output_features = cfg.input_dim

        self.layer1 = nn.Sequential(nn.Linear(self.input_features, self.hidden_dim))
        self.layer2 = nn.Sequential(nn.Linear(self.hidden_dim, self.input_features))
        self.layer3 = nn.Sequential(nn.Linear(self.input_features, self.output_features))

    def forward(self, x):
        hidden = self.layer1(x)
        mid = self.layer2(hidden)
        output = self.layer3(mid)
        return output, mid
