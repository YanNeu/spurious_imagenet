import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np

class TemperatureWrapper(nn.Module):
    def __init__(self, model, T=1.):
        super().__init__()

        self.train(model.training)

        self.model = model
        self.T = T

    def forward(self, x):
        logits = self.model(x)
        #try:
        #    self.T = self.T.to(logits.device)
        #except Exception as err:
        #    print(str(err))
        return logits / self.T

    @staticmethod
    def compute_temperature(model, loader, device):
        model.eval()
        logits = []
        labels = []
        with torch.no_grad():
            for data, target in loader:
                data = data.to(device)

                logits.append(model(data).detach().cpu())
                labels.append(target)

        logits = torch.cat(logits, 0)
        labels = torch.cat(labels, 0)

        ca = []
        log_T = torch.linspace(-3., 1., 2000)

        for t in log_T:
            ca.append(TemperatureWrapper._get_ece_inner(logits / np.exp(t), labels)[0])
            ece, idx = torch.stack(ca, 0).min(0)

        T = float(np.exp(log_T[idx]))
        return T

    @staticmethod
    def compute_ece(model, loader, device):
        model.eval()
        logits = []
        labels = []
        with torch.no_grad():
            for data, target in loader:
                data = data.to(device)

                logits.append(model(data).detach().cpu())
                labels.append(target)

        logits = torch.cat(logits, 0)
        labels = torch.cat(labels, 0)
        ece = TemperatureWrapper._get_ece_inner(logits, labels)[0]
        return ece

    @staticmethod
    def _get_ece_inner(logits, labels, n_bins=20):
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

