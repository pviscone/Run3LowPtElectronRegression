from abc import ABC, abstractmethod
import torch
import numpy as np

from MVENet.nn_utils import variance_transformation

class LossRecord:
    def __init__(self):
        self.loss_total = []
        self.loss_terms=[]

    def append(self, terms):
        if not isinstance(terms, list) or not isinstance(terms, tuple):
            terms = [terms]
        self.loss_total.append(sum(terms))
        self.loss_terms.append(terms)

    def __getitem__(self, idx):
        return self.loss_total[idx]

    def get_term(self, idx):
        return [terms[idx] for terms in self.loss_terms]



class Loss(ABC):
    @abstractmethod
    def loss(self):
        pass

    @abstractmethod
    def f(self):
        pass

    @abstractmethod
    def sigma(self):
        pass


class BetaLoss(Loss):
    def __init__(self, beta = 0, **kwargs):
        self.beta = beta
        self.n_out = 2
        self.n_targets = 1
        self.activations = ["Linear", "Linear"]

    def loss(self, targets, outputs):
        mu = outputs[..., 0:1]
        var = variance_transformation(outputs[..., 1:2], numpy=False)
        loss = ((targets - mu) ** 2) / var + torch.log(var)
        loss = loss * var.detach() ** self.beta
        return loss

    def f(self, outputs):
        return outputs[:, 0]

    def sigma(self, outputs):
        return np.sqrt(variance_transformation(outputs[:, 1], numpy=True))

    def rho(self, outputs):
        return torch.zeros_like(outputs[:, 0])

class L2Loss(Loss):
    def __init__(self, **kwargs):
        self.n_out = 2
        self.n_targets = 1
        self.activations = ["Linear", "Linear"]

    def loss(self, targets, outputs):
        mu = outputs[..., 0:1]
        var = variance_transformation(outputs[..., 1:2], numpy=False)
        y = targets[..., 0:1]
        loglik = -torch.log(var) - ((y - mu) ** 2) / var
        return -loglik

    def f(self, outputs):
        return outputs[:, 0]

    def sigma(self, outputs):
        return np.sqrt(variance_transformation(outputs[:, 1], numpy=True))

    def rho(self, outputs):
        return torch.zeros_like(outputs[:, 0])

    def nomalize_targets(self, Y):
        Y_mean = np.mean(Y, axis=0)
        Y_std = np.std(Y, axis=0)
        return Y_mean, Y_std


class L1Loss(Loss):
    def __init__(self, *args, **kwargs):
        self.n_out = 2
        self.n_targets = 1
        self.activations = ["Linear", "Linear"]

    def loss(self, targets, outputs):
        mu = outputs[..., 0:1]
        logsigma = outputs[..., 1:2]
        sigma = variance_transformation(logsigma, numpy=False)
        y = targets[..., 0:1]
        loglik = -logsigma - torch.sqrt(torch.tensor(2.0, device=outputs.device)) * (torch.abs(y - mu)) / sigma
        return -loglik

    def f(self, outputs):
        return outputs[:, 0]

    def sigma(self, outputs):
        return variance_transformation(outputs[:, 1], numpy=True)

    def rho(self, outputs):
        return torch.zeros_like(outputs[:, 0])

    def nomalize_targets(self, Y):
        Y_mean = np.median(Y, axis=0)
        Y_std = np.mean(np.abs(Y - Y_mean))
        return Y_mean, Y_std


class QuantileLoss(Loss):
    def __init__(self, quantiles, **kwargs):
        self.n_out = len(quantiles)
        self.n_targets = 1
        self.activations = ["Linear"] + ["Softplus"] * (self.n_out-1) #lean log quantiles
        self.quantiles = quantiles
        self.qs = torch.tensor(quantiles, device='cuda' if torch.cuda.is_available() else 'cpu').view(1, -1)

    def loss(self, targets, outputs):
        y = targets[..., 0:1]                                    # [B,1]
        qs = outputs.new_tensor(self.quantiles).view(1, -1)      # [1,Q]
        errors = y - outputs                                     # [B,Q]
        per_q = torch.maximum(qs * errors, (qs - 1.0) * errors)  # [B,Q]
        return per_q.mean(dim=-1)                                # [B]

    def nomalize_targets(self, Y):
        Y_mean = np.median(Y, axis=0)
        Y_std = np.mean(np.abs(Y - Y_mean))
        return Y_mean, Y_std

    def f(self, outputs, numpy=True):
        pass

    def sigma(self, outputs, numpy=True):
        pass



class BivariateL2(Loss):
    def __init__(self, **kwargs):
        self.n_out = 5
        self.activations = ["Softplus", "Linear", "Softplus", "Linear", "Tanh"]
        self.n_targets = 2

    #outputs: mu1, log sigma1, mu2, log sigma2, rho
    def loss(self, targets, outputs):
        y1 = targets[..., 0:1]
        y2 = targets[..., 1:2]

        mu1 = outputs[..., 0:1]
        logsigma1 = outputs[..., 1:2]
        sigma1 = variance_transformation(logsigma1, numpy=False)

        mu2 = outputs[..., 2:3]
        logsigma2 = outputs[..., 3:4]
        sigma2 = variance_transformation(logsigma2, numpy=False)

        rho = outputs[..., 4:5]
        rho = torch.clamp(rho, min=-0.999, max=0.999)

        z1 = (y1 - mu1) / sigma1
        z2 = (y2 - mu2) / sigma2

        zSz = ((z1**2) - 2*rho*z1*z2 + (z2**2))/(1-torch.pow(rho,2))

        loglik = - logsigma1 - logsigma2 - 0.5*torch.log1p(-torch.pow(rho,2)) - 0.5 * zSz
        return -loglik



    def f(self, outputs, numpy=True):
        if numpy:
            return np.stack((outputs[:, 0], outputs[:, 2]), axis=1)
        else:
            return torch.stack((outputs[:, 0], outputs[:, 2]), axis=1)

    def sigma(self, outputs, numpy=True):
        return variance_transformation(outputs[:, [1, 3]],numpy=numpy)

    def rho(self, outputs):
        return outputs[:, 4]

    def nomalize_targets(self, Y):
        Y_mean = np.mean(Y, axis=0)
        Y_std = np.std(Y, axis=0)
        return Y_mean, Y_std


class BivariateL1(Loss):
    def __init__(self, **kwargs):
        self.n_out = 5
        self.activations = ["Softplus", "Linear", "Softplus", "Linear", "Tanh"]
        self.n_targets = 2

    def loss(self, targets, outputs):
        y1 = targets[..., 0:1]
        y2 = targets[..., 1:2]

        mu1 = outputs[..., 0:1]
        logsigma1 = outputs[..., 1:2]
        sigma1 = variance_transformation(logsigma1, numpy=False)

        mu2 = outputs[..., 2:3]
        logsigma2 = outputs[..., 3:4]
        sigma2 = variance_transformation(logsigma2, numpy=False)

        rho = outputs[..., 4:5]
        rho = torch.clamp(rho, min=-0.999, max=0.999)

        z1 = (y1 - mu1) / sigma1
        z2 = (y2 - mu2) / sigma2

        zSz = ((z1**2) - 2*rho*z1*z2 + (z2**2))/(1-torch.pow(rho,2))

        bessel_term = torch.special.modified_bessel_k0(torch.sqrt(2*zSz))
        bessel_term = torch.clamp(bessel_term, min=1e-15, max=1e15)

        loglik = - logsigma1 - logsigma2 - 0.5*torch.log1p(-torch.pow(rho,2)) + torch.log(bessel_term)

        return -loglik

    def f(self, outputs, numpy=True):
        if numpy:
            return np.stack((outputs[:, 0], outputs[:, 2]), axis=1)
        else:
            return torch.stack((outputs[:, 0], outputs[:, 2]), axis=1)

    def sigma(self, outputs, numpy=True):
        return variance_transformation(outputs[:, [1, 3]],numpy=numpy)

    def rho(self, outputs):
        return outputs[:, 4]

    def nomalize_targets(self, Y):
        Y_mean = np.median(Y, axis=0)
        Y_std = np.mean(np.abs(Y - Y_mean))
        return Y_mean, Y_std


class MahalanobisL1(Loss):
    def __init__(self, **kwargs):
        self.n_out = 5
        self.activations = ["Softplus", "Linear", "Softplus", "Linear", "Tanh"]
        self.n_targets = 2

    def loss(self, targets, outputs):
        y1 = targets[..., 0:1]
        y2 = targets[..., 1:2]

        mu1 = outputs[..., 0:1]
        logsigma1 = outputs[..., 1:2]
        sigma1 = variance_transformation(logsigma1, numpy=False)

        mu2 = outputs[..., 2:3]
        logsigma2 = outputs[..., 3:4]
        sigma2 = variance_transformation(logsigma2, numpy=False)

        rho = outputs[..., 4:5]
        rho = torch.clamp(rho, min=-0.999, max=0.999)

        z1 = (y1 - mu1) / sigma1
        z2 = (y2 - mu2) / sigma2

        zSz = ((z1**2) - 2*rho*z1*z2 + (z2**2))/(1-torch.pow(rho,2))
        zSz = torch.clamp(zSz, min=1e-15)

        loglik = - logsigma1 - logsigma2 - 0.5*torch.log1p(-torch.pow(rho,2)) - torch.sqrt(zSz)

        return -loglik

    def f(self, outputs, numpy=True):
        if numpy:
            return np.stack((outputs[:, 0], outputs[:, 2]), axis=1)
        else:
            return torch.stack((outputs[:, 0], outputs[:, 2]), axis=1)

    def sigma(self, outputs, numpy=True):
        return variance_transformation(outputs[:, [1, 3]],numpy=numpy)

    def rho(self, outputs):
        return outputs[:, 4]

    def nomalize_targets(self, Y):
        Y_mean = np.median(Y, axis=0)
        Y_std = np.mean(np.abs(Y - Y_mean))
        return Y_mean, Y_std

class Rotated2DL1(Loss):
    def __init__(self, **kwargs):
        self.n_out = 5
        self.activations = ["Softplus", "Linear", "Softplus", "Linear", "Tanh"]
        self.n_targets = 2

    def loss(self, targets, outputs):
        y1 = targets[..., 0:1]
        y2 = targets[..., 1:2]

        mu1 = outputs[..., 0:1]
        logsigma1 = outputs[..., 1:2]
        sigma1 = variance_transformation(logsigma1, numpy=False)

        mu2 = outputs[..., 2:3]
        logsigma2 = outputs[..., 3:4]
        sigma2 = variance_transformation(logsigma2, numpy=False)

        rho = outputs[..., 4:5]
        rho = torch.clamp(rho, min=-0.999, max=0.999)

        s1 = (y1 - mu1)
        s2 = (y2 - mu2)
        z1 = torch.abs(s1-rho*s2)/ sigma1
        z2 = torch.abs(s2)/ (sigma2 * torch.sqrt(1 - rho**2))


        loglik = - logsigma1 - logsigma2 - 0.5*torch.log1p(-torch.pow(rho,2)) - z1 - z2

        return -loglik

    def f(self, outputs, numpy=True):
        if numpy:
            return np.stack((outputs[:, 0], outputs[:, 2]), axis=1)
        else:
            return torch.stack((outputs[:, 0], outputs[:, 2]), axis=1)

    def sigma(self, outputs, numpy=True):
        return variance_transformation(outputs[:, [1, 3]],numpy=numpy)

    def rho(self, outputs):
        return outputs[:, 4]

    def nomalize_targets(self, Y):
        Y_mean = np.median(Y, axis=0)
        Y_std = np.mean(np.abs(Y - Y_mean))
        return Y_mean, Y_std


loss_dictionary = {
    "BetaLoss": BetaLoss,
    "L2Loss": L2Loss,
    "L1Loss": L1Loss,
    "BivariateL2": BivariateL2,
    "BivariateL1": BivariateL1,
    "MahalanobisL1": MahalanobisL1,
    "Rotated2DL1": Rotated2DL1,
    "QuantileLoss": QuantileLoss,
}