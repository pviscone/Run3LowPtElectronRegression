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




#TODO REMEMBER TO APPLY THE MEAN LATER
#TODO Apply sample weight directly in the train method
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
    def __init__(self, **kwargs):
        self.n_out = 2
        self.n_targets = 1
        self.activations = ["Linear", "Linear"]

    def loss(self, targets, outputs):
        mu = outputs[..., 0:1]
        sigma = variance_transformation(outputs[..., 1:2], numpy=False)
        y = targets[..., 0:1]
        loglik = -torch.log(sigma) - (torch.abs(y - mu)) / sigma
        return -loglik

    def f(self, outputs):
        return outputs[:, 0]

    def sigma(self, outputs):
        return np.sqrt(2) * variance_transformation(outputs[:, 1], numpy=True)

    def rho(self, outputs):
        return torch.zeros_like(outputs[:, 0])

    def nomalize_targets(self, Y):
        Y_mean = np.median(Y, axis=0)
        Y_std = np.mean(np.abs(Y - Y_mean))
        return Y_mean, Y_std


class BivariateL2(Loss):
    def __init__(self, **kwargs):
        self.n_out = 5
        self.activations = ["Linear", "Linear", "Linear", "Linear", "Sigmoid"]
        self.n_targets = 2

    def loss(self, targets, outputs):
        y1 = targets[..., 0:1]
        y2 = targets[..., 1:2]

        mu1 = outputs[..., 0:1]
        var1 = variance_transformation(outputs[..., 1:2], numpy=False)
        sigma1 = torch.sqrt(var1)

        mu2 = outputs[..., 2:3]
        var2 = variance_transformation(outputs[..., 3:4], numpy=False)
        sigma2 = torch.sqrt(var2)

        rho = outputs[..., 4:5]
        loglik_var = -torch.log(sigma1) - torch.log(sigma2) - 0.5*torch.log(1-torch.pow(rho,2))
        loglik_mu = - (1/(1-torch.pow(rho,2))) * ( ((y1 - mu1) ** 2) / var1 + ((y2 - mu2) ** 2) / var2 - (2*rho*(y1 - mu1)*(y2 - mu2))/(sigma1*sigma2) )

        loglik = loglik_var + loglik_mu
        return -loglik


    def f(self, outputs):
        return torch.stack((outputs[:, 0], outputs[:, 2]), axis=1)

    def sigma(self, outputs):
        return torch.stack((outputs[:, 1], outputs[:, 3]), axis=1)

    def rho(self, outputs):
        return outputs[:, 4]

    def nomalize_targets(self, Y):
        Y_mean = np.mean(Y, axis=0)
        Y_std = np.std(Y, axis=0)
        return Y_mean, Y_std


class BivariateL1(Loss):
    def __init__(self, **kwargs):
        self.n_out = 5
        self.activations = ["Linear", "Linear", "Linear", "Linear", "sigmoid"]
        self.n_targets = 2
        self.l1_loss = L1Loss()
        self.laplace_cdf = lambda x, mu, sigma : torch.where(((x-mu)/sigma) < 0, 0.5 * torch.exp(((x-mu)/sigma)), 1 - 0.5 * torch.exp(-((x-mu)/sigma)))

    def loss(self, targets, outputs):
        mu1 = outputs[..., 0:1]
        sigma1 = variance_transformation(outputs[..., 1:2], numpy=False)
        mu2 = outputs[..., 2:3]
        sigma2 = variance_transformation(outputs[..., 3:4], numpy=False)
        rho = outputs[..., 4:5]

        y1 = targets[..., 0:1]
        y2 = targets[..., 1:2]

        nll1 = self.l1_loss.loss(y1, outputs[..., 0:2])
        nll2 = self.l1_loss.loss(y2, outputs[..., 2:4])

        z1 = self.laplace_cdf(y1, mu1, sigma1)
        z2 = self.laplace_cdf(y2, mu2, sigma2)

        nll_corr = 0.5*torch.log(1-torch.pow(rho,2)) + rho*z1*z2
        return nll1 + nll2 + nll_corr

    def f(self, outputs):
        return torch.stack((outputs[:, 0], outputs[:, 2]), axis=1)

    def sigma(self, outputs):
        return np.sqrt(2) * torch.stack((outputs[:, 1], outputs[:, 3]), axis=1)

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
    "BivariateL1": BivariateL1
}