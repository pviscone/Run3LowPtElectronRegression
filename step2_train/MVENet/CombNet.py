import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from MVENet import Losses
from MVENet import nn_utils


class CombNet(nn.Module):
    def __init__(
        self,
        *,
        input_shape,
        n_hidden,
        dropout_input=0.,
        dropout = 0.,
        loss,
        loss_kwargs={},
        n_epochs=0,
        device=None,
        features=None,
    ):
        super().__init__()
        if features is not None:
            assert len(features) == input_shape, (
                "Input shape must match number of features"
            )
        self._normalization = False

        self.model_kwargs = {
            "input_shape": input_shape,
            "n_hidden": n_hidden,
            "dropout_input": dropout_input,
            "dropout": dropout,
            "loss_kwargs": loss_kwargs,
            "n_epochs": n_epochs,
            "features": features,
            "loss": loss
        }
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.train_loss = []
        self.val_loss = []
        self.fit_kwargs = []
        self._normalization = False
        self.loss = getattr(Losses, loss)()
        self.freezed = []

        self.input_shape = input_shape
        layers = []
        in_features = input_shape
        # Dropout input
        if dropout_input > 0:
            layers.append(nn.Dropout(dropout_input))

        # Common layers
        for n in n_hidden:
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(in_features, n))
            layers.append(nn.ReLU())
            in_features = n
        layers.append(nn.Linear(in_features, 2))
        layers.append(nn.ReLU())
        self.common = nn.Sequential(*layers)

        self.to(self._device)

    def forward(self, x):
        return self.common(x)

    def normalize(self, X):
        self._normalization = True
        self._X_mean = np.mean(X, axis=0)
        self._X_std = np.std(X, axis=0)

    def f(self, X_test, quantile_low=0.16, quantile_high=0.84):
        """Return the correlation prediction"""
        if X_test.ndim == 1:
            X_test = X_test[:, None]
        if self._normalization:
            X_test = nn_utils.normalize(X_test, self._X_mean, self._X_std)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self._device)

        self.eval()
        with torch.no_grad():
            return self(X_test).detach().cpu().numpy()

    def show(self):
        print(self)

    def compile(self):
        self = torch.compile(self)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(
            self.state_dict(), os.path.join(path, f"model{self.__class__.__name__}.pt")
        )

        with open(os.path.join(path, "model_kwargs.pkl"), "wb") as f:
            pickle.dump(self.model_kwargs, f)
        with open(os.path.join(path, "fit_kwargs.pkl"), "wb") as f:
            pickle.dump(self.fit_kwargs, f)
        if self._normalization:
            np.savez(
                os.path.join(path, "normalization.npz"),
                X_mean=self._X_mean,
                X_std=self._X_std,
            )

    @classmethod
    def load(cls, path, device=None, n_epochs=None):
        with open(os.path.join(path, "model_kwargs.pkl"), "rb") as f:
            model_kwargs = pickle.load(f)
        model = cls(**model_kwargs, device=device)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        with open(os.path.join(path, "fit_kwargs.pkl"), "rb") as f:
            model.fit_kwargs = pickle.load(f)

        norm_path = os.path.join(path, "normalization.npz")
        if os.path.exists(norm_path):
            norm = np.load(norm_path)
            model._normalization = True
            model._X_mean = norm["X_mean"]
            model._X_std = norm["X_std"]

        model.load_state_dict(
            torch.load(
                os.path.join(path, f"model{cls.__name__}.pt"), map_location=device
            )
        )
        return model

    def freeze(self, patterns):
        """Freeze parameters matching patterns (list of regex)"""
        import re

        for name, param in self.named_parameters():
            for pattern in patterns:
                if re.match(pattern, name):
                    param.requires_grad = False
                    print(f"Freezing {name}")
                    self.freezed.append(name)
                    break

    def unfreeze(self, patterns=None):
        """Unfreeze parameters matching patterns (list of regex)"""
        import re

        for name, param in self.named_parameters():
            if patterns is None:
                param.requires_grad = True
                if name in self.freezed:
                    print(f"Unfreezing {name}")
                    self.freezed.remove(name)
            else:
                for pattern in patterns:
                    if re.match(pattern, name):
                        param.requires_grad = True
                        print(f"Unfreezing {name}")
                        self.freezed.remove(name)
                        break

    def prepare_tensors(
        self, X_train, Y_train, X_val=None, Y_val=None, sample_weight=None
    ):
        validation = X_val is not None and Y_val is not None
        if X_train.ndim == 1:
            X_train = X_train[:, None]
        if Y_train.ndim == 1:
            Y_train = Y_train[:, None]
        if validation:
            if X_val.ndim == 1:
                X_val = X_val[:, None]
            if Y_val.ndim == 1:
                Y_val = Y_val[:, None]

        if self._normalization:
            X_train = nn_utils.normalize(X_train, self._X_mean, self._X_std)
            if validation:
                X_val = nn_utils.normalize(X_val, self._X_mean, self._X_std)

        X_train = torch.tensor(X_train, dtype=torch.float32).to(self._device)
        Y_train = torch.tensor(Y_train, dtype=torch.float32).to(self._device)
        if validation:
            X_val = torch.tensor(X_val, dtype=torch.float32).to(self._device)
            Y_val = torch.tensor(Y_val, dtype=torch.float32).to(self._device)

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float32)
            assert sample_weight.shape[0] == X_train.shape[0], (
                "Sample weight length must match number of samples"
            )
            sample_weight = torch.tensor(sample_weight, dtype=torch.float32).to(
                self._device
            )
        return X_train, Y_train, X_val, Y_val, sample_weight

    def get_loss(self):
        train_loss = [
            item for fit_kwarg in self.fit_kwargs for item in fit_kwarg["train_loss"]
        ]
        val_loss = [
            item
            for fit_kwarg in self.fit_kwargs
            for item in fit_kwarg.get("val_loss", [])
        ]
        return train_loss, val_loss

    def init_biases(self):
        with torch.no_grad():
            self.common[-2].bias.data = torch.ones_like(self.common[-2].bias.data) * 0.5


    def train_model(
        self,
        X_train,
        Y_train,
        X_val=None,
        Y_val=None,
        sample_weight=None,  # Accept sample weights
        learn_rate=0.001,
        n_epochs=100,
        batch_size=32,
        verbose=True,
        reg=0.,
        lambda_var=0.,
        weight_decay=0.01,
        max_norm=None,
        checkpoint=None,
        callback_every=None,
        callback=None,
        savefolder="model_checkpoints",
        diagnostics=False,
    ):
        if (
            len(self.fit_kwargs) > 0
            and learn_rate == self.fit_kwargs[-1]["learn_rate"]
            and batch_size == self.fit_kwargs[-1]["batch_size"]
            and reg == self.fit_kwargs[-1]["reg"]
            and weight_decay == self.fit_kwargs[-1]["weight_decay"]
            and max_norm == self.fit_kwargs[-1]["max_norm"]
            and self.freezed == self.fit_kwargs[-1]["freezed"]
        ):
            self.fit_kwargs[-1]["n_epochs"] += n_epochs
        else:
            fit_kwarg = {
                "learn_rate": learn_rate,
                "n_epochs": n_epochs,
                "batch_size": batch_size,
                "reg": reg,
                "lambda_var": lambda_var,
                "weight_decay": weight_decay,
                "max_norm": max_norm,
                "train_loss": [],
                "val_loss": [],
                "freezed": self.freezed.copy(),
            }
            self.fit_kwargs.append(fit_kwarg)

        validation = X_val is not None and Y_val is not None
        initial_epoch = self.model_kwargs["n_epochs"]

        X_train, Y_train, X_val, Y_val, sample_weight = self.prepare_tensors(
            X_train, Y_train, X_val, Y_val, sample_weight
        )

        if self.model_kwargs["n_epochs"] == 0:
            self.init_biases()

        optimizer = torch.optim.AdamW(self.parameters(), lr=learn_rate, weight_decay=weight_decay)

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            self.train(True)
            idx = np.arange(X_train.shape[0])
            np.random.shuffle(idx)
            for start in range(0, X_train.shape[0], batch_size):
                end = start + batch_size
                batch_idx = idx[start:end]
                xb = X_train[batch_idx]
                yb = Y_train[batch_idx]
                wb = sample_weight[batch_idx] if sample_weight is not None else None
                optimizer.zero_grad()
                out = self(xb)

                loss_samples = self.loss.loss(yb, out)
                if wb is not None:
                    loss_samples = loss_samples * wb.unsqueeze(-1)
                loss = loss_samples.mean()
                epoch_loss += loss.item() * xb.shape[0]

                l2_common = 0.0
                var_penalty = 0.0
                if reg > 0:
                    l2_common = reg * nn_utils.get_l2_reg(self.common)
                if lambda_var >0:
                    rb = torch.sum(out*yb, axis=1)
                    if wb is None:
                        mu = rb.mean()
                        var_penalty = lambda_var * ((rb - mu)**2).mean()
                    else:
                        sw = wb.sum()
                        mu = (wb * rb).sum() / (sw + 1e-10)
                        var_penalty = (wb * (rb - mu)**2).sum() / (sw + 1e-10)
                        if "L1" in self.model_kwargs["loss"]:
                            var_penalty = torch.sqrt(var_penalty)
                        var_penalty = lambda_var * var_penalty

                full_loss = loss + l2_common + var_penalty
                full_loss.backward()
                if max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_norm)
                optimizer.step()
            self.train_loss.append(epoch_loss / X_train.shape[0])


            if validation:
                self.eval()
                with torch.no_grad():
                    out_val = self(X_val)
                    loss_val = self.loss.loss(Y_val, out_val).mean()
                    self.val_loss.append(loss_val.item())
                    self.fit_kwargs[-1]["val_loss"].append(self.val_loss[-1])
            self.fit_kwargs[-1]["train_loss"].append(self.train_loss[-1])

            if verbose:
                if validation:
                    print(
                        f"Epoch {self.model_kwargs['n_epochs']}/{initial_epoch + n_epochs}; Train Loss: {self.train_loss[-1]:.4f}, Val Loss: {self.val_loss[-1]:.4f}"
                    )
                else:
                    print(
                        f"Epoch {self.model_kwargs['n_epochs']}/{initial_epoch + n_epochs}; Train Loss: {self.train_loss[-1]:.4f}"
                    )

            if checkpoint is not None and (epoch + 1) % checkpoint == 0:
                os.makedirs(savefolder, exist_ok=True)
                self.save(
                    os.path.join(savefolder, f"epoch_{self.model_kwargs['n_epochs']}")
                )
                print(
                    f"Model checkpoint saved at epoch {self.model_kwargs['n_epochs']} to {savefolder}"
                )
            if (
                callback_every is not None
                and (epoch + 1) % callback_every == 0
                and callback is not None
            ):
                callback(
                    self,
                    os.path.join(savefolder, f"epoch_{self.model_kwargs['n_epochs']}"),
                    self.model_kwargs["n_epochs"],
                )

            self.model_kwargs["n_epochs"] += 1

