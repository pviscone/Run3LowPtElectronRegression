import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from MVENet import Losses
from MVENet import nn_utils


class SingleMVENet(nn.Module):
    def __init__(
        self,
        *,
        input_shape,
        n_hidden_common,
        n_hidden_mean1,
        n_hidden_var1,
        dropout_input=0.0,
        dropout_common=0.0,
        dropout_mean1=0.0,
        dropout_var1=0.0,
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
            "n_hidden_common": n_hidden_common,
            "n_hidden_mean1": n_hidden_mean1,
            "n_hidden_var1": n_hidden_var1,
            "dropout_input": dropout_input,
            "dropout_common": dropout_common,
            "dropout_mean1": dropout_mean1,
            "dropout_var1": dropout_var1,
            "loss": loss,
            "loss_kwargs": loss_kwargs,
            "n_epochs": n_epochs,
            "features": features,
        }
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.train_loss = []
        self.val_loss = []
        self.fit_kwargs = []
        self._normalization = False
        self.loss = getattr(Losses, loss)(**loss_kwargs)
        self.freezed = []

        self.input_shape = input_shape
        layers = []
        in_features = input_shape
        # Dropout input
        if dropout_input > 0:
            layers.append(nn.Dropout(dropout_input))

        # Common layers
        for n in n_hidden_common:
            if dropout_common > 0:
                layers.append(nn.Dropout(dropout_common))
            layers.append(nn.Linear(in_features, n))
            layers.append(nn.ELU())
            in_features = n
        self.common = nn.Sequential(*layers)

        # Mean branch
        mean1_layers = []
        mean1_in = in_features
        for n in n_hidden_mean1:
            if dropout_mean1 > 0:
                mean1_layers.append(nn.Dropout(dropout_mean1))
            mean1_layers.append(nn.Linear(mean1_in, n))
            mean1_layers.append(nn.ELU())
            mean1_in = n
        mean1_layers.append(nn.Linear(mean1_in, 1))
        self.mean1_branch = nn.Sequential(*mean1_layers)

        # Variance branch
        var1_layers = []
        var1_in = in_features
        for n in n_hidden_var1:
            if dropout_var1 > 0:
                var1_layers.append(nn.Dropout(dropout_var1))
            var1_layers.append(nn.Linear(var1_in, n))
            var1_layers.append(nn.ELU())
            var1_in = n
        var1_layers.append(nn.Linear(var1_in, 1))
        self.var1_branch = nn.Sequential(*var1_layers)

        self.to(self._device)

    def forward(self, x):
        inter = self.common(x)
        mean1_out = self.mean1_branch(inter)
        var1_out = self.var1_branch(inter)
        mean1_out = torch.exp(mean1_out)
        return torch.cat([mean1_out, var1_out], dim=-1)

    def normalize(self, X):
        self._normalization = True
        self._X_mean = np.mean(X, axis=0)
        self._X_std = np.std(X, axis=0)

    def mu(self, X_test):
        """Return the mean prediction"""
        if X_test.ndim == 1:
            X_test = X_test[:, None]
        if self._normalization:
            X_test = nn_utils.normalize(X_test, self._X_mean, self._X_std)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self._device)

        self.eval()
        with torch.no_grad():
            out = self(X_test)
            return self.loss.f(out.cpu().numpy())

    def sigma(self, X_test):
        """Return the standard deviation prediction"""
        if X_test.ndim == 1:
            X_test = X_test[:, None]
        if self._normalization:
            X_test = nn_utils.normalize(X_test, self._X_mean, self._X_std)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self._device)

        self.eval()
        with torch.no_grad():
            out = self(X_test)
            return self.loss.sigma(out.cpu().numpy())

    def f(self, X_test):
        """Return the correlation prediction"""
        if X_test.ndim == 1:
            X_test = X_test[:, None]
        if self._normalization:
            X_test = nn_utils.normalize(X_test, self._X_mean, self._X_std)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self._device)

        self.eval()
        with torch.no_grad():
            out = self(X_test)
            return (
                self.loss.f(out.cpu().numpy()),
                self.loss.sigma(out.cpu().numpy()),
            )

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

    def init_variance(self, X_train, Y_train):
        # Set variance branch output bias to logmse
        with torch.no_grad():
            out = self(X_train).cpu().numpy()
            mu1 = out[:, 0].flatten()
            y1_true = Y_train.cpu().numpy().flatten()
            if "L2" in self.model_kwargs["loss"]:
                logmse1 = 0.5 * np.log(np.mean(np.square(mu1 - y1_true)))
                self.var1_branch[-1].bias.data = torch.tensor(
                    [logmse1],
                    dtype=torch.float32,
                    device=self.var1_branch[-1].bias.device,
                )
            elif "L1" in self.model_kwargs["loss"]:
                logmae1 = np.log(np.mean(np.abs(mu1 - y1_true)))
                self.var1_branch[-1].bias.data = torch.tensor(
                    [logmae1],
                    dtype=torch.float32,
                    device=self.var1_branch[-1].bias.device,
                )

    def init_mu(self):
        # Set mean branch output bias to target mean
        with torch.no_grad():
            self.mean1_branch[-1].bias.data = torch.zeros_like(
                self.mean1_branch[-1].bias.data
            )
            self.mean1_branch[-1].weight.data = torch.zeros_like(
                self.mean1_branch[-1].weight.data
            )

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
        reg_common=0,
        reg_mean1=0,
        reg_var1=0,
        max_norm=1.0,
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
            and reg_common == self.fit_kwargs[-1]["reg_common"]
            and reg_mean1 == self.fit_kwargs[-1]["reg_mean1"]
            and reg_var1 == self.fit_kwargs[-1]["reg_var1"]
            and max_norm == self.fit_kwargs[-1]["max_norm"]
            and self.freezed == self.fit_kwargs[-1]["freezed"]
        ):
            self.fit_kwargs[-1]["n_epochs"] += n_epochs
        else:
            fit_kwarg = {
                "learn_rate": learn_rate,
                "n_epochs": n_epochs,
                "batch_size": batch_size,
                "reg_common": reg_common,
                "reg_mean1": reg_mean1,
                "reg_var1": reg_var1,
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
            self.init_variance(X_train, Y_train)
            self.init_mu()

        optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate)

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

                l2_common = reg_common * nn_utils.get_l2_reg(self.common)
                l2_mean1 = reg_mean1 * nn_utils.get_l2_reg(self.mean1_branch)
                l2_var1 = reg_var1 * nn_utils.get_l2_reg(self.var1_branch)
                full_loss = loss + l2_common + l2_mean1 + l2_var1
                full_loss.backward()
                if max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_norm)
                optimizer.step()
            self.train_loss.append(epoch_loss / X_train.shape[0])

            if diagnostics:
                if np.isnan(epoch_loss) or np.isinf(epoch_loss):
                    if np.isnan(epoch_loss):
                        print(f"NaN detected in loss at epoch {epoch}")
                    if np.isinf(epoch_loss):
                        print(f"Inf detected in loss at epoch {epoch}")
                    for name, param in self.named_parameters():
                        if torch.isnan(param).any():
                            print(f"NaN detected in parameter: {name} at epoch {epoch}")
                        if torch.isinf(param).any():
                            print(f"Inf detected in parameter: {name} at epoch {epoch}")
                    break

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
