import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import gc


def get_l2_reg(module):
    return sum(
        torch.sum(param**2)
        for param in module.parameters()
        if param.requires_grad and param.dim() > 1
    )


def normalize(x, mean=None, std=None):
    return (x - mean) / std


def reverse_normalized(x_normalized, mean, std):
    return x_normalized * std + mean


def variance_transformation(b, numpy=True):
    if numpy:
        return np.exp(b) + 1e-6
    else:
        return torch.exp(b) + 1e-6


def get_loss(beta=None, metric="L2"):
    if beta is not None:

        def beta_nll_loss(targets, outputs, sample_weight=None):
            mu = outputs[..., 0:1]
            var = variance_transformation(outputs[..., 1:2], numpy=False)
            # Stop gradient not directly supported, but can use detach
            loss = ((targets - mu) ** 2) / var + torch.log(var)
            loss = loss * var.detach() ** beta
            if sample_weight is not None:
                loss = loss * sample_weight.unsqueeze(-1)
            return loss.mean()

        return beta_nll_loss
    else:
        if metric == "L2":

            def l2_loss(targets, outputs, sample_weight=None):
                mu = outputs[..., 0:1]
                var = variance_transformation(outputs[..., 1:2], numpy=False)
                y = targets[..., 0:1]
                loglik = -torch.log(var) - ((y - mu) ** 2) / var
                if sample_weight is not None:
                    loglik = loglik * sample_weight.unsqueeze(-1)
                return -loglik.mean()
            return l2_loss
        elif metric == "L1":
            def l1_loss(targets, outputs, sample_weight=None):
                mu = outputs[..., 0:1]
                sigma = variance_transformation(outputs[..., 1:2], numpy=False)
                y = targets[..., 0:1]
                loglik = -torch.log(sigma) - (torch.abs(y - mu)) / sigma
                if sample_weight is not None:
                    loglik = loglik * sample_weight.unsqueeze(-1)
                return -loglik.mean()
            return l1_loss
        else:
            raise ValueError(f"Unknown metric: {metric}")


class MVENet(nn.Module):
    def __init__(
        self,
        input_shape,
        n_hidden_common,
        n_hidden_mean,
        n_hidden_var,
        dropout_input=0,
        dropout_common=0,
        dropout_mean=0,
        dropout_var=0,
    ):
        super().__init__()

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
        mean_layers = []
        mean_in = in_features
        for n in n_hidden_mean:
            if dropout_mean > 0:
                mean_layers.append(nn.Dropout(dropout_mean))
            mean_layers.append(nn.Linear(mean_in, n))
            mean_layers.append(nn.ELU())
            mean_in = n
        mean_layers.append(nn.Linear(mean_in, 1))
        self.mean_branch = nn.Sequential(*mean_layers)

        # Variance branch
        var_layers = []
        var_in = in_features
        for n in n_hidden_var:
            if dropout_var > 0:
                var_layers.append(nn.Dropout(dropout_var))
            var_layers.append(nn.Linear(var_in, n))
            var_layers.append(nn.ELU())
            var_in = n
        var_layers.append(nn.Linear(var_in, 1))
        self.var_branch = nn.Sequential(*var_layers)

    def forward(self, x):
        inter = self.common(x)
        mean_out = self.mean_branch(inter)
        var_out = self.var_branch(inter)
        return torch.cat([mean_out, var_out], dim=-1)


class MVENetwork:
    def __init__(
        self,
        *,
        input_shape,
        n_hidden_common,
        n_hidden_mean,
        n_hidden_var,
        dropout_input=0,
        dropout_common=0,
        dropout_mean=0,
        dropout_var=0,
        device=None,
        metric = "L2",
    ):
        self._normalization = False
        self.model_kwargs = {
            "input_shape": input_shape,
            "n_hidden_common": n_hidden_common,
            "n_hidden_mean": n_hidden_mean,
            "n_hidden_var": n_hidden_var,
            "dropout_input": dropout_input,
            "dropout_common": dropout_common,
            "dropout_mean": dropout_mean,
            "dropout_var": dropout_var,
            "metric": metric,
        }
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MVENet(
            input_shape=input_shape,
            n_hidden_common=n_hidden_common,
            n_hidden_mean=n_hidden_mean,
            n_hidden_var=n_hidden_var,
            dropout_input=dropout_input,
            dropout_common=dropout_common,
            dropout_mean=dropout_mean,
            dropout_var=dropout_var,
        ).to(self.device)
        self.train_loss = []
        self.val_loss = []
        self.fit_kwargs = []
        self.metric = metric

    def init_variance(self, X_train, Y_train):
        # Set variance branch output bias to logmse
        with torch.no_grad():
            mean_preds = self.model(X_train).cpu().numpy()[:, 0].flatten()
            y_true = Y_train.cpu().numpy().flatten()
            logmse = np.log(np.mean(np.square(mean_preds - y_true)))
            self.model.var_branch[-1].bias.data = torch.tensor(
                [logmse],
                dtype=torch.float32,
                device=self.model.var_branch[-1].bias.device,
            )

    def normalize(self, X, Y):
        self._normalization = True
        self._X_mean = np.mean(X, axis=0)
        self._X_std = np.std(X, axis=0)
        if self.metric == "L2":
            self._Y_mean = np.mean(Y, axis=0)
            self._Y_std = np.std(Y, axis=0)
        elif self.metric == "L1":
            self._Y_mean = np.median(Y, axis=0)
            self._Y_std = np.mean(np.abs(Y - self._Y_mean))

    def f(self, X_test):
        """Return the mean prediction"""
        if X_test.ndim == 1:
            X_test = X_test[:, None]
        if self._normalization:
            X_test = normalize(X_test, self._X_mean, self._X_std)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test).cpu().numpy()[:, 0]
            _ = gc.collect()
            if self._normalization:
                return reverse_normalized(predictions, self._Y_mean, self._Y_std)
            else:
                return predictions

    def sigma(self, X_test):
        """Return the standard deviation prediction"""
        if X_test.ndim == 1:
            X_test = X_test[:, None]
        if self._normalization:
            X_test = normalize(X_test, self._X_mean, self._X_std)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            out = self.model(X_test).cpu().numpy()
            var = variance_transformation(out[:, 1], numpy=True)

            _ = gc.collect()

            if self.metric == "L2":
                sigma = np.sqrt(var)
            elif self.metric == "L1":
                sigma = np.sqrt(2) * var

            if self._normalization:
                sigma = sigma * self._Y_std
            return sigma

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
        with open(os.path.join(path, "model_kwargs.pkl"), "wb") as f:
            pickle.dump(self.model_kwargs, f)
        with open(os.path.join(path, "fit_kwargs.pkl"), "wb") as f:
            pickle.dump(self.fit_kwargs, f)
        if self._normalization:
            np.savez(
                os.path.join(path, "normalization.npz"),
                X_mean=self._X_mean,
                X_std=self._X_std,
                Y_mean=self._Y_mean,
                Y_std=self._Y_std,
            )

    def show(self):
        print(self.model)

    @staticmethod
    def load(path, device=None):
        with open(os.path.join(path, "model_kwargs.pkl"), "rb") as f:
            model_kwargs = pickle.load(f)
        model = MVENetwork(**model_kwargs, device=device)
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
            model._Y_mean = norm["Y_mean"]
            model._Y_std = norm["Y_std"]

        model.model.load_state_dict(
            torch.load(os.path.join(path, "model.pt"), map_location=device)
        )
        return model

    def compile(self):
        self.model = torch.compile(self.model)

    def train(
        self,
        X_train,
        Y_train,
        X_val=None,
        Y_val=None,
        sample_weight=None,  # Accept sample weights
        beta=None,
        fixed_mean=False,
        learn_rate=0.001,
        warmup=None,
        n_epochs=100,
        batch_size=32,
        verbose=False,
        reg_common=0,
        reg_mean=0,
        reg_var=0,
    ):
        fit_kwarg = {
            "beta": beta,
            "fixed_mean": fixed_mean,
            "learn_rate": learn_rate,
            "warmup": warmup,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "verbose": verbose,
            "reg_common": reg_common,
            "reg_mean": reg_mean,
            "reg_var": reg_var,
        }
        self.fit_kwargs.append(fit_kwarg)
        validation = X_val is not None and Y_val is not None
        batch_size = batch_size

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
            X_train = normalize(X_train, self._X_mean, self._X_std)
            Y_train = normalize(Y_train, self._Y_mean, self._Y_std)
            if validation:
                X_val = normalize(X_val, self._X_mean, self._X_std)
                Y_val = normalize(Y_val, self._Y_mean, self._Y_std)

        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        Y_train = torch.tensor(Y_train, dtype=torch.float32).to(self.device)
        if validation:
            X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            Y_val = torch.tensor(Y_val, dtype=torch.float32).to(self.device)

        # Prepare sample_weight tensor
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float32)
            assert sample_weight.shape[0] == X_train.shape[0], (
                "Sample weight length must match number of samples"
            )
            sample_weight = torch.tensor(sample_weight, dtype=torch.float32).to(
                self.device
            )

        loss_fn = get_loss(beta, metric=self.metric)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learn_rate)

        def run_epoch(X, Y, W=None):
            epoch_loss = 0.0
            self.model.train(True)
            idx = np.arange(X.shape[0])
            np.random.shuffle(idx)
            for start in range(0, X.shape[0], batch_size):
                end = start + batch_size
                batch_idx = idx[start:end]
                xb = X[batch_idx]
                yb = Y[batch_idx]
                wb = W[batch_idx] if W is not None else None
                optimizer.zero_grad()
                out = self.model(xb)
                loss = loss_fn(yb, out, sample_weight=wb)

                l2_common = reg_common * get_l2_reg(self.model.common)
                l2_mean = reg_mean * get_l2_reg(self.model.mean_branch)
                l2_var = reg_var * get_l2_reg(self.model.var_branch)
                full_loss = loss + l2_common + l2_mean + l2_var
                full_loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            return epoch_loss / X.shape[0]

        def run_val(X, Y, W=None):
            self.model.eval()
            with torch.no_grad():
                out = self.model(X)
                loss = loss_fn(Y, out, sample_weight=W)
                return loss.item()

        # If no warmup, train both mean and variance together
        if not warmup:
            for epoch in range(n_epochs):
                train_loss = run_epoch(X_train, Y_train, sample_weight)
                self.train_loss.append(train_loss)
                if validation:
                    val_loss = run_val(X_val, Y_val, None)
                    self.val_loss.append(val_loss)
                if verbose and (epoch % 1 == 0 or epoch == n_epochs - 1):
                    print(
                        f"Epoch {epoch + 1}/{n_epochs} Train Loss: {train_loss:.4f}",
                        end="",
                    )
                    if validation:
                        print(f" Val Loss: {val_loss:.4f}", end="")
                    print()
            return self.model

        # Warmup: freeze variance branch, train only mean
        for param in self.model.var_branch.parameters():
            param.requires_grad = False
        for epoch in range(warmup):
            train_loss = run_epoch(X_train, Y_train, sample_weight)
            self.train_loss.append(train_loss)
            if validation:
                val_loss = run_val(X_val, Y_val, None)
                self.val_loss.append(val_loss)
            if verbose and (epoch % 1 == 0 or epoch == warmup - 1):
                print(
                    f"Warmup Epoch {epoch + 1}/{warmup} Train Loss: {train_loss:.4f}",
                    end="",
                )
                if validation:
                    print(f" Val Loss: {val_loss:.4f}", end="")
                print()

        if len(self.fit_kwargs) == 1:
            self.init_variance(X_train, Y_train)

        # Unfreeze variance branch
        for param in self.model.var_branch.parameters():
            param.requires_grad = True
        # Freeze mean branch if fixed_mean
        if fixed_mean:
            for param in self.model.mean_branch.parameters():
                param.requires_grad = False
        # Train both branches
        for epoch in range(n_epochs):
            train_loss = run_epoch(X_train, Y_train, sample_weight)
            self.train_loss.append(train_loss)
            if validation:
                val_loss = run_val(X_val, Y_val, None)
                self.val_loss.append(val_loss)
            if verbose and (epoch % 1 == 0 or epoch == n_epochs - 1):
                print(
                    f"Epoch {epoch + 1}/{n_epochs} Train Loss: {train_loss:.4f}", end=""
                )
                if validation:
                    print(f" Val Loss: {val_loss:.4f}", end="")
                print()
        return self.model
