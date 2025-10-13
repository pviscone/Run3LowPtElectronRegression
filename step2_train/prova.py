# %%
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use("CMS")


mu = lambda x: (x**2) / 25
sig = lambda x: 10 / (x**2 + 1)

X = np.random.uniform(-5, 5, 10000)
Y = np.random.normal(loc=mu(X), scale=sig(X))

X_val = np.random.uniform(-5, 5, 2000)
Y_val = np.random.normal(loc=mu(X_val), scale=sig(X_val))

# %%

from torchNet import MVENetwork

model = MVENetwork(
    input_shape=1,
    n_hidden_common=[],
    n_hidden_mean=[128, 64, 32],
    n_hidden_var=[128, 64, 32],
)
model.normalize(X, Y)
model.train(
    X,
    Y,
    sample_weight=np.ones(len(X)),
    beta=None,
    X_val=X_val,
    Y_val=Y_val,
    batch_size=1028,
    learn_rate=1e-3,
    warmup=200,
    n_epochs=200,
    reg_mean=1e-3,
    reg_var=0.,
    fixed_mean=False,
    verbose=True,
)


# %%


X_sort = np.sort(X)
plt.figure(dpi=300)
plt.scatter(X, Y, alpha=0.1)
means = model.f(X_sort)
sigmas = model.sigma(X_sort)
plt.fill_between(X_sort, means - sigmas, means + sigmas, alpha=0.5)
plt.plot(X_sort, means, label=r"$\hat{\mu}$")
plt.plot(X_sort, sigmas, label=r"$\hat{\sigma}$")
plt.plot(X_sort, mu(X_sort), label=r"True $ \mu$", linestyle="dashed")
plt.plot(X_sort, sig(X_sort), label=r"True $ \sigma$", linestyle="dashed")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend(loc="upper right")
plt.grid()
plt.tight_layout()
plt.show()


# %%
