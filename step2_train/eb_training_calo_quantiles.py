
# %%
import pandas as pd
from MVENet.QuantileNet import QuantileNet
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import os
import sys
import torch

sys.path.append("..")
from utils.features import caloEB_features, common_features, to_log
from utils.plots import response_resolution, plot_quantile_curve

hep.style.use("CMS")

df = pd.read_pickle("../data/full_data_splitted_w.pkl")
df = df[df["LPEle_isEB"] == 1]
#df = df[df["LPEle_caloTarget"] < 10]
# df = df[df["LPEle_tkTarget"] < 5]


for feature in to_log:
    if feature.startswith("-"):
        df[feature] = np.log(1e-8 - df[feature[1:]].values)
    else:
        df[feature] = np.log(1e-8 + df[feature].values)
df["target"] = np.log(1e-8 + df["LPEle_caloTarget"].values)
#df["target"] = df["LPEle_caloTarget"].values

features_eb = common_features + caloEB_features
train_data = df[features_eb].to_numpy()
targets = df["target"].to_numpy()
w = df["LPEle_w"].to_numpy()

df_train, df_test, X_train, X_val, Y_train, Y_val, w, _ = train_test_split(
    df, train_data, targets, w, test_size=0.2, random_state=666
)
# %%
quantiles = [0.05, 0.16, 0.5, 0.84, 0.95]

def callback(model, savefolder, nepochs):
    os.makedirs(f"{savefolder}/plots", exist_ok=True)

    #!loss
    train_loss, val_loss = model.get_loss()
    fig, ax = plt.subplots()
    ax.plot(train_loss, label="train")
    if len(val_loss) > 0:
        ax.plot(val_loss, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.savefig(f"{savefolder}/plots/loss_eb.pdf")

    #!Evaluate
    out = model.f(X_val)
    out = np.exp(out)- 1e-8
    mu = out[:, quantiles.index(0.5)]
    sigma = 0.5 * (out[:, quantiles.index(0.84)] - out[:, quantiles.index(0.16)])

    df_test["LPEle_ERatio"] = df_test["LPEle_energy"] / df_test["LPEle_Gen_p"]
    df_test["LPEle_corrFact"] = mu
    df_test["LPEle_sigma"] = sigma

    df_test["LPEle_ECorrRatio"] = df_test["LPEle_corrFact"] * df_test["LPEle_ERatio"]

    pratio_dict_calo = {
        "No Regression": "LPEle_ERatio",
        "Regressed": "LPEle_ECorrRatio",
    }

    response_resolution(
        df_test,
        pratio_dict_calo,
        "LPEle_Gen_p",
        "LPEle_Gen_eta",
        plot_distributions=False,
        eta_bins=[0, 0.25, 0.5, 0.75, 1, 1.25, 1.47],
        lab="E_{\\text{calo}}",
        savefolder=f"{savefolder}/plots/calo",
    )

    plot_quantile_curve(
        df_test,
        "LPEle_caloTarget",
        out,
        quantiles,
        genp_bins=np.arange(0,600,50),
        eta_bins=[0, 0.5, 0.8, 1.2, 1.44],
        savefolder=f"{savefolder}/plots/calo",
        plot_distributions=True
    )

model = QuantileNet(
    input_shape=len(features_eb),
    n_hidden=[512, 256, 128, 64, 32, 16],
    quantiles=quantiles,
    dropout_input=0.,
    dropout=0.1,
    loss="QuantileLoss",
)
model.init_biases(Y_train)
model.normalize(X_train)
model.train_model(
    X_train,
    Y_train,
    #sample_weight=w,
    X_val=X_val,
    Y_val=Y_val,
    batch_size=10280,
    learn_rate=3e-4,
    weight_decay=1e-4,
    n_epochs=1000,
    reg=0.,
    reg_sep=1,
    sep_gap=5e-3,
    max_norm=None,
    checkpoint=50,
    callback=callback,
    callback_every=50,
    savefolder="model_caloEB_quantiles",
)
# %%
# model.save("model_eb")
# %%

# %%
# model = SingleMVENet.load("model_eb")
#%%

