
# %%
import sys

sys.path.append("../../..")
sys.path.append("../..")
import pandas as pd
from MVENet.CombNet import CombNet
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import os
import sys
from utils.features import combined_features, common_features, to_log, caloEB_features, tk_features
from utils.plots import response_resolution
import xgboost as xgb

hep.style.use("CMS")

df = pd.read_pickle("../../../data/full_data_splitted_w.pkl")
df = df[df["LPEle_isEB"] == 1]
#df = df[df["LPEle_caloTarget"] < 10]
# df = df[df["LPEle_tkTarget"] < 5]


for feature in to_log:
    if feature.startswith("-"):
        df[feature] = np.log(1e-8 - df[feature[1:]].values)
    else:
        df[feature] = np.log(1e-8 + df[feature].values)


caloFeatures = common_features + caloEB_features
tkFeatures = common_features + tk_features

caloModel = xgb.Booster()
caloModel.load_model("BDT_caloEB_quantile.json")
tkModel = xgb.Booster()
tkModel.load_model("BDT_tkEB_quantile.json")

quantiles_calo = caloModel.inplace_predict(df[caloFeatures].to_numpy())
quantiles_tk = tkModel.inplace_predict(df[tkFeatures].to_numpy())

features_eb = []
for i,q in enumerate([0.05, 0.16, 0.5, 0.84, 0.95]):
    df[f"calo_quantile_{q}"] = quantiles_calo[:, i]
    df[f"tk_quantile_{q}"] = quantiles_tk[:, i]
    features_eb.append(f"calo_quantile_{q}")
    features_eb.append(f"tk_quantile_{q}")

features_eb += common_features + combined_features
train_data = df[features_eb].to_numpy()
target1 = df["LPEle_energy"].to_numpy()/df["LPEle_Gen_p"].to_numpy()
target2 = df["LPEle_Tk_p"].to_numpy()/df["LPEle_Gen_p"].to_numpy()
targets = np.vstack((target1, target2)).T

w = df["LPEle_w"].to_numpy()

df_train, df_test, X_train, X_val, Y_train, Y_val, w, _ = train_test_split(
    df, train_data, targets, w, test_size=0.2, random_state=666
)

# %%
def callback(model, savefolder, nepochs):
    train_loss, val_loss = model.get_loss()
    fig, ax = plt.subplots()
    ax.plot(train_loss, label="train")
    if len(val_loss) > 0:
        ax.plot(val_loss, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.savefig(f"{savefolder}/loss.pdf")


    df_test["originalRatio"] = np.cosh(df_test["LPEle_eta"].values) * df_test["LPEle_pt"].values / df_test["LPEle_Gen_p"].values
    E = df_test["LPEle_energy"].values
    p = df_test["LPEle_Tk_p"].values
    ep = np.stack((E, p), axis=1)

    out = model.f(df_test[features_eb].to_numpy())
    res = np.sum(out*ep, axis=1)

    df_test["LPEle_corrRatio"] = res / df_test["LPEle_Gen_p"].values

    pratio_dict = {
        "No Regression": "originalRatio",
        "Regressed": "LPEle_corrRatio",
    }

    response_resolution(
        df_test,
        pratio_dict,
        "LPEle_Gen_p",
        "LPEle_Gen_eta",
        plot_distributions=False,
        eta_bins=[0, 0.25, 0.5, 0.75, 1, 1.25, 1.47],
        lab="E_{\\text{comb}}",
        savefolder=f"{savefolder}",
    )


model = CombNet(
    input_shape=len(features_eb),
    n_hidden=[512,256,128,64,32],
    dropout_input=0.,
    dropout=0.,
    loss="CombineLossL1",
)

model.normalize(X_train)
model.train_model(
    X_train,
    Y_train,
    sample_weight=w,
    X_val=X_val,
    Y_val=Y_val,
    batch_size=10280,
    learn_rate=1e-5,
    n_epochs=500,
    reg=0.,
    lambda_var=1,
    weight_decay=0.1,
    max_norm=None,
    checkpoint=50,
    callback=callback,
    callback_every=50,
    savefolder="plots/combinedL1",
)
# %%
# model.save("model_eb")
# %%

# %%
# model = SingleMVENet.load("model_eb")
