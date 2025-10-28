#%%
import sys
sys.path.append("../..")
sys.path.append("../../..")
import pandas as pd
from MVENet.SingleMVENet import SingleMVENet
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import os

from utils.features import tk_features, common_features, to_log
from utils.plots import response_resolution, plot_calibration_curve

hep.style.use("CMS")

df = pd.read_pickle("../../../data/full_data_splitted_w.pkl")
df = df[df["LPEle_isEB"] == 0]
#df = df[df["LPEle_tkTarget"] < 10]
# df = df[df["LPEle_tkTarget"] < 5]


for feature in to_log:
    if feature.startswith("-"):
        df[feature] = np.log(1e-8 - df[feature[1:]].values)
    else:
        df[feature] = np.log(1e-8 + df[feature].values)


features_ee = common_features + tk_features
train_data = df[features_ee].to_numpy()
targets = df["LPEle_tkTarget"].to_numpy()
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
    fig.savefig(f"{savefolder}/loss_ee.pdf")

    df_test["LPEle_ERatio"] = df_test["LPEle_Tk_p"] / df_test["LPEle_Gen_p"]
    mu, sigma = model.f(df_test[features_ee].to_numpy())
    df_test["LPEle_corrFact"] = mu
    df_test["LPEle_sigma"] = sigma

    df_test["LPEle_ECorrRatio"] = df_test["LPEle_corrFact"] * df_test["LPEle_ERatio"]

    pratio_dict_tk = {
        "No Regression": "LPEle_ERatio",
        "Regressed": "LPEle_ECorrRatio",
    }

    response_resolution(
        df_test,
        pratio_dict_tk,
        "LPEle_Gen_p",
        "LPEle_Gen_eta",
        plot_distributions=False,
        genp_bins=np.arange(1, 500, 15),
        eta_bins=[1.47, 1.6, 1.8, 2, 2.2, 2.4],
        res_ymax=1,
        lab="E_{\\text{tk}}",
        savefolder=f"{savefolder}",
    )

    plot_calibration_curve(
        df_test,
        "LPEle_tkTarget",
        "LPEle_corrFact",
        "LPEle_sigma",
        sigma_bins=np.arange(0, 2, 0.05),
        #genp_bins=[0, 100, 350, 600],
        #eta_bins=[0, 0.8, 1.44],
        savefolder=f"{savefolder}",
        metric="L1",
        plot_distributions=True
    )


    plot_calibration_curve(
        df_test,
        "LPEle_tkTarget",
        "LPEle_corrFact",
        "LPEle_sigma",
        sigma_bins=np.arange(0, 0.1, 0.005),
        genp_bins=[0, 100, 350, 600],
        eta_bins=[1.47, 1.95, 2.4],
        savefolder=f"{savefolder}",
        metric="L1",
        plot_distributions=True
    )



model = SingleMVENet(
    input_shape=len(features_ee),
    n_hidden_common=[],
    n_hidden_mean1=[256, 256, 256, 128, 64, 32, 16],
    n_hidden_var1=[256, 256, 256, 128, 64, 32, 16],
    dropout_input=0.,
    dropout_mean1=0.,
    dropout_var1=0.,
    loss="L1Loss",
)

model.normalize(X_train)
model.train_model(
    X_train,
    Y_train,
    sample_weight=w,
    X_val=X_val,
    Y_val=Y_val,
    batch_size=10280,
    learn_rate=5e-4,
    n_epochs=1000,
    reg_var1=1e-2,
    reg_mean1=1e-4,
    max_norm=None,
    checkpoint=100,
    callback=callback,
    callback_every=100,
    savefolder="plots/track",
)
# %%
# model.save("model_ee")
# %%

# %%
# model = SingleMVENet.load("model_ee")
