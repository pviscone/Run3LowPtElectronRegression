# %%
import pandas as pd
from MVENet.DoubleMVENet import DoubleMVENet
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mplhep as hep
import os
import sys
import numpy as np

sys.path.append("..")
from utils.features import features_eb
from utils.plots import response_resolution

hep.style.use("CMS")

os.makedirs("plots_eb", exist_ok=True)

df = pd.read_pickle("../data/full_data_splitted_w.pkl")
df = df[df["LPEle_isEB"] == 1]
df = df[df["LPEle_caloTarget"] < 5]
df = df[df["LPEle_tkTarget"] < 5]

train_data = df[features_eb].to_numpy()
targets = df[["LPEle_caloTarget", "LPEle_tkTarget"]].to_numpy()
w = df["LPEle_w"].to_numpy()

df_train, df_test, X_train, X_val, Y_train, Y_val, w, _ = train_test_split(
    df, train_data, targets, w, test_size=0.2, random_state=666
)

# %%
def callback(model, savefolder, nepochs):
    os.makedirs(f"{savefolder}/plots_eb", exist_ok=True)
    train_loss, val_loss = model.get_loss()
    fig, ax = plt.subplots()
    ax.plot(train_loss, label="train")
    if len(val_loss)>0:
        ax.plot(val_loss, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.savefig(f"{savefolder}/plots_eb/loss_eb.pdf")


    df_test["LPEle_ERatio"] = df_test["LPEle_energy"] / df_test["LPEle_Gen_p"]
    df_test["LPEle_pRatio"] = df_test["LPEle_Tk_p"] / df_test["LPEle_Gen_p"]
    df_test["LPEle_combPRatio"] = df_test["LPEle_pt"] * np.cosh(df_test["LPEle_eta"])/ df_test["LPEle_Gen_p"]

    df_test["LPEle_ECorrRatio"] = (
        model.mu(df_test[features_eb].to_numpy())[:,0] * df_test["LPEle_ERatio"]
    )
    df_test["LPEle_pCorrRatio"] = (
        model.mu(df_test[features_eb].to_numpy())[:,1] * df_test["LPEle_pRatio"]
    )

    blue_mu, _ = model.BLUE(df_test[features_eb].to_numpy(), df_test["LPEle_energy"].to_numpy(), df_test["LPEle_Tk_p"].to_numpy())
    df_test["LPEle_combPCorrRatio"] = blue_mu / df_test["LPEle_Gen_p"]

    pratio_dict_calo = {"No Regression": "LPEle_ERatio", "Regressed": "LPEle_ECorrRatio"}
    pratio_dict_tk = {"No Regression": "LPEle_pRatio", "Regressed": "LPEle_pCorrRatio"}
    pratio_dict_comb = {"No Regression": "LPEle_combPRatio", "Regressed": "LPEle_combPCorrRatio"}

    response_resolution(
        df_test,
        pratio_dict_calo,
        "LPEle_Gen_p",
        "LPEle_Gen_eta",
        plot_distributions=False,
        eta_bins=[0, 0.25, 0.5, 0.75, 1, 1.25, 1.47],
        lab="E_{\\text{calo}}",
        savefolder=f"{savefolder}/plots_eb/calo",
    )

    response_resolution(
        df_test,
        pratio_dict_tk,
        "LPEle_Gen_p",
        "LPEle_Gen_eta",
        plot_distributions=False,
        eta_bins=[0, 0.25, 0.5, 0.75, 1, 1.25, 1.47],
        lab="p_{\\text{Tk}}",
        savefolder=f"{savefolder}/plots_eb/tk",
    )

    #TODO FIX BLUE
    response_resolution(
        df_test,
        pratio_dict_comb,
        "LPEle_Gen_p",
        "LPEle_Gen_eta",
        plot_distributions=False,
        eta_bins=[0, 0.25, 0.5, 0.75, 1, 1.25, 1.47],
        lab="p_{\\text{BLUE}}",
        savefolder=f"{savefolder}/plots_eb/comb",
    )

    #TODO implement calibration and correlation curves



model = DoubleMVENet(
    input_shape=len(features_eb),
    n_hidden_common=[],
    n_hidden_mean1=[512, 256, 128, 64, 32],
    n_hidden_var1=[512, 256, 128, 64, 32],
    n_hidden_mean2=[512, 256, 128, 64, 32],
    n_hidden_var2=[512, 256, 128, 64, 32],
    n_hidden_rho=[512, 256, 128, 64, 32],
    loss="BivariateL2",
)

model.normalize(X_train, Y_train)
model.train_model(
    X_train,
    Y_train,
    sample_weight=w,
    X_val=X_val,
    Y_val=Y_val,
    batch_size=10280,
    learn_rate=1e-6,
    n_epochs=200,
    reg_var1=1e-3,
    reg_mean1=1e-4,
    reg_var2=1e-3,
    reg_mean2=1e-4,
    reg_rho=1e-2,
    max_norm=None,
    checkpoint=10,
    callback=callback,
    callback_every=10,
    savefolder="modelL2_eb",
)

#%%
#model.save("model_eb")
# %%

# %%
#model = DoubleMVENet.load("model_eb")

