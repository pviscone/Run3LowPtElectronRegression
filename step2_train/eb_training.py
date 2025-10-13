# %%
import pandas as pd
from torchNet import MVENetwork
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mplhep as hep
import os

hep.style.use("CMS")

os.makedirs("plots_eb", exist_ok=True)


#TODO add gsf variables
# make metric more consistent (network attribute?)
features = [
    "LPEle_rawEnergy",
    "LPEle_etaWidth",
    "LPEle_phiWidth",
    "LPEle_seedEnergyFraction",
    "LPEle_rho",
    "LPEle_hcalOverEcalBc",
    "LPEle_seedClusterEtaDiff",
    "LPEle_seedClusterPhiDiff",
    "LPEle_r9",
    "LPEle_sigmaIetaIeta",
    "LPEle_sigmaIetaIphi",
    "LPEle_sigmaIphiIphi",
    "LPEle_eMaxOverE5x5",
    "LPEle_e2ndOverE5x5",
    "LPEle_eTopOverE5x5",
    "LPEle_eBottomOverE5x5",
    "LPEle_eLeftOverE5x5",
    "LPEle_eRightOverE5x5",
    "LPEle_e2x5MaxOverE5x5",
    "LPEle_e2x5LeftOverE5x5",
    "LPEle_e2x5RightOverE5x5",
    "LPEle_e2x5TopOverE5x5",
    "LPEle_e2x5BottomOverE5x5",
    # "LPEle_nSaturatedXtals", (always 0)
    "LPEle_numberOfClusters",
    "LPEle_iEtaOrX",
    "LPEle_iPhiOrY",
    "LPEle_iEtaMod5",  # (barrel only)
    "LPEle_iPhiMod2",  # (barrel only)
    "LPEle_iEtaMod20",  # (barrel only)
    "LPEle_iPhiMod20",  # (barrel only)
    # "LPEle_rawESEnergy", (endcap only)
    # "LPEle_isAlsoPF"
]


df = pd.read_pickle("../data/full_data_splitted_w.pkl")
df = df[df["LPEle_isEB"] == 1]
df = df[df["LPEle_target"] < 4]
df = df[df["LPEle_target"] > 0.25]

df_train, df_test = train_test_split(df, test_size=0.2, random_state=666)

X_train = df_train[features].to_numpy()
Y_train = df_train["LPEle_target"].to_numpy()
w = df_train["LPEle_w"].to_numpy()
X_val = df_test[features].to_numpy()
Y_val = df_test["LPEle_target"].to_numpy()


# %%
model = MVENetwork(
    input_shape=len(features),
    n_hidden_common=[],
    n_hidden_mean=[128, 128, 64, 32],
    n_hidden_var=[128, 128, 64, 32],
)
model.normalize(X_train, Y_train, metric="L1")
model.train(
    X_train,
    Y_train,
    sample_weight=w,
    X_val=X_val,
    Y_val=Y_val,
    beta=None,
    batch_size=10280,
    learn_rate=1e-2,
    warmup=10,
    n_epochs=100,
    reg_var=5e-3,
    fixed_mean=False,
    verbose=True,
    metric="L1",
)
model.save("model_eb")
# %%
fig, ax = plt.subplots()
ax.plot(model.train_loss, label="train")
if hasattr(model, "val_loss"):
    ax.plot(model.val_loss, label="val")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
fig.savefig("plots_eb/loss_eb.pdf")
# %%
model = MVENetwork.load("model_eb")

import sys

sys.path.append("..")
from utils.plots import response_resolution
import pandas as pd

df_test["LPEle_ptRatio"] = df_test["LPEle_pt"] / df_test["LPEle_Gen_pt"]
df_test["LPEle_ptCorrRatio"] = (
    model.f(df_test[features].to_numpy()) * df_test["LPEle_ptRatio"]
)
ptratio_dict = {"No Regression": "LPEle_ptRatio", "Regressed": "LPEle_ptCorrRatio"}


response_resolution(
    df_test,
    ptratio_dict,
    "LPEle_Gen_pt",
    "LPEle_Gen_eta",
    plot_distributions=False,
    eta_bins=[0, 0.25, 0.5, 0.75, 1, 1.25, 1.47],
    savefolder="plots_eb",
)
# %%
