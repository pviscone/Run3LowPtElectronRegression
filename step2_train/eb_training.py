# %%
import pandas as pd
from torchNet import MVENetwork
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mplhep as hep
import os
import sys

sys.path.append("..")
from utils.features import features_eb

hep.style.use("CMS")

os.makedirs("plots_eb", exist_ok=True)

df = pd.read_pickle("../data/full_data_splitted_w.pkl")
df = df[df["LPEle_isEB"] == 1]
# df = df[df["LPEle_target"] < 4]
# df = df[df["LPEle_target"] > 0.25]

df_train, df_test = train_test_split(df, test_size=0.2, random_state=666)

X_train = df_train[features_eb].to_numpy()
Y_train = df_train["LPEle_caloTarget"].to_numpy()
w = df_train["LPEle_w"].to_numpy()
X_val = df_test[features_eb].to_numpy()
Y_val = df_test["LPEle_caloTarget"].to_numpy()


# %%
model = MVENetwork(
    input_shape=len(features_eb),
    n_hidden_common=[128],
    n_hidden_mean=[128, 64, 32],
    n_hidden_var=[128, 64, 32],
    metric = "L1"
)

model.normalize(X_train, Y_train)
model.train(
    X_train,
    Y_train,
    sample_weight=w,
    X_val=X_val,
    Y_val=Y_val,
    beta=None,
    batch_size=10280,
    learn_rate=1e-3,
    warmup=0,
    n_epochs=140,
    reg_var=0.0,
    fixed_mean=False,
    verbose=True,
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

#%%
import sys

sys.path.append("..")
from utils.plots import response_resolution
import pandas as pd

df_test["LPEle_pRatio"] = df_test["LPEle_energy"] / df_test["LPEle_Gen_p"]
df_test["LPEle_pCorrRatio"] = (
    model.f(df_test[features_eb].to_numpy()) * df_test["LPEle_pRatio"]
)
pratio_dict = {"No Regression": "LPEle_pRatio", "Regressed": "LPEle_pCorrRatio"}


response_resolution(
    df_test,
    pratio_dict,
    "LPEle_Gen_p",
    "LPEle_Gen_eta",
    plot_distributions=False,
    eta_bins=[0, 0.25, 0.5, 0.75, 1, 1.25, 1.47],
    savefolder="plots_eb",
)
# %%
