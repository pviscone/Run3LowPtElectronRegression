
# %%
import pandas as pd
from MVENet.SingleMVENet import SingleMVENet
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

model = SingleMVENet(
    input_shape=len(features_eb),
    n_hidden_common=[512, 256, 128],
    n_hidden_mean=[128, 128, 64, 32],
    n_hidden_var=[128, 128, 64, 32],
    loss="L1Loss",
)

model.normalize(X_train, Y_train)
model.train_model(
    X_train,
    Y_train,
    sample_weight=w,
    X_val=X_val,
    Y_val=Y_val,
    batch_size=10280,
    learn_rate=1e-4,
    n_epochs=80,
    reg_var=1e-3,
    reg_mean=1e-4,
    verbose=True,
)


model.save("model_eb")
# %%
fig, ax = plt.subplots()
ax.plot(model.train_loss.loss_total, label="train")
if hasattr(model, "val_loss"):
    ax.plot(model.val_loss.loss_total, label="val")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
fig.savefig("plots_eb/loss_eb.pdf")
# %%
model = SingleMVENet.load("model_eb")

# %%
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
