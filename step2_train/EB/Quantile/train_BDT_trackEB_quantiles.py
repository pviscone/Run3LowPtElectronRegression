# %%
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import os
import sys
import xgboost as xgb
from typing import Dict

sys.path.append("../../..")
from utils.features import tk_features, common_features, to_log
from utils.plots import response_resolution, plot_quantile_curve

hep.style.use("CMS")

df = pd.read_pickle("../../../data/full_data_splitted_w.pkl")
df = df[df["LPEle_isEB"] == 1]
# df = df[df["LPEle_caloTarget"] < 10]
# df = df[df["LPEle_tkTarget"] < 5]


for feature in to_log:
    if feature.startswith("-"):
        df[feature] = np.log(1e-8 - df[feature[1:]].values)
    else:
        df[feature] = np.log(1e-8 + df[feature].values)
df["target"] = df["LPEle_tkTarget"].values


features_eb = common_features + tk_features
train_data = df[features_eb].to_numpy()
targets = df["target"].to_numpy()
w = df["LPEle_w"].to_numpy()

df_train, df_test, X_train, X_val, Y_train, Y_val, w, _ = train_test_split(
    df, train_data, targets, w, test_size=0.2, random_state=666
)

# %%
quantiles = [0.05, 0.16, 0.5, 0.84, 0.95]
Xy = xgb.QuantileDMatrix(X_train, Y_train, weight=w)
Xy_test = xgb.QuantileDMatrix(X_val, Y_val, ref=Xy)

evals_result: Dict[str, Dict] = {}
booster = xgb.train(
    {
        "objective": "reg:quantileerror",
        "tree_method": "hist",
        "quantile_alpha": np.array(quantiles),
        "learning_rate": 0.4,
        "max_depth": 6,
        "device": "cuda",
    },
    Xy,
    num_boost_round=1000,
    early_stopping_rounds=10,
    evals=[(Xy, "Train"), (Xy_test, "Test")],
    evals_result=evals_result,
)

# %%
out = booster.inplace_predict(X_val)
# out = np.atanh(np.clip(out, -0.999999, 0.999999)) + 1  # inverse of tanh
mu = out[:, quantiles.index(0.5)]

df_test["LPEle_pRatio"] = df_test["LPEle_Tk_p"] / df_test["LPEle_Gen_p"]
df_test["LPEle_corrFact"] = mu

df_test["LPEle_pCorrRatio"] = df_test["LPEle_corrFact"] * df_test["LPEle_pRatio"]

pratio_dict = {
    "No Regression": "LPEle_pRatio",
    "Regressed": "LPEle_pCorrRatio",
}

os.makedirs("plots/track", exist_ok=True)
# plot loss for train and test
fig, ax = plt.subplots()
ax.plot(evals_result["Train"]["quantile"], label="Train")
ax.plot(evals_result["Test"]["quantile"], label="Test")
ax.set_xlabel("Boosting Round")
ax.set_ylabel("Quantile Error")
ax.legend()
fig.savefig("plots/track/loss_eb.pdf")


response_resolution(
    df_test,
    pratio_dict,
    "LPEle_Gen_p",
    "LPEle_Gen_eta",
    plot_distributions=False,
    eta_bins=[0, 0.25, 0.5, 0.75, 1, 1.25, 1.47],
    lab="p_{\\text{Tk}}",
    savefolder="plots/track",
)

plot_quantile_curve(
    df_test,
    "LPEle_tkTarget",
    out,
    quantiles,
    genp_bins=np.arange(0, 600, 50),
    eta_bins=[0, 0.5, 0.8, 1.2, 1.44],
    savefolder="plots/track",
    plot_distributions=True,
)


#%%

#booster.save_model("BDT_tkEB_quantile.json")