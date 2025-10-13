#%%
import pandas as pd
from torchNet import MVENetwork
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")

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
    #"LPEle_nSaturatedXtals", (always 0)
    "LPEle_numberOfClusters",
    "LPEle_iEtaOrX",
    "LPEle_iPhiOrY",
    "LPEle_iEtaMod5", #(barrel only)
    "LPEle_iPhiMod2", #(barrel only)
    "LPEle_iEtaMod20", #(barrel only)
    "LPEle_iPhiMod20", #(barrel only)
    #"LPEle_rawESEnergy", (endcap only)
    #"LPEle_isAlsoPF"
]


df = pd.read_pickle("../data/full_data_splitted_w.pkl")
df = df[df["LPEle_isEB"]==1]

df_train, df_test = train_test_split(df, test_size=0.2, random_state=666)

X_train = df_train[features].to_numpy()
Y_train = df_train["LPEle_target"].to_numpy()
w = df_train["LPEle_w"].to_numpy()
X_val = df_test[features].to_numpy()
Y_val = df_test["LPEle_target"].to_numpy()


#%%

model = MVENetwork(
    input_shape=len(features),
    n_hidden_common=[],
    n_hidden_mean=[128, 64, 32],
    n_hidden_var=[128, 64, 32],
)

model.normalize(X_train, Y_train)
model.train(
    X_train, Y_train, sample_weight=w,
    X_val = X_val, Y_val = Y_val,
    beta=None,
    batch_size=1028, learn_rate=1e-3,
    warmup = 10, n_epochs = 10,
    fixed_mean=False, verbose = True
)

#%%
fig, ax = plt.subplots()
ax.plot(model.train_loss, label="train")
if hasattr(model, 'val_loss'):
    ax.plot(model.val_loss, label="val")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()

# %%
