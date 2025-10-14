#%%
import sys
sys.path.append("..")
from utils.plots import plot_distributions
from utils.features import features_eb, features_ee
import pandas as pd

df = pd.read_pickle("../data/full_data_splitted_w.pkl")

#%%
plot_distributions(df[df["LPEle_isEB"]==1], features=features_eb+["LPEle_caloTarget", "LPEle_tkTarget"], log=True, weight="LPEle_w", savefolder="plots/features/eb_w")
plot_distributions(df[df["LPEle_isEB"]==1], features=features_eb+["LPEle_caloTarget", "LPEle_tkTarget", "LPEle_w"], log=True, savefolder="plots/features/eb")

#%%
plot_distributions(df[df["LPEle_isEB"]==0], features=features_ee+["LPEle_caloTarget", "LPEle_tkTarget"], log=True, weight="LPEle_w", savefolder="plots/features/ee_w")
plot_distributions(df[df["LPEle_isEB"]==0], features=features_ee+["LPEle_caloTarget", "LPEle_tkTarget", "LPEle_w"], log=True, savefolder="plots/features/ee")
# %%
