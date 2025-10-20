#%%
import sys
sys.path.append("..")
from utils.plots import plot_distributions
from utils.features import caloEB_features, caloEE_features, common_features, combined_features, tk_features
import pandas as pd

df = pd.read_pickle("../data/full_data_splitted_w.pkl")

#%%
ebfeatures = common_features + combined_features + tk_features + caloEB_features
plot_distributions(df[df["LPEle_isEB"]==1], features=ebfeatures+["LPEle_caloTarget", "LPEle_tkTarget"], log=True, weight="LPEle_w", savefolder="plots/features/eb_w")
plot_distributions(df[df["LPEle_isEB"]==1], features=ebfeatures+["LPEle_caloTarget", "LPEle_tkTarget", "LPEle_w"], log=True, savefolder="plots/features/eb")

#%%
eefeatures = common_features + combined_features + tk_features + caloEE_features
plot_distributions(df[df["LPEle_isEB"]==0], features=eefeatures+["LPEle_caloTarget", "LPEle_tkTarget"], log=True, weight="LPEle_w", savefolder="plots/features/ee_w")
plot_distributions(df[df["LPEle_isEB"]==0], features=eefeatures+["LPEle_caloTarget", "LPEle_tkTarget", "LPEle_w"], log=True, savefolder="plots/features/ee")
# %%
