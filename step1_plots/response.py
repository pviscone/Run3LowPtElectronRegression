#%%
import sys
sys.path.append("..")
from utils.plots import response_resolution
import pandas as pd

df = pd.read_pickle("../data/full_data_splitted_w.pkl")
df["LPEle_ptRatio"] = df["LPEle_pt"]/df["LPEle_Gen_pt"]
ptratio_dict= {"No Regression":"LPEle_ptRatio"}

#%%
response_resolution(
    df,
    ptratio_dict,
    "LPEle_Gen_pt",
    "LPEle_Gen_eta",
    plot_distributions = True,
    eta_bins = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.47, 1.52, 1.75, 2, 2.25, 2.5],
)
# %%
