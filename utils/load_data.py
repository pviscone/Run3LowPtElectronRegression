#%%
import uproot
import awkward as ak
import pandas as pd
import numpy as np
import hist

def load_data( datasets_dict, columns=None, genpt="GenEle_pt", genidx="LPEle_GenIdx" ,pt="LPEle_pt", n = -1):
    final_dict = {}
    for year, dataset in datasets_dict.items():
        tree = uproot.open(dataset)["Events"]
        keys = tree.keys()
        for col in keys:
            if columns is not None and col not in columns:
                keys.remove(col)
        arrays = tree.arrays(keys)
        if n>0:
            arrays = arrays[:n]
        arrays["LPEle_year"] = ak.ones_like(arrays[pt]) * float(year)
        if genpt:
            arrays["LPEle_target"] = arrays[genpt][arrays[genidx]] / arrays[pt]
        final_dict[year] = arrays
    return final_dict

def get_h(values, bins, arr):
    bin_idxs = np.digitize(arr, bins[:-1])-1
    res = values[bin_idxs]
    return res

#Lpele_branch needed only to pass the lp structure
def weights_and_merge(arrays_dict, balance_year=True, balance_genpt=True, genpt="GenEle_pt", genidx="LPEle_GenIdx", bins=None, lpele_branch="LPEle_pt"):
    for year, arrays in arrays_dict.items():
        arrays["LPEle_w"] = ak.ones_like(arrays[lpele_branch])
        if balance_year:
            mean_events = np.mean(len(arrays))
            arrays["LPEle_year_w"] = ak.ones_like(arrays[lpele_branch])*(mean_events/len(arrays))
            arrays["LPEle_w"] = arrays["LPEle_w"]*arrays["LPEle_year_w"]

        if balance_genpt:
            if bins is None:
                bins = np.linspace(1, 100, 200)
            elif isinstance(bins, np.ndarray):
                pass
            elif isinstance(bins, list) or isinstance(bins, tuple):
                bins = np.linspace(bins[1], bins[2], bins[0])

            getpt_hist = hist.Hist(hist.axis.Variable(bins, name="gen_pt", label="gen_pt"))
            pt_hist = hist.Hist(hist.axis.Variable(bins, name="genmatch_pt", label="genmatch_pt"))
            unbalance_hist = hist.Hist(hist.axis.Variable(bins, name="gen_pt", label="gen_pt"))
            unbalance_hist.fill(bins)

            pt = ak.flatten(arrays[genpt][arrays[genidx]]).to_numpy()
            gen_pt = ak.flatten(arrays[genpt]).to_numpy()
            getpt_hist.fill(gen_pt)
            pt_hist.fill(pt)
            eff_hist = pt_hist/getpt_hist.values()
            unbalance_hist = unbalance_hist/(eff_hist/eff_hist.integrate(0)).values()

            pt_w=get_h(unbalance_hist.values(), unbalance_hist.axes[0].edges, ak.ravel(pt).to_numpy())
            pt_w = ak.unflatten(pt_w, ak.num(arrays[lpele_branch], axis=1))
            arrays["LPEle_pt_w"] = ak.ones_like(arrays[lpele_branch])*len(arrays)*pt_w/np.sum(pt_w)
            arrays["LPEle_w"] = arrays["LPEle_w"] * arrays["LPEle_pt_w"]

    arrays = ak.concatenate(list(arrays_dict.values()), axis=0)
    return arrays

def convert_to_df(arrays, columns=None):
    if columns is None:
        df = ak.to_pandas(arrays)
    else:
        df= pd.DataFrame({key: ak.flatten(arrays[key]).to_numpy().astype(np.float32) for key in columns})
    return df


#%%
