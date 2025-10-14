#%%
import uproot
import awkward as ak
import pandas as pd
import numpy as np
import hist

def load_data( datasets_dict, columns=None, gen=True, genidx="LPEle_GenIdx" , n = -1):
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
        arrays["LPEle_year"] = ak.ones_like(arrays[genidx]) * float(year)
        arrays["LPEle_energy"] = arrays["LPEle_rawEnergy"] + arrays["LPEle_rawESEnergy"]
        arrays["LPEle_caloTkRatio"] = arrays["LPEle_energy"]/arrays["LPEle_Tk_p"]

        if gen:
            gen_p = arrays["GenEle_p"][arrays[genidx]]
            arrays["LPEle_caloTarget"] = gen_p / arrays["LPEle_energy"]
            arrays["LPEle_tkTarget"] = gen_p / arrays["LPEle_Tk_p"]
            arrays["LPEle_Gen_p"] = gen_p
            arrays["LPEle_Gen_pt"] = arrays["GenEle_pt"][arrays[genidx]]
            arrays["LPEle_Gen_eta"] = arrays["GenEle_eta"][arrays[genidx]]
            arrays["LPEle_Gen_phi"] = arrays["GenEle_phi"][arrays[genidx]]
        final_dict[year] = arrays
    return final_dict

def get_h(values, bins, arr):
    bin_idxs = np.digitize(arr, bins[:-1])-1
    res = values[bin_idxs]
    return res

#Lpele_branch needed only to pass the lp structure
def weights_and_merge(arrays_dict, balance_year=True, balance_genpt="splitted", genpt="GenEle_pt", genidx="LPEle_GenIdx", bins=None, lpele_branch="LPEle_pt"):
    n_electrons = [len(ak.ravel(arrays[lpele_branch]).to_numpy()) for arrays in arrays_dict.values()]
    mean_n_electrons = np.mean(n_electrons)

    for year, arrays in arrays_dict.items():
        arrays["LPEle_w"] = ak.ones_like(arrays[lpele_branch])
        if balance_year:
            arrays["LPEle_year_w"] = ak.ones_like(arrays[lpele_branch])*(mean_n_electrons/len(ak.ravel(arrays[lpele_branch]).to_numpy()))
            arrays["LPEle_w"] = arrays["LPEle_w"]*arrays["LPEle_year_w"]

        def compute_pt_w(arrays, bins, genpt, genidx, selection=None):
            if bins is None:
                bins = np.linspace(1, 100, 200)
            elif isinstance(bins, np.ndarray):
                pass
            elif isinstance(bins, list) or isinstance(bins, tuple):
                bins = np.linspace(bins[1], bins[2], bins[0])

            getp_hist = hist.Hist(hist.axis.Variable(bins, name="gen_pt", label="gen_pt"))
            pt_hist = hist.Hist(hist.axis.Variable(bins, name="genmatch_pt", label="genmatch_pt"))
            unbalance_hist = hist.Hist(hist.axis.Variable(bins, name="gen_pt", label="gen_pt"))
            unbalance_hist.fill(bins)

            gen_idx = arrays[genidx]
            if selection:
                gen_idx = eval(f"gen_idx[{selection}]")
                selector = eval(selection)
            pt = ak.flatten(arrays[genpt][gen_idx]).to_numpy()
            gen_pt = ak.flatten(arrays[genpt]).to_numpy()
            getp_hist.fill(gen_pt)
            pt_hist.fill(pt)
            eff_hist = pt_hist/getp_hist.values()
            unbalance_hist = unbalance_hist/(eff_hist/eff_hist.integrate(0)).values()

            pt_weights=get_h(unbalance_hist.values(), unbalance_hist.axes[0].edges, ak.ravel(pt).to_numpy())

            pt_w = ak.ones_like(arrays[lpele_branch])
            pt_w = ak.ravel(pt_w).to_numpy()
            if selection:
                pt_w[ak.ravel(selector).to_numpy()] = pt_weights
            pt_w = ak.unflatten(pt_w, ak.num(arrays[lpele_branch], axis=1))
            return pt_w

        if balance_genpt == "full":
            pt_w = compute_pt_w(arrays, bins, genpt, genidx)
            arrays["LPEle_pt_w"] = ak.ones_like(arrays[lpele_branch])*len(ak.ravel(arrays[lpele_branch]).to_numpy())*pt_w/np.sum(pt_w)
            arrays["LPEle_w"] = arrays["LPEle_w"] * arrays["LPEle_pt_w"]
        elif balance_genpt == "splitted":
            pt_w_eb = compute_pt_w(arrays, bins, genpt, genidx, selection = 'arrays["LPEle_isEB"]==1')
            pt_w_ee = compute_pt_w(arrays, bins, genpt, genidx, selection = 'arrays["LPEle_isEB"]==0')
            arrays["LPEle_pt_w"] = ak.where(arrays["LPEle_isEB"]==1, pt_w_eb, pt_w_ee)
            arrays["LPEle_pt_w"] = arrays["LPEle_pt_w"]*len(ak.ravel(arrays[lpele_branch]).to_numpy())/np.sum(ak.ravel(arrays["LPEle_pt_w"]))
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
