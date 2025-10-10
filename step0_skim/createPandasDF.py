


#%%
import sys
sys.path.append("..")

from utils.load_data import load_data, weights_and_merge, convert_to_df

datasets = {
    "2024" : "../data/dataset2024.root",
    "2023" : "../data/dataset2023.root",
    "2022" : "../data/dataset2022.root",
}

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
    "LPEle_nSaturatedXtals",
    "LPEle_numberOfClusters",
    "LPEle_iEtaOrX",
    "LPEle_iPhiOrY",
    "LPEle_iEtaMod5", #(barrel only)
    "LPEle_iPhiMod2", #(barrel only)
    "LPEle_iEtaMod20", #(barrel only)
    "LPEle_iPhiMod20", #(barrel only)
    "LPEle_rawESEnergy", #(endcap only)
    "LPEle_isAlsoPF"
]

aux = [
    "LPEle_target",
    "LPEle_isEB",
    "LPEle_pt",
    "LPEle_phi",
    "LPEle_eta",
    "LPEle_year",
    "LPEle_w",
    "LPEle_year_w",
    "LPEle_pt_w",
]


data = load_data(datasets)
data = weights_and_merge(data, balance_year=True, balance_genpt="splitted")
df = convert_to_df(data, columns=features+aux)
df.to_pickle("../data/full_data_splitted_w.pkl")

#%%
