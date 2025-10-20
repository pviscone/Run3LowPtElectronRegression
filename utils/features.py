common_features = [
    #"LPEle_absEta",
    #"LPEle_caloTkRatio",
]


combined_features = [
    "LPEle_Tk_dEtaSeedClusterAtCalo",
    "LPEle_Tk_dPhiSeedClusterAtCalo",
    "LPEle_Tk_dEtaSeedClusterAtVtx",
    "LPEle_Tk_eSeedClusterOverP",
    "LPEle_Tk_eSeedClusterOverPout",
    "LPEle_Tk_dEtaEleClusterAtCalo",
    "LPEle_Tk_dPhiEleClusterAtCalo",
    "LPEle_Tk_dEtaSuperClusterAtVtx",
    "LPEle_Tk_dPhiSuperClusterAtVtx",
    "LPEle_Tk_eSuperClusterOverP",
    "LPEle_Tk_eEleClusterOverPout",
    "LPEle_isAlsoPF",
]

tk_features = [
    "LPEle_Tk_p",
    "LPEle_Tk_eta",
    "LPEle_Tk_validFraction",
    "LPEle_Tk_nValidHits",
    "LPEle_Tk_chi2",
    "LPEle_Tk_charge",
    "LPEle_Tk_qoverp",
    "LPEle_Tk_fbrem",  # (leads to nans, require small learning rate)
    "LPEle_Tk_errPRatio",  # (small numbers, leads to nans, require small learning rate)
    "LPEle_Tk_nLostHits",
    "LPEle_Tk_nLostInnerHits",
    "LPEle_Tk_nLostOuterHits",
]

calo_features = [
    "LPEle_rawEnergy",
    "LPEle_nOfBrems",
    "LPEle_superClusterFBrem",
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
    "LPEle_numberOfClusters",
    "LPEle_iEtaOrX",
    "LPEle_iPhiOrY",
    # "LPEle_ecalDrivenSeed", (always 0)
    # "LPEle_nSaturatedXtals", (always 0)
]

caloEB_features = calo_features + [
    "LPEle_iEtaMod5",
    "LPEle_iPhiMod2",
    "LPEle_iEtaMod20",
    "LPEle_iPhiMod20",
]

caloEE_features = calo_features + [
    "LPEle_rawESEnergy",
]


to_log=[
    "LPEle_hcalOverEcalBc",
    "LPEle_r9",
    "LPEle_Tk_chi2",
    "LPEle_Tk_eEleClusterOverPout",
    "LPEle_Tk_errPRatio",
    "LPEle_Tk_eSeedClusterOverP",
    "LPEle_Tk_eSeedClusterOverPout",
    "LPEle_Tk_eSuperClusterOverP",
    "-LPEle_Tk_fbrem",
    #"LPEle_caloTarget",
    #"LPEle_tkTarget",
]