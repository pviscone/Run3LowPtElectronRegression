common = [
    "LPEle_Tk_validFraction",
    "LPEle_Tk_nValidHits",
    "LPEle_Tk_nLostHits",
    "LPEle_Tk_nLostInnerHits",
    "LPEle_Tk_nLostOuterHits",
    "LPEle_nOfBrems",
    "LPEle_superClusterFBrem",
    "LPEle_Tk_chi2",
    "LPEle_Tk_charge",
    "LPEle_Tk_qoverp",
    "LPEle_Tk_dEtaEleClusterAtCalo",
    "LPEle_Tk_dPhiEleClusterAtCalo",
    "LPEle_Tk_dEtaSuperClusterAtVtx",
    "LPEle_Tk_dPhiSuperClusterAtVtx",
    "LPEle_Tk_dEtaSeedClusterAtCalo",
    "LPEle_Tk_dPhiSeedClusterAtCalo",
    "LPEle_Tk_dEtaSeedClusterAtVtx",
    "LPEle_Tk_eSuperClusterOverP",
    "LPEle_Tk_eSeedClusterOverP",
    "LPEle_Tk_eSeedClusterOverPout",
    "LPEle_Tk_eEleClusterOverPout",
    "LPEle_Tk_fbrem", #(leads to nans, require small learning rate)
    "LPEle_Tk_errPRatio", #(small numbers, leads to nans, require small learning rate)
    "LPEle_Tk_p",

    "LPEle_Tk_eta",
    "LPEle_caloTkRatio",
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
    "LPEle_numberOfClusters",
    "LPEle_iEtaOrX",
    "LPEle_iPhiOrY",
    "LPEle_isAlsoPF",
    #"LPEle_ecalDrivenSeed", (always 0)
    #"LPEle_nSaturatedXtals", (always 0)
]



features_ee = common +[
    #"LPEle_iEtaMod5", (barrel only)
    #"LPEle_iPhiMod2", (barrel only)
    #"LPEle_iEtaMod20", (barrel only)
    #"LPEle_iPhiMod20", (barrel only)
    "LPEle_rawESEnergy", #(endcap only)
]

features_eb = common + [
    "LPEle_iEtaMod5", #(barrel only)
    "LPEle_iPhiMod2", #(barrel only)
    "LPEle_iEtaMod20", #(barrel only)
    "LPEle_iPhiMod20", #(barrel only)
    #"LPEle_rawESEnergy", #(endcap only)
]