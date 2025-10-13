import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import Var
import argparse


def LazyVar(expr, valtype, doc=None, precision=-1):
    return Var(expr, valtype, doc, precision, lazyEval=True)


parser = argparse.ArgumentParser()
# input and output are mandatory
parser.add_argument(
    "-i",
    "--input",
    type=str,
    required=True,
    help="Input root file or txt file containing the list of files to process",
)
parser.add_argument("-o", "--output", type=str, required=True, help="Output file name")
parser.add_argument(
    "--maxGenPU", type=int, default=70, help="Maximum number of PU interactions"
)
parser.add_argument(
    "--maxGenPt", type=int, default=100, help="Maximum number of PU interactions"
)
parser.add_argument(
    "--maxEvents", type=int, default=-1, help="Maximum number of events to process"
)
parser.add_argument(
    "--year", type=str, default="2023", help="Year (needed for setting the GT)"
)
parser.add_argument(
    "--data", action="store_true", help="If set, the input file is data"
)
parser.add_argument(
    "--storeUnmatched",
    action="store_true",
    help="Store also LP electrons not matched to GenEle (only for MC)",
)
args = parser.parse_args()

if "," in args.input:
    filelist = args.input.split(",")
elif args.input.endswith(".root"):
    filelist = "file:" + args.input
elif args.input.endswith(".txt"):
    with open(args.input, "r") as f:
        files = f.read().splitlines()
    filelist = ["root://cms-xrd-global.cern.ch/" + f for f in files]

if "2023" in args.year or "2022" in args.year:
    gt = "140X_dataRun3_v17"
elif "2024" in args.year:
    gt = "140X_dataRun3_v20"

process = cms.Process("SKIM")
process.load("PhysicsTools.PatAlgos.patSequences_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring(filelist))
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(args.maxEvents))

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = gt


# Add all variables
process.extendedLPEle = cms.EDProducer(
    "ElectronExtenderProducer",
    src=cms.InputTag("slimmedLowPtElectrons"),
    rho=cms.InputTag("fixedGridRhoFastjetAll"),
)

process.extendedPFEle = cms.EDProducer(
    "ElectronExtenderProducer",
    src=cms.InputTag("slimmedElectrons"),
    rho=cms.InputTag("fixedGridRhoFastjetAll"),
)


if not args.data:
    process.selectedGenPart = cms.EDProducer(
        "GenParticlePruner",
        src=cms.InputTag("prunedGenParticles"),
        select=cms.vstring(
            "drop *",
            f"keep abs(pdgId) == 11 && isPromptFinalState && isLastCopy && pt <= {args.maxGenPt} && abs(eta) < 2.5",
        ),
    )

    process.nonEmptyGenPartFilter = cms.EDFilter(
        "CandViewCountFilter",
        src=cms.InputTag("selectedGenPart"),
        minNumber=cms.uint32(1),
    )

    process.genMatchedLPEle = cms.EDProducer(
        "GenElectronMatcher",
        src1=cms.InputTag("extendedLPEle"),
        src2=cms.InputTag("selectedGenPart"),
        dRmax=cms.double(0.05),
        storeUnmatched=cms.bool(args.storeUnmatched),
        varLabel=cms.string("Gen"),
    )

    process.genMatchedPFEle = cms.EDProducer(
        "GenElectronMatcher",
        src1=cms.InputTag("extendedPFEle"),
        src2=cms.InputTag("selectedGenPart"),
        dRmax=cms.double(0.05),
        storeUnmatched=cms.bool(args.storeUnmatched),
        varLabel=cms.string("Gen"),
    )

PFsource = "genMatchedPFEle" if not args.data else "extendedPFEle"
LPsource = "genMatchedLPEle" if not args.data else "extendedLPEle"
process.extendedLPEleMatchedToPF = cms.EDProducer(
    "ElectronMatcher",
    src1=cms.InputTag(LPsource),
    src2=cms.InputTag(PFsource),
    dRmax=cms.double(0.05),
    storeUnmatched=cms.bool(True),
    varLabel=cms.string("PF"),
)


process.extendedPFEleMatchedToLP = cms.EDProducer(
    "ElectronMatcher",
    src1=cms.InputTag(PFsource),
    src2=cms.InputTag(LPsource),
    dRmax=cms.double(0.05),
    storeUnmatched=cms.bool(True),
    varLabel=cms.string("LP"),
)

#! Training vars
vars = cms.PSet(
    # Aux
    pt=LazyVar("pt", float),
    eta=LazyVar("eta", float),
    phi=LazyVar("phi", float),
    isEB=LazyVar("isEB", bool),
    isPF=LazyVar("isPF", bool),
    # Training vars
    rawEnergy=LazyVar("userFloat('rawEnergy')", float, doc="Supercluster raw energy"),
    etaWidth=LazyVar("userFloat('etaWidth')", float, doc="Supercluster eta width"),
    phiWidth=LazyVar("userFloat('phiWidth')", float, doc="Supercluster phi width"),
    seedEnergyFraction=LazyVar(
        "userFloat('seedEnergyFraction')",
        float,
        doc="Supercluster seed energy / supercluster raw energy",
    ),
    rho=LazyVar("userFloat('rho')", float, doc="Energy density rho"),
    hcalOverEcalBc=LazyVar("userFloat('hcalOverEcalBc')", float, doc="H/E BC"),
    seedClusterEtaDiff=LazyVar(
        "userFloat('seedClusterEtaDiff')",
        float,
        doc="Seed cluster eta - supercluster eta",
    ),
    seedClusterPhiDiff=LazyVar(
        "userFloat('seedClusterPhiDiff')",
        float,
        doc="Seed cluster phi - supercluster phi",
    ),
    r9=LazyVar("userFloat ('r9')", float, doc="R9"),
    sigmaIetaIeta=LazyVar(
        "userFloat('sigmaIetaIeta')", float, doc="Full5x5 sigmaIetaIeta"
    ),
    sigmaIetaIphi=LazyVar(
        "userFloat('sigmaIetaIphi')", float, doc="Full5x5 sigmaIetaIphi"
    ),
    sigmaIphiIphi=LazyVar(
        "userFloat('sigmaIphiIphi')", float, doc="Full5x5 sigmaIphiIphi"
    ),
    eMaxOverE5x5=LazyVar("userFloat('eMaxOverE5x5')", float, doc="eMax / e5x5"),
    e2ndOverE5x5=LazyVar("userFloat('e2ndOverE5x5')", float, doc="e2nd / e5x5"),
    eTopOverE5x5=LazyVar("userFloat('eTopOverE5x5')", float, doc="eTop / e5x5"),
    eBottomOverE5x5=LazyVar(
        "userFloat('eBottomOverE5x5')", float, doc="eBottom / e5x5"
    ),
    eLeftOverE5x5=LazyVar("userFloat('eLeftOverE5x5')", float, doc="eLeft / e5x5"),
    eRightOverE5x5=LazyVar("userFloat('eRightOverE5x5')", float, doc="eRight / e5x5"),
    e2x5MaxOverE5x5=LazyVar(
        "userFloat('e2x5MaxOverE5x5')", float, doc="e2x5Max / e5x5"
    ),
    e2x5LeftOverE5x5=LazyVar(
        "userFloat('e2x5LeftOverE5x5')", float, doc="e2x5Left / e5x5"
    ),
    e2x5RightOverE5x5=LazyVar(
        "userFloat('e2x5RightOverE5x5')", float, doc="e2x5Right / e5x5"
    ),
    e2x5TopOverE5x5=LazyVar(
        "userFloat('e2x5TopOverE5x5')", float, doc="e2x5Top / e5x5"
    ),
    e2x5BottomOverE5x5=LazyVar(
        "userFloat('e2x5BottomOverE5x5')", float, doc="e2x5Bottom / e5x5"
    ),
    nSaturatedXtals=LazyVar(
        "userInt('nSaturatedXtals')",
        int,
        doc="Number of saturated crystals in the supercluster",
    ),
    numberOfClusters=LazyVar(
        "userInt('numberOfClusters')", int, doc="Number of clusters in the supercluster"
    ),
    iEtaOrX=LazyVar(
        "userInt('iEtaOrX')", int, doc="iEta (EB) or iX (EE) of the seed crystal"
    ),
    iPhiOrY=LazyVar(
        "userInt('iPhiOrY')", int, doc="iPhi (EB) or iY (EE) of the seed crystal"
    ),
    iEtaMod5=LazyVar("userInt('iEtaMod5')", int, doc="iEta mod 5 (EB only)"),
    iPhiMod2=LazyVar("userInt('iPhiMod2')", int, doc="iPhi mod 2 (EB only)"),
    iEtaMod20=LazyVar("userInt('iEtaMod20')", int, doc="iEta mod 20 (EB only)"),
    iPhiMod20=LazyVar("userInt('iPhiMod20')", int, doc="iPhi mod 20 (EB only)"),
    rawESEnergy=LazyVar(
        "userFloat('rawESEnergy')",
        float,
        doc="Preshower energy / supercluster raw energy (EE only)",
    ),

    ecalDrivenSeed =LazyVar(
        "userFloat('ecalDrivenSeed')",
        float,
        doc="Ecal driven seed",
    ),

    Tk_p=LazyVar(
        "userFloat('Tk_p')",
        float,
        doc="Gsf track momentum",
    ),
    Tk_eta=LazyVar(
        "userFloat('Tk_eta')",
        float,
        doc="Gsf track eta",
    ),
    Tk_fbrem=LazyVar(
        "userFloat('Tk_fbrem')",
        float,
        doc="Gsf track fbrem",
    ),
    Tk_errPRatio=LazyVar(
        "userFloat('Tk_errPRatio')",
        float,
        doc="Gsf track momentum error / momentum",
    ),
)

process.LPEleTable = cms.EDProducer(
    "SimpleCandidateFlatTableProducer",
    src=cms.InputTag("extendedLPEleMatchedToPF"),
    cut=cms.string(""),  # already filtered
    name=cms.string("LPEle"),
    doc=cms.string("Selected LP electrons"),
    singleton=cms.bool(False),
    extension=cms.bool(False),
    variables=vars.clone(
        PFIdx=LazyVar("userInt('PFIdx')", int, doc="Index of the matched PF electron"),
        isAlsoPF=LazyVar("userInt('isPF')", bool, doc="Is matched to a PF electron"),
    ),
)

process.PFEleTable = cms.EDProducer(
    "SimpleCandidateFlatTableProducer",
    src=cms.InputTag("extendedPFEleMatchedToLP"),
    cut=cms.string(""),  # already filtered
    name=cms.string("PFEle"),
    doc=cms.string("Selected PF electrons"),
    singleton=cms.bool(False),
    extension=cms.bool(False),
    variables=vars.clone(
        LPIdx=LazyVar("userInt('LPIdx')", int, doc="Index of the matched LP electron"),
        isAlsoLP=LazyVar("userInt('isLP')", bool, doc="Is matched to a LP electron"),
    ),
)


process.filterPath = cms.Path()
process.out = cms.OutputModule(
    "NanoAODOutputModule",
    fileName=cms.untracked.string(args.output),
    SelectEvents=cms.untracked.PSet(SelectEvents=cms.vstring("filterPath")),
    outputCommands=cms.untracked.vstring(
        "drop *",
        "keep *_LPEleTable_*_*",
        "keep *_PFEleTable_*_*",
    ),
)

process.p = cms.Path(
    process.extendedLPEle
    + process.extendedPFEle
    + process.extendedLPEleMatchedToPF
    + process.extendedPFEleMatchedToLP
    + process.LPEleTable
    + process.PFEleTable
)

if not args.data:
    # TODO Add gen variables to lp table
    process.PFEleTable.variables = process.PFEleTable.variables.clone(
        GenIdx=LazyVar(
            "userInt('GenIdx')", int, doc="Index of the matched Gen electron"
        ),
        isGenMatched=LazyVar(
            "userInt('isGen')", bool, doc="Is matched to a Gen electron"
        ),
    )

    process.LPEleTable.variables = process.LPEleTable.variables.clone(
        GenIdx=LazyVar(
            "userInt('GenIdx')", int, doc="Index of the matched Gen electron"
        ),
        isGenMatched=LazyVar(
            "userInt('isGen')", bool, doc="Is matched to a Gen electron"
        ),
    )

    process.GenPartTable = cms.EDProducer(
        "SimpleCandidateFlatTableProducer",
        src=cms.InputTag("selectedGenPart"),
        cut=cms.string(""),
        name=cms.string("GenEle"),
        doc=cms.string("Selected gen electrons"),
        singleton=cms.bool(False),
        extension=cms.bool(False),
        variables=cms.PSet(
            p=LazyVar("p", float),
            pt=LazyVar("pt", float),
            eta=LazyVar("eta", float),
            phi=LazyVar("phi", float),
            mass=LazyVar("mass", float),
            pdgId=LazyVar("pdgId", int),
            status=Var("status", int, doc="Particle status. 1=stable"),
        ),
    )

    process.nTrueIntFilter = cms.EDFilter(
        "PileupTrueNumIntFilter",
        src=cms.InputTag("slimmedAddPileupInfo"),
        maxTrueNumInteractions=cms.double(args.maxGenPU),
    )

    process.puTable = cms.EDProducer(
        "NPUTablesProducer",
        src=cms.InputTag("slimmedAddPileupInfo"),
        pvsrc=cms.InputTag("offlineSlimmedPrimaryVertices"),
        zbins=cms.vdouble([0.0, 1.7, 2.6, 3.0, 3.5, 4.2, 5.2, 6.0, 7.5, 9.0, 12.0]),
        savePtHatMax=cms.bool(True),
    )
    # TODO add matching process
    process.p = cms.Path(
        process.extendedLPEle
        + process.extendedPFEle
        + process.selectedGenPart
        + process.genMatchedLPEle
        + process.genMatchedPFEle
        + process.extendedLPEleMatchedToPF
        + process.extendedPFEleMatchedToLP
        + process.LPEleTable
        + process.PFEleTable
        + process.GenPartTable
        + process.puTable
    )
    process.filterPath += process.nonEmptyGenPartFilter
    process.filterPath += process.nTrueIntFilter
    process.out.outputCommands.append("keep *_GenPartTable_*_*")
    process.out.outputCommands.append("keep *_puTable_*_*")
process.e = cms.EndPath(process.out)
