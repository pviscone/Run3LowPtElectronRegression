#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/Common/interface/View.h"

#include "vdt/vdtMath.h"

class ElectronExtenderProducer : public edm::stream::EDProducer<> {
public:
  explicit ElectronExtenderProducer(const edm::ParameterSet& iConfig)
      : EleToken_(consumes<pat::ElectronCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      rhoToken_(consumes<double>(iConfig.getParameter<edm::InputTag>("rho"))) {
    produces<pat::ElectronCollection>();
  }
  void produce(edm::Event& iEvent, const edm::EventSetup&) override {
    auto const& Eles = iEvent.get(EleToken_);
    float rho = iEvent.get(rhoToken_);
    auto out = std::make_unique<pat::ElectronCollection>();



    for (auto const& ele : Eles) {
      pat::Electron Ele(ele);
      auto sc = Ele.superCluster();
      auto seed = sc->seed();
      auto full5x5 = Ele.full5x5_showerShape();
      double e5x5_inverse = Ele.full5x5_showerShape().e5x5 != 0. ? vdt::fast_inv(Ele.full5x5_showerShape().e5x5) : 0.;
      auto gsfTrk = ele.gsfTrack();


      //! NEW Vars here
      Ele.addUserFloat("rawEnergy", sc->rawEnergy());
      Ele.addUserFloat("etaWidth", sc->etaWidth());
      Ele.addUserFloat("phiWidth", sc->phiWidth());
      Ele.addUserFloat("seedEnergyFraction", seed->energy() / sc->rawEnergy());
      Ele.addUserFloat("hcalOverEcalBc", Ele.full5x5_hcalOverEcalBc());
      Ele.addUserFloat("rho", rho);
      Ele.addUserFloat("seedClusterEtaDiff", sc->seed()->eta() - sc->position().Eta());
      Ele.addUserFloat("seedClusterPhiDiff", reco::deltaPhi(sc->seed()->phi(), sc->position().Phi()));
      Ele.addUserFloat("r9", Ele.full5x5_showerShape().r9);
      Ele.addUserFloat("sigmaIetaIeta", full5x5.sigmaIetaIeta);
      Ele.addUserFloat("sigmaIetaIphi", full5x5.sigmaIetaIphi);
      Ele.addUserFloat("sigmaIphiIphi", full5x5.sigmaIphiIphi);
      Ele.addUserFloat("eMaxOverE5x5", full5x5.eMax * e5x5_inverse);
      Ele.addUserFloat("e2ndOverE5x5", full5x5.e2nd * e5x5_inverse);
      Ele.addUserFloat("eTopOverE5x5", full5x5.eTop * e5x5_inverse);
      Ele.addUserFloat("eBottomOverE5x5", full5x5.eBottom * e5x5_inverse);
      Ele.addUserFloat("eLeftOverE5x5", full5x5.eLeft * e5x5_inverse);
      Ele.addUserFloat("eRightOverE5x5", full5x5.eRight * e5x5_inverse);
      Ele.addUserFloat("e2x5MaxOverE5x5", full5x5.e2x5Max * e5x5_inverse);
      Ele.addUserFloat("e2x5LeftOverE5x5", full5x5.e2x5Left * e5x5_inverse);
      Ele.addUserFloat("e2x5RightOverE5x5", full5x5.e2x5Right * e5x5_inverse);
      Ele.addUserFloat("e2x5TopOverE5x5", full5x5.e2x5Top * e5x5_inverse);
      Ele.addUserFloat("e2x5BottomOverE5x5", full5x5.e2x5Bottom * e5x5_inverse);
      Ele.addUserInt("nSaturatedXtals", Ele.nSaturatedXtals());
      Ele.addUserInt("numberOfClusters", std::max(0, int(sc->clusters().size())));

      Ele.addUserFloat("ecalDrivenSeed", Ele.ecalDrivenSeed());
      //!gsf track vars
      const float trkP = gsfTrk->pMode();
      const float trkEta = gsfTrk->etaMode();
      const float trkPErr = std::abs(gsfTrk->qoverpModeError()) * trkP * trkP;
      const float fbrem = ele.fbrem();
      Ele.addUserFloat("Tk_p", trkP);
      Ele.addUserFloat("Tk_eta", trkEta);
      Ele.addUserFloat("Tk_fbrem", fbrem);
      Ele.addUserFloat("Tk_errPRatio", trkPErr / trkP);

      Ele.addUserFloat("Tk_validFraction", gsfTrk->validFraction());
      Ele.addUserInt("Tk_nValidHits", gsfTrk->numberOfValidHits());
      Ele.addUserInt("Tk_nLostHits", gsfTrk->numberOfLostHits());
      Ele.addUserInt("Tk_nLostInnerHits", gsfTrk->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS));
      Ele.addUserInt("Tk_nLostOuterHits", gsfTrk->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_OUTER_HITS));

      Ele.addUserInt("nOfBrems", ele.numberOfBrems());
      Ele.addUserFloat("superClusterFBrem", ele.superClusterFbrem());

      Ele.addUserFloat("Tk_chi2", gsfTrk->normalizedChi2());
      Ele.addUserFloat("Tk_charge", gsfTrk->chargeMode());
      Ele.addUserFloat("Tk_qoverp", gsfTrk->qoverpMode());


      Ele.addUserFloat("Tk_dEtaEleClusterAtCalo", ele.deltaEtaEleClusterTrackAtCalo());
      Ele.addUserFloat("Tk_dPhiEleClusterAtCalo", ele.deltaPhiEleClusterTrackAtCalo());
      Ele.addUserFloat("Tk_dEtaSuperClusterAtVtx", ele.deltaEtaSuperClusterTrackAtVtx());
      Ele.addUserFloat("Tk_dPhiSuperClusterAtVtx", ele.deltaPhiSuperClusterTrackAtVtx());
      Ele.addUserFloat("Tk_dEtaSeedClusterAtCalo", ele.deltaEtaSeedClusterTrackAtCalo());
      Ele.addUserFloat("Tk_dPhiSeedClusterAtCalo", ele.deltaPhiSeedClusterTrackAtCalo());
      Ele.addUserFloat("Tk_dEtaSeedClusterAtVtx", ele.deltaEtaSeedClusterTrackAtVtx());
      Ele.addUserFloat("Tk_eSuperClusterOverP", ele.eSuperClusterOverP());
      Ele.addUserFloat("Tk_eSeedClusterOverP", ele.eSeedClusterOverP());
      Ele.addUserFloat("Tk_eSeedClusterOverPout", ele.eSeedClusterOverPout());
      Ele.addUserFloat("Tk_eEleClusterOverPout", ele.eEleClusterOverPout());


      if (Ele.isEB()){
        EBDetId detId((*seed).seed());
        Ele.addUserInt("iEtaOrX", detId.ieta());
        Ele.addUserInt("iPhiOrY", detId.iphi());
        Ele.addUserInt("iEtaMod5", (detId.ieta() - (detId.ieta() > 0 ? +1 : -1)) % 5);
        Ele.addUserInt("iPhiMod2", (detId.iphi() - 1) % 2);
        Ele.addUserInt("iEtaMod20", (std::abs(detId.ieta()) <= 25 ?
                                    (detId.ieta() - (detId.ieta() > 0 ? +1 : -1)) % 20 :
                                    (detId.ieta() - (detId.ieta() > 0 ? +26 : -26)) % 20));
        Ele.addUserInt("iPhiMod20", (detId.iphi() - 1) % 20);
        Ele.addUserFloat("rawESEnergy", 0.);

      } else{
        EEDetId detId((*seed).seed());
        Ele.addUserInt("iEtaOrX", detId.ix());
        Ele.addUserInt("iPhiOrY", detId.iy());
        Ele.addUserInt("iEtaMod5",-999);
        Ele.addUserInt("iPhiMod2",-999);
        Ele.addUserInt("iEtaMod20",-999);
        Ele.addUserInt("iPhiMod20",-999);
        Ele.addUserFloat("rawESEnergy", sc->preshowerEnergy()/sc->rawEnergy());
      }

      out->push_back(Ele);
    }
    iEvent.put(std::move(out));
  }
private:
  edm::EDGetTokenT<pat::ElectronCollection> EleToken_;
  edm::EDGetTokenT<double> rhoToken_;
};


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ElectronExtenderProducer);
