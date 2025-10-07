#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/Common/interface/View.h"

template <typename T, typename U>
class Matcher : public edm::stream::EDProducer<> {
public:
  explicit Matcher(const edm::ParameterSet& iConfig)
      : Coll1Token_(consumes<std::vector<T>>(iConfig.getParameter<edm::InputTag>("src1"))),
        Coll2Token_(consumes<std::vector<U>>(iConfig.getParameter<edm::InputTag>("src2"))),
        dRMax(iConfig.getParameter<double>("dRmax")),
        storeUnmatched(iConfig.getParameter<bool>("storeUnmatched")),
        varLabel(iConfig.getParameter<std::string>("varLabel")) {
    produces<std::vector<T>>();
  }
  void produce(edm::Event& iEvent, const edm::EventSetup&) override {
    auto const& coll1 = iEvent.get(Coll1Token_);
    auto const& coll2 = iEvent.get(Coll2Token_);
    auto out = std::make_unique<std::vector<T>>();

    for (auto const& _cand1 : coll1) {
      T cand1(_cand1);
      float dRmin = 9999;
      int idx = -1;
      bool matched = false;
      for (int i2 = 0; i2 < (int)coll2.size(); ++i2) {
        float dR = reco::deltaR(cand1, coll2[i2]);
        if (dR < dRMax && dR < dRmin) {
          dRmin = dR;
          idx = i2;
          matched = true;
        }
      }
      if (matched)
        cand1.addUserInt(varLabel+"Idx", idx);
      else
        cand1.addUserInt(varLabel+"Idx", -1);
      cand1.addUserInt("is"+varLabel, matched);
      if (storeUnmatched or matched){
        out->push_back(cand1);
      }
    }
    iEvent.put(std::move(out));
  }
private:
  edm::EDGetTokenT<std::vector<T>> Coll1Token_;
  edm::EDGetTokenT<std::vector<U>> Coll2Token_;
  double dRMax;
  bool storeUnmatched;
  std::string varLabel;
};

typedef Matcher<pat::Electron, pat::Electron> ElectronMatcher;
typedef Matcher<pat::Electron, reco::GenParticle> GenElectronMatcher;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ElectronMatcher);
DEFINE_FWK_MODULE(GenElectronMatcher);