#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/View.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

class PileupTrueNumIntFilter : public edm::stream::EDFilter<> {
public:
  explicit PileupTrueNumIntFilter(const edm::ParameterSet& iConfig)
    : src_(consumes<std::vector<PileupSummaryInfo>>(iConfig.getParameter<edm::InputTag>("src"))),
      maxTrueInt_(iConfig.getParameter<double>("maxTrueNumInteractions")) {}

  bool filter(edm::Event& iEvent, const edm::EventSetup&) override {
    auto const& pileupInfos = iEvent.get(src_);
    for (auto const& pu : pileupInfos) {
      if (pu.getBunchCrossing() == 0) {
        if (pu.getTrueNumInteractions() > maxTrueInt_) return false;
        return true;
      }
    }
    // If no BX=0 object, reject the event
    return false;
  }

private:
  edm::EDGetTokenT<std::vector<PileupSummaryInfo>> src_;
  double maxTrueInt_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PileupTrueNumIntFilter);