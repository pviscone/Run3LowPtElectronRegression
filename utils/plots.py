import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mplhep as hep
import numpy as np
import hist
import os

hep.style.use("CMS")

colors=[
    "#3f90da",
    "#ffa90e",
    "#bd1f01",
    "#94a4a2",
    "#832db6",
    "#a96b59",
    "#e76300",
    "#b9ac70",
    "#92dadd",
    "#717581",
]

markers = ["s", "P", "X"]


def plot_distributions(df, features = None, weight=None, log=False, savefolder="plots/distributions"):
    if features is None:
        keys = df.columns
    else:
        keys = features
    if isinstance(weight, str):
        weight = df[weight].values
    nfeatures = len(keys)
    os.makedirs(savefolder, exist_ok=True)
    for i in range(nfeatures):
        key = keys[i]

        fig, ax = plt.subplots()
        ax.hist(df[key], bins=(100), weights=weight, density=True)
        ax.set_xlabel(key)
        if log:
            ax.set_yscale("log")
        fig.savefig(f"{savefolder}/{key}.pdf")
        fig.savefig(f"{savefolder}/{key}.png")
        plt.close(fig)



def response_resolution(df,
                               pratio_dict,
                               genp_,
                               eta_,
                               eta_bins = None,
                               genp_bins = None,
                               plot_distributions = False,
                               savefolder = "plots",):
    os.makedirs(savefolder, exist_ok=True)
    if eta_bins is None:
        eta_bins = np.array([0, 0.7, 1.2, 1.5])
    if genp_bins is None:
        genp_bins = np.arange(1,500,5)
    eta = np.abs(df[eta_].values)
    pratio_dict = {key: df[value].values for key, value in pratio_dict.items()}
    genp = df[genp_].values

    for eta_min, eta_max in zip(eta_bins[:-1], eta_bins[1:]):
        if plot_distributions:
            os.makedirs(f"{savefolder}/distributions", exist_ok=True)
        os.makedirs(f"{savefolder}/response", exist_ok=True)
        os.makedirs(f"{savefolder}/resolution", exist_ok=True)
        mask = (eta >= eta_min) & (eta < eta_max)
        genp_eta = genp[mask]
        centers={key:np.array([]) for key in pratio_dict.keys()}
        medians={key:np.array([]) for key in pratio_dict.keys()}
        perc16s={key:np.array([]) for key in pratio_dict.keys()}
        perc84s={key:np.array([]) for key in pratio_dict.keys()}
        perc5s={key:np.array([]) for key in pratio_dict.keys()}
        perc95s={key:np.array([]) for key in pratio_dict.keys()}
        residuals = {key:np.array([]) for key in pratio_dict.keys()}
        variances = {key:np.array([]) for key in pratio_dict.keys()}
        n = {key:np.array([]) for key in pratio_dict.keys()}
        for genp_min, genp_max in zip(genp_bins[:-1], genp_bins[1:]):
            skip = False
            for idx, (label, pratio) in enumerate(pratio_dict.items()):
                pratio_eta = pratio[mask]
                mask_genp = (genp_eta >= genp_min) & (genp_eta < genp_max)
                pratio_masked = pratio_eta[mask_genp]
                if len(pratio_masked) ==0:
                    skip = True
                    break
                median = np.median(pratio_masked)
                perc16 = np.percentile(pratio_masked, 16)
                perc84 = np.percentile(pratio_masked, 84)
                perc5 = np.percentile(pratio_masked, 5)
                perc95 = np.percentile(pratio_masked, 95)
                res = np.median(genp_eta[mask_genp]*np.abs(pratio_masked - 1))
                var = np.sum(((genp_eta[mask_genp]*np.abs(pratio_masked - 1))**2)/(len(genp_eta[mask_genp]) - 1))
                centers[label] = np.append(centers[label],((genp_min + genp_max) / 2))
                medians[label] = np.append(medians[label],(median))
                perc16s[label] = np.append(perc16s[label],(perc16))
                perc84s[label] = np.append(perc84s[label],(perc84))
                perc5s[label] = np.append(perc5s[label],(perc5))
                perc95s[label] = np.append(perc95s[label],(perc95))
                residuals[label] = np.append(residuals[label],(res))
                variances[label] = np.append(variances[label],(var))
                n[label] = np.append(n[label],(len(pratio_masked)))
            if skip:
                continue

            if plot_distributions:
                fig, ax = plt.subplots()
                for idx, (label, pratio) in enumerate(pratio_dict.items()):

                    ax.axvline(1, color='black', alpha=0.3)
                    ax.set_title(f"$| \eta |$: [{eta_min},{eta_max}], GenP: [{genp_min},{genp_max}]")
                    ax.set_xlabel("Reco $p$ / $p^{\\text{Gen}}$")
                    ax.set_ylabel("Density")
                    ax.legend(fontsize=15)
                    h = hist.Hist(hist.axis.Regular(29, 0.3, 1.7, name="pratio", label="Reco $p$ / $p^{\\text{Gen}}$"))
                    h.fill(pratio_masked)
                    hep.histplot(h, density=True, alpha=0.75, histtype='step', label=label, linewidth=2, color=colors[idx], ax=ax)
                    ax.axvline(median, color=colors[idx], linestyle='--', label=f'Median {label}: {median:.2f}', alpha=0.7)
                    ax.fill_betweenx(y=[0, ax.get_ylim()[1]], x1=perc16, x2=perc84, color=colors[idx], alpha=0.2, label=f'16-84% {label}: [{perc16:.2f}, {perc84:.2f}]')
                ax.legend(fontsize=12)
                #fig.savefig(f"{savefolder}/distributions/pratio_eta_{str(eta_min).replace('.','')}_{str(eta_max).replace('.','')}_genp_{str(genp_min).replace('.','')}_{str(genp_max).replace('.','')}.png")
                fig.savefig(f"{savefolder}/distributions/pratio_eta_{str(eta_min).replace('.','')}_{str(eta_max).replace('.','')}_genp_{str(genp_min).replace('.','')}_{str(genp_max).replace('.','')}.pdf")
                plt.close(fig)


        legend_order = []
        counter=0
        for i in range(3*len(pratio_dict)):
            if i%3==0:
                legend_order.append(counter)
            else:
                legend_order.append(counter+len(pratio_dict))
                counter+=1
        os.makedirs(savefolder, exist_ok=True)
        fig, ax = plt.subplots()
        ax.axhline(1, color='gray', linestyle='-', alpha=0.5, zorder=99)
        ax.text(0, 1.3, f"${eta_min}< | \eta | < {eta_max}$")
        ax.set_xlabel("$p^{\\text{Gen}}$ [GeV]")
        ax.set_ylabel("$p^{\\text{Reco}}$ / $p^{\\text{Gen}}$")
        for idx, label in enumerate(pratio_dict.keys()):
            diff = centers[label][1:]-centers[label][:-1]
            diff = np.append(diff, diff[-1])
            median_label=label.replace(' ',r'\ ')
            ax.plot(centers[label]+idx*diff*0.3-diff/4, medians[label], ".", marker=markers[idx], label=f"$\\bf{{{median_label}}}$\nMedian", color=colors[idx], markeredgecolor='black', markeredgewidth=1, markersize=9, zorder=10)
            ax.errorbar(centers[label]+idx*diff*0.3-diff/4, medians[label],
                                    yerr=[medians[label] - perc16s[label], perc84s[label] - medians[label]],
                                    color=colors[idx], alpha=1, label=f"{label} 16/84%", linewidth=3, fmt="o", markersize=0)
            ax.errorbar(centers[label]+idx*diff*0.3-diff/4, medians[label],
                        yerr=[medians[label] - perc5s[label], perc95s[label] - medians[label]],
                        color=colors[idx], alpha=0.5, label=f"{label} 5/95% Quantiles", fmt = "o", markersize=0, linewidth=2)

            for ii in range(len(centers[label])):
                ax.axvline(centers[label][ii]-diff[ii]/2, alpha=0.05, color="gray", linestyle=':')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[idx] for idx in legend_order],
                     [labels[idx] for idx in legend_order],
                     fontsize=20, loc='lower right')
        ax.set_ylim(0.3,1.4)
        ax.set_xlabel("$p^{\\text{Gen}}$ [GeV]")
        hep.cms.text("Preliminary", ax=ax, loc=0, fontsize=22)
        #hep.cms.lumitext("PU 0-70 (13.6 TeV)", ax=ax[0], fontsize=22)
        fig.savefig(f"{savefolder}/response/aresponse_eta_{str(eta_min).replace('.','')}_{str(eta_max).replace('.','')}.pdf")
        fig.savefig(f"{savefolder}/response/aresponse_eta_{str(eta_min).replace('.','')}_{str(eta_max).replace('.','')}.png")
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.axhline(0, color='black', linestyle=':', alpha=0.3)
        ax.set_xlabel("$p^{\\text{Gen}}$ [GeV]")
        ax.set_ylabel("$\sigma_{eff}$($p^{\\text{Reco}}$ / $p^{\\text{Gen}}$)")
        for idx, label in enumerate(pratio_dict.keys()):
            width = (perc84s[label] - perc16s[label])# / centers[eta_idx][label]
            diff=(centers[label][1:]-centers[label][:-1])/2
            yerr = np.sqrt(3.715* variances[label]/n[label])[:-1]/centers[label][:-1]
            ax.errorbar(centers[label][:-1], width[:-1], xerr=diff, yerr=yerr/2, marker='o', label=label, color=colors[idx], markeredgecolor='black', markeredgewidth=1, markersize=5, ls="none")
        ax.legend()
        ax.set_ylim(0, 0.5)
        ax.text(0, 0.45, f"${eta_min} < | \eta | < {eta_max}$")
        #ax.set_yscale("log")
        hep.cms.text("Preliminary", ax=ax, loc=0, fontsize=22)
        #hep.cms.lumitext("PU 200 (14 TeV)", ax=ax, fontsize=22)
        fig.savefig(f"{savefolder}/resolution/resolution_eta_{str(eta_min).replace('.','')}_{str(eta_max).replace('.','')}.pdf")
        fig.savefig(f"{savefolder}/resolution/resolution_eta_{str(eta_min).replace('.','')}_{str(eta_max).replace('.','')}.png")
        plt.close(fig)

