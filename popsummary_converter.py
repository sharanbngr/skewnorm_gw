import sys
import numpy as np
from bilby.core.result import read_in_result
from gwpopulation.models.spin import gaussian_chi_eff, skewnorm_chi_eff, eps_skewnorm_chi_eff
import matplotlib.pyplot as plt
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from skewnorm import skewnorm_mixture_model, eps_skewnorm_mixture_model
from popsummary.popresult import PopulationResult

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.size":16,
})

def spinplot(chi_eff_arr, posterior, draw):

    if 'eta_chi_eff' in posterior.keys():
        p_chi_eff = skewnorm_chi_eff(chi_eff_arr,
                            mu_chi_eff=posterior['mu_chi_eff'][draw],
                            sigma_chi_eff= posterior['sigma_chi_eff'][draw],
                            eta_chi_eff=posterior['eta_chi_eff'][draw],)

    elif 'eps_chi_eff' in posterior.keys():
        p_chi_eff = eps_skewnorm_chi_eff(chi_eff_arr,
                                        mu_chi_eff=posterior['mu_chi_eff'][draw],
                                        sigma_chi_eff=posterior['sigma_chi_eff'][draw],
                                        eps_chi_eff=posterior['eps_chi_eff'][draw])
    else:
        raise ValueError('unsupported model type for popsummary conversion')


    return p_chi_eff



def plotter(rundir):

    result = read_in_result(rundir + '/GWTC-3_result.json')

    chi_eff_arr = {'chi_eff':np.arange(-1, 1.01, 0.01)}

    p_chi_effs = np.zeros((chi_eff_arr['chi_eff'].size, result.posterior['beta'].size))


    plt.grid(ls=':', lw=0.5)
    for draw in range(result.posterior['beta'].size):

        p_chi_eff = spinplot(chi_eff_arr, result.posterior, draw)


        plt.plot(chi_eff_arr['chi_eff'], p_chi_eff, color='cyan', alpha=0.05, lw=0.25)

        p_chi_effs[:, draw] = p_chi_eff

    plt.plot(chi_eff_arr['chi_eff'], np.quantile(p_chi_effs, 0.05, axis=1), color='k', ls='--', lw=1.0)
    plt.plot(chi_eff_arr['chi_eff'], np.quantile(p_chi_effs, 0.95, axis=1), color='k', ls='--', lw=1.0)
    plt.plot(chi_eff_arr['chi_eff'], np.median(p_chi_effs, axis=1), label='median values', color='k', lw=1.5)

    plt.xlim([-0.6, 0.6])
    plt.ylim([0, 6])
    plt.legend(frameon=False)
    plt.ylabel('$p(\\chi_{eff} )$')
    plt.xlabel('$\\chi_{eff}$')
    plt.savefig(rundir + "/p_chi_eff.png", dpi=300)
    plt.close()

    dRdchi_effs = np.array(result.posterior['rate'])[None, :] * p_chi_effs
    for draw in range(result.posterior['beta'].size):
        plt.plot(chi_eff_arr['chi_eff'], dRdchi_effs[:, draw],
                color='cyan', alpha=0.05, lw=0.25)

    plt.plot(chi_eff_arr['chi_eff'], np.quantile(dRdchi_effs, 0.05, axis=1), color='k', ls='--', lw=1.0)
    plt.plot(chi_eff_arr['chi_eff'], np.quantile(dRdchi_effs, 0.95, axis=1), color='k', ls='--', lw=1.0)
    plt.plot(chi_eff_arr['chi_eff'], np.median(dRdchi_effs, axis=1) , label='median values', color='k', lw=1.5 )

    plt.grid(ls=':', lw=0.5)
    plt.legend(frameon=False)
    plt.yscale('log')
    plt.xlim([-0.6, 0.6])
    plt.ylim([1e0, 1000])
    plt.ylabel('$\\frac{dR}{d \\chi_{eff}}$')
    plt.xlabel('$\\chi_{eff}$')
    plt.savefig(rundir + "/dRdchi_eff.png", dpi=300)
    plt.close()


    popsummary_result = PopulationResult(
            fname=rundir + '/popsummary_output.h5',
            hyperparameters = list(result.posterior.keys()),
            hyperparameter_latex_labels=result.get_latex_labels_from_parameter_keys(result.posterior.keys()),
            )


    popsummary_result.set_hyperparameter_samples(result.posterior)
    popsummary_result.set_rates_on_grids('Effective inspiral spin',
                                        grid_params='chi_eff',
                                        positions=chi_eff_arr['chi_eff'],
                                        rates=p_chi_effs,
                                        attribute_keys='description',
                                        attributes='the rates array is a 2d array with each colum denoting a seperate ppd of p(chi_eff)')
    
if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise ValueError('Provide the path to the run directory')
    else:
        plotter(sys.argv[1])