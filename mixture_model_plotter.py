import sys
import numpy as np
from bilby.core.result import read_in_result
from gwpopulation.models.spin import gaussian_chi_eff, skewnorm_chi_eff, eps_skewnorm_chi_eff
import matplotlib.pyplot as plt
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from skewnorm import skewnorm_mixture_model, eps_skewnorm_mixture_model




def spinplot(chi_eff_arr, posterior, draw):

    if 'eta_al' in posterior.keys():
        p_chi_eff = skewnorm_mixture_model(chi_eff_arr,
                                                mu_al=posterior['mu_al'][draw],
                                                sigma_al=posterior['sigma_al'][draw],
                                                eta_al=posterior['eta_al'][draw],
                                                lam_al=posterior['lam_al'][draw],
                                                sigma_dy=posterior['sigma_dy'][draw])

    elif 'eps_al' in posterior.keys():
        p_chi_eff = eps_skewnorm_mixture_model(chi_eff_arr,
                                                mu_al=posterior['mu_al'][draw],
                                                sigma_al=posterior['sigma_al'][draw],
                                                eps_al=posterior['eps_al'][draw],
                                                lam_al=posterior['lam_al'][draw],
                                                sigma_dy=posterior['sigma_dy'][draw])

    else:
        # defaulting to a truncated normal model
        raise ValueError('Unsupported model. Only pass mixture model runs')


    return p_chi_eff



def plotter(rundir):

    result = read_in_result(rundir + '/GWTC-3_result.json')

    chi_eff_arr = {'chi_eff':np.arange(-1, 1.01, 0.01)}

    p_chi_effs = np.zeros((chi_eff_arr['chi_eff'].size, result.posterior['alpha'].size))

    for draw in range(result.posterior['alpha'].size):

        p_chi_eff = spinplot(chi_eff_arr, result.posterior, draw)

        p_chi_effs[:, draw] = p_chi_eff

    dRdchi_effs = np.array(result.posterior['rate'])[None, :] * p_chi_effs


    ## we want to plot the seperate modes a little bit.
    if 'eps_al' in result.posterior.keys() or 'eta_al' in result.posterior.keys():
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(8.0, 4.0)

        #for draw in range(result.posterior['alpha'].size):
        for ii in range(500):

            draw = np.random.randint(result.posterior['alpha'].size)


            if result.posterior['lam_al'][draw] <= 0.5:
                ax1.plot(chi_eff_arr['chi_eff'], dRdchi_effs[:, draw],
                     color='#56B4E9', alpha=0.2, lw=0.25)
            else:
                ax2.plot(chi_eff_arr['chi_eff'], dRdchi_effs[:, draw],
                     color='#56B4E9', alpha=0.2, lw=0.25)


    random_rate = np.zeros(p_chi_effs.shape)
    aligned_rate = np.zeros(p_chi_effs.shape)

    for draw in range(result.posterior['alpha'].size):

        random_rate[:, draw] = result.posterior['rate'][draw] *\
                gaussian_chi_eff(chi_eff_arr, mu_chi_eff=0, sigma_chi_eff=result.posterior['sigma_dy'][draw],) *\
                (1 - result.posterior['lam_al'][draw])


        if 'eta_al' in result.posterior.keys():
            aligned_rate[:, draw] = result.posterior['rate'][draw] *\
                    skewnorm_chi_eff(chi_eff_arr,
                            mu_chi_eff=result.posterior['mu_al'][draw],
                            sigma_chi_eff=result.posterior['sigma_al'][draw],
                            eta_chi_eff=result.posterior['eta_al'][draw],) *\
                            result.posterior['lam_al'][draw]


        elif 'eps_al' in result.posterior.keys():
            aligned_rate[:, draw] = result.posterior['rate'][draw] *\
                    eps_skewnorm_chi_eff(chi_eff_arr,
                            mu_chi_eff=result.posterior['mu_al'][draw],
                            sigma_chi_eff=result.posterior['sigma_al'][draw],
                            eps_chi_eff=result.posterior['eps_al'][draw],) *\
                                    result.posterior['lam_al'][draw]


    ## for the low modes
    low_indx = result.posterior['lam_al'] <= 0.5
    high_indx = result.posterior['lam_al'] > 0.5


    import pdb; pdb.set_trace()


    ax1.plot(chi_eff_arr['chi_eff'], np.median(dRdchi_effs[:, low_indx], axis=1),  color='k', label='Low $\\lambda_{\\rm al}$ mode')
    ax1.plot(chi_eff_arr['chi_eff'], np.median(random_rate[:, low_indx], axis=1), color='#D55E00', label='Random')
    ax1.plot(chi_eff_arr['chi_eff'], np.median(aligned_rate[:, low_indx], axis=1), color='#009e74', label='Aligned')



    ax1.grid(ls=':', lw=0.5)
    ax1.legend(frameon=False)
    ax1.set_yscale('log')
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([1e0, 1000])
    ax1.tick_params(which='both', direction='in')

    ax2.plot(chi_eff_arr['chi_eff'], np.median(dRdchi_effs[:, high_indx], axis=1),  color='k', label='Low $\\lambda_{\\rm al}$ mode')
    ax2.plot(chi_eff_arr['chi_eff'], np.median(random_rate[:, high_indx], axis=1), color='#D55E00', label='Random')
    ax2.plot(chi_eff_arr['chi_eff'], np.median(aligned_rate[:, high_indx], axis=1), color='#009e74', label='Aligned')



    ax2.grid(ls=':', lw=0.5)
    ax2.legend(frameon=False)
    ax2.set_yscale('log')
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([1e0, 1000])
    ax2.set_yticklabels([])
    ax2.tick_params(which='both', direction='in')

    ax1.set_ylabel('$\\frac{dR}{d \\chi_{eff}}$')
    ax1.set_xlabel('$\\chi_{eff}$')
    ax2.set_xlabel('$\\chi_{eff}$')

    plt.subplots_adjust(wspace=0.1)
    plt.tight_layout()

    plt.savefig(rundir + "/dRdchi_eff_median_modal.png", dpi=300)
    plt.close()







if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise ValueError('Provide the path to the run directory')
    else:
        plotter(sys.argv[1])


