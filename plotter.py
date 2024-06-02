import sys
import numpy as np
from bilby.core.result import read_in_result
from gwpopulation.models.spin import gaussian_chi_eff, skewnorm_chi_eff, eps_skewnorm_chi_eff
import matplotlib.pyplot as plt
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from skewnorm import skewnorm_mixture_model, eps_skewnorm_mixture_model




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
    elif 'eta_al' in posterior.keys():
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
        p_chi_eff = skewnorm_chi_eff(chi_eff_arr,
                            mu_chi_eff=posterior['mu_chi_eff'][draw],
                            sigma_chi_eff= posterior['sigma_chi_eff'][draw],
                            eta_chi_eff=0,)

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

    ## we want to plot the seperate modes a little bit.
    if 'eps_al' in result.posterior.keys() or 'eta_al' in result.posterior.keys():
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(8.0, 4.0)

        #for draw in range(result.posterior['beta'].size):
        for ii in range(500):

            draw = np.random.randint(result.posterior['beta'].size)


            if result.posterior['lam_al'][draw] <= 0.5:
                ax1.plot(chi_eff_arr['chi_eff'], dRdchi_effs[:, draw],
                     color='#56B4E9', alpha=0.2, lw=0.25)
            else:
                ax2.plot(chi_eff_arr['chi_eff'], dRdchi_effs[:, draw],
                     color='#56B4E9', alpha=0.2, lw=0.25)


        low_mode_index = np.argmax(np.where(result.posterior['lam_al'] <= 0.5, result.posterior['log_likelihood'], -1e80))
        high_mode_index = np.argmax(np.where(result.posterior['lam_al'] >0.5, result.posterior['log_likelihood'], -1e80))

        random_channel_low_mode_rate = result.posterior['rate'][low_mode_index] * gaussian_chi_eff(chi_eff_arr, mu_chi_eff=0,
                                                                            sigma_chi_eff=result.posterior['sigma_dy'][low_mode_index],) * \
                                                                                    (1 - result.posterior['lam_al'][low_mode_index])
        random_channel_high_mode_rate = result.posterior['rate'][high_mode_index] * gaussian_chi_eff(chi_eff_arr, mu_chi_eff=0,
                                                                            sigma_chi_eff=result.posterior['sigma_dy'][high_mode_index],) * \
                                                                                    (1 - result.posterior['lam_al'][high_mode_index])



        if 'eta_al' in result.posterior.keys():
            aligned_channel_low_mode_rate = result.posterior['rate'][low_mode_index] * skewnorm_chi_eff(chi_eff_arr,
                                                                            mu_chi_eff=result.posterior['mu_al'][low_mode_index],
                                                                            sigma_chi_eff=result.posterior['sigma_al'][low_mode_index],
                                                                            eta_chi_eff=result.posterior['eta_al'][low_mode_index],) * result.posterior['lam_al'][low_mode_index]

            aligned_channel_high_mode_rate = result.posterior['rate'][high_mode_index] * skewnorm_chi_eff(chi_eff_arr,
                                                                            mu_chi_eff=result.posterior['mu_al'][high_mode_index],
                                                                            sigma_chi_eff=result.posterior['sigma_al'][high_mode_index],
                                                                            eta_chi_eff=result.posterior['eta_al'][high_mode_index],) * result.posterior['lam_al'][high_mode_index]

        elif 'eps_al' in result.posterior.keys():
            aligned_channel_low_mode_rate = result.posterior['rate'][low_mode_index] * eps_skewnorm_chi_eff(chi_eff_arr,
                                                                            mu_chi_eff=result.posterior['mu_al'][low_mode_index],
                                                                            sigma_chi_eff=result.posterior['sigma_al'][low_mode_index],
                                                                            eps_chi_eff=result.posterior['eps_al'][low_mode_index],) * result.posterior['lam_al'][low_mode_index]

            aligned_channel_high_mode_rate = result.posterior['rate'][high_mode_index] * eps_skewnorm_chi_eff(chi_eff_arr,
                                                                            mu_chi_eff=result.posterior['mu_al'][high_mode_index],
                                                                            sigma_chi_eff=result.posterior['sigma_al'][high_mode_index],
                                                                            eps_chi_eff=result.posterior['eps_al'][high_mode_index],) * result.posterior['lam_al'][high_mode_index]


        ax1.plot(chi_eff_arr['chi_eff'], dRdchi_effs[:, low_mode_index],  color='k', label='Low $\\lambda_{\\rm al}$ mode')
        ax1.plot(chi_eff_arr['chi_eff'], random_channel_low_mode_rate, color='#D55E00', label='Random orientation')
        ax1.plot(chi_eff_arr['chi_eff'], aligned_channel_low_mode_rate, color='#009e74', label='Aligned')



        ax1.grid(ls=':', lw=0.5)
        ax1.legend(frameon=False)
        ax1.set_yscale('log')
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([1e0, 1000])
        ax1.tick_params(which='both', direction='in')

        ax2.plot(chi_eff_arr['chi_eff'], dRdchi_effs[:, high_mode_index], color='k', label='High $\\lambda_{\\rm al}$ mode')
        ax2.plot(chi_eff_arr['chi_eff'], random_channel_high_mode_rate, color='#D55E00', label='Random orientation')
        ax2.plot(chi_eff_arr['chi_eff'], aligned_channel_high_mode_rate, color='#009e74', label='Aligned')


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

        plt.savefig(rundir + "/dRdchi_eff_modal.png", dpi=300)
        plt.close()



    ## make mass plots
    mass_pdf = SinglePeakSmoothedMassDistribution()

    m1_arr = {'mass_1':np.arange(1, 100, 0.5)}

    p_m1s = np.zeros((m1_arr['mass_1'].size, result.posterior['beta'].size))

    for draw in range(result.posterior['beta'].size):

        p_m1 = mass_pdf.p_m1(m1_arr, alpha=result.posterior['alpha'][draw],
                            mmax=result.posterior['mmax'][draw],
                            mmin=result.posterior['mmin'][draw],
                            delta_m=result.posterior['delta_m'][draw],
                            lam=result.posterior['lam'][draw],
                            mpp=result.posterior['mpp'][draw],
                            sigpp=result.posterior['sigpp'][draw],)

        plt.plot(m1_arr['mass_1'], p_m1, color='cyan', alpha=0.05, lw=0.25)

        p_m1s[:, draw] = p_m1

    plt.plot(m1_arr['mass_1'], np.quantile(p_m1s, 0.05, axis=1), color='k', ls='--', lw=1.0)
    plt.plot(m1_arr['mass_1'], np.quantile(p_m1s, 0.95, axis=1), color='k', ls='--', lw=1.0)
    plt.plot(m1_arr['mass_1'], np.median(p_m1s, axis=1) , label='median values', color='k', lw=1.5 )


    plt.legend(frameon=False)
    plt.yscale('log')
    plt.ylabel('$p(m_1)$')
    plt.xlabel('$m_1 [M_{\odot}]$')
    plt.xlim([1, 100])
    plt.ylim([1e-6, 1e0])
    plt.savefig(rundir + "/p_m1.png", dpi=300)
    plt.close()


    dRdm1s = np.array(result.posterior['rate'])[None, :] * p_m1s
    for draw in range(result.posterior['beta'].size):
        plt.plot(m1_arr['mass_1'], dRdm1s[:, draw],
                 color='cyan', alpha=0.05, lw=0.25)

    plt.plot(m1_arr['mass_1'], np.quantile(dRdm1s, 0.05, axis=1), color='k', ls='--', lw=1.0)
    plt.plot(m1_arr['mass_1'], np.quantile(dRdm1s, 0.95, axis=1), color='k', ls='--', lw=1.0)
    plt.plot(m1_arr['mass_1'], np.median(dRdm1s, axis=1) , label='median values', color='k', lw=1.5 )

    plt.legend(frameon=False)
    plt.yscale('log')
    plt.ylabel('$\\frac{dR}{dm_1}$')
    plt.xlabel('$m_1 [M_{\odot}]$')
    plt.xlim([1, 100])
    plt.ylim([1e-4, 1e1])
    plt.savefig(rundir + "/dRm1.png", dpi=300)
    plt.close()



    ## plot rate histogram
    plt.hist(result.posterior['rate'], bins=50, histtype="step", density=True)
    plt.xlabel("$ {\\mathcal{R}_0}$")
    plt.ylabel("$p({\mathcal{R}_0 | d})$")
    plt.savefig(rundir + '/R0.png', dpi=300)
    plt.close()






if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise ValueError('Provide the path to the run directory')
    else:
        plotter(sys.argv[1])


