import sys
import numpy as np
from bilby.core.result import read_in_result
from gwpopulation.models.spin import gaussian_chi_eff, skewnorm_chi_eff, eps_skewnorm_chi_eff
import matplotlib.pyplot as plt
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution

def norm_mixture_model_chi_eff(dataset, mu_al, sigma_al, eta_al, lam_al, sigma_dy):

    pdf = mixture_model_chi_eff(dataset, mu_al, sigma_al, eta_al, lam_al, sigma_dy)

    # calc norm
    chi_arr = {'chi_eff': np.array([0.005 * i for i in range(-200, 201, 1)])}
    norm = 0.005 * np.sum(mixture_model_chi_eff(chi_arr, mu_al, sigma_al, eta_al, lam_al, sigma_dy))

    return pdf/norm

def mixture_model_chi_eff(dataset, mu_al, sigma_al, eta_al, lam_al, sigma_dy):

    '''
    Mixture model combining an aligned and dynamic popualtion

    Parameters
    ----------
    dataset: dict
        Input data, must contain `chi_eff` 
    mu_al: float
        Mean parameter of the aligned population 
    sigma_al: float
        Scale parameter of the aligned population
    eta_al: float
        skewness parameter of the aligned population
    lam_al: float
        fraction of systems in the aligned population
    sigma_dy: float
        Scale parameter for the dynamic population (zero mean)

    Returns
    -------
    array-like: The probability


    '''

    # Should be normalized because gaussian_chi_eff and skewnorm_chi_eff already are
    pdf = (1 - lam_al) * gaussian_chi_eff(dataset, mu_chi_eff=0, sigma_chi_eff=sigma_dy) +  \
            lam_al * skewnorm_chi_eff(dataset, mu_chi_eff=mu_al, sigma_chi_eff=sigma_al, eta_chi_eff=eta_al)

    return pdf




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
    elif 'sigma_dy' in posterior.keys():
        p_chi_eff = norm_mixture_model_chi_eff(chi_eff_arr, 
                                                mu_al=posterior['mu_al'][draw], 
                                                sigma_al=posterior['sigma_al'][draw],
                                                eta_al=posterior['eta_al'][draw], 
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
        #p_chi_eff = skewnorm_chi_eff(chi_eff_arr,
        #                            mu_chi_eff=result.posterior['mu_chi_eff'][draw],
        #                            sigma_chi_eff= result.posterior['sigma_chi_eff'][draw],
        #                            eta_chi_eff=eta_x[draw],)

        plt.plot(chi_eff_arr['chi_eff'], p_chi_eff, color='cyan', alpha=0.05, lw=0.25)

        p_chi_effs[:, draw] = p_chi_eff

    plt.plot(chi_eff_arr['chi_eff'], np.quantile(p_chi_effs, 0.05, axis=1), color='k', ls='--', lw=1.0)
    plt.plot(chi_eff_arr['chi_eff'], np.quantile(p_chi_effs, 0.95, axis=1), color='k', ls='--', lw=1.0)
    plt.plot(chi_eff_arr['chi_eff'], np.median(p_chi_effs, axis=1), label='median values', color='k', lw=1.5)

    plt.xlim([-1, 1])
    plt.legend(frameon=False)
    plt.ylabel('$p(\\chi_{eff} )$')
    plt.xlabel('$\\chi_{eff}$')
    plt.savefig(rundir + "/chi_eff_plot.png", dpi=300)
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
    plt.ylim([1e-6, 1e0])
    plt.savefig(rundir + "/dRm1.png", dpi=300)
    plt.close()



    ## plot rate histogram
    plt.hist(result.posterior['rate'], bins=50, histtype="step", density=True)
    plt.xlabel("$ {\\mathcal{R}_0}$")
    plt.ylabel("$p({\mathcal{R}_0 | d})$")
    plt.savefig(rundir + '/R0.png', dpi=300)
    plt.close()
    import pdb; pdb.set_trace()






if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise ValueError('Provide the path to the run directory')
    else:
        plotter(sys.argv[1])


