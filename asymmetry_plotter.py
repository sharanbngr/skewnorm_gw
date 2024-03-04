import corner
from bilby.core.result import read_in_result
from gwpopulation.models.spin import gaussian_chi_eff, skewnorm_chi_eff
import matplotlib.pyplot as plt
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
import numpy as np
import sys


def asymmetry_plotter(rundir):

    result = read_in_result(rundir + '/GWTC-3_result.json')

    pe = result.posterior

    corner.corner(np.array(pe['eta_chi_eff']).reshape((pe['eta_chi_eff'].size, 1)), color='b')
    plt.xlim([-10, 10])
    plt.axvline(0, ls='--', color='k')
    plt.xlabel('$\\eta_{\\chi}$')
    plt.ylabel('$p(\\eta_{\\chi} | d)$')
    plt.savefig(rundir + "/eta_chi_eff.png", dpi=250)
    plt.tight_layout()
    plt.close()


    delta = pe['eta_chi_eff'] / np.sqrt(1 + pe['eta_chi_eff']**2)

    def calc_mode(mu_chi_eff=0, sigma_chi_eff=0.5, eta_chi_eff=0):
        delta = eta_chi_eff / np.sqrt(1 + eta_chi_eff**2)
        norm_mode = np.sqrt(2/np.pi) - ((1 - np.pi/4) * (np.sqrt(2/np.pi) * delta)**3 )/(1 - (2*delta**2)/np.pi ) - 0.5 * np.sign(eta_chi_eff) * np.exp(-2*np.pi/np.abs(eta_chi_eff))
        return mu_chi_eff + sigma_chi_eff * norm_mode


    def calc_mean_variance(mu_chi_eff=0, sigma_chi_eff=0.5, eta_chi_eff=0):
        delta = eta_chi_eff / np.sqrt(1 + eta_chi_eff**2)

        mean =  mu_chi_eff + sigma_chi_eff *  delta * np.sqrt(2/np.pi)

        variance =  (1 - (2/np.pi) * delta**2) * sigma_chi_eff**2

        return mean, variance

    def calc_skewness(mu_chi_eff=0, sigma_chi_eff=0.5, eta_chi_eff=0):

        delta = eta_chi_eff / np.sqrt(1 + eta_chi_eff**2)


        gamma = (2 - np.pi/2) * ((delta * np.sqrt(2/np.pi))**3)  / (1 - 2 * delta**2 / np.pi)**1.5

        return gamma


    mode = calc_mode(pe['mu_chi_eff'], pe['sigma_chi_eff'], pe['eta_chi_eff'])
    mean, variance = calc_mean_variance(pe['mu_chi_eff'], pe['sigma_chi_eff'], pe['eta_chi_eff'])


    chi_eff_arr = {'chi_eff':np.arange(-1, 1.01, 0.01)}

    skewness = calc_skewness(pe['mu_chi_eff'], pe['sigma_chi_eff'], pe['eta_chi_eff'])

    mode_asymmetry = []
    central_asymmetry = []



    if 'trunc' in rundir:
        eta_chi_eff = np.zeros(result.posterior['beta'].shape)
    elif 'skew' in rundir:
        eta_chi_eff = result.posterior['eta_chi_eff']


    for ii in range(mode.size):
        chi_eff_pdf = skewnorm_chi_eff(chi_eff_arr, 
                                    mu_chi_eff=pe['mu_chi_eff'][ii], 
                                    sigma_chi_eff=np.exp(pe['sigma_chi_eff'][ii]), 
                                    eta_chi_eff=pe['eta_chi_eff'][ii])

        central_asymmetry.append( (chi_eff_pdf[chi_eff_arr['chi_eff']>=0].sum() -  chi_eff_pdf[chi_eff_arr['chi_eff']<0].sum()) / chi_eff_pdf.sum() )
        mode_asymmetry.append( (chi_eff_pdf[chi_eff_arr['chi_eff']>=mode[ii]].sum() -  chi_eff_pdf[chi_eff_arr['chi_eff']<mode[ii]].sum()) / chi_eff_pdf.sum() )



    hyperparams = {'$\\eta_{\\chi}$':np.array(pe['eta_chi_eff']) , 'mode':np.array(mode), '$\\gamma$' : np.array(skewness),
                'mode_asymmetry':np.array(mode_asymmetry), 'central_asymmetry':np.array(central_asymmetry)}
    corner.corner(hyperparams, show_titles=True, color='b') #, range=[(-2, 10), (-0.2, 0.25), (0.2, 1), (-1, 0.8), (-0.4, 1)])
    plt.savefig(rundir + "/asymmetry.png", dpi=250)




if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise ValueError('Provide the path to the run directory')
    else:
        asymmetry_plotter(sys.argv[1])