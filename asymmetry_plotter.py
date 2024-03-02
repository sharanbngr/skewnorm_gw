import corner
from bilby.core.result import read_in_result
from gwpopulation.models.spin import gaussian_chi_eff, skewnorm_chi_eff
import matplotlib.pyplot as plt
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
import numpy as np


rundir = './skew_norm_jax_inc_rate'



result = read_in_result(rundir + '/GWTC-3_result.json')



pe = result.posterior

corner.corner(np.array(pe['eta_x']).reshape((pe['eta_x'].size, 1)), color='b')
plt.xlim([-10, 10])
plt.axvline(0, ls='--', color='k')
plt.xlabel('$\\eta_{\\chi}$')
plt.ylabel('$p(\\eta_{\\chi} | d)$')
plt.savefig(rundir + "/eta_x.png", dpi=250)
plt.tight_layout()
plt.close()


delta = pe['eta_x'] / np.sqrt(1 + pe['eta_x']**2)

def calc_mode(mu_x=0, log_sig_x=0.5, eta_x=0):
    delta = eta_x / np.sqrt(1 + eta_x**2)
    norm_mode = np.sqrt(2/np.pi) - ((1 - np.pi/4) * (np.sqrt(2/np.pi) * delta)**3 )/(1 - (2*delta**2)/np.pi ) - 0.5 * np.sign(eta_x) * np.exp(-2*np.pi/np.abs(eta_x))
    return mu_x + np.exp(log_sig_x) * norm_mode


def calc_mean_variance(mu_x=0, log_sig_x=0.5, eta_x=0):
    delta = eta_x / np.sqrt(1 + eta_x**2)

    mean =  mu_x + np.exp(log_sig_x) *  delta * np.sqrt(2/np.pi)

    variance =  (1 - (2/np.pi) * delta**2) * np.exp(log_sig_x)**2

    return mean, variance

def calc_skewness(mu_x=0, log_sig_x=0.5, eta_x=0):

    delta = eta_x / np.sqrt(1 + eta_x**2)


    gamma = (2 - np.pi/2) * ((delta * np.sqrt(2/np.pi))**3)  / (1 - 2 * delta**2 / np.pi)**1.5

    return gamma


mode = calc_mode(pe['mu_x'], pe['log_sig_x'], pe['eta_x'])
mean, variance = calc_mean_variance(pe['mu_x'], pe['log_sig_x'], pe['eta_x'])


chi_eff_arr = {'chi_eff':np.arange(-1, 1.01, 0.01)}

skewness = calc_skewness(pe['mu_x'], pe['log_sig_x'], pe['eta_x'])

mode_asymmetry = []
central_asymmetry = []



if 'trunc' in rundir:
    eta_x = np.zeros(result.posterior['beta'].shape)
elif 'skew' in rundir:
    eta_x = result.posterior['eta_x']


for ii in range(mode.size):
    chi_eff_pdf = skewnorm_chi_eff(chi_eff_arr, 
                                   mu_chi_eff=pe['mu_x'][ii], 
                                   sigma_chi_eff=np.exp(pe['log_sig_x'][ii]), 
                                   eta_chi_eff=pe['eta_x'][ii])

    central_asymmetry.append( (chi_eff_pdf[chi_eff_arr['chi_eff']>=0].sum() -  chi_eff_pdf[chi_eff_arr['chi_eff']<0].sum()) / chi_eff_pdf.sum() )
    mode_asymmetry.append( (chi_eff_pdf[chi_eff_arr['chi_eff']>=mode[ii]].sum() -  chi_eff_pdf[chi_eff_arr['chi_eff']<mode[ii]].sum()) / chi_eff_pdf.sum() )



hyperparams = {'$\\eta_{\\chi}$':np.array(pe['eta_x']) , 'mode':np.array(mode), '$\\gamma$' : np.array(skewness),
               'mode_asymmetry':np.array(mode_asymmetry), 'central_asymmetry':np.array(central_asymmetry)}
corner.corner(hyperparams, show_titles=True, color='b') #, range=[(-2, 10), (-0.2, 0.25), (0.2, 1), (-1, 0.8), (-0.4, 1)])
plt.savefig(rundir + "/asymmetry.png", dpi=250)


import pdb; pdb.set_trace()


