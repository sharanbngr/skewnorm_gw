import numpy as np
from bilby.core.result import read_in_result
from gwpopulation.models.spin import gaussian_chi_eff, skewnorm_chi_eff
import matplotlib.pyplot as plt
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution




rundir = './trunc_norm_jax_inc_rate'



result = read_in_result(rundir + '/GWTC-3_result.json')

import pdb; pdb.set_trace()
chi_eff_arr = {'chi_eff':np.arange(-1, 1.01, 0.01)}

p_chi_effs = np.zeros((chi_eff_arr['chi_eff'].size, result.posterior['beta'].size))


if 'trunc' in rundir:
    eta_x = np.zeros(result.posterior['beta'].shape)
elif 'skew' in rundir:
    eta_x = result.posterior['eta_x']


plt.grid(ls=':', lw=0.5)
for draw in range(result.posterior['beta'].size):

    p_chi_eff = skewnorm_chi_eff(chi_eff_arr,
                                 mu_chi_eff=result.posterior['mu_x'][draw],
                                 sigma_chi_eff=np.exp(result.posterior['log_sig_x'][draw]),
                                 eta_chi_eff=eta_x[draw],)

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
plt.ylim([1e-4, 1e0])
plt.savefig(rundir + "/p_m1.png", dpi=300)
plt.close()
