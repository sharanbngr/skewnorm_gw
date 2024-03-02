from gwpopulation.utils import skewnorm
from gwpopulation.models.spin import skewnorm_chi_eff
import numpy as np
import pickle
import matplotlib.pyplot as plt


mu_x = 0
log_sig_xs = [-4, -1.5, 0]


chi_eff_arr = {'chi_eff':np.arange(-1, 1.0001, 0.0001)}
eta_x_arr = np.arange(0, 100.1, 0.1)


## loading injections. 
with open('/projects/p31963/sharan/pop/O3_injections.pkl', 'rb') as f:
    injs = pickle.load(f)

#### need to calc chi_eff for injs
injs['chi_eff'] = (injs['mass_1']*injs['a_1']*injs['cos_tilt_1'] + injs['mass_2']*injs['a_2']*injs['cos_tilt_2'] ) / (injs['mass_1'] + injs['mass_2'])




for log_sig_x in log_sig_xs:

    num_injs = []

    for eta_x in eta_x_arr:

        pdf = skewnorm_chi_eff(chi_eff_arr, mu_x, np.exp(log_sig_x), eta_x)
        
        #max value of the pdf
        max_pdf = np.amax(pdf)
        max_idx = np.where(pdf == max_pdf)[0][0] + 1

        chi_015 = np.interp(0.16*max_pdf, pdf[:max_idx], chi_eff_arr['chi_eff'][:max_idx])
        chi_086 = np.interp(0.84*max_pdf, pdf[:max_idx], chi_eff_arr['chi_eff'][:max_idx])


        try:
            num_inj = np.logical_and(injs['chi_eff'] >= chi_015, injs['chi_eff'] <= chi_086).sum()
        except:
            import pdb; pdb.set_trace()

        num_injs.append(num_inj)



    # make plot
    plt.plot(eta_x_arr, np.array(num_injs), label='$\\log \\sigma_{\\chi} = $' + str(log_sig_x), lw='1.0')


plt.axhline(500, ls='-.', color='k', lw=1)
plt.grid(ls='--', lw=0.5)
plt.legend(frameon=False)
plt.xlabel('$\\eta_{\\chi}$')
plt.ylabel('Number of injections covering 1-sigma of the asymmetric rise')
plt.yscale('log')
plt.tight_layout()
plt.savefig('asymmetry_limits.png', dpi=300)
import pdb; pdb.set_trace()