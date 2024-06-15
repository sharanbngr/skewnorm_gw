import sys
import numpy as np
from bilby.core.result import read_in_result
from gwpopulation.models.spin import gaussian_chi_eff, skewnorm_chi_eff, eps_skewnorm_chi_eff
import matplotlib.pyplot as plt
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution

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
        # defaulting to a truncated normal model
        p_chi_eff = skewnorm_chi_eff(chi_eff_arr,
                            mu_chi_eff=posterior['mu_chi_eff'][draw],
                            sigma_chi_eff= posterior['sigma_chi_eff'][draw],
                            eta_chi_eff=0,)

    return p_chi_eff





def limits(rundir):

    result = read_in_result(rundir + '/GWTC-3_result.json')

    chi_eff_arr = {'chi_eff':np.arange(-1, 1.001, 0.001)}

    p_chi_effs = np.zeros((chi_eff_arr['chi_eff'].size, result.posterior['alpha'].size))


    for draw in range(result.posterior['alpha'].size):

        p_chi_effs[:, draw] = spinplot(chi_eff_arr, result.posterior, draw)


    del_chi = chi_eff_arr['chi_eff'][1] - chi_eff_arr['chi_eff'][0]

    import pdb; pdb.set_trace()
    # asymmetry about zero
    central_asymmetry = np.sum(del_chi * p_chi_effs[chi_eff_arr['chi_eff'] > 0] , axis=0) -\
            np.sum(del_chi * p_chi_effs[chi_eff_arr['chi_eff'] <= 0] , axis=0)


    negative_wt = np.sum(del_chi * p_chi_effs[chi_eff_arr['chi_eff'] <= 0] , axis=0)


    q5, q10, median, q95 = np.quantile(central_asymmetry, [0.05, 0.1, 0.5, 0.95])


    with open(f"{rundir}/limits.txt", "w") as file:
        file.write(f"The 90\% lower limit on asymmetry about zero is {q10:.2f} \n")
        file.write(f"Median asymmetry about zero is {median:.2f}^{q95-median:.2f}_{median-q5:.2f} \n")
        file.write(f"The probability mass at chi_eff <0 is at least {np.quantile(negative_wt, 0.1):.2f} \n")

    if 'eps_chi_eff' in result.posterior.keys():
        mode = np.array(result.posterior['mu_chi_eff'])

        high_index = chi_eff_arr['chi_eff'][:, None] > mode[None, :]
        low_index = chi_eff_arr['chi_eff'][:, None] <= mode[None, :]

        modal_asymmetry = np.sum(del_chi*p_chi_effs[high_index], axis=0) - np.sum(del_chi*p_chi_effs[low_index], axis=0)

        q5_modal, q10_modal, median_modal, q95_modal = np.quantile(central_asymmetry, [0.05, 0.1, 0.5, 0.95])

        with open(f"{rundir}/limits.txt", "a") as file:
            file.write(f"The 90\% lower limit on asymmetry about the model is {q10_modal:.2f} \n")
            file.write(f"Median asymmetry about the mode is {median_modal:.2f}^{q95_modal-median_modal:.2f}_{median_modal-q5_modal:.2f} \n")


    elif 'eta_chi_eff' in result.posterior.keys():

        eta = result.posterior['eta_chi_eff']
        b = np.sqrt(2/np.pi)
        delta = np.array(eta / np.sqrt(1 + eta**2))

        mu_z = b*delta
        sigma_z = np.sqrt(1 - mu_z**2)
        gamma_1  = (2  - np.pi/2) * (mu_z / sigma_z)**3


        mode = np.array( mu_z -\
                0.5 * gamma_1 * sigma_z -\
                0.5 * np.sign(eta) * np.exp(- 2*np.pi/np.abs(eta)) )


        high_index = chi_eff_arr['chi_eff'][:, None] > mode[None, :]
        low_index = chi_eff_arr['chi_eff'][:, None] <= mode[None, :]

        modal_asymmetry = np.sum(del_chi*p_chi_effs[high_index], axis=0) - np.sum(del_chi*p_chi_effs[low_index], axis=0)

        q5_modal, q10_modal, median_modal, q95_modal = np.quantile(central_asymmetry, [0.05, 0.1, 0.5, 0.95])

        with open(f"{rundir}/limits.txt", "a") as file:
            file.write(f"The 90\% lower limit on asymmetry about the model is {q10_modal:.2f} \n")
            file.write(f"Median asymmetry about the mode is {median_modal:.2f}^{q95_modal-median_modal:.2f}_{median_modal-q5_modal:.2f} \n")




if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise ValueError('Provide the path to the run directory')
    else:
        limits(sys.argv[1])


