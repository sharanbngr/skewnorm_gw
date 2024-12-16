import sys
import numpy as np
from bilby.core.result import read_in_result
from gwpopulation.models.spin import gaussian_chi_eff, skewnorm_chi_eff, eps_skewnorm_chi_eff
import matplotlib.pyplot as plt
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from skewnorm import skewnorm_mixture_model, eps_skewnorm_mixture_model
import corner

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "Helvetica",
    "font.size":16,
    "contour.linewidth":2.5,
})



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

    # markers to seperate the modes
    lambda_low, lambda_high = 0.25, 0.75


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
        for ii in range(1500):

            draw = np.random.randint(result.posterior['alpha'].size)


            if result.posterior['lam_al'][draw] <= lambda_low:
                ax1.plot(chi_eff_arr['chi_eff'], dRdchi_effs[:, draw],
                     color='#56B4E9', alpha=0.2, lw=0.25)
            elif result.posterior['lam_al'][draw] > lambda_high:
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
    low_indx = result.posterior['lam_al'] <= lambda_low
    high_indx = result.posterior['lam_al'] > lambda_high

    lambda_10 = np.quantile(result.posterior['lam_al'], 0.1)

    with open(f"{rundir}/limits.txt", "w") as file:
        file.write(f"The 90 percent lower limit on lambda_al is {lambda_10:.2f} \n")



    ax1.plot(chi_eff_arr['chi_eff'], np.median(dRdchi_effs[:, low_indx], axis=1),  color='k', label='Full population, $\\lambda_{\\rm al} \leq' +  f'{lambda_low}$')
    ax1.plot(chi_eff_arr['chi_eff'], np.median(random_rate[:, low_indx], axis=1), color='#D55E00', label='$\\textsc{random}$')
    ax1.plot(chi_eff_arr['chi_eff'], np.median(aligned_rate[:, low_indx], axis=1), color='#009e74', label='$\\textsc{aligned}$')


    ax1.tick_params(direction='in')
    ax1.grid(ls=':', lw=0.5)
    ax1.legend(frameon=False, loc='upper left')
    ax1.set_yscale('log')
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([1e0, 1000])
    ax1.tick_params(which='both', direction='in')

    ax2.plot(chi_eff_arr['chi_eff'], np.median(dRdchi_effs[:, high_indx], axis=1),  color='k', label='Full poulation, $\\lambda_{\\rm al} >' + f' {lambda_high}$')
    ax2.plot(chi_eff_arr['chi_eff'], np.median(random_rate[:, high_indx], axis=1), color='#D55E00', label='$\\textsc{random}$')
    ax2.plot(chi_eff_arr['chi_eff'], np.median(aligned_rate[:, high_indx], axis=1), color='#009e74', label='$\\textsc{aligned}$')



    ax2.grid(ls=':', lw=0.5)
    ax2.legend(frameon=False, loc='upper left')
    ax2.set_yscale('log')
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([1e0, 1000])
    ax2.set_yticklabels([])
    ax2.tick_params(which='both', direction='in')



    ax1.set_ylabel('$\\frac{dR}{d \\chi_{\\rm eff}}$')
    ax2.set_xlabel('$\\chi_{\\rm eff}$')
    ax1.set_xlabel('$\\chi_{\\rm eff}$')

    plt.subplots_adjust(wspace=0.1)
    plt.tight_layout()

    plt.savefig(rundir + "/dRdchi_eff_median_modal.pdf", dpi=300)
    plt.close()


    spin_params = ['lam_al', 'mu_al', 'sigma_al', 'eta_al', 'sigma_dy']
    labels = [
            '$\\lambda_{\\rm al}$',
            '$\\mu_{\\rm al}$',
            '$\\sigma_{\\rm al}$',
            '$\\eta_{\\rm al}$',
            '$\\sigma_{\\rm r}$',
            ]

    if 'beta1' in result.posterior.keys():
        spin_params.append('beta1')
        spin_params.append('beta2')

        labels.append('$\\beta_{\\textsc{random}}$')
        labels.append('$\\beta_{\\textsc{aligned}}$')

    contour_kwargs = {'linewidth':2.5}

    result.plot_corner(
            save=True,
            parameters=spin_params,
            color='#0072B2',
            no_fill_contours=True,
            fill_contours=False,
            smooth=1.5,
            quantiles=None,
            levels=[0.68, 0.95],
            hist_kwargs=contour_kwargs,
            labels=labels,
            titles=False,
            filename=rundir + '/spin_corner.pdf')

    if 'beta1' in result.posterior.keys():
        result.plot_corner(
            save=True,
            parameters=['beta1', 'beta2'],
            color='#0072B2',
            no_fill_contours=True,
            fill_contours=False,
            smooth=1.5,
            labels = ['$\\beta_{\\rm R}$', '$\\beta_{\\rm A}$'],
            quantiles=None,
            show_titles=False,
            levels=[0.68, 0.95],

            label_kwargs={"fontsize":18},
            hist_kwargs={"linewidth":2},
            filename=rundir + '/beta_corner.pdf')

        beta_comp = np.sum(result.posterior['beta1'] > result.posterior['beta2']) / result.posterior['beta1'].size


        with open(f"{rundir}/limits.txt", "a") as file:
            file.write(f"beta 1 greater than beta 2 at {beta_comp:.2f}")


    p_chi_effs_low = p_chi_effs[:, low_indx]
    p_chi_effs_high = p_chi_effs[:, high_indx]


    del_chi = chi_eff_arr['chi_eff'][1] - chi_eff_arr['chi_eff'][0]

    # asymmetry about zero
    central_asymmetry_low = np.sum(del_chi * p_chi_effs_low[chi_eff_arr['chi_eff'] > 0] , axis=0) -\
            np.sum(del_chi * p_chi_effs_low[chi_eff_arr['chi_eff'] <= 0] , axis=0)

    central_asymmetry_high = np.sum(del_chi * p_chi_effs_high[chi_eff_arr['chi_eff'] > 0] , axis=0) -\
            np.sum(del_chi * p_chi_effs_high[chi_eff_arr['chi_eff'] <= 0] , axis=0)



    q5_low, q10_low, median_low, q95_low = np.quantile(central_asymmetry_low, [0.05, 0.1, 0.5, 0.95])
    q5_high, q10_high, median_high, q95_high = np.quantile(central_asymmetry_high, [0.05, 0.1, 0.5, 0.95])



    negative_wt_low = np.sum(del_chi * p_chi_effs_low[chi_eff_arr['chi_eff'] <= 0] , axis=0)
    negative_wt_high = np.sum(del_chi * p_chi_effs_high[chi_eff_arr['chi_eff'] <= 0] , axis=0)



    with open(f"{rundir}/limits.txt", "a") as file:
        file.write(f"The 90\% lower limit on asymmetry about zero for the low lambda mode is {q10_low:.2f} \n")
        file.write(f"The 90\% lower limit on asymmetry about zero for the high lambda mode is {q10_high:.2f} \n")

        file.write(f"Median asymmetry about zero for low lambda mode is {median_low:.2f}^{q95_low-median_low:.2f}_{median_low-q5_low:.2f} \n")
        file.write(f"Median asymmetry about zero for high lambda mode is {median_high:.2f}^{q95_high-median_high:.2f}_{median_high-q5_high:.2f} \n")


        file.write(f"The probability mass at chi_eff <0 for the low lambda mode is at least {np.quantile(negative_wt_low, 0.1):.2f} \n")
        file.write(f"The probability mass at chi_eff <0 for the high lambda mode is at least {np.quantile(negative_wt_high, 0.1):.2f} \n")







    mode_low_indx = np.argmax(p_chi_effs_low, axis=0)
    mode_high_indx = np.argmax(p_chi_effs_high, axis=0)


    modal_asymmetry_high = []
    modal_asymmetry_low = []


    for ii in range(p_chi_effs_low.shape[1]):

        mode = chi_eff_arr['chi_eff'][mode_low_indx[ii]]
        modal_asymmetry_low.append(
                 np.sum(del_chi * p_chi_effs_low[chi_eff_arr['chi_eff'] > mode, ii] , axis=0) -\
                         np.sum(del_chi * p_chi_effs_low[chi_eff_arr['chi_eff'] <= mode, ii] , axis=0)
                )

    for ii in range(p_chi_effs_high.shape[1]):

        mode = chi_eff_arr['chi_eff'][mode_high_indx[ii]]
        asymmetry = np.sum(del_chi * p_chi_effs_high[chi_eff_arr['chi_eff'] > mode, ii] , axis=0) -\
                         np.sum(del_chi * p_chi_effs_high[chi_eff_arr['chi_eff'] <= mode, ii] , axis=0)


        if np.abs(asymmetry) > 0.95:
            import pdb; pdb.set_trace()
        else:
            modal_asymmetry_high.append(asymmetry)

    modal_asymmetry_high = np.array(modal_asymmetry_high)
    modal_asymmetry_low = np.array(modal_asymmetry_low)



    # asymmetry about zero
    central_asymmetry = np.sum(del_chi * p_chi_effs[chi_eff_arr['chi_eff'] > 0] , axis=0) -\
            np.sum(del_chi * p_chi_effs[chi_eff_arr['chi_eff'] <= 0] , axis=0)


    negative_wt = np.sum(del_chi * p_chi_effs[chi_eff_arr['chi_eff'] <= 0] , axis=0)


    q5, q10, median, q95 = np.quantile(central_asymmetry, [0.05, 0.1, 0.5, 0.95])


    with open(f"{rundir}/limits.txt", "a") as file:
        file.write(f"The 90\% lower limit on asymmetry about zero is {q10:.2f} \n")
        file.write(f"Median asymmetry about zero is {median:.2f}^{q95-median:.2f}_{median-q5:.2f} \n")
        file.write(f"The probability mass at chi_eff <0 is at least {np.quantile(negative_wt, 0.1):.2f} \n")

    if 'eps_chi_eff' in result.posterior.keys():
        mode = np.array(result.posterior['mu_chi_eff'])

        high_index = chi_eff_arr['chi_eff'][:, None] > mode[None, :]
        low_index = chi_eff_arr['chi_eff'][:, None] <= mode[None, :]


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
        plotter(sys.argv[1])


