from gwpopulation.models.spin import gaussian_chi_eff, skewnorm_chi_eff, gaussian_chi_p, eps_skewnorm_chi_eff


def eps_skewnorm_q_mixture_model(dataset, 
                                 mu_al,
                                 sigma_al,
                                 eps_al, 
                                 lam_al,
                                 sigma_dy, 
                                 beta1, 
                                 beta2):
    


    # Should be normalized because gaussian_chi_eff and skewnorm_chi_eff already are
    pdf = (1 - lam_al) * gaussian_chi_eff(dataset, mu_chi_eff=0, sigma_chi_eff=sigma_dy) +  \
            lam_al * eps_skewnorm_chi_eff(dataset, mu_chi_eff=mu_al, sigma_chi_eff=sigma_al, eps_chi_eff=eps_al)

    return pdf  





def skewnorm_mixture_model(dataset, mu_al, sigma_al, eta_al, lam_al, sigma_dy):

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

def eps_skewnorm_mixture_model(dataset, mu_al, sigma_al, eps_al, lam_al, sigma_dy):

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
            lam_al * eps_skewnorm_chi_eff(dataset, mu_chi_eff=mu_al, sigma_chi_eff=sigma_al, eps_chi_eff=eps_al)

    return pdf


