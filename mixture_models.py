from gwpopulation.models.spin import *
from gwpopulation.models.mass import *
from gwpopulation.utils import powerlaw, truncnorm
import inspect


def q_skewnorm_mixture_model(dataset, beta1, beta2, mmin, mu_al, sigma_al, eta_al, lam_al, sigma_dy):

        # Should be normalized because gaussian_chi_eff and skewnorm_chi_eff already are
        p_dynamic = gaussian_chi_eff(dataset, mu_chi_eff=0, sigma_chi_eff=sigma_dy) * \
        powerlaw(dataset["mass_ratio"], beta1, 1, mmin / dataset["mass_1"])

        p_aligned = skewnorm_chi_eff(dataset, mu_chi_eff=mu_al, sigma_chi_eff=sigma_al, eta_chi_eff=eta_al) * \
        powerlaw(dataset["mass_ratio"], beta2, 1, mmin / dataset["mass_1"])

        p_dynamic *= (1 - lam_al)
        p_aligned *= lam_al

        #pdf = (1 - lam_al) * p_dyanmic + lam_al * p_aligned

        return p_dynamic, p_aligned


def q_eps_skewnorm_mixture_model(dataset, beta1, beta2, mmin, mu_al, sigma_al, eps_al, lam_al, sigma_dy):

        # Should be normalized because gaussian_chi_eff and skewnorm_chi_eff already are
        p_dynamic = gaussian_chi_eff(dataset, mu_chi_eff=0, sigma_chi_eff=sigma_dy) * \
        powerlaw(dataset["mass_ratio"], beta1, 1, mmin / dataset["mass_1"])

        p_aligned = skewnorm_chi_eff(dataset, mu_chi_eff=mu_al, sigma_chi_eff=sigma_al, eps_chi_eff=eps_al) * \
        powerlaw(dataset["mass_ratio"], beta2, 1, mmin / dataset["mass_1"])

        p_dynamic *= (1 - lam_al)
        p_aligned *= lam_al

        #pdf = (1 - lam_al) * p_dyanmic + lam_al * p_aligned

        return p_dynamic, p_aligned




class BaseDoubleqchieff:
    """
    Smoothed mass distribution base class with two q component
    for two chi_eff compoents

    Implements the low-mass smoothing and power-law mass ratio
    distribution. Requires p_m1 to be implemented.

    Parameters
    ==========
    mmin: float
        The minimum mass considered for numerical normalization
    mmax: float
        The maximum mass considered for numerical normalization

    Adapted from SinglePeakSmoothedMassDistribution in gwpopulation
    """

    primary_model = None
    q_chieff_model = None

    @property
    def variable_names(self):
        vars = getattr(
            self.primary_model,
            "variable_names",
            inspect.getfullargspec(self.primary_model).args[1:],
        )
        vars += getattr(
                self.q_chieff_model,
                "variable_names",
                inspect.getfullargspec(self.q_chieff_model).args[1:],
                )

        vars += ["beta1", "beta2", "lam_al", "delta_m"]
        vars = set(vars).difference(self.kwargs.keys())
        return vars

    @property
    def kwargs(self):
        return dict()

    def __init__(self, mmin=2, mmax=100, normalization_shape=(1000, 500), cache=True):
        self.mmin = mmin
        self.mmax = mmax
        self.m1s = xp.linspace(mmin, mmax, normalization_shape[0])
        self.qs = xp.linspace(0.001, 1, normalization_shape[1])
        self.dm = self.m1s[1] - self.m1s[0]
        self.dq = self.qs[1] - self.qs[0]
        self.m1s_grid, self.qs_grid = xp.meshgrid(self.m1s, self.qs)
        self.cache = cache

    def __call__(self, dataset, *args, **kwargs):
        from gwpopulation.utils import xp

        m1_keys = ['delta_m', 'alpha', 'mpp', 'sigpp', 'lam',  'mmax',]
        m1_kwargs = {key:kwargs[key] for key in m1_keys}
        m1_kwargs['mmin'] = kwargs['mmin']

        mmin = kwargs.get("mmin", self.mmin)
        mmax = kwargs.get("mmax", self.mmax)

        ## retain only the chi_eff parameters
        for m1_key in m1_keys:
            kwargs.pop(m1_key)

        if "jax" not in xp.__name__:
            if mmin < self.mmin:
                raise ValueError(
                    "{self.__class__}: mmin ({mmin}) < self.mmin ({self.mmin})"
                )
            if mmax > self.mmax:
                raise ValueError(
                    "{self.__class__}: mmax ({mmax}) > self.mmax ({self.mmax})"
                )

        p_m1 = self.p_m1(dataset, **m1_kwargs, **self.kwargs)
        p_q_chieff = self.p_q_chieff(dataset, **kwargs)
        prob = p_m1 * p_q_chieff
        return prob

    def p_m1(self, dataset, **kwargs):
        mmin = kwargs.get("mmin", self.mmin)
        delta_m = kwargs.pop("delta_m", 0)
        p_m = self.__class__.primary_model(dataset["mass_1"], **kwargs)
        p_m *= self.smoothing(
            dataset["mass_1"], mmin=mmin, mmax=self.mmax, delta_m=delta_m
        )
        norm = self.norm_p_m1(delta_m=delta_m, **kwargs)
        return p_m / norm

    def norm_p_m1(self, delta_m, **kwargs):
        """Calculate the normalisation factor for the primary mass"""

        from gwpopulation.utils import xp

        mmin = kwargs.get("mmin", self.mmin)

        if "jax" not in xp.__name__ and delta_m == 0:
            return 1
        p_m = self.__class__.primary_model(self.m1s, **kwargs)
        p_m *= self.smoothing(self.m1s, mmin=mmin, mmax=self.mmax, delta_m=delta_m)

        norm = xp.where(xp.array(delta_m) > 0, xp.trapz(p_m, self.m1s), 1)
        return norm

    def p_q_chieff(self, dataset, **kwargs):
        from gwpopulation.utils import xp

        mmin = kwargs.get("mmin", self.mmin)
        mmax = kwargs.get("mmin", self.mmin)
        delta_m = kwargs.get("delta_m", self.mmin)
        beta1 = kwargs.get("beta1")
        beta2 = kwargs.get("beta2")

        p_dynamic, p_aligned = self.__class__.q_chieff_model(dataset, **kwargs)

        p_smooth = self.smoothing(
            dataset["mass_1"] * dataset["mass_ratio"],
            mmin=mmin,
            mmax=dataset["mass_1"],
            delta_m=delta_m,
        )

        try:
            if self.cache:
                norm1, norm2 = self.norm_p_q(beta1=beta1, beta2=beta2, mmin=mmin, delta_m=delta_m)
            else:
                self._cache_q_norms(dataset["mass_1"])
                norm1, norm2 = self.norm_p_q(beta1=beta1, beta2=beta2, mmin=mmin, delta_m=delta_m)
        except (AttributeError, TypeError, ValueError):
            self._cache_q_norms(dataset["mass_1"])
            norm1, norm2 = self.norm_p_q(beta1=beta1, beta2=beta2, mmin=mmin, delta_m=delta_m)


        p_dynamic /= norm1
        p_aligned /= norm2

        p_q = p_dynamic + p_aligned

        return xp.nan_to_num(p_q)

    def norm_p_q(self, beta1, beta2, mmin, delta_m):
        """Calculate the mass ratio normalisation by linear interpolation"""

        from gwpopulation.utils import xp

        p1_q = powerlaw(self.qs_grid, beta1, 1, mmin / self.m1s_grid)
        p2_q = powerlaw(self.qs_grid, beta2, 1, mmin / self.m1s_grid)

        p_smooth = self.smoothing(
            self.m1s_grid * self.qs_grid, mmin=mmin, mmax=self.m1s_grid, delta_m=delta_m
        )

        p1_q *= p_smooth
        p2_q *= p_smooth

        norm1 = xp.where(
            xp.array(delta_m) > 0,
            xp.nan_to_num(xp.trapz(p1_q, self.qs, axis=0)),
            xp.ones(self.m1s.shape),
        )

        norm2 = xp.where(
            xp.array(delta_m) > 0,
            xp.nan_to_num(xp.trapz(p2_q, self.qs, axis=0)),
            xp.ones(self.m1s.shape),
        )

        return self._q_interpolant(norm1), self._q_interpolant(norm2) 

    def _cache_q_norms(self, masses):
        """
        Cache the information necessary for linear interpolation of the mass
        ratio normalisation
        """
        from gwpopulation.utils import xp

        from gwpopulation.models.interped import _setup_interpolant

        self._q_interpolant = _setup_interpolant(
            self.m1s, masses, kind="cubic", backend=xp
        )

    @staticmethod
    def smoothing(masses, mmin, mmax, delta_m):
        """
        Apply a one sided window between mmin and mmin + delta_m to the
        mass pdf.

        The upper cut off is a step function,
        the lower cutoff is a logistic rise over delta_m solar masses.

        See T&T18 Eqs 7-8
        Note that there is a sign error in that paper.

        S = (f(m - mmin, delta_m) + 1)^{-1}
        f(m') = delta_m / m' + delta_m / (m' - delta_m)

        See also, https://en.wikipedia.org/wiki/Window_function#Planck-taper_window
        """
        from gwpopulation.utils import xp, scs

        if "jax" in xp.__name__ or delta_m > 0.0:
            shifted_mass = xp.nan_to_num((masses - mmin) / delta_m, nan=0)
            shifted_mass = xp.clip(shifted_mass, 1e-6, 1 - 1e-6)
            exponent = 1 / shifted_mass - 1 / (1 - shifted_mass)
            window = scs.expit(-exponent)
            window *= (masses >= mmin) * (masses <= mmax)
            return window
        else:
            return xp.ones(masses.shape)


class DoubleqPLP_skewnorm(BaseDoubleqchieff):
    """
    Powerlaw + peak model for two-dimensional mass distribution with low
    mass smoothing.

    https://arxiv.org/abs/1801.02699 Eq. (11) (T&T18)

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' and 'mass_ratio'.
    alpha: float
        Powerlaw exponent for more massive black hole.
    beta: float
        Power law exponent of the mass ratio distribution.
    mmin: float
        Minimum black hole mass.
    mmax: float
        Maximum mass in the powerlaw distributed component.
    lam: float
        Fraction of black holes in the Gaussian component.
    mpp: float
        Mean of the Gaussian component.
    sigpp: float
        Standard deviation of the Gaussian component.
    delta_m: float
        Rise length of the low end of the mass distribution.

    Notes
    -----
    The Gaussian component is bounded between [`mmin`, `self.mmax`].
    This means that the `mmax` parameter is _not_ the global maximum.
    """

    primary_model = two_component_single
    q_chieff_model = q_skewnorm_mixture_model

    @property
    def kwargs(self):
        return dict(gaussian_mass_maximum=self.mmax)


class DoubleqPLP_eps_skewnorm(BaseDoubleqchieff):
    """
    Powerlaw + peak model for two-dimensional mass distribution with low
    mass smoothing.

    https://arxiv.org/abs/1801.02699 Eq. (11) (T&T18)

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' and 'mass_ratio'.
    alpha: float
        Powerlaw exponent for more massive black hole.
    beta: float
        Power law exponent of the mass ratio distribution.
    mmin: float
        Minimum black hole mass.
    mmax: float
        Maximum mass in the powerlaw distributed component.
    lam: float
        Fraction of black holes in the Gaussian component.
    mpp: float
        Mean of the Gaussian component.
    sigpp: float
        Standard deviation of the Gaussian component.
    delta_m: float
        Rise length of the low end of the mass distribution.

    Notes
    -----
    The Gaussian component is bounded between [`mmin`, `self.mmax`].
    This means that the `mmax` parameter is _not_ the global maximum.
    """

    primary_model = two_component_single
    q_chieff_model = q_eps_skewnorm_mixture_model

    @property
    def kwargs(self):
        return dict(gaussian_mass_maximum=self.mmax)







### -------------- mixed q-chi_eff distributionve above  ^^^^^^
#### -------------- simple chi_eff mixture models distirbutions below ------







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


