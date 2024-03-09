from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.redshift import PowerLawRedshift
from gwpopulation.models.spin import gaussian_chi_eff, skewnorm_chi_eff
from gwpopulation.hyperpe import HyperparameterLikelihood, RateLikelihood
from gwpopulation.vt import ResamplingVT
import deepdish as dd
import pickle 
import bilby
from bilby.core.prior import PriorDict, Uniform, LogUniform
from gwpopulation.backend import set_backend
import numpy as np




# numpy, cupy or jax
backend = 'jax'


if backend == 'numpy':
    xp = np
elif backend == 'cupy':
    import cupy
    xp = cupy
elif backend == 'jax':
    import jax.numpy as jnp
    from gwpopulation.experimental.jax import NonCachingModel, JittedLikelihood
    xp = jnp

set_backend(backend=backend)

def norm_mixture_model_chi_eff(dataset, mu_al, sigma_al, eta_al, lam_al, sigma_dy):

    pdf = mixture_model_chi_eff(dataset, mu_al, sigma_al, eta_al, lam_al, sigma_dy)

    # calc norm
    chi_arr = {'chi_eff': xp.array([0.005 * i for i in range(-200, 201, 1)])}
    norm = 0.005 * xp.sum(mixture_model_chi_eff(chi_arr, mu_al, sigma_al, eta_al, lam_al, sigma_dy))

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


## ^^^^^^^^ functions and methods ^^^^^^^^^^^^
## vvvvvvv runing pop inference vvvvvvvvv

def get_model(models, backend):
    if type(models) not in (list, tuple):
        models = [models]
    Model = NonCachingModel if backend == 'jax' else bilby.hyper.model.Model
    return Model(
        [model() if type(model) is type else model for model in models]
    )





runargs = { 'pe_file':'/projects/p31963/sharan/pop/GW_PE_samples.h5',
    'inj_file':'/projects/p31963/sharan/pop/O3_injections.pkl',
    'chains':1, 'samples':7000, 'thinning':1, 'warmup':3000, 
    'skip_inference':False, 'spin_model':'skew_norm'}

#runargs['outdir'] = './trunc_norm_cuda'
runargs['outdir'] = './mixture_model_' + backend + '_no_rate'

# extract posterior
post = dd.io.load(runargs['pe_file'])

# get injections
with open(runargs['inj_file'], 'rb') as f:
    injs = pickle.load(f)

    #### need to calc chi_eff for injs
    injs['chi_eff'] = (injs['mass_1']*injs['a_1']*injs['cos_tilt_1'] + injs['mass_2']*injs['a_2']*injs['cos_tilt_2'] ) / (injs['mass_1'] + injs['mass_2'])


for key in injs.keys():
     injs[key] = xp.array(injs[key])

posteriors = []

for event in post.keys():
    posteriors.append(post[event])
    
#rate = LogUniform(minimum=1e-1, maximum=1e3, name='rate', latex_label='$R$')

priors = PriorDict(
    dict(
    lamb = Uniform(minimum=-6, maximum=6, name='lamb', latex_label='$\\kappa_z$'),
    alpha = Uniform(minimum=-4, maximum=12, latex_label='$\\alpha$'),
    beta = Uniform(minimum=-4, maximum=12, name='beta', latex_label='$\\beta_{q}$'),
    mmax = Uniform(minimum=30, maximum=100, name='mmax', latex_label='$m_{\\max}$'),
    mmin = Uniform(minimum=2, maximum=10, name='mmin', latex_label='$m_{\\min}$'),
    delta_m = Uniform(minimum=0.01, maximum=10, name='delta_m', latex_label='$\\delta_{m}$'),
    lam = Uniform(minimum=0, maximum=1, name='lam', latex_label='$\\lambda_{\\rm peak}$'),
    mpp = Uniform(minimum=20, maximum=50, name='mpp', latex_label='$\\mu_{\\rm peak}$'),
    sigpp = Uniform(minimum=1, maximum=10, name='sigpp', latex_label='$\\sigma_{\\rm peak}$'),
    mu_al = Uniform(minimum=0, maximum=1, name='mu_al', latex_label='$\\mu_{\\rm al}$'),
    sigma_al = LogUniform(minimum=0.01, maximum=4, name='sigma_al', latex_label='$\\sigma_{\\rm al}$'),
    eta_al = Uniform(minimum=-20, maximum=20, name='eta_al', latex_label='$\\eta_{\\rm al}$'),
    lam_al = Uniform(minimum=0, maximum=1, name='lam_al', latex_label='$lam_{\\rm al}$'),
    sigma_dy = LogUniform(minimum=0.01, maximum=4, name='sigma_dy', latex_label='$\\sigma_{\\rm dy}$'),
    ))

models = [SinglePeakSmoothedMassDistribution, PowerLawRedshift, norm_mixture_model_chi_eff]

hyperprior = get_model(models, backend)

VTs = ResamplingVT(model=get_model(models, backend), data=injs, n_events=len(posteriors), 
                        marginalize_uncertainty=False, enforce_convergence=True)

likelihood = HyperparameterLikelihood(posteriors = posteriors, hyper_prior = get_model(models, backend), selection_function = VTs)


if backend == 'jax':
    likelihood = JittedLikelihood(likelihood)


result = bilby.run_sampler(likelihood = likelihood, 
                  nlive=500, resume=True,
                  priors = priors, 
                  label = 'GWTC-3',  
                  outdir = runargs['outdir'])

spin_params = ['mu_al', 'sigma_al', 'eta_al', 'lam_al', 'sigma_dy']

# plot corner plot
result.plot_corner(save=True)

# plot spin only corner plot
result.plot_corner(save=True, parameters=spin_params, filename=runargs['outdir'] + '/spin_corner.png')




    