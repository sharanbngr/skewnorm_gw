import sys, glob, json
import h5py
import pandas as pd
import gwpopulation as gwpop
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
import matplotlib.pyplot as plt
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

#mass_pdf = SinglePeakSmoothedMassDistribution()
#z_pdf = PowerLawRedshift()



def trun_norm_prob(
          dataset, kappa,
          alpha, beta,
          mmax, mmin, 
          delta_m, lam, 
          mpp, sigpp, 
          mu_x, log_sig_x):

    p_mass = mass_pdf(dataset, alpha=alpha, beta=beta, 
                      mmax=mmax, mmin=mmin, delta_m=delta_m, 
                      lam=lam, mpp=mpp, sigpp=sigpp,) 

    p_spins = gaussian_chi_eff(dataset, mu_x, xp.exp(log_sig_x))


    p_z = z_pdf(dataset, lamb=kappa)

    prob = p_mass * p_spins * p_z

    return prob



def skew_norm_prob(
        dataset, kappa,
        alpha, beta,
        mmax, mmin, 
        delta_m, lam, 
        mpp, sigpp, 
        mu_x, log_sig_x, eta_x):
     
    p_mass = mass_pdf(dataset, alpha=alpha, beta=beta, 
                      mmax=mmax, mmin=mmin, delta_m=delta_m, 
                      lam=lam, mpp=mpp, sigpp=sigpp,) 

    p_spins = skewnorm_chi_eff(dataset, mu_x, xp.exp(log_sig_x), eta_x)

    p_z = z_pdf(dataset, lamb=kappa)

    return p_mass * p_spins * p_z

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
runargs['outdir'] = './' + runargs['spin_model'] + '_' + backend + '_no_rate'

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
    mu_chi_eff = Uniform(minimum=-1, maximum=1, name='mu_chi_eff', latex_label='$\\mu_{\\chi}$'),
    sigma_chi_eff = LogUniform(minimum=0.01, maximum=4, name='sigma_chi_eff', latex_label='$\\log \\sigma_{\\chi}$'),
    ))



if runargs['spin_model'] == 'skew_norm':
    priors['eta_chi_eff']= bilby.prior.Uniform(minimum=-60, maximum=60, name='eta_chi_eff', latex_label='$\\eta_{\\chi}$')
    models = [SinglePeakSmoothedMassDistribution, PowerLawRedshift, skewnorm_chi_eff]
    #spin_prob = skew_norm_prob
elif runargs['spin_model'] == 'trunc_norm':
    models = [SinglePeakSmoothedMassDistribution, PowerLawRedshift, gaussian_chi_eff]
    #spin_prob = trun_norm_prob

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

spin_params = []

if runargs['spin_model'] == 'trunc_norm':
    spin_params.append("mu_chi_eff")
    spin_params.append("sigma_chi_eff")

elif runargs['spin_model'] == 'skew_norm':
    spin_params.append("mu_chi_eff")
    spin_params.append("sigma_chi_eff")
    spin_params.append("eta_chi_eff")




# plot corner plot
result.plot_corner(save=True)

# plot spin only corner plot
result.plot_corner(save=True, parameters=spin_params, filename=runargs['outdir'] + '/spin_corner.png')




    