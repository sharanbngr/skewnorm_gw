import sys, os
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.redshift import PowerLawRedshift
from gwpopulation.models.spin import gaussian_chi_eff, skewnorm_chi_eff, gaussian_chi_p, eps_skewnorm_chi_eff
from gwpopulation.hyperpe import HyperparameterLikelihood, RateLikelihood
from gwpopulation.vt import ResamplingVT
import deepdish as dd
import pickle 
import bilby
from bilby.core.prior import PriorDict, Uniform, LogUniform
from gwpopulation.backend import set_backend
import numpy as np
from configparser import ConfigParser
from gwpopulation.experimental.jax import NonCachingModel, JittedLikelihood
from prior_conversions import chi_effective_prior_from_isotropic_spins as chi12_to_chieff
from prior_conversions import chi_p_prior_from_isotropic_spins as chi12_to_chip



def get_model(models, backend):
    if type(models) not in (list, tuple):
        models = [models]
    Model = NonCachingModel if backend == 'jax' else bilby.hyper.model.Model
    return Model(
        [model() if type(model) is type else model for model in models]
    )





def spinfit(runargs):

    if runargs['backend'] == 'numpy':
        xp = np
    elif runargs['backend'] == 'cupy':
        import cupy
        xp = cupy
    elif runargs['backend'] == 'jax':
        import jax.numpy as jnp
        xp = jnp

    set_backend(backend=runargs['backend'])

    runargs['outdir'] = './' + runargs['spin_model'] + '_' + runargs['backend'] + '_' + runargs['rundix']


    # create directory and copy the config file
    os.system('mkdir -p ' + runargs['outdir'])
    os.system('cp '  + runargs['configfile']  + ' ' + runargs['outdir'] + '/config.ini')



    # extract posterior
    post = dd.io.load(runargs['pe_file'])

    posteriors = []

    ## do prior conversions
    if runargs['fit_chip']:
        
        print('converting PE priors to chi_eff, chi_p ...')
        for event in post.keys():
            sin_tilt_1, sin_tilt_2 = np.sqrt(1 - post[event]['cos_tilt_1']**2), np.sqrt(1 - post[event]['cos_tilt_2']**2)
            mass_factor = (4*post[event]['mass_ratio'] + 3) / (4 + 3*post[event]['mass_ratio'])

            # np.maximum calculates element-wise maxima
            post[event]['chi_p'] = np.maximum(post[event]['a_1']*sin_tilt_1, 
                                                    mass_factor * post[event]['mass_ratio']*post[event]['a_2']*sin_tilt_2)

            post[event]['prior']  *= 4.0 * chi12_to_chieff(post[event]['mass_ratio'], 1.0, post[event]['chi_eff'])                                         
            post[event]['prior'] *= chi12_to_chip(post[event]['mass_ratio'], 1.0, post[event]['chi_p'])

            posteriors.append(post[event])

    else:
        print('converting PE priors to chi_eff ...')
        for event in post.keys():
            post[event]['prior']  *= 4.0 * chi12_to_chieff(post[event]['mass_ratio'], 1.0, post[event]['chi_eff']) 
            posteriors.append(post[event])

        

    # get injections
    with open(runargs['inj_file'], 'rb') as f:
        injs = pickle.load(f)

    for key in injs.keys():
        injs[key] = xp.array(injs[key])

    if runargs['fit_chip']:
        print('converting inj priors to chi_eff, chi_p ...')
        injs['chi_eff'] = (injs['mass_1']*injs['a_1']*injs['cos_tilt_1'] + injs['mass_2']*injs['a_2']*injs['cos_tilt_2'] ) / (injs['mass_1'] + injs['mass_2'])
        sin_tilt_1, sin_tilt_2 = np.sqrt(1 - injs['cos_tilt_1']**2), np.sqrt(1 - injs['cos_tilt_2']**2)
        mass_factor = (4*injs['mass_ratio'] + 3) / (4 + 3*injs['mass_ratio'])

        injs['chi_p'] = np.maximum(injs['a_1']*sin_tilt_1, 
                                        mass_factor*injs['mass_ratio']*injs['a_2']*sin_tilt_2)        
        
        injs['prior'] *= 4.0 * chi12_to_chieff(injs['mass_ratio'], 1.0, injs['chi_eff'])   
        injs['prior'] *= chi12_to_chip(injs['mass_ratio'], 1.0, injs['chi_p'])
    
    else:
        print('converting inj priors to chi_eff ...')
        injs['chi_eff'] = (injs['mass_1']*injs['a_1']*injs['cos_tilt_1'] + injs['mass_2']*injs['a_2']*injs['cos_tilt_2'] ) / (injs['mass_1'] + injs['mass_2'])
        injs['prior'] *= 4.0 * chi12_to_chieff(injs['mass_ratio'], 1.0, injs['chi_eff'])    
        
    priors = PriorDict(filename=runargs['priors'])

    models = [SinglePeakSmoothedMassDistribution, PowerLawRedshift]

    if runargs['spin_model'] == 'skewnorm':
        models.append(skewnorm_chi_eff)
    elif runargs['spin_model'] == 'truncnorm':
        models.append(gaussian_chi_eff)
    elif runargs['spin_model'] == 'eps_skewnorm':
        models.append(eps_skewnorm_chi_eff)

    if runargs['fit_chip']:
        models.append(gaussian_chi_p)
    else:
        priors.pop('mu_chi_p')
        priors.pop('sigma_chi_p')

    if runargs['fit_rate']:
        hyperlikelihood = RateLikelihood
    else:
        hyperlikelihood = HyperparameterLikelihood
        priors.pop('rate')

    VTs = ResamplingVT(model=get_model(models, 
                                       runargs['backend']), 
                                       data=injs,
                                         n_events=len(posteriors), 
                            marginalize_uncertainty=False, 
                            enforce_convergence=True)

    likelihood = hyperlikelihood(posteriors = posteriors, 
                                 hyper_prior = get_model(models, runargs['backend']), 
                                 selection_function = VTs)


    if runargs['backend'] == 'jax':
        jit_likelihood = JittedLikelihood(likelihood)


    result = bilby.run_sampler(likelihood = jit_likelihood, 
                    nlive=runargs['nlive'], resume=True,
                    priors = priors, 
                    label = 'GWTC-3',  
                    check_point_delta_t = 300,
                    outdir = runargs['outdir'])

    ## calculate rates in post-processing
    rates = list()
    for ii in range(len(result.posterior)):
        likelihood.parameters.update(dict(result.posterior.iloc[ii]))
        rates.append(float(likelihood.generate_rate_posterior_sample()))
    result.posterior["rate"] = rates
    result.save_to_file(overwrite=True, extension='json')

    spin_params = []

    if runargs['spin_model'] == 'truncnorm':
        spin_params.append("mu_chi_eff")
        spin_params.append("sigma_chi_eff")

    elif runargs['spin_model'] == 'skewnorm':
        spin_params.append("mu_chi_eff")
        spin_params.append("sigma_chi_eff")
        spin_params.append("eta_chi_eff")
    elif runargs['spin_model'] == 'eps_skewnorm':
        spin_params.append("mu_chi_eff")
        spin_params.append("sigma_chi_eff")
        spin_params.append("eps_chi_eff")


    if runargs['fit_chip']:
        spin_params.append('mu_chi_p')
        spin_params.append('sigma_chi_p')



    # plot corner plot
    result.plot_corner(save=True)

    # plot spin only corner plot
    result.plot_corner(save=True, parameters=spin_params, filename=runargs['outdir'] + '/spin_corner.png')



if __name__ == "__main__":
    if len(sys.argv) != 2:
            raise ValueError('Provide the config file as an argument')
    else:   


        config = ConfigParser()
        config.read(sys.argv[1])

        runargs = {}

        # collate args
        runargs['doMixture'] = bool(int(config.get('model', 'doMixture')))
        runargs['spin_model'] = config.get('model', 'spin_model')
        runargs['priors'] = config.get('model', 'priors')
        runargs['skewness_prior'] = config.get('model', 'skewness_prior')
        runargs['doqBinning'] = bool(int(config.get('model', 'doqBinning')))
        runargs['fit_chip'] = bool(int(config.get('model', 'fit_chip')))
        runargs['fit_rate'] = bool(int(config.get('model', 'fit_rate')))
        runargs['backend'] = config.get('params', 'backend')

        runargs['pe_file'] = config.get('params', 'pe_file')
        runargs['inj_file'] = config.get('params', 'inj_file')
        runargs['nlive'] = int(config.get('params', 'nlive'))
        runargs['dlogz'] = float(config.get('params', 'dlogz'))
        runargs['rundix'] = config.get('params', 'rundix')
        runargs['configfile'] = sys.argv[1] 

        spinfit(runargs)