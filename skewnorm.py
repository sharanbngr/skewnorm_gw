import sys, os
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.redshift import PowerLawRedshift
from gwpopulation.models.spin import gaussian_chi_eff, skewnorm_chi_eff, gaussian_chi_p, eps_skewnorm_chi_eff
from gwpopulation.hyperpe import HyperparameterLikelihood, RateLikelihood
from gwpopulation.vt import ResamplingVT
import pickle, json
import bilby
from bilby.core.prior import PriorDict
from gwpopulation.backend import set_backend
import numpy as np
from configparser import ConfigParser
from gwpopulation.experimental.jax import NonCachingModel, JittedLikelihood
from mixture_models import *

def get_model(models, backend):
    if type(models) not in (list, tuple):
        models = [models]
    Model = NonCachingModel if backend == 'jax' else bilby.hyper.model.Model
    return Model(
        [model() if type(model) is type else model for model in models]
    )


def q_binning_skewnorm(dataset,
                       mu1, sigma1, eta1,
                       mu2, sigma2, eta2):


    pdf = xp.where(dataset['mass_ratio'] >= 0.8,
                   skewnorm_chi_eff(dataset, mu_chi_eff=mu1, sigma_chi_eff=sigma1, eta_chi_eff=eta1),
                   skewnorm_chi_eff(dataset, mu_chi_eff=mu2, sigma_chi_eff=sigma2, eta_chi_eff=eta2),)

    return pdf


def q_binning_eps_skewnorm(dataset,
                       mu1, sigma1, eps1,
                       mu2, sigma2, eps2):


    pdf = xp.where(dataset['mass_ratio'] >= 0.8,
                   eps_skewnorm_chi_eff(dataset, mu_chi_eff=mu1, sigma_chi_eff=sigma1, eps_chi_eff=eps1),
                   eps_skewnorm_chi_eff(dataset, mu_chi_eff=mu2, sigma_chi_eff=sigma2, eps_chi_eff=eps2),)

    return pdf


def spinfit(runargs):

    global xp
    if runargs['backend'] == 'numpy':
        xp = np
    elif runargs['backend'] == 'cupy':
        import cupy
        xp = cupy
    elif runargs['backend'] == 'jax':
        import jax.numpy as jnp
        xp = jnp

    set_backend(backend=runargs['backend'])

    runargs['outdir'] = './' + runargs['spin_model'] + '_' + runargs['sampler'] + '_' + runargs['backend'] + '_' + runargs['rundix']


    # create directory and copy the config file
    os.system('mkdir -p ' + runargs['outdir'])
    os.system('cp '  + runargs['configfile']  + ' ' + runargs['outdir'] + '/config.ini')



    # extract posterior
    with open(runargs['pe_file'], 'rb') as f:
        post = pickle.load(f)
    #post = dd.io.load(runargs['pe_file'])

    # this event is tooo massive. remove for now. 
    try:
        post.pop('S231020bw')
    except:
        print('S231020bw does not exist in this catalog')

    # get injections
    with open(runargs['inj_file'], 'rb') as f:
        injs = pickle.load(f)

    for key in injs.keys():
        injs[key] = xp.array(injs[key])

    posteriors = []


    ## do prior conversions
    if runargs['fit_chip']:

        print('converting PE priors to chi_eff, chi_p ...')
        for event in post.keys():

            post[event]['prior'] *= 4.0 * post[event]['chieff_chip_prior']

            posteriors.append(post[event])

    else:
        print('converting PE priors to chi_eff ...')
        for event in post.keys():
            post[event]['prior'] *= 4.0 * post[event]['chieff_prior']

 
            try:
                post[event]  = post[event].drop(columns=['chieff_chip_prior'])               
            except:
                print('No chi_p prior column in ' + event)

            posteriors.append(post[event])


    if runargs['fit_chip']:
        print('new - converting inj priors to chi_eff, chi_p ...')
        injs['prior'] *= 4*injs['chieff_chip_prior'] * (injs['a_1'] * injs['a_2'])**2

    else:
        print('new - converting inj priors to chi_eff ...')
        try:
            injs.pop('chieff_chip_prior')
        except:
            pass
        injs['prior'] *= 4*injs['chieff_prior'] * (injs['a_1'] * injs['a_2'])**2


    priors = PriorDict(filename=runargs['priors'])

    models = [SinglePeakSmoothedMassDistribution, PowerLawRedshift]

    if runargs['spin_model'] == 'skewnorm' and not runargs['fit_chip']:
        models.append(skewnorm_chi_eff)
        priors.pop('mu_chi_p')
        priors.pop('sigma_chi_p')

    elif runargs['spin_model'] == 'skewnorm' and runargs['fit_chip']:
        models.append(skewnorm_chi_eff)
        models.append(gaussian_chi_p)

    elif runargs['spin_model'] == 'truncnorm' and not runargs['fit_chip']:
        models.append(gaussian_chi_eff)
        priors.pop('mu_chi_p')
        priors.pop('sigma_chi_p')
        priors.pop('spin_covariance')

    elif runargs['spin_model'] == 'truncnorm' and runargs['fit_chip']:
        models.append(gaussian_chi_eff)
        models.append(gaussian_chi_p)
        priors.pop('spin_covariance')

    elif runargs['spin_model'] == 'eps_skewnorm' and runargs['fit_chip']:
        models.append(eps_skewnorm_chi_eff)
        models.append(gaussian_chi_p)

    elif runargs['spin_model'] == 'eps_skewnorm' and not runargs['fit_chip']:
        models.append(eps_skewnorm_chi_eff)
        priors.pop('mu_chi_p')
        priors.pop('sigma_chi_p')

    elif runargs['spin_model'] == 'skewnorm_mixture':
        models.append(skewnorm_mixture_model)
        priors.pop('eps_al')

    elif runargs['spin_model'] == 'eps_skewnorm_mixture':
        models.append(eps_skewnorm_mixture_model)
        priors.pop('eta_al')

    elif runargs['spin_model'] == 'qbinning_skewnorm':
        models.append(q_binning_skewnorm)
        priors.pop('eps1')
        priors.pop('eps2')

    elif runargs['spin_model'] == 'q_binning_eps_skewnorm':
        models.append(q_binning_eps_skewnorm)
        priors.pop('eta1')
        priors.pop('eta2')


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

    if runargs['sampler'] == 'numpyro':
        result = bilby.run_sampler(likelihood = likelihood,
            resume=True,
            priors = priors,
            label = 'GWTC-3',
            sampler=runargs['sampler'],
            use_ratio=True,
            num_warmup=500,
            num_samples=4000,
            check_point=True,
            n_check_point=250,
            thinning=1,
            num_chains=1,
            outdir = runargs['outdir'])

    elif runargs['backend'] == 'jax' and runargs['sampler'] == 'dynesty':
        jitted_likelihood = JittedLikelihood(likelihood)
        result = bilby.run_sampler(likelihood = jitted_likelihood,
                    nlive=runargs['nlive'], resume=True,
                    priors = priors,
                    label = 'GWTC-3',
                    sampler=runargs['sampler'],
                    use_ratio=True,
                    check_point_delta_t = 300,
                    outdir = runargs['outdir'])
        
    elif runargs['sampler'] == 'dynesty':
        result = bilby.run_sampler(likelihood = likelihood,
            resume=True,
            nlive=runargs['nlive'],
            priors = priors,
            label = 'GWTC-3',
            sampler=runargs['sampler'],
            use_ratio=True,
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


    for key in result.priors.keys():
        latex_label = result.priors[key].latex_label

        if '\\rm' in latex_label:

            idx = latex_label.find('\\rm')
            result.priors[key].latex_label = latex_label[:idx] + latex_label[idx + 3:]
            result._priors[key].latex_label = latex_label[:idx] + latex_label[idx + 3:]


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
        runargs['spin_model'] = config.get('model', 'spin_model')
        runargs['priors'] = config.get('model', 'priors')
        runargs['skewness_prior'] = config.get('model', 'skewness_prior')
        #runargs['doqBinning'] = bool(int(config.get('model', 'doqBinning')))
        runargs['fit_chip'] = bool(int(config.get('model', 'fit_chip')))
        runargs['fit_rate'] = bool(int(config.get('model', 'fit_rate')))
        runargs['backend'] = config.get('params', 'backend')
        runargs['sampler'] = config.get('params', 'sampler')
        #if runargs['doqBinning']:
        #    runargs['qBins'] = json.loads(config.get('model', 'qBins'))

        runargs['pe_file'] = config.get('params', 'pe_file')
        runargs['inj_file'] = config.get('params', 'inj_file')
        runargs['nlive'] = int(config.get('params', 'nlive'))
        runargs['dlogz'] = float(config.get('params', 'dlogz'))
        runargs['rundix'] = config.get('params', 'rundix')
        runargs['configfile'] = sys.argv[1]

        spinfit(runargs)
