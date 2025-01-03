import sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NPROC'] = '1'


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
from get_o4a_data import load_posteriors, load_injections

import jax
jax.config.update('jax_enable_x64', True)
from gwpopulation.experimental.jax import NonCachingModel, JittedLikelihood
from mixture_models import *




def get_model(models, backend):
    if type(models) not in (list, tuple):
        models = [models]
    Model = NonCachingModel if backend == 'jax' else bilby.hyper.model.Model
    return Model(
        [model() if type(model) is type else model for model in models]
    )



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



    print('loading posteriors...')
    posteriors, events = load_posteriors(exclude=['GW200129_065458', 'GW231123_135430'])

    nmin = int(3e3)

    for ii in range(len(events)):
        post = posteriors[ii]
        if post.shape[0] < nmin:
            import pdb; pdb.set_trace()
        else:
            post = post.sample(n=nmin, axis=0)


    print('loading injections...')
    injs = load_injections()
    for key in injs:
        injs[key] = xp.array(injs[key])

    ## do prior conversions
    if runargs['fit_chip']:

        print('converting PE priors to chi_eff, chi_p ...')
        for post in posteriors:
            post['prior'] = post['prior_effective_spin']
            post['mass_1'] = post['mass_1_source']

        print('converting inj priors to chi_eff, chi_p ...')
        #injs['prior'] *= 4*injs['chieff_chip_prior'] * (injs['a_1'] * injs['a_2'])**2
        injs['prior'] = injs['prior_effective_spin']
        injs['mass_1'] = injs['mass_1_source']


    priors = PriorDict(filename=runargs['priors'])

    models = [PowerLawRedshift]

    ## are we doing joint q-chi_eff mixture models?
    if runargs['spin_model'] == 'q_skewnorm_mixture':
        models.append(DoubleqPLP_skewnorm)
        priors.pop('eps_al')
    elif runargs['spin_model'] == 'q_eps_skewnorm_mixture':
        models.append(DoubleqPLP_eps_skewnorm)
        priors.pop('eta_al')
    else:
        models.append(SinglePeakSmoothedMassDistribution)


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
                            enforce_convergence=False)


    likelihood = hyperlikelihood(posteriors = posteriors,
                                 hyper_prior = get_model(models, runargs['backend']),
                                 selection_function = VTs, maximum_uncertainty=1)

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
                    outdir = runargs['outdir'],
                    sample = 'acceptance-walk',
                    naccept = 10,)

    elif runargs['sampler'] == 'dynesty':
        result = bilby.run_sampler(likelihood = likelihood,
            resume=True,
            nlive=runargs['nlive'],
            priors = priors,
            label = 'GWTC-3',
            sampler=runargs['sampler'],
            use_ratio=True,
            check_point_delta_t = 300,
            outdir = runargs['outdir'],)
            #sample = 'acceptance-walk',
            #naccept = 10,)

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

        #runargs['pe_file'] = config.get('params', 'pe_file')
        #runargs['inj_file'] = config.get('params', 'inj_file')
        runargs['nlive'] = int(config.get('params', 'nlive'))
        runargs['dlogz'] = float(config.get('params', 'dlogz'))
        runargs['rundix'] = config.get('params', 'rundix')
        runargs['configfile'] = sys.argv[1]

        spinfit(runargs)
