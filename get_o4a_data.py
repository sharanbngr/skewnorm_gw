from glob import glob
import os
import tqdm
import h5ify, h5py
import pandas
from tools.joint_prior_analytical import Joint_prob_Xeff_Xp
import numpy as np
import jax.numpy as jnp
gwtc_events = (
    'GW150914_095045', 'GW151012_095443', 'GW151226_033853', 'GW170104_101158',
    'GW170608_020116', 'GW170729_185629', 'GW170809_082821', 'GW170814_103043',
    'GW170818_022509', 'GW170823_131358', 'GW190408_181802', 'GW190412_053044',
    'GW190413_052954', 'GW190413_134308', 'GW190421_213856', 'GW190503_185404',
    'GW190512_180714', 'GW190513_205428', 'GW190517_055101', 'GW190519_153544',
    'GW190521_030229', 'GW190521_074359', 'GW190527_092055', 'GW190602_175927',
    'GW190620_030421', 'GW190630_185205', 'GW190701_203306', 'GW190706_222641',
    'GW190707_093326', 'GW190708_232457', 'GW190719_215514', 'GW190720_000836',
    'GW190725_174728', 'GW190727_060333', 'GW190728_064510', 'GW190731_140936',
    'GW190803_022701', 'GW190805_211137', 'GW190828_063405', 'GW190828_065509',
    'GW190910_112807', 'GW190915_235702', 'GW190924_021846', 'GW190925_232845',
    'GW190929_012149', 'GW190930_133541', 'GW191103_012549', 'GW191105_143521',
    'GW191109_010717', 'GW191127_050227', 'GW191129_134029', 'GW191204_171526',
    'GW191215_223052', 'GW191216_213338', 'GW191222_033537', 'GW191230_180458',
    'GW200112_155838', 'GW200128_022011', 'GW200129_065458', 'GW200202_154313',
    'GW200208_130117', 'GW200209_085452', 'GW200216_220804', 'GW200219_094415',
    'GW200224_222234', 'GW200225_060421', 'GW200302_015811', 'GW200311_115853',
    'GW200316_215756',
)



def load_posteriors(exclude = [], save = True):
    files = sorted([
        glob(f'/home/rp.o4/catalogs/GWTC-*/data-release/*{event}*_cosmo.h5')[0]
        for event in gwtc_events
    ])

    files += sorted(glob(
        # '/home/rp.o4/catalogs/O4a_prelim/2024-04-01/*.h5',
        # '/home/aditya.vijaykumar/rp_o4_87df25a0_59/*.hdf5',
        '/home/rp.o4/catalogs/GWTC-4/FourthDraftRelease/5b450400_362/*.hdf5',
    ))

    posteriors = []
    events = []

    for file in tqdm.tqdm(files):
        filename = file.split('/')[-1]

        path = (
            '/home/matthew.mould/o4a-astro-dist-model-comparison-study/data/'
            'posteriors/'
        )
        path_to_repo = '/home/sharan.banagiri/O4a/O4a_astrodist/skewnorm/data/posteriors/'


        if os.path.exists(path_to_repo + filename):
            posterior = h5ify.load(path_to_repo + filename)

        else:

            if os.path.exists(path + filename):
                posterior = h5ify.load(path + filename)

            else:
                posterior = {}

                with h5py.File(file, 'r') as f:
                    exp = 'C01:Mixed'
                    if exp not in f:
                        exp = (set(f) - {'history', 'version'}).pop()
                    data = f[exp]['posterior_samples'][:]

                    for key in (
                        'mass_1_source', 'mass_ratio', 'redshift',
                        'a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2',
                        'chi_eff', 'chi_p',
                    ):
                        posterior[key] = data[key]

                # priors are:
                # - flat in detectors frame component masses
                # - uniform in Kerr parameters
                # - isotropic in spin directions
                # - comoving redshift prior
                # normalization constants don't matter so just ignore them
                # include Jacobian from:
                # - detector-frame masses -> source frame masses
                # - secondary mass to mass ratio
                posterior['prior'] = (
                    bilby.gw.prior.UniformSourceFrame(
                        minimum = posterior['redshift'].min(),
                        maximum = posterior['redshift'].max(),
                        name = 'redshift',
                    ).prob(posterior['redshift'])
                    * (1 + posterior['redshift']) ** 2
                    * posterior['mass_1_source']
                )

            if 'chi_eff' not in posterior:
                posterior['chi_eff'] = (posterior['a_1'] * posterior['cos_tilt_1'] +\
                                        posterior['mass_ratio'] * posterior['a_2'] * posterior['cos_tilt_2']) / (1 + posterior['mass_ratio'])

                chi_p_term1 = posterior['a_1'] * (1 - posterior['cos_tilt_1']**2)**0.5
                chi_p_term2 = posterior['mass_ratio'] * (4 * posterior['mass_ratio'] + 3) / (4 + 3*posterior['mass_ratio']) * posterior['a_2'] * (1 - posterior['cos_tilt_2']**2)**0.5

                posterior['chi_p'] = np.maximum(chi_p_term1, chi_p_term2)
                        

            posterior['prior_effective_spin'] = posterior['prior'] * Joint_prob_Xeff_Xp(posterior['chi_eff'], 
                                                                                        posterior['chi_p'], 
                                                                                        posterior['mass_ratio'], 
                                                                                        amax=1)
            if save:
                h5ify.save(path_to_repo + filename, posterior)


        event = 'GW' + file.split('GW')[-1].split('-')[0].split('_PE')[0]

        if event in exclude:
            print('excluding', file)
            print(event, 'in excldue')

        elif posterior['mass_1_source'].max() < 3:
            print('excluding', file)
            print(event, 'maximum mass_1_source < 3')

        elif (posterior['mass_1_source'] * posterior['mass_ratio']).max() < 3:
            print('excluding', file)
            print(event, 'maximum mass_2_source < 3')

        elif np.min(posterior['mass_1_source']) > 80:
            import pdb; pdb.set_trace()

        else:

            posteriors.append(pandas.DataFrame.from_dict(posterior))
            events.append(event)

    return posteriors, events


def load_injections(far_cut = 1, snr_cut = 10, save = True):
    file = (
        '/home/rp.o4/offline-injections/mixtures/'
        # 'T2400110-v2/'
        # 'rpo1234-cartesian_spins-semianalytic_o1_o2_o4a-real_o3.hdf'
        # 'T2400314-v2/'
        # 'rpo1234-cartesian_spins-semi_o1_o2-real_o3-T2400314_v2.hdf'
        'T2400373-v1/'
        'rpo1234-cartesian_spins-semi_o1_o2-real_o3_o4a-no_er15-T2400373_v1.hdf'
    )

    filename = '-'.join(map(str, [far_cut, snr_cut, file.split('/')[-1]]))

    path = (
        '/home/matthew.mould/o4a-astro-dist-model-comparison-study/data/'
        'injections/'
    )
    path_to_repo = '/'.join(__file__.split('/')[:-2]) + '/data/injections/'

    #if os.path.exists(path + filename):
    #    return h5ify.load(path + filename)

    if os.path.exists(path_to_repo + filename):
        return h5ify.load(path_to_repo + filename)

    injections = {}

    with h5py.File(file, 'r') as f:
        events = f['events'][:]
        fars = [events[key] for key in events.dtype.names if 'far' in key]
        min_fars = np.min(fars, axis = 0)
        snrs = events['semianalytic_observed_phase_maximized_snr_net']
        found = (min_fars < far_cut) | (snrs > snr_cut)
        events = events[found]

        injections['mass_1_source'] = events['mass1_source']
        injections['mass_ratio'] = \
            events['mass2_source'] / injections['mass_1_source']
        injections['redshift'] = events['redshift']
        injections['a_1'] = (
            events['spin1x']**2 + events['spin1y']**2 + events['spin1z']**2
        )**0.5
        injections['a_2'] = (
            events['spin2x']**2 + events['spin2y']**2 + events['spin2z']**2
        )**0.5
        injections['cos_tilt_1'] = events['spin1z'] / injections['a_1']
        injections['cos_tilt_2'] = events['spin2z'] / injections['a_2']

        ln_prior = events[
            'lnpdraw_mass1_source_mass2_source_redshift'
            '_spin1x_spin1y_spin1z_spin2x_spin2y_spin2z'
        ]
        prior = (
            np.exp(ln_prior)
            * injections['mass_1_source']
            * 4 * np.pi**2 * injections['a_1']**2 * injections['a_2']**2
        )
        injections['prior'] = prior / events['weights']

        q = injections['mass_ratio']
        a1 = injections['a_1']
        a2 = injections['a_2']
        c1 = injections['cos_tilt_1']
        c2 = injections['cos_tilt_2']
        s1 = np.sin(np.arccos(c1))
        s2 = np.sin(np.arccos(c2))
        injections['chi_eff'] = (a1 * c1 + q * a2 * c2) / (1 + q)
        injections['chi_p'] = np.max(
            [a1 * s1, a2 * s2 * (4 * q + 3) / (4 + 3 * q)], axis = 0,
        )
        injections['prior_effective_spin'] = injections['prior'] * Joint_prob_Xeff_Xp(injections['chi_eff'], 
                                                                                      injections['chi_p'], 
                                                                                      injections['mass_ratio'], 
                                                                                      amax=1)
        
        zeroidx = injections['prior_effective_spin'] == 0
        injections['prior_effective_spin'][zeroidx] = 1e-12

        injections['found'] = found.sum()
        injections['total_generated'] = f.attrs['total_generated']

        for key in 'analysis_time', 'total_analysis_time', 'analysis_time_s':
            if key in f.attrs:
                injections['analysis_time'] = f.attrs[key]
        if 'analysis_time' not in injections:
            print('analysis_time not found')
        else:
            injections['analysis_time'] /= 60 * 60 * 24 * 365.25


    for key in injections:
        injections[key] = np.asarray(injections[key])

    for key in injections:
        injections[key] = jnp.array(injections[key])

    if save:
        h5ify.save(path_to_repo + filename, injections)
    
    return injections

