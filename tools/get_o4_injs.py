import numpy as np
import h5py
import pickle
import pandas as pd
from cosmology import PLANCK_2015_Cosmology as cosmo
import sys

def detected_injections(snr, snr_threshold):

    found_index = snr > snr_threshold

    return found_index


def get_o1o2o3o4_injs(snr_threshold):

    injs = h5py.File('./rpo1234-cartesian_spins-semianalytic_o1_o2_o4a-real_o3.hdf', 'r')
    savefile = './o1o2o3o4_prelim_injs_snr-' + str(snr_threshold) + '.pkl'


    o1o2o3o4_injs = {
                    'mass_1': injs['events']['mass1_source'],
                    'mass_2': injs['events']['mass2_source'],
                    'redshift': injs['events']['redshift'],
                    'snrs': injs['events']['semianalytic_observed_phase_maximized_snr_net'],
                    }


    a1x, a1y, a1z = injs['events']['spin1x'], injs['events']['spin1y'], injs['events']['spin1z']
    a2x, a2y, a2z = injs['events']['spin2x'], injs['events']['spin2y'], injs['events']['spin2z']

    o1o2o3o4_injs['mass_ratio'] = o1o2o3o4_injs['mass_2'] / o1o2o3o4_injs['mass_1']

    snr_indx = detected_injections(o1o2o3o4_injs['snrs'], snr_threshold)
    gstlal_indx = injs['events']['far_gstlal'] <= 1
    mbta_indx = injs['events']['far_mbta'] <= 1
    pycbc_bbh_indx = injs['events']['far_pycbc_bbh'] <= 1
    pycbc_hyperbank_indx = injs['events']['far_pycbc_hyperbank'] <= 1

    detected_idx = snr_indx | gstlal_indx | mbta_indx | pycbc_bbh_indx | pycbc_hyperbank_indx

    log_prior = injs['events']['lnpdraw_mass1_source_mass2_source_redshift_spin1x_spin1y_spin1z_spin2x_spin2y_spin2z']

    ## log prior in (m1, m2, z, a1, a2, t1, t2, phi1, phi2)
    #log_prior = injs['events']['lnpdraw_mass1_source_mass2_source_redshift_spin1_magnitude_spin1_polar_angle_spin1_azimuthal_angle_spin2_magnitude_spin2_polar_angle_spin2_azimuthal_angle']

    ## tranform into a prior on (Mc, q, z, a1, a2, cos t1, cos t2)
    #log_prior += np.log(o1o2o3o4_injs['mass_1']) + \
    #            np.log(4*np.pi**2) - \
    #            np.log(np.sin(injs['events']['spin1_polar_angle'])) - \
    #            np.log(np.sin(injs['events']['spin2_polar_angle']))

    ## add the mixture model weights in
    o1o2o3o4_injs['prior'] = np.exp(log_prior) / injs['events']['weights']

    o1o2o3o4_injs['a_1'] = np.sqrt(a1x**2 + a1y**2 + a1z**2)
    o1o2o3o4_injs['a_2'] = np.sqrt(a2x**2 + a2y**2 + a2z**2)

    o1o2o3o4_injs['cos_tilt_1'] = a1z / o1o2o3o4_injs['a_1']
    o1o2o3o4_injs['cos_tilt_2'] = a2z / o1o2o3o4_injs['a_2']

    jacobian_spin1 = np.sqrt(1 - o1o2o3o4_injs['cos_tilt_1']**2)
    jacobian_spin2 = np.sqrt(1 - o1o2o3o4_injs['cos_tilt_2']**2)

    #jacobian_spin1 = 2 * np.pi * o1o2o3o4_injs['a_1']**2
    #jacobian_spin2 = 2 * np.pi * o1o2o3o4_injs['a_2']**2
    o1o2o3o4_injs['prior'] *= jacobian_spin1 * jacobian_spin2 * o1o2o3o4_injs['mass_1']

    for key in o1o2o3o4_injs.keys():
        o1o2o3o4_injs[key] = o1o2o3o4_injs[key][detected_idx]

    bbh_idx = o1o2o3o4_injs['mass_2'] >= 3

    for key in o1o2o3o4_injs.keys():
        o1o2o3o4_injs[key] = o1o2o3o4_injs[key][bbh_idx]


    o1o2o3o4_injs['total_generated'] = np.round(injs.attrs['total_generated'])
    o1o2o3o4_injs['analysis_time'] = injs.attrs['analysis_time'] /60/60/24/365.24

    with open(savefile, 'wb') as f:
        pickle.dump(o1o2o3o4_injs, f)


    return


def get_o4a_semianalytical(snr_threshold):


    injs = h5py.File('/home/rp.o4/offline-injections/semianalytic/T2400073-v2/samples-rpo4a_v1_1ifo-1366933504-23846400.hdf', 'r')
    savefile = './o4_prelim_injs_snr-' + str(snr_threshold) + '.pkl'

    o4a_injs = {
                    'mass_1': injs['events']['mass1_source'],
                    'mass_2': injs['events']['mass2_source'],
                    'redshift': injs['events']['z'],
                    'a_1': injs['events']['spin1_magnitude'],
                    'a_2': injs['events']['spin2_magnitude'],
                    'cos_tilt_1': np.cos(injs['events']['spin1_polar_angle']),
                    'cos_tilt_2': np.cos(injs['events']['spin2_polar_angle']),
                    'chi_eff': injs['events']['chi_eff'],
                    'chi_p': injs['events']['chi_p'],
                    'snrs': injs['events']['observed_phase_maximized_snr_net'],
                    }


    o4a_injs['cos_tilt_1'] = injs['events']['spin1z'] / o4a_injs['a_1']
    o4a_injs['cos_tilt_2'] = injs['events']['spin2z'] / o4a_injs['a_2']


    ## this is the log prior in (m1, m2, z, a1, a2, t1, t2)
    log_prior = injs['events']['lnpdraw_mass1_source'] + \
                injs['events']['lnpdraw_mass2_source_GIVEN_mass1_source'] + \
                injs['events']['lnpdraw_z'] + \
                injs['events']['lnpdraw_spin1_magnitude'] + \
                injs['events']['lnpdraw_spin2_magnitude'] + \
                injs['events']['lnpdraw_spin1_polar_angle'] + \
                injs['events']['lnpdraw_spin2_polar_angle']


    ## tranform into a prior on (Mc, q, z, a1, a2, cos t1, cos t2)
    log_prior += np.log(o4a_injs['mass_1']) - \
            np.log(np.sin(injs['events']['spin1_polar_angle'])) - \
            np.log(np.sin(injs['events']['spin2_polar_angle']))


    o4a_injs['prior'] = np.exp(log_prior)

    detected_idx = detected_injections(o4a_injs['snrs'], snr_threshold)

    for key in o4a_injs.keys():
        o4a_injs[key] = o4a_injs[key][detected_idx]


    bbh_idx = o4a_injs['mass_2'] >= 3

    for key in o4a_injs.keys():
        o4a_injs[key] = o4a_injs[key][bbh_idx]


    o4a_injs['total_generated'] = np.round(injs.attrs['total_generated'])
    o4a_injs['analysis_time'] = injs.attrs['total_analysis_time'] /60/60/24/365.24

    with open(savefile, 'wb') as f:
        pickle.dump(o4a_injs, f)


    return



if __name__ == "__main__":

    print('using SNR threshold of ' + sys.argv[2])

    if sys.argv[1] == 'o4a':
        get_o4a_semianalytical(float( sys.argv[2]))
    elif sys.argv[1] == 'o1-o4':
        get_o1o2o3o4_injs(float( sys.argv[2]))
    else:
        print('unknown input string.')
