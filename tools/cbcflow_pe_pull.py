import numpy as np
import cbcflow
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from cosmology import PLANCK_2015_Cosmology as cosmo

def construct_prior_dl_to_z(data):
    z_max = 1.9

    if max(data["redshift"]) > z_max:
        z_max = max(data["redshift"])

    zs = np.linspace(0, z_max * 1.01, 1000)
    
    ## Convert to dl
    dl = cosmo.z2DL(zs) / 1e3
    p_z = dl**2 * (dl / (1 + zs) + (1 + zs) * cosmo.dDcdz(zs, mpc=True) / 1e3)
    p_z /= np.trapz(p_z, zs)
    
    
    data["prior"] = np.interp(np.array(data["redshift"]), zs, p_z) * data["mass_1"] * (1 + data["redshift"]) ** 2
    
    # For spins
    data["prior"] /= 4
    
    return data



def construct_O4a_prelim_prior(data):

    z_max = 1.9

    if max(data["redshift"]) > z_max:
        z_max = max(data["redshift"])

    zs = np.linspace(0, z_max * 1.01, 1000)

    ## calc Dc
    dc = cosmo.z2DL(zs) / 1e3 / (1 + zs)

    p_z = dc**2 * cosmo.dDcdz(zs, mpc=True) / 1e3  / (1 + zs)
    p_z /= np.trapz(p_z, zs)

    p_z =  np.interp(np.array(data["redshift"]), zs, p_z)

    ## calc the symmetric mass ratio
    eta = data['mass_ratio'] / (1 + data['mass_ratio'])**2

    data["prior"] = p_z * (1 + data["redshift"]) * (1 + data['mass_ratio']) * eta**(3/5)

    return data


## the LocalLibraryDatabase object
library = cbcflow.database.LocalLibraryDatabase('/home/cbc/cbcflow/O4a/cbc-workflow-o4a')


o4pe = {}

for sevent in library.superevents_in_library:
    #print('extracting info for ' + sevent)
    seventdata = cbcflow.get_superevent(sevent)

    isretracted = False
    if len(seventdata["Info"]["Notes"]) !=0:
        for note in seventdata["Info"]["Notes"]:
            if "retracted:" in note.lower():
                isretracted = True


    if not isretracted:

        far = 10
        for uevent in seventdata['GraceDB']['Events']:
            if uevent['FAR'] < far:
                far = uevent['FAR']
            if uevent['State'] == 'preferred':
                snr = uevent['NetworkSNR']

        if snr < 10 or far > 3.1688e-08:
            print(sevent + ' does not pass the snr/far threshold')
        else:
            try:
                illustrative = seventdata['ParameterEstimation']['IllustrativeResult']
                for pe in seventdata['ParameterEstimation']['Results']:
                    if pe['UID'].casefold() == illustrative.casefold():
                        illustrative_exists = True

                        pe_data = h5py.File(pe['ResultFile']['Path'][4:], 'r')


                        df = {
                            'mass_1': pe_data['posterior']['mass_1_source'][:], 
                            'mass_2': pe_data['posterior']['mass_2_source'][:],
                            'mass_ratio': pe_data['posterior']['mass_ratio'][:],
                            'redshift': pe_data['posterior']['redshift'][:],            
                            'a_1': pe_data['posterior']['a_1'][:],
                            'a_2': pe_data['posterior']['a_2'][:], 
                            'cos_tilt_1': pe_data['posterior']['cos_tilt_1'][:],
                            'cos_tilt_2': pe_data['posterior']['cos_tilt_2'][:],
                            'chi_eff': pe_data['posterior']['chi_eff'][:],
                            'chi_p': pe_data['posterior']['chi_p'][:],
                            }


                        df = construct_O4a_prelim_prior(df)
                        df = pd.DataFrame(data=df)

                        o4pe[sevent] = df
            except:
                print('no illustrative PE for ' + sevent)

    else:
        print(sevent + ' has been retracted')

print('PE obtained for ' + str(len(o4pe.keys())) + ' triggers')

import pdb; pdb.set_trace()