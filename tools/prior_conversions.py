import numpy as np
from scipy.stats import gaussian_kde
from scipy.special import spence as PL
#import deepdish as dd
import pickle
import sys


## adapted from Tom's repository https://github.com/tcallister/effective-spin-priors

def Di(z):

    """
    Wrapper for the scipy implmentation of Spence's function.
    Note that we adhere to the Mathematica convention as detailed in:
    https://reference.wolfram.com/language/ref/PolyLog.html

    Inputs
    z: A (possibly complex) scalar or array

    Returns
    Array equivalent to PolyLog[2,z], as defined by Mathematica
    """

    return PL(1.-z+0j)

def chi_effective_prior_from_aligned_spins(q,aMax,xs):

    """
    Function defining the conditional priors p(chi_eff|q) corresponding to
    uniform, aligned component spin priors.

    Inputs
    q: Mass ratio value (according to the convention q<1)
    aMax: Maximum allowed dimensionless component spin magnitude
    xs: Chi_effective value or values at which we wish to compute prior

    Returns:
    Array of prior values
    """

    # Ensure that `xs` is an array and take absolute value
    xs = np.reshape(xs,-1)

    # Set up various piecewise cases
    pdfs = np.zeros(xs.size)
    caseA = (xs>aMax*(1.-q)/(1.+q))*(xs<=aMax)
    caseB = (xs<-aMax*(1.-q)/(1.+q))*(xs>=-aMax)
    caseC = (xs>=-aMax*(1.-q)/(1.+q))*(xs<=aMax*(1.-q)/(1.+q))

    # Select relevant effective spins
    x_A = xs[caseA]
    x_B = xs[caseB]
    x_C = xs[caseC]

    pdfs[caseA] = (1.+q)**2.*(aMax-x_A)/(4.*q*aMax**2)
    pdfs[caseB] = (1.+q)**2.*(aMax+x_B)/(4.*q*aMax**2)
    pdfs[caseC] = (1.+q)/(2.*aMax)

    return pdfs

def chi_effective_prior_from_isotropic_spins(q,aMax,xs):

    """
    Function defining the conditional priors p(chi_eff|q) corresponding to
    uniform, isotropic component spin priors.

    Inputs
    q: Mass ratio value (according to the convention q<1)
    aMax: Maximum allowed dimensionless component spin magnitude
    xs: Chi_effective value or values at which we wish to compute prior

    Returns:
    Array of prior values
    """

    # Ensure that `xs` is an array and take absolute value
    xs = np.reshape(np.abs(xs),-1)

    # Set up various piecewise cases
    pdfs = np.ones(xs.size,dtype=complex)*(-1.)
    caseZ = (xs==0)
    caseA = (xs>0)*(xs<aMax*(1.-q)/(1.+q))*(xs<q*aMax/(1.+q))
    caseB = (xs<aMax*(1.-q)/(1.+q))*(xs>q*aMax/(1.+q))
    caseC = (xs>aMax*(1.-q)/(1.+q))*(xs<q*aMax/(1.+q))
    caseD = (xs>aMax*(1.-q)/(1.+q))*(xs<aMax/(1.+q))*(xs>=q*aMax/(1.+q))
    caseE = (xs>aMax*(1.-q)/(1.+q))*(xs>aMax/(1.+q))*(xs<aMax)
    caseF = (xs>=aMax)

    # Select relevant effective spins
    x_A = xs[caseA]
    x_B = xs[caseB]
    x_C = xs[caseC]
    x_D = xs[caseD]
    x_E = xs[caseE]

    # the corresponding mass ratio values
    qZ = q[caseZ]
    qA = q[caseA]
    qB = q[caseB]
    qC = q[caseC]
    qD = q[caseD]
    qE = q[caseE]



    pdfs[caseZ] = (1.+qZ)/(2.*aMax)*(2.-np.log(qZ))

    pdfs[caseA] = (1.+qA)/(4.*qA*aMax**2)*(
                    qA*aMax*(4.+2.*np.log(aMax) - np.log(qA**2*aMax**2 - (1.+qA)**2*x_A**2))
                    - 2.*(1.+qA)*x_A*np.arctanh((1.+qA)*x_A/(qA*aMax))
                    + (1.+qA)*x_A*(Di(-qA*aMax/((1.+qA)*x_A)) - Di(qA*aMax/((1.+qA)*x_A)))
                    )

    pdfs[caseB] = (1.+qB)/(4.*qB*aMax**2)*(
                    4.*qB*aMax
                    + 2.*qB*aMax*np.log(aMax)
                    - 2.*(1.+qB)*x_B*np.arctanh(qB*aMax/((1.+qB)*x_B))
                    - qB*aMax*np.log((1.+qB)**2*x_B**2 - qB**2*aMax**2)
                    + (1.+qB)*x_B*(Di(-qB*aMax/((1.+qB)*x_B)) - Di(qB*aMax/((1.+qB)*x_B)))
                    )

    pdfs[caseC] = (1.+qC)/(4.*qC*aMax**2)*(
                    2.*(1.+qC)*(aMax-x_C)
                    - (1.+qC)*x_C*np.log(aMax)**2.
                    + (aMax + (1.+qC)*x_C*np.log((1.+qC)*x_C))*np.log(qC*aMax/(aMax-(1.+qC)*x_C))
                    - (1.+qC)*x_C*np.log(aMax)*(2. + np.log(qC) - np.log(aMax-(1.+qC)*x_C))
                    + qC*aMax*np.log(aMax/(qC*aMax-(1.+qC)*x_C))
                    + (1.+qC)*x_C*np.log((aMax-(1.+qC)*x_C)*(qC*aMax-(1.+qC)*x_C)/qC)
                    + (1.+qC)*x_C*(Di(1.-aMax/((1.+qC)*x_C)) - Di(qC*aMax/((1.+qC)*x_C)))
                    )

    pdfs[caseD] = (1.+qD)/(4.*qD*aMax**2)*(
                    -x_D*np.log(aMax)**2
                    + 2.*(1.+qD)*(aMax-x_D)
                    + qD*aMax*np.log(aMax/((1.+qD)*x_D-qD*aMax))
                    + aMax*np.log(qD*aMax/(aMax-(1.+qD)*x_D))
                    - x_D*np.log(aMax)*(2.*(1.+qD) - np.log((1.+qD)*x_D) - qD*np.log((1.+qD)*x_D/aMax))
                    + (1.+qD)*x_D*np.log((-qD*aMax+(1.+qD)*x_D)*(aMax-(1.+qD)*x_D)/qD)
                    + (1.+qD)*x_D*np.log(aMax/((1.+qD)*x_D))*np.log((aMax-(1.+qD)*x_D)/qD)
                    + (1.+qD)*x_D*(Di(1.-aMax/((1.+qD)*x_D)) - Di(qD*aMax/((1.+qD)*x_D)))
                    )

    pdfs[caseE] = (1.+qE)/(4.*qE*aMax**2)*(
                    2.*(1.+qE)*(aMax-x_E)
                    - (1.+qE)*x_E*np.log(aMax)**2
                    + np.log(aMax)*(
                        aMax
                        -2.*(1.+qE)*x_E
                        -(1.+qE)*x_E*np.log(qE/((1.+qE)*x_E-aMax))
                        )
                    - aMax*np.log(((1.+qE)*x_E-aMax)/qE)
                    + (1.+qE)*x_E*np.log(((1.+qE)*x_E-aMax)*((1.+qE)*x_E-qE*aMax)/qE)
                    + (1.+qE)*x_E*np.log((1.+qE)*x_E)*np.log(qE*aMax/((1.+qE)*x_E-aMax))
                    - qE*aMax*np.log(((1.+qE)*x_E-qE*aMax)/aMax)
                    + (1.+qE)*x_E*(Di(1.-aMax/((1.+qE)*x_E)) - Di(qE*aMax/((1.+qE)*x_E)))
                    )

    pdfs[caseF] = 0.

    # Deal with spins on the boundary between cases
    if np.any(pdfs==-1):
        boundary = (pdfs==-1)
        pdfs[boundary] = 0.5*(chi_effective_prior_from_isotropic_spins(q,aMax,xs[boundary]+1e-6)\
                        + chi_effective_prior_from_isotropic_spins(q,aMax,xs[boundary]-1e-6))

    return np.real(pdfs)

def chi_p_prior_from_isotropic_spins(q,aMax,xs):

    """
    Function defining the conditional priors p(chi_p|q) corresponding to
    uniform, isotropic component spin priors.

    Inputs
    q: Mass ratio value (according to the convention q<1)
    aMax: Maximum allowed dimensionless component spin magnitude
    xs: Chi_p value or values at which we wish to compute prior

    Returns:
    Array of prior values
    """

    # Ensure that `xs` is an array and take absolute value
    xs = np.reshape(xs,-1)

    # Set up various piecewise cases
    pdfs = np.zeros(xs.size)
    caseA = xs<q*aMax*(3.+4.*q)/(4.+3.*q)
    caseB = (xs>=q*aMax*(3.+4.*q)/(4.+3.*q))*(xs<aMax)

    # Select relevant effective spins
    x_A = xs[caseA]
    x_B = xs[caseB]

    # the corresponding mass ratio values
    qA = q[caseA]
    qB = q[caseB]

    pdfs[caseA] = (1./(aMax**2*qA))*((4.+3.*qA)/(3.+4.*qA))*(
                    np.arccos((4.+3.*qA)*x_A/((3.+4.*qA)*qA*aMax))*(
                        aMax
                        - np.sqrt(aMax**2-x_A**2)
                        + x_A*np.arccos(x_A/aMax)
                        )
                    + np.arccos(x_A/aMax)*(
                        aMax*qA*(3.+4.*qA)/(4.+3.*qA)
                        - np.sqrt(aMax**2*qA**2*((3.+4.*qA)/(4.+3.*qA))**2 - x_A**2)
                        + x_A*np.arccos((4.+3.*qA)*x_A/((3.+4.*qA)*aMax*qA))
                        )
                    )

    pdfs[caseB] = (1./aMax)*np.arccos(x_B/aMax)

    return pdfs

def joint_prior_from_isotropic_spins(mratio,aMax,xeffs,xps,ndraws=10000,bw_method='scott'):

    """
    Function to calculate the conditional priors p(xp|xeff,q) on a set of {xp,xeff,q} posterior samples.

    INPUTS
    qs: Mass ratio
    aMax: Maximimum spin magnitude considered
    xeffs: Effective inspiral spin samples
    xps: Effective precessing spin values
    ndraws: Number of draws from the component spin priors used in numerically building interpolant

    RETURNS
    p_chi_p: Array of priors on xp, conditioned on given effective inspiral spins and mass ratios
    """

    # Convert to arrays for safety
    xeffs = np.reshape(xeffs,-1)
    xps = np.reshape(xps,-1)
    mratio = np.reshape(mratio, -1)
    # Compute marginal prior on xeff, conditional prior on xp, and multiply to get joint prior!
    p_chi_eff = chi_effective_prior_from_isotropic_spins(mratio, aMax, xeffs)
    p_chi_p_given_chi_eff = np.array([chi_p_prior_given_chi_eff_q(mratio[i], aMax, xeffs[i], xps[i], ndraws,bw_method) for i in range(len(xeffs))])

    joint_p_chi_p_chi_eff = p_chi_eff*p_chi_p_given_chi_eff

    return joint_p_chi_p_chi_eff

def chi_p_prior_given_chi_eff_q(q,aMax,xeff,xp,ndraws=10000,bw_method='scott'):

    """
    Function to calculate the conditional prior p(xp|xeff,q) on a single {xp,xeff,q} posterior sample.
    Called by `joint_prior_from_isotropic_spins`.

    INPUTS
    q: Single posterior mass ratio sample
    aMax: Maximimum spin magnitude considered
    xeff: Single effective inspiral spin sample
    xp: Single effective precessing spin value
    ndraws: Number of draws from the component spin priors used in numerically building interpolant

    RETURNS
    p_chi_p: Prior on xp, conditioned on given effective inspiral spin and mass ratio
    """

    # Draw random spin magnitudes.
    # Note that, given a fixed chi_eff, a1 can be no larger than (1+q)*chi_eff,
    # and a2 can be no larger than (1+q)*chi_eff/q
    a1 = np.random.random(ndraws)*aMax
    a2 = np.random.random(ndraws)*aMax

    # Draw random tilts for spin 2
    cost2 = 2.*np.random.random(ndraws)-1.

    # Finally, given our conditional value for chi_eff, we can solve for cost1
    # Note, though, that we still must require that the implied value of cost1 be *physical*
    cost1 = (xeff*(1.+q) - q*a2*cost2)/a1

    # While any cost1 values remain unphysical, redraw a1, a2, and cost2, and recompute
    # Repeat as necessary
    while np.any(cost1<-1) or np.any(cost1>1):
        to_replace = np.where((cost1<-1) | (cost1>1))[0]
        a1[to_replace] = np.random.random(to_replace.size)*aMax
        a2[to_replace] = np.random.random(to_replace.size)*aMax
        cost2[to_replace] = 2.*np.random.random(to_replace.size)-1.
        cost1 = (xeff*(1.+q) - q*a2*cost2)/a1

    # Compute precessing spins and corresponding weights, build KDE
    # See `Joint-ChiEff-ChiP-Prior.ipynb` for a discussion of these weights
    Xp_draws = chi_p_from_components(a1,a2,cost1,cost2,q)
    jacobian_weights = (1.+q)/a1
    prior_kde = gaussian_kde(Xp_draws,weights=jacobian_weights,bw_method=bw_method)

    # Compute maximum chi_p
    if (1.+q)*np.abs(xeff)/q<aMax:
        max_Xp = aMax
    else:
        max_Xp = np.sqrt(aMax**2 - ((1.+q)*np.abs(xeff)-q)**2.)

    # Set up a grid slightly inside (0,max chi_p) and evaluate KDE
    reference_grid = np.linspace(0.05*max_Xp,0.95*max_Xp,50)
    reference_vals = prior_kde(reference_grid)

    # Manually prepend/append zeros at the boundaries
    reference_grid = np.concatenate([[0],reference_grid,[max_Xp]])
    reference_vals = np.concatenate([[0],reference_vals,[0]])
    norm_constant = np.trapz(reference_vals,reference_grid)

    # Interpolate!
    p_chi_p = np.interp(xp,reference_grid,reference_vals/norm_constant)
    return p_chi_p

def chi_p_from_components(a1,a2,cost1,cost2,q):

    """
    Helper function to define effective precessing spin parameter from component spins

    INPUTS
    a1: Primary dimensionless spin magnitude
    a2: Secondary's spin magnitude
    cost1: Cosine of the primary's spin-orbit tilt angle
    cost2: Cosine of the secondary's spin-orbit tilt
    q: Mass ratio

    RETRUNS
    chi_p: Corresponding precessing spin value
    """

    sint1 = np.sqrt(1.-cost1**2)
    sint2 = np.sqrt(1.-cost2**2)

    return np.maximum(a1*sint1,((3.+4.*q)/(4.+3.*q))*q*a2*sint2)



def convert_priors(pe_file='./GW_PE_samples.h5',
                   inj_file='./O3_injections.pkl',
                   dochip=False):

    if dochip:
        inj_savefile = inj_file[:-4] + '_chip_prior_calc.pkl'
        pe_savefile = pe_file[:-4] + '_chip_prior_calc.pkl'
    else:
        inj_savefile = inj_file[:-4] + '_prior_calc.pkl'
        pe_savefile = pe_file[:-4] + '_prior_calc.pkl'



    with open(inj_file, 'rb') as f:
        injs = pickle.load(f)

    print('calculating inj priors to chi_eff, chi_p ...')


    try:
        injs['chieff_prior'] = chi_effective_prior_from_isotropic_spins(injs['mass_ratio'], 1.0, injs['chi_eff'])

        if dochip:
            injs['chieff_chip_prior'] = joint_prior_from_isotropic_spins(injs['mass_ratio'], 1.0, injs['chi_eff'], injs['chi_p'])

    except:

        injs['chi_eff'] = (injs['mass_1']*injs['a_1']*injs['cos_tilt_1'] + injs['mass_2']*injs['a_2']*injs['cos_tilt_2'] ) / (injs['mass_1'] + injs['mass_2'])
        injs['chi_p'] = chi_p_from_components(injs['a_1'], injs['a_2'], injs['cos_tilt_1'], injs['cos_tilt_2'], injs['mass_ratio'])

        injs['chieff_prior'] = chi_effective_prior_from_isotropic_spins(injs['mass_ratio'], 1.0, injs['chi_eff'])

        if dochip:
            injs['chieff_chip_prior'] = joint_prior_from_isotropic_spins(injs['mass_ratio'], 1.0, injs['chi_eff'], injs['chi_p'])



    #injs['chieff_prior'] = chi_effective_prior_from_isotropic_spins(injs['mass_ratio'], 1.0, injs['chi_eff']) / ((2 * np.pi * injs["a_1"]** 2) * (2 * np.pi * injs["a_2"]** 2))
    #injs['chieff_chip_prior'] = joint_prior_from_isotropic_spins(injs['mass_ratio'], 1.0, injs['chi_eff'], injs['chi_p']) / ((2 * np.pi * injs["a_1"]** 2) * (2 * np.pi * injs["a_2"]** 2))

    with open(inj_savefile, 'wb') as file:
        pickle.dump(injs, file)



    with open(pe_file, 'rb') as f:
        post = pickle.load(f)


    print('calculating PE priors to chi_eff, chi_p ...')
    for event in post.keys():

            print('converting for event ' + event + ' ...')
            post[event]['chi_p'] = chi_p_from_components(post[event]['a_1'],
                                                         post[event]['a_2'],
                                                         post[event]['cos_tilt_1'],
                                                         post[event]['cos_tilt_2'],
                                                         post[event]['mass_ratio'])


            if dochip:
                post[event]['chieff_chip_prior'] = joint_prior_from_isotropic_spins(post[event]['mass_ratio'], 1.0,
                                                                           post[event]['chi_eff'],
                                                                           post[event]['chi_p'])


            post[event]['chieff_prior']  = chi_effective_prior_from_isotropic_spins(post[event]['mass_ratio'],
                                                                                    1.0, post[event]['chi_eff'])

    with open(pe_savefile, 'wb') as file:
        pickle.dump(post, file)

    #dd.io.save('/projects/p31963/sharan/pop/o1o2o3_pe_prior_calculated.h5', post)


    return

if __name__ == "__main__":

    if int(sys.argv[3]) ==0:
        convert_priors(sys.argv[1], sys.argv[2], dochip=False)
    elif int(sys.argv[3]) ==1:
        convert_priors(sys.argv[1], sys.argv[2], dochip=True)



