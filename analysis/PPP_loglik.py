'''
Modules for computing the likelihood of the observed CMD
as function of a given set of parametrs
'''

import numpy as np
from sim_utils.GeneralRandom import GeneralRandom
from sim_utils.getmags import get_absolute_mags,get_observable_mags,get_noisy_mags
from sim_utils.oneCMD import oneCMD
import copy,time,sys
from sklearn.neighbors import KDTree


def set_GR(alpha,bf):

    pml = np.linspace(np.log(0.4),np.log(8),500)
    pmv = np.exp(pml*(alpha+1))
    mss_GR = GeneralRandom(pml,pmv,1000)

    pbl = np.array([-1,-1e-6,0.,1])
    pbv = np.array([1-bf,1-bf,bf,bf])
    bin_GR = GeneralRandom(pbl,pbv,1000)

    return mss_GR, bin_GR


def GKDE_lik(noisymags,data,bwcol,bwmag):

    colmod = noisymags[:,0]-noisymags[:,1]
    magmod = noisymags[:,1]
    coldat = data[:,0]-data[:,1]
    magdat = data[:,1]

    ndata  = magdat.size
    nmodel = magmod.size

    exponents = np.zeros([ndata,nmodel])
    
    for i in range(ndata):
        dx2 = ((colmod-coldat[i])/bwcol)**2
        dy2 = ((magmod-magdat[i])/bwmag)**2
        exponents[i,:] =  -0.5*(dx2+dy2)

    exponentials = np.exp(exponents)
    indlnlik     = np.log(np.sum(exponentials,axis=1))
    alllnlik     = np.sum(indlnlik)

    return -ndata*(np.log(nmodel*2*np.pi) + 2*np.log(bwcol*bwmag)) +alllnlik

def loglik(nsim,alpha,bf,**kwargs):

    '''
    First set the new mass and binary priors
    '''

    mss_GR_h, bin_GR_h = set_GR(alpha,bf)

    GRdic_h = copy.deepcopy(kwargs['GRdic'])
    GRdic_h['logM'] = mss_GR_h
    GRdic_h['BinQ'] = bin_GR_h

    '''
    Obtain a CMD for the given parameters
    '''
    params, absmags, obsmags, noisymags, ntrue, indGR = oneCMD(nsim,GRdic_h,
                                                               kwargs['iso_int'],kwargs['ASKDtree'],kwargs['AS_mag1_in'],kwargs['AS_mag2_in'],
                                                               kwargs['AS_mag1_out'],kwargs['AS_mag2_out'],kwargs['AS_det'])

    '''
    Compute the likelihood using a gaussian kde
    '''

    lnlik = GKDE_lik(noisymags,kwargs['data'],kwargs['bwcol'],kwargs['bwmag'])
    lnlik = -nsim+lnlik+kwargs['ndata']*np.log(nsim)

    return lnlik

def logpost(pars,**kwargs):

    lnpri = np.log(kwargs['hp_alpha'].getpdf(pars[1])) + np.log(kwargs['hp_bf'].getpdf(pars[2]))
    if(pars[0] <= 0):
        lnpri = -np.inf
    
    if (np.isfinite(lnpri) == True):    
        lnlik = loglik(np.floor(pars[0]).astype(np.int),pars[1],pars[2],**kwargs)
    else:
        lnlik = -np.inf

    return lnlik+lnpri


def logpost_nb(pars,**kwargs):

    lnpri = np.log(kwargs['hp_alpha'].getpdf(pars[1]))
    if(pars[0] <= 0):
        lnpri = -np.inf

    if (np.isfinite(lnpri) == True):    
        lnlik = loglik(np.floor(pars[0]).astype(np.int),pars[1],kwargs['fix_bfval'],**kwargs)
    else:
        lnlik = -np.inf

    return lnlik+lnpri