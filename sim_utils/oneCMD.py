'''
Simulate an observed CMD, given the total number of stars and
a dictionary of priors on (logM, Age, FeH, DM, BinQ, Alam1, Alam2)
'''

import numpy as np
from sim_utils.GeneralRandom import GeneralRandom
from sim_utils.getmags import get_absolute_mags,get_observable_mags,get_noisy_mags
import time

def oneCMD(nstars,GRdic,iso_int,ASKDtree,AS_mag1_in,AS_mag2_in,AS_mag1_out,AS_mag2_out,AS_det,verbose=False):

    #Determine which parameters are fixed

    GR_yes   = []
    indGR    = []
    defaults = np.zeros(7,dtype=np.float_)
    
    if (isinstance(GRdic['logM'],GeneralRandom)):
        GR_yes.append(GRdic['logM'])
        indGR.append(0)
    else:
        defaults[0] = GRdic['logM']
    
    if (isinstance(GRdic['Age'],GeneralRandom)):
        GR_yes.append(GRdic['Age'])
        indGR.append(1) 
    else:
        defaults[1] = GRdic['Age']

    if (isinstance(GRdic['FeH'],GeneralRandom)):
        GR_yes.append(GRdic['FeH'])
        indGR.append(2) 
    else:
        defaults[2] = GRdic['FeH']

    if (isinstance(GRdic['DM'],GeneralRandom)):
        GR_yes.append(GRdic['DM'])
        indGR.append(3)
    else:
        defaults[3] = GRdic['DM']

    if (isinstance(GRdic['BinQ'],GeneralRandom)):
        GR_yes.append(GRdic['BinQ'])
        indGR.append(4)
    else:
        defaults[4] = GRdic['BinQ']

    if (isinstance(GRdic['Alam1'],GeneralRandom)):
        GR_yes.append(GRdic['Alam1'])
        indGR.append(5)
    else:
        defaults[5] = GRdic['Alam1']

    if (isinstance(GRdic['Alam2'],GeneralRandom)):
        GR_yes.append(GRdic['Alam2'])
        indGR.append(6)
    else:
        defaults[6] = GRdic['Alam2']


    #Start the simulation

    ndone = 0
    ntrue = 0
    pars  = np.zeros([nstars,7])
    absmags = np.zeros([nstars,2])
    obsmags = np.zeros([nstars,2])
    noisymags = np.zeros([nstars,2])

    if (verbose == True):
        tstart = time.process_time()
    
    while ndone < nstars:

        ntrue = ntrue + 1
        #Extract th eparameters that are not defaulted
        pars1 = defaults
        for GR,ind in zip(GR_yes,indGR):
            pars1[ind] = GR.random()

        #Get the magnitudes
        absmags1 = get_absolute_mags(pars1[0],pars1[1],pars1[2],pars1[4],iso_int)

        if (np.any(np.isnan(absmags1)) == True):
            continue

        obsmags1 = get_observable_mags(absmags1, DM=pars1[3], ext=pars1[5:])
        noisymags1 = get_noisy_mags(obsmags1,ASKDtree,AS_mag1_in,AS_mag2_in,AS_mag1_out,AS_mag2_out,AS_det)


        if (np.any(np.isnan(noisymags1)) == True):
            continue

        pars[ndone,:] = pars1
        absmags[ndone,:] = absmags1
        obsmags[ndone,:] = obsmags1
        noisymags[ndone,:] = noisymags1

        if (verbose == True):
            if ( ndone%(nstars//10) == 0 ):
                print('Done ',(100.*ndone)//nstars,' % of stars')
            
        ndone = ndone + 1


    if (verbose == True):
        tend = time.process_time()
        print('Elapsed_time (min)', (tend - tstart)/60.)
        print('Elapsed_time  per star (millisec)', 1000.*(tend - tstart)/ndone)


    return  pars, absmags, obsmags, noisymags, ntrue, indGR
 
