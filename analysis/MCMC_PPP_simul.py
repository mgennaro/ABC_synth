'''
Code to test the KDE based method to estimate the posterior
of (N,alpha,bf). It uses the artificial stars and the priors
on age,feh, Av, DM for hercules (optical data)
'''

import numpy as np
import pickle,bz2,sys
import pandas as pd
import extinction
import corner
import copy
import emcee
import time

from scipy.io import readsav
from scipy.stats import norm

from sim_utils.GeneralRandom import GeneralRandom
from sim_utils.getmags import get_absolute_mags,get_observable_mags,get_noisy_mags
from sim_utils.oneCMD import oneCMD

from analysis.PPP_loglik import loglik,GKDE_lik,set_GR,logpost,logpost_nb,TGKDE_lik,UKDE_lik

################################
#Initial setup

root = '/user/gennaro/ABC_synth/WORK/herc_simul/'

case = 'trg_sk'  # suffix for the output files

tru_N, tru_alpha, tru_bf = 2500, -2.0, 0.0

fix_bf, fix_bfval = True, tru_bf # If fix_bf true do not fit for binary fraction but keep it equal to  fix_bfval

ra_min,ra_max = -4, -0.5 # Range for the prior in alpha (i.e. range where the uniform prior on alpha extends)
rb_min,rb_max = 0., 1.  #  Range for the prior in bf (i.e. range where the uniform prior on bf extends)

#Limits on the catalogs                    
mag2lim = 28.25

#Width of the magnitude and color kernels

bwcol,bwmag = 0.01,0.025

#emcee parameters
nthreads = 10
nwalkers = 200
nsteps = 1000

################################

with bz2.BZ2File(root+'iso_int.pbz2', 'rb') as f:
    iso_int = pickle.load(f)

with bz2.BZ2File(root+'ASKDtree.pbz2', 'rb') as f:
    ASKDtree = pickle.load(f)

with bz2.BZ2File(root+'AScat.pbz2', 'rb') as f:
    AScat = pickle.load(f)

AS_mag1_in  = AScat['AS_mag1_in']
AS_mag2_in  = AScat['AS_mag2_in']
AS_mag1_out = AScat['AS_mag1_out']
AS_mag2_out = AScat['AS_mag2_out']
AS_det      = AScat['AS_det']

# Limit the detections to brigther than a certain magnitude
AS_det = AS_det & (AS_mag2_out < mag2lim)

#############################
# Fixed values of Distance modulus and exinction

DM = 20.64
Av = 0.279

ext1 = Av*(extinction.ccm89(np.array([5921.1]),1,3.1))[0]
ext2 = Av*(extinction.ccm89(np.array([8057.0]),1,3.1))[0]


# Setup the random numbers generators

isoPD = pickle.load(open( '/user/gennaro/UFDs_OPT/shelves/isoACS.pickle', "rb" ) )
age_vals = isoPD['age_vals']
feh_vals = isoPD['feh_vals']
del(isoPD)

limits_all = [(np.log(0.4),np.log(8)),
              (np.amin(age_vals),np.amax(age_vals)),
              (np.amin(feh_vals),np.amax(feh_vals)),
              (20.64-1,20.64+1),
              (-1,1.),
              (0,1.),
              (0,1.)]

#Age
sfh = pd.read_table('/user/gennaro/UFDs_OPT/herc/hercsfh.txt',header=None,sep='\s+',
                   names=['age','feh','weights'])
ages_sfh = np.unique(sfh.age.values)
marg_wgt = np.zeros_like(ages_sfh)
for i,aaa in enumerate(ages_sfh):
    marg_wgt[i] = np.sum(sfh[sfh.age == aaa].weights.values)
    
pal = np.linspace(limits_all[1][0],limits_all[1][1],250)
pav = np.zeros_like(pal)
for i,aaa in enumerate(ages_sfh):
    pav = pav + marg_wgt[i]*norm.pdf(pal,loc=aaa,scale=0.1)
age_GR = GeneralRandom(pal,pav,1000)
    
#Metallicity
dicMDF = readsav('/user/gennaro/UFDs_OPT/MDFS/Summary_grid0p2_herc_adp.sav')
pfl = feh_vals
pfv = dicMDF.mednmdf
feh_GR = GeneralRandom(pfl,pfv,1000)

#DM
DMT_GR = DM

#A_606 - A_814
ex1_GR = ext1

#A_814
ex2_GR = ext2

GRdic = {'logM':np.nan,
         'Age':age_GR,
         'FeH':feh_GR,
         'DM':DM,
         'BinQ':np.nan,
         'Alam1':ex1_GR,
         'Alam2':ex2_GR
        }

################################
## Setup the priors on alpha and binary fraction


hpal = np.array([ra_min,ra_max])
hpav = np.array([1.,1])
hpa_GR = GeneralRandom(hpal,hpav,500)

hpbl = np.array([rb_min,rb_max])
hpbv = np.array([1.,1])
hpb_GR = GeneralRandom(hpbl,hpbv,500)

#Define the dictionary of all parameters

dictopass ={'GRdic':GRdic,
            'iso_int':iso_int,
            'ASKDtree':ASKDtree,
            'AS_mag1_in':AS_mag1_in,
            'AS_mag2_in':AS_mag2_in,
            'AS_mag1_out':AS_mag1_out,
            'AS_mag2_out':AS_mag2_out,
            'AS_det':AS_det,
            'data':[],
            'ndata':[],
            'bwcol':bwcol,
            'bwmag':bwmag,
            'mag2lim':mag2lim,
            'hp_alpha': hpa_GR,
            'hp_bf':hpb_GR
            }

###################
#Simulate a reference CMD to be used as "Data"

nsimdata = tru_N
mss_GR_h, bin_GR_h = set_GR(tru_alpha,tru_bf)
kwargsh = copy.deepcopy(dictopass)

GRdic_h = kwargsh['GRdic']
GRdic_h['logM'] = mss_GR_h
GRdic_h['BinQ'] = bin_GR_h

t0  = time.time()

print('Generating fake catalog')
sys.stdout.flush()

pars, absmags, obsmags, simdata, nall, indGR = oneCMD(nsimdata,GRdic_h,
                                                       kwargsh['iso_int'],kwargsh['ASKDtree'],kwargsh['AS_mag1_in'],kwargsh['AS_mag2_in'],
                                                       kwargsh['AS_mag1_out'],kwargsh['AS_mag2_out'],kwargsh['AS_det'],verbose=True)

dic = {'pars':pars,
       'absmags':absmags,
       'obsmags':obsmags,
       'simdata':simdata,
       'nall':nall,
       'inputs':(tru_N, tru_alpha, tru_bf),
       'indGR':indGR
       }

with bz2.BZ2File(root+'/Results/simcat_'+case+'.pbz2', 'w') as f:
    pickle.dump(dic,f)

print('Time elapsed (sec)',time.time()-t0)
sys.stdout.flush()

dictopass['data'] = simdata
dictopass['ndata'] = nsimdata

###################
#Initialize and run emcee

pn = np.random.uniform(low=0.2*nsimdata,high=5*nsimdata,size=nwalkers)
pa = np.random.uniform(low=ra_min,high=ra_max,size=nwalkers)
pb = np.random.uniform(low=rb_min,high=rb_max,size=nwalkers)

if (fix_bf == True):
    ndim = 2
    p0 = np.array([pn,pa]).T
    lp = logpost_nb
    dictopass['fix_bfval'] = fix_bfval
    f = open(root+'/Results/pr_'+case+'.dat', "w")
    f.write('          N                         alpha                   logpost\n')
else:
    ndim = 3
    p0 = np.array([pn,pa,pb]).T
    lp=logpost
    f = open(root+'/Results/pr_'+case+'.dat', "w")
    f.write('          N                         alpha                    bf                    logpost\n')

f.close()
tt = time.time()    
sampler = emcee.EnsembleSampler(nwalkers, ndim, lp, kwargs=dictopass, threads=nthreads)
print('Initializing time',time.time()-tt)
sys.stdout.flush()

print('Starting emcee run')
sys.stdout.flush()
tt  = time.time()    
tt1 = tt

for i, result in enumerate(sampler.sample(p0, iterations=nsteps)):
    if ((i+1) % (nsteps/100) == 0):
        print(100.*(1.+i)/nsteps,' % done')
        print('Partial time',time.time()-tt)
        sys.stdout.flush()
        tt = time.time()
    f = open(root+'/Results/pr_'+case+'.dat', "ab")
    np.savetxt(f,np.column_stack((result[0],result[1])))
    f.close()
        
print('Total time (hours)',(time.time()-tt1)/3600.)
sys.stdout.flush()

print('Saving results on:'+root+'/Results/chain_'+case+'.pbz2')
sys.stdout.flush()

dic = {'a':sampler.a,
       'acceptance_fraction': sampler.acceptance_fraction,
       'chain':sampler.chain,
       'lnprobability' : sampler.lnprobability,
       'naccepted': sampler.naccepted,
       'bwcol':bwcol,
       'bwmag':bwmag,
       'mag2lim':mag2lim,
       'fix_bf':fix_bf,
       'fix_bfval':fix_bfval,
       'ra_min':ra_min,
       'ra_max':ra_max,
       'rb_min':rb_min,
       'rb_max':rb_max  
       }

with bz2.BZ2File(root+'/Results/chain_'+case+'.pbz2', 'w') as f:
    pickle.dump(dic,f)




