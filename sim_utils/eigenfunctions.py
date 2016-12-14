'''
Routine to run multiple instances of oneCMD.py in parallel
and produce a set of CMD "eigenfunctions"
'''

import numpy as np
import pandas as pd
import pickle,bz2,extinction,itertools,copy,sys
from sim_utils.oneCMD import oneCMD
from joblib import Parallel, delayed
from analysis.PPP_loglik import set_GR
from scipy.io import readsav
from scipy.stats import norm
from sim_utils.GeneralRandom import GeneralRandom

nstars = 1e6
bfs    = np.linspace(0.,1,21)   #fraction of systems that are binaries
alphas = np.arange(-0.500,-0.499,0.025)  # IMF slope
ncores = 7
root = '/user/gennaro/ABC_synth/WORK/herc_simul/'


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
AS_det = AS_det & (AS_mag1_out < 28.25)

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

###################
#Simulate a reference CMD to be used as "Data"

dict ={'GRdic':GRdic,
       'nsimdata':nstars,
       'iso_int':iso_int,
       'ASKDtree':ASKDtree,
       'AS_mag1_in':AS_mag1_in,
       'AS_mag2_in':AS_mag2_in,
       'AS_mag1_out':AS_mag1_out,
       'AS_mag2_out':AS_mag2_out,
       'AS_det':AS_det
       }


def make_1EF(alpha,bf,dict,root):

    print('Started (alpha,bf) = ',( ("%.3f" % alpha)+' ,  '+("%.3f" % bf)))
    sys.stdout.flush()

    mss_GR_h, bin_GR_h = set_GR(alpha,bf)
    dicth = copy.deepcopy(dict)
    GRdic_h = dicth['GRdic']
    GRdic_h['logM'] = mss_GR_h
    GRdic_h['BinQ'] = bin_GR_h

    pars, absmags, obsmags, simdata, nall, indGR = oneCMD(dicth['nsimdata'],GRdic_h,
                                                          dicth['iso_int'],dicth['ASKDtree'],dicth['AS_mag1_in'],dicth['AS_mag2_in'],
                                                          dicth['AS_mag1_out'],dicth['AS_mag2_out'],dicth['AS_det'])


    dts = {'simdata':simdata,
           'nall':nall}

    file2s = '/Eigenfunctions/EF_alpha'+("%.3f" % alpha)+'_bf'+("%.3f" % bf)+'.pbz2'

    with bz2.BZ2File(root+file2s, 'w') as f:
        pickle.dump(dts,f)

    print('Done (alpha,bf) = ',( ("%.3f" % alpha)+' ,  '+("%.3f" % bf)))
    sys.stdout.flush()



Parallel(n_jobs=ncores)(delayed(make_1EF)(alpha,bf,dict,root) for alpha,bf in itertools.product(alphas,bfs))
