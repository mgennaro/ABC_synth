'''
Author: Mario Gennaro

Define a class that can be used to interpolate isocrones
in mass, age and metallicity.
Initialization requires a pandas dataframe of isocrones,
with multiindex ('[Fe/H]', 'age','mass') and
a photband object
'''

from Utils.findNN_arr import find_nearest_idx
import numpy as np
from scipy.interpolate import interp1d

class intNN:
    
    def __init__(self, isoPD, photband):

        self.isoages = np.unique(np.asarray([isoPD.index.get_level_values('age')]).T)
        self.isomets = np.unique(np.asarray([isoPD.index.get_level_values('[Fe/H]')]).T)
        self.iso_intp = [[0 for x in range(len(self.isomets))] for y in range(len(self.isoages))]
        self.iso_mrng = [[0 for x in range(len(self.isomets))] for y in range(len(self.isoages))]
        self.photband = photband

        for aa, age in enumerate(self.isoages):
            for zz, met in enumerate(self.isomets):
                isomass = np.asarray(isoPD.ix[met].ix[age].index.get_level_values('mass'))
                isomag  = np.asarray(isoPD.ix[met].ix[age][self.photband.name])
                self.iso_mrng[aa][zz] = ([np.amin(isomass),np.amax(isomass)])
                self.iso_intp[aa][zz] = interp1d(isomass, isomag, kind='linear', assume_sorted=True,
                                                 bounds_error=False, fill_value=np.nan)
    def __call__(self,mss,age,met):
        nage_idx = find_nearest_idx(self.isoages,age)
        nmet_idx = find_nearest_idx(self.isomets,met)
        isointp  = self.iso_intp[nage_idx][nmet_idx]

        return isointp(mss)


class intNN_logM:
    
    def __init__(self, isoPD, photband):

        self.isoages = np.unique(np.asarray([isoPD.index.get_level_values('age')]).T)
        self.isomets = np.unique(np.asarray([isoPD.index.get_level_values('[Fe/H]')]).T)
        self.iso_intp = [[0 for x in range(len(self.isomets))] for y in range(len(self.isoages))]
        self.iso_mrng = [[0 for x in range(len(self.isomets))] for y in range(len(self.isoages))]
        self.photband = photband

        for aa, age in enumerate(self.isoages):
            for zz, met in enumerate(self.isomets):
                isomass = np.asarray(np.log(isoPD.ix[met].ix[age].index.get_level_values('mass')))
                isomag  = np.asarray(isoPD.ix[met].ix[age][self.photband.name])
                self.iso_mrng[aa][zz] = ([np.amin(isomass),np.amax(isomass)])
                self.iso_intp[aa][zz] = interp1d(isomass, isomag, kind='linear', assume_sorted=True,
                                                 bounds_error=False, fill_value=np.nan)
    def __call__(self,mss,age,met):
        nage_idx = find_nearest_idx(self.isoages,age)
        nmet_idx = find_nearest_idx(self.isomets,met)
        isointp  = self.iso_intp[nage_idx][nmet_idx]

        return isointp(np.log(mss))

