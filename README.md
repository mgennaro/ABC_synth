This project deals with the analysis of CMD using
synthetic stellar populations.

The prep_utils folder contains all the code that is necessary to setup a new
case in which the analysis is needed:
formatting catalogs, models and artificial stars according to a standard interface.

The sim_utils folder contains the code to actually make a CMD realization
according to given parameters

The analysis folder contains the code to fit the data using the simulated CMDs

Each subfolder in the WORK folder contains the actual catalog.pbz2 (the data), iso_int.pbz2 (the isocrhone scipy interpolators) and AScat.pbz2 (the AS catalog) and ASKDtree.pbz2 (input magnitudes KD tree) for an individual case.

*******************
catalog.pbz2

A dictionary that must contain 3 keys:
dat_mag1 : the magnitude of the data in band 1 (numpy ndarray of floats)
dat_mag2 : the magnitude of the data in band 2 (numpy ndarray of floats)
dat_det : a flag to mark whether the star has to be used or not (if the user wants to use all their data, 
	  they should just set this column to all '1's) (numpy ndarray of np.bool_)

*******************
iso_int.pbz2

A list of 2 intNN objects (one per photometric band used)
intNN objects return a magnitude as function of (mass, age, metallicity)

*******************
ASKDtree.pbz2

A sklearn.neighbors.kd_tree.BinaryTree object
(the KD Tree of the input AS magnitudes)

*******************
AScat.pbz2

A dictionary with the following columns:

AS_mag1_in : the input magnitude for band 1 (numpy ndarray of floats)
AS_mag2_in : the input magnitude for band 2 (numpy ndarray of floats)
AS_mag1_out: the output magnitude for band 1 (numpy ndarray of floats)
AS_mag2_out: the output magnitude for band 2 (numpy ndarray of floats)
AS_det     : a flag to mark whether the AS has to be considered detected (True) or not (False) (numpy ndarray of np.bool_)