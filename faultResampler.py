#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:41:31 2017
@author: bstressler

########################### faultResampler.py #################################

This program generates fault models for inversions of slip from GPS and InSAR 
observed surface displacements. It then conducts a final inversion of slip using
the optimal fault model. This code is based on the faultResampler MATLAB code
(available at http://myweb.uiowa.edu/wbarnhart/programs.html) from Barnhart &
Lohman (2010), using a different method to downsample a coarse starting fault
discretization to generate an optimally discretized fault model rather than the 
complete model remeshing used by the original MATLAB version.

Citation:  Barnhart, W.D., R.B. Lohman (2010) Automated fault model
discretization for inversions for coseismic slip distributions. Journ. 
Geophys. Res. V.115, B10419

###############################################################################

"""

# Import requisite python libraries/functions
import numpy as np
from fault_funcs import *
from inversion_funcs import *
from plot_utils import *
import pickle

# Load in data, fault files, and prepare necessary variables for inversions
from loadResampData import *

# Generate starting fault model discretization
if disc == 'uniform':
    '''Perform simple inversion by using a uniform fault model discretization. Adjust
       second input parameter to makeStartFault_Uniform to control patch sizes'''
    patchstruct, triId, triCoords, plotCoords, nt = makeStartFault_Uniform(faultstruct, disc_param)
    nPatch = int(np.size(patchstruct['xfault'])/3)
    print('\n \nNumber of fault patches = ', nPatch, '\n')
    
    if invDict['rake_type'] == 'free':
        gsmooth, G, Gg, slip, synth, mil, rake = inversionFreeRake(patchstruct, plotCoords, triId, invDict)
    
    else:
        gsmooth, G, Gg, mil, synth = inversionFixedRake(patchstruct, plotCoords, triId, invDict)
        slip = mil[0:nPatch]
    
    Newscales, scales, resamp = calcSmoothScales(G, Gg, rake_type, patchstruct, nt, nPatch)
    plot_slip_res(plotCoords, triId, slip, rake, scales, Newscales)     

else:
    #Generate coarse starting fault model discretization to be downsampled
    patchstruct, triId, triCoords, plotCoords, nt = makeStartFault_Uniform(faultstruct, disc_param)
    nPatch = int(np.size(patchstruct['xfault'])/3)
    print('\n \nStarting number of fault patches = ', nPatch, '\n \n')
 
    # Iterate through fault re-meshing until model is properly downsampled 
    patchstruct, gsmooth, G, Gg, slip, synth, mil, rake, plotCoords, triCoords, triId = do_invert_Remesh(patchstruct, nt, plotCoords, triCoords, triId, invDict)
    nPatch = int(np.size(patchstruct['xfault'])/3)

# Ask user if they desire to save inversion results
dump = input('To save inversion results, enter ''yes'' ... to quit hit enter : ')

saveDict = {}
saveDict['resampstruct'] = resampstruct
saveDict['data']         = data
saveDict['patchstruct']  = patchstruct
saveDict['invDict']      = invDict
saveDict['synth']        = synth
saveDict['slip']         = slip
saveDict['mil']          = mil
saveDict['triCoords']    = triCoords
saveDict['plotCoords']    = plotCoords
saveDict['triId']        = triId
saveDict['faultstruct']  = faultstruct
saveDict['rake']         = rake

if dump == 'yes':
    pickle.dump(saveDict, open(saveFile, 'wb'))



