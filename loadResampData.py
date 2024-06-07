#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:36:15 2017

@author: bstressler

######################### loadResampData.py ####################################################

    Loads in input data, fault model files, and defines parameters/variables for performing
    fault slip inversions with faultResampler.
    
    datafiles : files containing saved pickle object- 'savestruct' 
        savestruct: dictionary containing keys:
                   
                - data: dictionary containing keys:
                    - X: x position (UTM) of each data point
                    - Y: y position (UTM) of each data point
                    - S: 3 x numpts --> each column is the look vector for the corresponding data point
                    - data: displacement (m) at each data point in the corresponding look vector
                        
                - covstruct: dictionary contraining key
                    - cov: (numpts x numpts) covariance matrix
                    
                - dataType: string- 'GPS' or 'InSAR'
                    
                - numpts: number of data points
                
                - zone: UTM zone (string)
                
    To save/load files with pickle:
    
        # save
        pickle.dump(savestruct, open('filename', 'wb'))
        
        # load 
        savestruct = pickle.load(open('filename', 'rb'))
        
    *** Use/reference writeInputs.py for creating/formatting input files
        
    faultfile: files containing saved pickle object- 'faultstruct'
        faultstruct: dictionary containing keys corresponding to each fault geometry
            
            Each key contains:
                -L: Fault length (m) along strike
                -W: Fault width (m) down dip
                -vertices: 2 x 2 array containing x-y (UTM) coordinates of the top fault vertices
                           [x1, x2; y1, y2]
                -strike: Fault strike
                -dip: Fault dip
                -zt: depth to the top of the fault plane (m)
                
            -Note: key name doesn't matter
        
        To save/load files with pickle:
        
        # save
        pickle.dump(faultstruct, open('filename', 'wb'))
        
        # load
        faultstruct = pickle.load(open('filename', 'rb'))
        
"""

import numpy as np
import scipy as sp
import pickle
import matplotlib.pylab as plt

datafiles      = ['examples/illapel/illapel_gps.in','examples/illapel/p156_1.in','examples/illapel/p156_2.in']
faultfile      = ['examples/illapel/faultstruct_illapel']
saveFile       = 'out.fr' 
rake_type      = 'fixed' # fixed or free
smooth_method  = 'mm' # minimum moment (mm) or laplacian (laplacian) smoothing--> only use laplacian smoothing with uniform meshes
reg_method     = 'lcurve' # jRi or lcurve
ramp_type      = 'none' # Type of ramp to invert for: none, linear, bilinear, quadratic
rake           = 95
ss_constraints = 'l' # 'l' or 'r' for left or right lateral strike slip rake constraints
ds_constraints = 'r' # 'r' or 'n' for reverse or normal reverse slip rake constraints
disc           = 'uniform' # 'variable' or 'uniform' discretization
disc_param     = 50
plot_flag      = 1 #Plotting flag: 1= plots after each iteration, 0= plots only at end
check_inputs   = 'n'

resampstruct = {}
rampg        = []
allnp        = [0]*len(datafiles)
data_type    = [0]*len(datafiles)

#Load in data files

for i in range(0, len(datafiles)):
    
    savestruct                 = pickle.load(open(datafiles[i],'rb'))                
    covstruct                  = savestruct['covstruct']
    allnp[i]                   = savestruct['numpts']
    data_type[i]               = savestruct['dataType']
    
    if i == 0:
        covd       = covstruct['cov']
        datastruct = savestruct['data']
    else:
        tmpcov = covstruct['cov']
        covd   = sp.linalg.block_diag(covd, tmpcov)
        dstruct1 = datastruct
        dstruct2 = savestruct['data']
        datastruct={}
        for k in dstruct1.keys():
            key1 = dstruct1[k]
            key2 = dstruct2[k]
            if np.ndim(key1)==1:
                key1=key1.reshape([1, len(key1)])
            if np.ndim(key2)==1:
                key2=key2.reshape([1, len(key2)])
            datastruct[k] = np.column_stack([key1,key2])
    allnp[i]=savestruct['numpts']
    
X = np.squeeze(datastruct['X'])    
Y = np.squeeze(datastruct['Y'])
S = datastruct['S']
data = datastruct['data']

# Check if cov matrix is positive definite
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

temp = is_pos_def(covd)

# If covariance matrix not pos def, apply correction
if temp == False:
    val,vec   = np.linalg.eig(covd)
    idz = val == 0
    eps = np.spacing(val[idz])
    valNew = np.array(val)
    valNew[idz] = eps
    
    idn = val < 0
    val[idn] = -val[idn]
    
    tmp  = np.dot(vec,np.diag(val))
    covd = np.dot(tmp,vec.transpose())


# Write structure with proper dimensionality for calculating greens functions 
resampstruct['X']    = X.transpose() # Easting coordinates of data points
resampstruct['Y']    = Y # Northing coordinates of data points
resampstruct['S']    = S # Look vector for each data point (3 x n)
resampstruct['data'] = data # Displacement for each data point
ch                   = np.linalg.cholesky(covd) # For scaling data
Cdinv                = np.linalg.inv(ch.transpose())
Dnoise               = np.dot(Cdinv,data.squeeze())  # Weighted Data
sortt                = 0
EQt                  = 0

# Load faultstruct

faultstruct = pickle.load(open(faultfile[0],'rb'))

for i in faultstruct.keys():
    tmp      = faultstruct[i]
    strike   = tmp['strike']
    dip      = tmp['dip']
    L        = tmp['L']
    W        = tmp['W']
    zt       = tmp['zt']
    vertices = tmp['vertices']

#if dip == 90:
#    dip = 89.99
    
if strike < 0:
    strike = strike + 360
    
# Generate Ramp Parameters

if ramp_type:
    
    for j in range(0, len(datafiles)):
    
        id    = np.arange(0,allnp[j],1)+np.sum(allnp[0:j])
        id    = id.astype('int')
        Xtmp  = (X[id]-np.min(X))/1e4
        Ytmp  = (Y[id]-np.min(Y))/1e4
        XXtmp = Xtmp**2
        YYtmp = Ytmp**2
        XYtmp = Xtmp*Ytmp
        XXtmp = XXtmp/np.max(XXtmp)
        YYtmp = YYtmp/np.max(YYtmp)
        XYtmp = XYtmp/np.max(XYtmp)
        
        if ramp_type == 'linear':
            if j < 1:
                rampg = np.row_stack([Xtmp, Ytmp, np.ones([1,allnp[j]])])
            else:
                rampg = sp.linalg.block_diag(rampg, np.row_stack([Xtmp, Ytmp, np.ones([1,allnp[j]])]))
                
        elif ramp_type == 'bilinear':
            if j < 1:
                rampg = np.row_stack([Xtmp, Ytmp, np.ones([1,allnp[j]]), XYtmp, XXtmp, YYtmp])
            else:
                rampg =  sp.linalg.block_diag(rampg, np.row_stack([Xtmp, Ytmp, np.ones([1,allnp[j]]), XYtmp, XXtmp, YYtmp]))

        elif ramp_type == 'quadratic':
            if j < 1:
                rampg = np.row_stack([Xtmp, Ytmp, np.ones([1,allnp[j]]), Xtmp*Ytmp, Xtmp**2*Ytmp**2])
            else:
                rampg = sp.linalg.block_diag(rampg, np.row_stack([Xtmp, Ytmp, np.ones([1,allnp[j]]), Xtmp*Ytmp, Xtmp**2*Ytmp**2]))
        else:
            rampg = []
        
nramp  = np.shape(rampg)
nramp  = nramp[0]
numpts = np.sum(allnp) 

lambdas = 10**np.linspace(-2,2,15) # Regularization coefficients to test
tol     = 0.1 # 10% tolerance for dislocation area change     
covd2   = np.tile(np.diagflat(np.ones([1,numpts])),[2,2]) #Used for jRi method

# Define dictionary containing parameters needed to set up inversions 
invDict                   = {}
invDict['rake']           = rake
invDict['rake_type']      = rake_type
invDict['reg_method']     = reg_method
invDict['smooth_method']  = smooth_method
invDict['data_type']      = data_type
invDict['rampg']          = rampg
invDict['nramp']          = nramp
invDict['Cdinv']          = Cdinv
invDict['lambdas']        = lambdas
invDict['covd2']          = covd2  
invDict['ss_constraints'] = ss_constraints
invDict['ds_constraints'] = ds_constraints
invDict['faultstruct']    = faultstruct
invDict['data']           = data
invDict['allnp']          = allnp
invDict['resampstruct']   = resampstruct
invDict['Dnoise']         = Dnoise
invDict['plot_flag']      = plot_flag
    
def qualityCheckInputs():
    
    plt.scatter(X,Y,c=data,s=2, cmap='jet')
    plt.colorbar()
    
    for i in faultstruct.keys():
        Faultstruct = faultstruct[i]
        vertices = Faultstruct['vertices']
        plt.plot(vertices[0,:],vertices[1,:],'-r')
    plt.axis('image')

    plt.show()

if check_inputs == 'y':
    qualityCheckInputs()



