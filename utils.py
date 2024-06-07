#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues July 25 09:58:18 2017

@author: bstressler
"""
import numpy as np
import utm
import pickle
import matplotlib.pylab as plt
    
def dataFromText(fname,dtype):
    
    lon,lat,data,sx,sy,sz=np.loadtxt(fname ,delimiter=' ', skiprows=1, usecols=(0,1,2,3,4,5), unpack=True)
    
    return lon,lat,data,sx,sy,sz

def writeKite2Savestruct(infile, covfile, outfile, utm_number=[]):
    ''' Function to write and save  outputs of downsampleInSAR.py into proper format for pyFaultResampler

        Inputs: infile--> .txt file output from downsampleInSAR.py
                covfile --> cov.npy data covariance matrix output from downsampleInSAR.py
                outfile --> name to save file for pyFaultResampler

    '''

    # Load in output text file from quadtree downsampling
    lon,lat,disp,sx,sy,sz =  dataFromText(infile, 'float')
    cov = np.load(covfile)

    X=np.zeros(len(lon))
    Y=np.zeros(len(lat))
    if np.size(utm_number) > 0:
        for i in range(0, len(X)):
            tmp = utm.from_latlon(lat[i],lon[i], force_zone_number=utm_number)
            X[i] = tmp[0]
            Y[i] = tmp[1]
    else:
        out = utm.from_latlon(lat[0], lon[0])
        utm_number=out[2]
        for i in range(0, len(X)):
            tmp = utm.from_latlon(lat[i],lon[i], force_zone_number=utm_number)
            X[i] = tmp[0]
            Y[i] = tmp[1]
    zone_number = tmp[2]
    zone_letter = tmp[3]
    zone = str(zone_number)+zone_letter

    print('UTM Zone: ', zone)

    S=np.row_stack([sx,sy,sz])
    
    #Prepare and save file for pyFaultResampler
    savestruct = {}
    data = {}
    data['X']    = X
    data['Y']    = Y
    data['S']    = S
    data['data'] = disp
    
    covstruct    = {}
    covstruct['cov'] = cov
    
    savestruct['data']      = data
    savestruct['numpts']    = len(X)
    savestruct['dataType']  = 'InSAR'
    savestruct['zone']      = zone
    savestruct['covstruct'] = covstruct
    
    file = open(outfile,'wb')
    pickle.dump(savestruct, file)

    return savestruct



    
