#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 15:46:18 2017

@author: bstressler
"""
import numpy as np
import pyproj
import utm
import pickle
import matplotlib.pylab as plt
from plot_utils import plot_data_resid


def importResults(fname):
    ''' Load in file containing inversion results and plot slip model and residuals'''
    
    #Load pickles file containing inversion results
    out = pickle.load(open(fname,'rb'))
    
    #Pull out values from save dictionary
    resampstruct = out['resampstruct'] 
    data         = out['data']         
    patchstruct  = out['patchstruct']  
    invDict      = out['invDict']       
    synth        = out['synth']        
    slip         = out['slip']         
    mil          = out['mil']
    triCoords    = out['triCoords']    
    triId        = out['triId']        
    faultstruct  = out['faultstruct']  
    rake         = out['rake']         
    
    #plot slip model and data residuals
    fig = plt.figure()
    plt.tripcolor(triCoords[:,0], triCoords[:,1], triangles = triId, facecolors=slip, shading='flat')
    plt.axis('image')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title('Inverted Slip')
    
    plot_data_resid(resampstruct['X'], resampstruct['Y'], resampstruct['S'], data, synth, invDict['allnp'], invDict['data_type'])
    
    return out

def makeFaultFile(strike, dip, Mw, x, y, z, utm_ot_ll, name, utm_zone_number=None):
    ''' Given a fault geometry and a hypocentral location, create a fault geometry
        file for use in pyFaultResampler using earthquake scaling relationships to
        define fault dimensions 

        Inputs: strike, dip, Mw: self explanatory
                x,y: either UTM or lon/lat coordinates of hypocenter
                z: depth (m)
                utm_or_ll : 'utm' or 'll'--> for converting to UTM
                name: name for saving fault file
                utm_zone_number: optional, add to fix UTM zone number

         Output: faultstruct: fault geometry dictionary based on moment tensor/focal mechanism solution
                               --> saved with pickle

    '''
    
    '''if utm_or_ll != 'utm':
        X,Y = my_utm2ll(x, y, utm_or_ll='ll', utm_zone='', hem = 'north')'''
    
    if utm_ot_ll != 'utm':
        tmp = utm.from_latlon(y, x, force_zone_number=utm_zone_number)
        X = tmp[0]
        Y = tmp[1]
        zone_number = tmp[2]
        zone_letter = tmp[3]
        utm_zone = str(zone_number)+zone_letter

    else:
        X = x
        Y = y

    L,W = makeFaultDims(Mw, mechanism='')
    L = 2*L
    W = 2*W
    xc = X
    yc = Y
    
    zt = z-((W/2)*np.sin(dip*np.pi/180))
    xt = xc-W/2*np.cos(dip*np.pi/180)*np.cos(strike*np.pi/180)
    yt = yc+W/2*np.cos(dip*np.pi/180)*np.sin(strike*np.pi/180)

    if zt < 0:
        neg = zt
        zt = 0
        xt = xc-W/2*np.cos(dip*np.pi/180)*np.cos(strike*np.pi/180)
        yt = yc+W/2*np.cos(dip*np.pi/180)*np.sin(strike*np.pi/180)
 
        

    vertices = np.array([[xt-np.sin(strike*np.pi/180)*L/2, xt+np.sin(strike*np.pi/180)*L/2], 
                          [yt-np.cos(strike*np.pi/180)*L/2, yt+np.cos(strike*np.pi/180)*L/2]])      

    
    f1={}
    f1['strike']=strike
    f1['dip']=dip
    f1['vertices']=vertices
    f1['zt'] = zt
    f1['L']=L
    f1['W']=W
    faultstruct = {}
    faultstruct['f1']=f1
    
    pickle.dump(faultstruct, open(name,'wb'))

    return faultstruct
    
    
def makeFaultDims(Mw, mechanism):
    ''' Use Wells & Coppersmith to define fault dimensions ''' 
   
    if mechanism == 'reverse':
        aW=-1.61
        saW=0.2
        bW=0.41
        sbW=0.03
        aL=-2.42
        saL=0.21
        bL=0.58
        sbL=0.03
    elif mechanism == 'normal':
        aW=-1.14
        saW=0.28
        bW=0.35
        sbW=0.05
        aL=-1.88
        saL=0.37
        bL=0.5
        sbL=0.06
    elif mechanism == 'ss':
        aW=-0.76
        saW=0.12
        bW=0.27
        sbW=0.02
        aL=-2.57
        saL=0.12
        bL=0.62
        sbL=0.02
    else:
        aW=-1.01
        saW=0.1
        bW=0.32
        sbW=0.02
        aL=-2.44
        saL=0.11
        bL=0.59
        sbL=0.02
        
    W=10**(aW+bW*Mw);
    L=10**(aL+bL*Mw);
    W=W*1000;
    L=L*1000;

    return L, W

def my_utm2ll(x, y, utm_or_ll='ll', utm_zone='', hem = 'north'):
    '''Inputs:
        
            x,y:       XY coordinates in either lon/lat or utm
            utm_or_ll: Specify starting coordinate system with 'll' or 'utm'
            utm_zone:  UTM zone
            hem:       hemisphere: 'north' or 'south'
            
        Note: To ensure that pyproj converts coordinates properly, append line <2392> in the
              epsg file in lib/pyproj/data/ to:
      
      <2392> +proj=tmerc +lat_0=0 +lon_0=24 +k=1.000000 +x_0=2500000 +y_0=0 +ellps=intl 
      +towgs84=-90.7,-106.1,-119.2,4.09,0.218,-1.05,1.37 +units=m +no_defs no_defs <>
      
      ***** Not currently being implemented, if utm module causes problems try to use this 
            function for conversions*****
      
    '''
        
    myProj = pyproj.Proj('+proj=utm +zone=' + utm_zone + ' +'+ hem + ' +ellps=WGS84 +datum=WGS84 +units=m +no_defs')

    if utm_or_ll == 'll':
        X,Y    = myProj(x, y)    
    else:
        X,Y    = myProj(x, y, inverse=True)
    
    return X,Y

def writeInputInSAR(x, y, utm_or_ll, sx, sy, sz, data, covd, fname, utm_zone = None):
 
    '''if np.mean(lat)<0:
        hem = '+south'
    else:
        hem = '+north'
    
    if utm_or_ll == 'll':
        myProj = pyproj.Proj('+proj=utm +zone=' + utm_zone + ' '+ hem + ' +ellps=WGS84 +datum=WGS84 +units=m +no_defs')
        X,Y    = myProj(lon, lat)    
    else:
        X = x
        Y = y'''
    
    if utm_or_ll == 'll':
        X = np.zeros(len(x))
        Y = np.zeros(len(y))
        for i in range(0, len(X)):
            tmp = utm.from_latlon(x[i], y[i], force_zone_number=utm_zone)
            X[i] = tmp[0]
            Y[i] = tmp[1]
        zone_number = tmp[2]
        zone_letter = tmp[3]
        utm_zone = str(zone_number)+zone_letter
    else:
        X = x
        Y = y

    S      = np.zeros([3,len(sx)])
    S[0,:] = sx
    S[1,:] = sy
    S[2,:] = sz
    
    Data        = {}
    Data['X']   = X
    Data['Y']   = Y
    Data['S']   = S
    Data['data']= data
    
    out = np.shape(covd)
    if len(out)==1:
        covd = np.diag(covd)
        
    covstruct = {}
    covstruct['cov']= covd
    
    savestruct              = {}
    savestruct['data']      = Data
    savestruct['covstruct'] = covstruct
    savestruct['zone']      = utm_zone
    savestruct['numpts']    = len(data)
    savestruct['dataType']  = 'InSAR'
    pickle.dump(savestruct, open(fname,'wb'))

def writeInputGPS(x, y, utm_or_ll, dE, dN, dU, eE, eN, eU, fname, nComponents = 3, utm_zone = None):
    
    '''if np.mean(lat)<0:
        hem = '+south'
    else:
        hem = '+north'
    
    if utm_or_ll == 'll':
        myProj = pyproj.Proj('+proj=utm +zone=' + utm_zone + ' '+ hem + ' +ellps=WGS84 +datum=WGS84 +units=m +no_defs')
        tmpX,tmpY    = myProj(lon, lat, inverse=False)    
    else:
        tmpX = x
        tmpY = y'''
    
    if utm_or_ll == 'll':
        X = np.zeros(len(x))
        Y = np.zeros(len(y))
        for i in range(0, len(X)):
            tmp = utm.from_latlon(x[i], y[i], force_zone_number=utm_zone)
            X[i] = tmp[0]
            Y[i] = tmp[1]
        zone_number = tmp[2]
        zone_letter = tmp[3]
        utm_zone = str(zone_number)+zone_letter
    else:
        X = x
        Y = y
    
    eE = np.array(eE)**2
    eN = np.array(eN)**2
    eU = np.array(eU)**2
    
    if nComponents == 2:
        
        S = np.zeros([3,len(dE)*2])
        S[0,np.arange(0, len(dE)*2,2)] = 1
        S[1,np.arange(1, len(dE)*2,2)] = 1
        
        X = np.zeros(len(dE)*2)
        Y = np.zeros(len(dE)*2)
        data = np.zeros(len(dE)*2)
        covd = np.zeros(len(dE)*2)

        for i in range(1, len(dE)+1):
            X[2*i-2]  = tmpX[i-1]
            X[2*i-1]  = tmpX[i-1]
            Y[2*i-2]  = tmpY[i-1]
            Y[2*i-1]  = tmpY[i-1]
            data[2*i-2] = dE[i-1]
            data[2*i-1] = dN[i-1]
            covd[2*i-2] = eE[i-1]
            covd[2*i-1] = eN[i-1]
    else:
        
        S = np.zeros([3,len(dE)*3])
        S[0,np.arange(0, len(dE)*3, 3)] = 1
        S[1,np.arange(1, len(dE)*3, 3)] = 1
        S[2,np.arange(2, len(dE)*3, 3)] = 1
        
        X   = np.zeros(len(dE)*3)
        Y   = np.zeros(len(dE)*3)
        data = np.zeros(len(dE)*3)
        covd = np.zeros(len(dE)*3)

        for i in range(1, len(dE)+1):
            X[3*i-3]  = tmpX[i-1]
            X[3*i-2]  = tmpX[i-1]            
            X[3*i-1]  = tmpX[i-1]
            Y[3*i-3]  = tmpY[i-1]
            Y[3*i-2]  = tmpY[i-1]
            Y[3*i-1]  = tmpY[i-1]
            data[3*i-3] = dE[i-1]
            data[3*i-2] = dN[i-1]
            data[3*i-1] = dU[i-1]
            covd[3*i-3] = eE[i-1]
            covd[3*i-2] = eN[i-1]
            covd[3*i-1] = eU[i-1]
            
    Data        = {}
    Data['X']   = X
    Data['Y']   = Y
    Data['S']   = S
    Data['data']= data
    
    covstruct = {}
    covd = np.diag(covd)
    covstruct['cov']= covd
    
    savestruct              = {}
    savestruct['data']      = Data
    savestruct['covstruct'] = covstruct
    savestruct['zone']      = utm_zone
    savestruct['numpts']    = len(data)
    savestruct['dataType']  = 'GPS'
 
    pickle.dump(savestruct,open(fname,'wb'))
    
def dataFromText(fname,dtype):
    
    lon,lat,data,sx,sy,sz=np.loadtxt(fname ,delimiter=' ', skiprows=1, usecols=(0,1,2,3,4,5), unpack=True)
    
    return lon,lat,data,sx,sy,sz

def writeKite2Savestruct(infile, covfile, outfile, utm_number=[]):

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
        for i in range(0, len(X)):
            tmp = utm.from_latlon(lat[i],lon[i])
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

def writeSlip2XY(results_file, filename_xy, utm_number, northern):
    ''' Write slip distribution to xy file for GMT plotting. Loads in results file from
        pyFaultResampler and writes a .xy file. This function requires a utm zone number
        and declaration of northern as True or False --> northern hemisphere '''

    out = pickle.load(open(results_file,'rb'))
    patchstruct = out['patchstruct']
    slip = out['slip']

    xf = patchstruct['xfault']
    yf = patchstruct['yfault']
    ntris = len(slip)
    
    file = open(filename_xy,'w')

    for i in range(0, ntris):
        file.writelines('>-Z%f \n' % (slip[i]))
        xtemp = xf[i,:]
        ytemp = yf[i,:]
        lon = np.zeros(3)
        lat = np.zeros(3)

        for j in range(0,3):
            tmp = utm.to_latlon(ytemp[j], xtemp[j], utm_number, strict=False, northern=northern)
            lon[j] = tmp[0]
            lat[j] = tmp[1]
            file.writelines('%f %f \n' % (lon[j], lat[j]))

    file.close()
        
        

        

    
