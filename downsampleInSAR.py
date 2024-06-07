# -*- coding: utf-8 -*-
"""

Script for downsampling InSAR observations using the Kite module
(https://pyrocko.github.io/kite/index.html). This module uses a quadtree 
downsampling method.Kite is written in python2-> this will not work in python3.

Citation: Isken, Marius; Sudhaus, Henriette; Heimann, Sebastian; Steinberg, 
Andreas; Daout, Simon; Vasyura-Bathke, Hannes (2017): Kite - Software for 
Rapid Earthquake Source Optimisation from InSAR Surface Displacement. V. 0.1.
GFZ Data Services. http://doi.org/10.5880/GFZ.2.1.2017.002

This script downsamples an interferogram and prepares and saves an input data 
file as a pickled object for use in pyFaultResampler. 
    
To use:
    1) Copy this script to a directory containing:
        - filt_topophase.unw.geo
        - filt_topophase.unw.geo.xml
        - los.rdr.geo
    
    2) Change any parameters as necessary:
        -e.g., change outputFile to whatever file name you want the downsampled
              data saved as

    3) Execute: run downsampleInSAR.py (spyder or other IDE), or "python2
       downsampleInSAR.py" from command line
            -Must be run in a python2 shell!
       
    4) If the downsampled data does not adequately represent the deformation 
       present in the interferogram, tweak the variables "thresh", "maxSize",
       and "minSize" as necessary and rerun
       
           - thresh: Parameter used to scale covariance threshold used to perform
                     quadtree decomposition. Increase thresh for greater sensitivity
                     to noise (more data points). Descrease thresh for lesser 
                     sensitivity to noise (fewer data points)
           - maxSize/minSize: Maximum and minimum leaf sizes allowed for quadtree
                              decomposition
 
       
Bryan Stressler 7/25/2017
   
"""

import kite
from kite import Scene
import numpy as np
import matplotlib.pylab as plt
import pickle
import utm

file_name = 'filt_topophase.unw.geo'
xml_file_name = 'filt_topophase.unw.geo.xml'
outputFile = 'example.txt'
lambd = 0.0550 # C band wavelength--> Sentinel
thresh = 30 # Used for scaling the covariance threshold used in quadtree decomposition
maxSize = 25e3 # Max leaf size for quadtree decomp. Default  = 25e3 (m)
minSize = 250 # Min leaf size for quadtree decomp. Default  = 250 (m)

def main():

    # Pull image dimensions from metadata files
    par = kite.scene_io.ISCEXMLParser(xml_file_name)
    c1 = par.getProperty('coordinate1')
    c2 = par.getProperty('coordinate2')
    width = int(par.getProperty('width'))
    length = int(par.getProperty('length'))
    
    # Read in unw.geo file along with LOS file
    sc = Scene.import_data(file_name)
    
    # Convert unwrapped phase from radians to meters
    disp = -sc.displacement
    disp = disp*lambd/(4*np.pi)
    disp = np.flipud(disp)
    sc.displacement=disp
    delDisp=np.float(np.nanmax(disp)-np.nanmin(disp))
    
    # Fix Line of Sight vectors
    vE = np.flipud(sc.los.unitE)
    vN = np.flipud(sc.los.unitN)
    vU = np.flipud(sc.los.unitU)
    
    vE=vE.flatten()
    id = vE == 1
    idn = vE != 1
    vE[id] = np.mean(vE[idn])
    vE = vE.reshape(length, width)
    del id, idn
    
    vN=vN.flatten()
    id = vN == 0
    idn = vN != 0
    vN[id] = np.mean(vN[idn])
    vN = vN.reshape(length, width)
    del id, idn
    
    vU=vU.flatten()
    id = np.ceil(vU) == 0
    idn = np.ceil(vU) != 0
    vU[id] = np.mean(vU[idn])
    vU = vU.reshape(length, width)
    
    sc.los.unitE = vE
    sc.los.unitN = vN
    sc.los.unitU = vU
    
    # Configure quadtree parameters
    conf = kite.quadtree.QuadtreeConfig
    conf.epsilon = delDisp/thresh # Threshold for quadtree downsampling based on node variance
    conf.nan_allowed = .9 # Max fraction of nans with quadtree leaf: default = 0.9
    conf.tile_size_max = maxSize # max tile size (m): default = 25000
    conf.tile_size_min = minSize # min tile size (m): default = 250
    conf.correction = 'median'
    conf.leaf_blacklist=[]
    
    # Initiate quadtree downsampling
    quad = kite.Quadtree(sc, config=conf)
    print '\nNumber of downsampled data points: ', quad.nleaves, '\n'
    
    # Pull out downsampled coordinates and data
    coords = quad.leaf_coordinates
    X = coords[:,0]
    Y = coords[:,1]
    X = X + quad.frame.llEutm
    Y = Y + quad.frame.llNutm
    zone = str(quad.frame.utm_zone)+quad.frame.utm_zone_letter
    means = quad.leaf_means
    medians = quad.leaf_medians
    
    lon=np.zeros(len(X))
    lat=np.zeros(len(X))
    for i in range(0, len(X)):
        out = utm.to_latlon(X[i], Y[i], zone_number = quad.frame.utm_zone, zone_letter=quad.frame.utm_zone_letter)
        lat[i] = out[0]
        lon[i] = out[1]
    
    #Plot raw interferogram and downsampled data
    fig=plt.figure()
    ax=fig.add_subplot(1,2,1)
    plt.imshow(disp)
    plt.axis('image')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title('Unwrapped Interferogram')
    ax=fig.add_subplot(1,2,2)
    plt.scatter(X,Y,c=medians,edgecolors='face',s=10)
    plt.axis('image')
    plt.colorbar()
    plt.title('Downsampled Interferogram')
    plt.pause(0.1)
    plt.show(block=False)

    # As user if they are satisfied with quadtree parameterization
    out = raw_input('Continue with covariance calculation? [y/n] ... ')
    if out == 'n':
        print('\n########## Adjust parameters as necessary and re-run ########## \n')
        raise SystemExit

    # Pull out look vectors
    phi = quad.leaf_phis #Horizontal angle towards satellite’ line of sight in radians--> 0 = east, pi/2 = north
    theta = quad.leaf_thetas #Theta is look vector elevation angle towards satellite from horizon in radians. -pi/2 = down
    theta = theta - np.pi/2
    
    # Set fixed values to means
    id = phi == 0
    idn = phi != 0 
    phi[id] = np.mean(phi[idn])
    
    id = theta == 0
    idn =  theta != 0
    theta[id] = np.mean(theta[idn])
    
    look = phi
    heading = theta
    
    sx = -np.sin(heading*np.pi/180)*np.sin(look*np.pi/180)
    sy = np.cos(heading*np.pi/180)*np.sin(look*np.pi/180)
    sz = np.cos(phi*np.pi/180)
    
    S=np.row_stack([sx,sy,sz])
    
    #Write downsampled data to text file
    file = open(outputFile,'w')
    file.writelines('Lon (deg) Lat (deg) displacement(m) Sx Sy Sz \n')  
    for i in range(0, len(X)):
        file.writelines('%f %f %f %f %f %f\n' % (lon[i], lat[i], medians[i], sx[i], sy[i], sz[i]))
    
    #Calculate data covariance
    sc.quadtree = quad
    covar = kite.covariance.Covariance(sc)
    print 'Calculating data covariance..... \n'
    cov = covar.covariance_matrix_focal
    
    np.save('cov', cov)
            
    return quad, sc, cov
    
if __name__ == '__main__':
    quad, sc, cov = main()
    
