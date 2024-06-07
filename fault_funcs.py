#!/usr/bin/env python3

### faultResampler utils #######
# -*- coding: utf-8 -*-
#

import numpy as np
import scipy as sp
import triangle
import matplotlib.pylab as plt
from tde import calc_tri_displacements
import mpl_toolkits.mplot3d.axes3d as p3
from plot_utils import *

def makeFaultStruct(strike, dip, zt, vertices, L, W):
    
    faultstruct             = {}
    faultstruct['strike']   = strike   #fault strike
    faultstruct['dip']      = dip      #fault dip
    faultstruct['zt']       = zt       #depth to top of fault plane
    faultstruct['vertices'] = vertices # x-y coordinates of updip fault vertices
    faultstruct['L']        = L        #Fault length (along strike)
    faultstruct['W']        = W        #Fault width (down dip)
        
    return faultstruct

def makeStartFault_multi(faultstruct):
    '''Create a starting fault model discretization using the Triangle mesh generator.
       Options can be changed to constrain maximum triangle areas ('a') and to generate
       quality meshes ('q') with angle constraints
       
       e.g., mesh = triangle.triangulate(tri, 'q30a1000') 
               -q30 --> quality mesh, minumum angle of 30 degrees
               -a1000 --> area contraint of 1000 units squared
               
       Inputs:
           faultstruct: dictionary containing keys corresponding to each fault geometry
            
            Each key contains:
                -L: Fault length (m) along strike
                -W: Fault width (m) down dip
                -vertices: 2 x 2 array containing x-y (UTM) coordinates of the top fault vertices
                           [x1, x2; y1, y2]
                -strike: Fault strike
                -dip: Fault dip
                -zt: depth to the top of the fault plane (m)
       
       Outputs:
           patchstruct: patchstruct: Dictionary containing keys 'xfault', 'yfault', and 'zfault'
                        which contain 3 x n arrays describing the vertices of each
                        triangular dislocation
          triCoords, triId: coordinates and indices of each triangle making up the fault model 
                            discretization
           nt: array consisting of the number of triangular dislocations per fault segment
       
        '''
    
    patchstruct = {}
    triId       = np.array([]).reshape(0,3)
    triCoords   = np.array([]).reshape(0,2)
    nt          = [0]
    keys        = faultstruct.keys()
    
    #For loop over length of faultstruct

    for i in faultstruct.keys():
        Faultstruct=faultstruct[i]
        strike = Faultstruct['strike']
        dip    = Faultstruct['dip']
        L      = Faultstruct['L']
        W      = Faultstruct['W']
        vertices = Faultstruct['vertices']
        xx       = vertices[0,:]
        yy       = vertices[1,:]        
        node          = np.array([[-L/2, 0],[ L/2, 0], [L/2, W], [-L/2, W], [-L/4, W/2], [L/4, W/2]]) 
        
        pSize         = L*W/20
        opt1           = 'q30a' + str(pSize)
        
        tri             = {}
        tri['vertices'] = node
        mesh            = triangle.triangulate(tri, 'q30')
        TriId           = mesh['triangles']
        TriCoords       = mesh['vertices']
        
        a,b         = TriId.shape
        nt.append(a)
        Patchstruct = ver2patchTri(Faultstruct, TriCoords, TriId)
        patchstruct = patch_append(patchstruct, Patchstruct)
        tmp         = len(triCoords)
        TriId       = TriId + tmp
        triCoords   = np.row_stack([triCoords, TriCoords])
        triId       = np.row_stack([triId,TriId])

    nt = np.array(nt)
    
    ####################################################################
    '''For plotting'''
    patch_rot = rotateFinal(faultstruct, patchstruct, nt)
    xf = patch_rot['xfault']
    yf = patch_rot['yfault']
    plotCoords = np.zeros(np.shape(triCoords))
    
    for i in range(0, len(triId)):
        for j in range(0,3):
            k = int(triId[i,j])
            plotCoords[k,0]=xf[i,j]
            plotCoords[k,1]=yf[i,j]
    ####################################################################
    
    return patchstruct, triId, triCoords, plotCoords, nt

def makeStartFault_Uniform(faultstruct, triParam):
    '''Create a uniform fault model discretization using the Triangle mesh generator.
       
    Inputs:
        faultstruct: dictionary containing keys corresponding to each fault geometry
                    
            Each key contains:
                -L: Fault length (m) along strike
                -W: Fault width (m) down dip
                -vertices: 2 x 2 array containing x-y (UTM) coordinates of the top fault vertices
                           [x1, x2; y1, y2]
                -strike: Fault strike
                -dip: Fault dip
                -zt: depth to the top of the fault plane (m)   
                
        triParam: parameter that can be adjusted to control the number/size of all
                  dislocations
                  
    Outputs:
        patchstruct: Dictionary containing keys 'xfault', 'yfault', and 'zfault'
                     which contain 3 x n arrays describing the vertices of each
                     triangular dislocation
        triCoords, triId: coordinates and indices of each triangle making up the fault model 
                          discretization
        nt: array consisting of the number of triangular dislocations per fault segment
       
        '''
        
    patchstruct = {}
    triId       = np.array([]).reshape(0,3)
    triCoords   = np.array([]).reshape(0,2)
    nt          = [0]
    keys        = faultstruct.keys()
    
    #For loop over length of faultstruct

    for i in faultstruct.keys():
        Faultstruct = faultstruct[i]
        strike      = Faultstruct['strike']
        dip         = Faultstruct['dip']
        L           = Faultstruct['L']
        W           = Faultstruct['W']
        node        = np.array([[-L/2, 0],[L/2, 0],[L/2, W],[-L/2, W]])

        pSize         = L*W/triParam
        opt1           = 'q35a' + str(pSize)
        
        tri             = {}
        tri['vertices'] = node
        mesh            = triangle.triangulate(tri, opt1)
        TriId           = mesh['triangles']
        TriCoords       = mesh['vertices']
        
        a,b         = TriId.shape
        nt.append(a)
        Patchstruct = ver2patchTri(Faultstruct, TriCoords, TriId)
        patchstruct = patch_append(patchstruct, Patchstruct)
        tmp         = len(triCoords)
        TriId       = TriId + tmp
        triCoords   = np.row_stack([triCoords, TriCoords])
        triId       = np.row_stack([triId,TriId])

    nt = np.array(nt)
    
    ####################################################################
    '''For plotting'''
    patch_rot = rotateFinal(faultstruct, patchstruct, nt)
    xf = patch_rot['xfault']
    yf = patch_rot['yfault']
    plotCoords = np.zeros(np.shape(triCoords))

    for i in range(0, len(triId)):
        for j in range(0,3):
            k = int(triId[i,j])
            plotCoords[k,0]=xf[i,j]
            plotCoords[k,1]=yf[i,j]
    ####################################################################
    
    return patchstruct, triId, triCoords, plotCoords, nt


def patch_append(patchstruct, Patchstruct):
    ''' Utility for concatenating patchstruct dictionaries'''
    
    if len(patchstruct) < 3:
        patchNew = Patchstruct
    else:
        xf1=patchstruct['xfault']
        yf1=patchstruct['yfault']
        zf1=patchstruct['zfault']
        xf2=Patchstruct['xfault']
        yf2=Patchstruct['yfault']
        zf2=Patchstruct['zfault']
        patchNew = {}
        patchNew['xfault'] = np.row_stack([xf1,xf2])
        patchNew['yfault'] = np.row_stack([yf1,yf2])
        patchNew['zfault'] = np.row_stack([zf1,zf2])
    
    return patchNew

def ver2patchTri(faultstruct, triCoords, triId):
    ''' Generates patchstruct dictionary containing triangular dislocation
        elements for inversions for fault slip.
        
        Inputs:
            faultstruct: dictionary containing geometric information about fault 
                         geometry and position
            triCoords, triId: triCoords, triId: coordinates and indices of each 
                              triangle making up the fault model discretization
        Outputs:
            patchstruct: Dictionary containing keys 'xfault', 'yfault', and 'zfault'
                         which contain 3 x n arrays describing the vertices of each
                         triangular dislocation
    '''
    
    a        = np.shape(triCoords)
    numver   = a[0]
    a        = np.shape(triId)
    numtri   = a[0]
    strike   = faultstruct['strike']
    strike   = 0.999999*strike
    dip      = faultstruct['dip']
    zt       = faultstruct['zt']
    vertices = faultstruct['vertices']
    xc       = sp.mean(vertices[0,:])
    yc       = sp.mean(vertices[1,:])
    x        = np.zeros([numver,1])
    y        = np.transpose([triCoords[:,0]])
    z        = np.transpose([triCoords[:,1]])
    p1       = np.hstack([x,y,z])
    
    #Rotate vertices clockwise about dip
    Qdip  = [[-1*sp.sin(dip*np.pi/180), 0, sp.cos(dip*np.pi/180)],[0, 1, 0], 
                [sp.cos(dip*np.pi/180), 0, sp.sin(dip*np.pi/180)]]
    p2       = p1.dot(Qdip)

    #Rotate vertices clockwise about strike
    Qstrike  = [[sp.cos(strike*np.pi/180), -1*sp.sin(strike*np.pi/180), 0],
                [sp.sin(strike*np.pi/180), sp.cos(strike*np.pi/180), 0], [0, 0, 1]]  
    p3       = p2.dot(Qstrike)
    
    p3[:,0]  = p3[:,0]+xc
    p3[:,1]  = p3[:,1]+yc
    p3[:,2]  = p3[:,2]+zt

    #Generate patchstruct dictionary 
    patchstruct={}
    xfault = np.zeros([3,numtri])
    yfault = np.zeros([3,numtri])
    zfault = np.zeros([3,numtri])
    
    triId=triId.transpose()
    xfault = np.transpose(p3[triId,0])
    yfault = np.transpose(p3[triId,1])
    zfault = np.transpose(p3[triId,2])
            
    patchstruct['xfault'] = xfault
    patchstruct['yfault'] = yfault
    patchstruct['zfault'] = zfault
    
    return patchstruct

def addNode(triCoords, TriId, resamp):
    '''Function to add a node at the center of a poorly sized dislocation
       for use in creating a new mesh. This approach is not implemented in 
       faultResampler.py '''
       
    if np.sum(resamp) == 0:
        return triCoords
    else:
        pts = triCoords[TriId]
        for i in range(0,len(TriId)):
            if resamp[i] == 1:
                tmp = pts[i,:,:]
                xnew = np.mean(tmp[:,0])
                ynew = np.mean(tmp[:,1])
                newNode = np.array([xnew, ynew])
                triCoords = np.row_stack([triCoords, newNode])
   
    return triCoords
    
def downsampleTri(triCoords, TriId, resamp, Faultstruct):
    '''Function for downsampling triangular dislocations. Each dislocation
       is subdivided into 4 ~equal sized triangles according to the smoothing scales
       calculated in calcSmoothScales.
       
       Inputs:
           triCoords, TriId: triangle coordinates and indices making up model discretization
           resamp: array of values used to downsample dislocations. 0 = keep dislocation, 1 = downsample 
           Faultstruct:
           
       Outputs:
           triCoords, triId: same as inputs --> Generated from downsampling
           
           patchstruct: Dictionary containing keys 'xfault', 'yfault', and 'zfault'
                        which contain 3 x n arrays describing the vertices of each
                        triangular dislocation
               
           '''
        
    TriId = TriId.astype(int)
    pts = triCoords[TriId]
    for i in range(0, len(TriId)):
        if resamp[i]==1:
            temp       = pts[i,:,:]
            tridOld    = TriId[i,:]
            p1         = temp[0,:]
            p2         = temp[1,:]
            p3         = temp[2,:]
            mid1       = np.mean(np.row_stack([p1,p2]),0)
            mid2       = np.mean(np.row_stack([p1,p3]),0)
            mid3       = np.mean(np.row_stack([p2,p3]),0)
            newCoords  = np.row_stack([mid1,mid2,mid3])
            oldLen     = len(triCoords)
            triCoords  = np.row_stack([triCoords, newCoords])
            newLen     = len(triCoords)
            #Add ids for 3 new triangles 
            TriId      = np.row_stack([TriId,[tridOld[0], newLen-3, newLen-2]])
            TriId      = np.row_stack([TriId,[tridOld[1], newLen-3, newLen-1]])
            TriId      = np.row_stack([TriId,[tridOld[2], newLen-2, newLen-1]])
            #Remove id for old triangle and add reference to coords of new central triangle--> new vertices are midpoints of previous triangle sides
            TriId[i,:] = [newLen-1, newLen-2, newLen-3]

    patchstruct = ver2patchTri(Faultstruct, triCoords, TriId)
    
    '''plt.figure()
    plt.triplot(triCoords[:,0],triCoords[:,1],TriId)
    plt.axis('equal')'''
    
    return triCoords, TriId, patchstruct 

def make_green_meade_tri(patchstruct,resampstruct):
    '''Calculate green's functions for slip on a buried triangular fault
    dislocation. This code uses a python adaptation of the codes developed 
    by Meade, 2007.  
    
    #ss += left-lateral
    #ds += thrust
    *** These conventions rely on an ENU coordinate system
    
    Inputs:
        patchstruct: Dictionary containing keys 'xfault', 'yfault', and 'zfault'
                     which contain 3 x n arrays describing the vertices of each
                     triangular dislocation
        resampstruct: Dictionary containing keys 'X', 'Y', and 'S' where
                      X and Y position are position (UTM) and S is the look vector
                      (3 x n) of each surface displacement observation
            
    Output:
        green: Green's functions describing the surface displacements at each data 
               point location due to unit slip on a triangular fault dislocation
    
    '''
    
    print("\nCalculating Green's functions ............. \n")
    
    nu = 0.25
    x  = []
    y  = []
    z  = []
    S  = []
    
    #numfiles = len(resampstruct)
    #Npatch   = len(patchstruct)
    
    x = np.squeeze(resampstruct['X'])
    y = np.squeeze(resampstruct['Y'])
    S = resampstruct['S']
    
    S = sp.transpose(S)
    
    xf = patchstruct['xfault']
    yf = patchstruct['yfault']
    zf = patchstruct['zfault']
    
    Npatch = np.shape(xf)
    Npatch = Npatch[0]
    
    numpts = len(x)
    
    green = sp.zeros([numpts, Npatch*2])
    
    if 'z' in resampstruct:
        z = resampstruct.get('Z')
    else:
        z=x*0
    
    ts = 0
    
    for j in range(2):
        if j==1:
            ds = 0
            ss = -1
        else: 
            ds = -1
            ss = 0
            
        for k in range(Npatch):
            id = k + (j-1)*Npatch
            vx = xf[k,:]
            vy = yf[k,:]
            vz = zf[k,:]
            U  = calc_tri_displacements(x, y, z, vx, vy, vz, nu, ss, ts, ds)
            green[:,id] = U['x']*S[:,0]+U['y']*S[:,1]-U['z']*S[:,2]
    
    return green


def calcMoment(patchstruct, slip):
    ''' Function to calculate seismic moment and moment magnitude for a given
        slip model.
        
        Inputs: 
            patchstruct: Dictionary containing keys 'xfault', 'yfault', and 'zfault'
                     which contain 3 x n arrays describing the vertices of each
                     triangular dislocation 
            slip: array of values containing the modeled slip (m) at each dislocation

        Outputs:
            m0: total seismic moment dyne/cm^2
            mw: moment magnitude
    '''
    
    mu   = 4e11 # shear modulus- dyne/cm^2
    slip = np.abs(slip)
    id_i = []
    
    for i in range(0, len(slip)):
        if slip[i] > 0:
            id_i.append(i)
            
    id_i = np.array(id_i)
    
    xf = patchstruct['xfault']
    yf = patchstruct['yfault']
    zf = patchstruct['zfault']
    
    eta = []
    
    ## Use cross product to calculate triangle areas
    for k in range(0,len(id_i)):
        x    = xf[id_i[k],:]
        y    = yf[id_i[k],:]        
        z    = zf[id_i[k],:]
        v    = np.row_stack([x,y,z])
        v1   = np.array(v[:, 0])
        v2   = np.array(v[:, 1])
        v3   = np.array(v[:, 2])
        w1   = v3-v1
        w2   = v3-v2
        out  = np.cross(w1.transpose(),w2.transpose())
        eta.append(slip[id_i[k]]*0.5*np.linalg.norm(out))
    
    #Scale and sum the dislocation areas    
    eta = np.array(eta)*1e6 #m^3 to cm^3
    eta = eta.sum()
      
    #Calculate seismic moment [dyne/cm^2] and magnitude
    m0 = eta*mu
    mw = np.log10(m0)/1.5-10.73
    
    print('\n \nm0 = ', m0)
    print('Mw = ', mw, '\n \n')
    
    return m0, mw
 
def calcTriArea(patchstruct):
    ''' Calculates the area of triangular dislocations with xyz coordinates defined
        by the dictionary patchstruct
        
        Inputs:
            patchstruct: Dictionary containing keys 'xfault', 'yfault', and 'zfault'
                         which contain 3 x n arrays describing the vertices of each
                         triangular dislocation
        Outputs:
            area: array of dislocation areas
    
    '''
    
    area = []
    xf=patchstruct['xfault']
    yf=patchstruct['yfault']
    zf=patchstruct['zfault']
    for k in range(0, int(np.size(xf)/3)):
        x = xf[k,:]
        y = yf[k,:]
        z = zf[k,:]
        v    = np.row_stack([x,y,z])
        v1   = np.array(v[:, 0])
        v2   = np.array(v[:, 1])
        v3   = np.array(v[:, 2])
        w1   = v3-v1
        w2   = v3-v2
        out  = np.cross(w1.transpose(),w2.transpose())
        area.append(.5*np.linalg.norm(out))
    
    return area

