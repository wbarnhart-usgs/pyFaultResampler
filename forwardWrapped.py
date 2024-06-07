#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:06:22 2017

@author: bstressler
"""

from fault_funcs import *
import matplotlib.pylab as plt

''' Enter input parameters to estimate a wrapped interferogram of an earthquake
    given an estimated magnitude, depth, and focal mechanism '''
    
Mw = 6.4
strike = 205
dip =57
rake = -5
mechanism = 'ss' # can be 'ss', 'reverse', 'normal', or ''
depth = 44.1e3
lambd = 0.055 # C-band wavelength (m)- Sentinel

def main():
    xx,yy,wrapped,data = forwardWrapped(Mw, strike, dip, depth, lambd=lambd, mechanism='reverse')
    
def forwardWrapped(Mw, strike, dip, z, lambd = 0.056, mechanism=''):
    
    fig = plt.figure()
    xc = 0
    yc = 0
    
    L,W = makeFaultDims(Mw, mechanism)
    
    zt = z-((W/2)*np.sin(dip*np.pi/180))
    xt = xc-W/2*np.cos(dip*np.pi/180)*np.cos(strike*np.pi/180)
    yt = yc+W/2*np.cos(dip*np.pi/180)*np.sin(strike*np.pi/180)
    vertices = np.array([[xt-np.sin(strike*np.pi/180)*L/2, xt+np.sin(strike*np.pi/180)*L/2], 
                          [yt-np.cos(strike*np.pi/180)*L/2, yt+np.cos(strike*np.pi/180)*L/2]])

    if zt < 0:
        zt = 0
    
    f1={}
    f1['strike']=strike
    f1['dip']=dip
    f1['vertices']=vertices
    f1['zt'] = zt
    f1['L']=L
    f1['W']=W
    faultstruct = {}
    faultstruct['f1']=f1
    patchstruct, triId, triCoords, nt = makeStartFault(faultstruct)
    nPatch = int(np.size(patchstruct['xfault'])/3)
    
    slip = np.ones(nPatch)
    test = np.linspace(0.01,10,1000)
    
    for i in range(0, np.size(test)):
        slip = test[i]*np.ones(nPatch)
        m0,mw = calcMoment(patchstruct, slip)
        if np.abs(mw-Mw)<0.005:
            break

    print('zt = ', zt)
    
    X = np.linspace(-4*L, 4*L, 300)
    Y = np.linspace(-4*L, 4*L, 300)
    xx, yy = np.meshgrid(X,Y)
    X=xx.flatten()
    Y=yy.flatten()
        
    for i in ['Ascending','Descending']:
        title = i

        if i == 'Ascending':
            sx = -.696
            sy = -.116
            sz = .708
            j=1
        elif i == 'Descending':
            sx = .558
            sy = -.104
            sz = .823
            j=2
        S = np.zeros([3,len(X)])
        S[0,:]=sx
        S[1,:]=sy
        S[2,:]=sz
        
        resampstruct = {}
        resampstruct['X']=X
        resampstruct['Y']=Y
        resampstruct['S']=S
    
        green = make_green_meade_tri(patchstruct, resampstruct)
        g1     = green[:,0:nPatch] #dip slip
        g2     = green[:, (nPatch+np.arange(0,nPatch))]#strike slip
        green  = np.cos(rake*np.pi/180)*g1.transpose()+np.sin(rake*np.pi/180)*g2.transpose()
        
        data = np.dot(green.transpose(),slip)
        dataSq = data.reshape(np.shape(xx))
        
        minD = np.min(data)
        wrapped = np.mod((data-minD)*4*np.pi/lambd, 2*np.pi)
        ax = fig.add_subplot(1,2,j)
        plt.pcolor(xx,yy,wrapped.reshape(np.shape(xx)), cmap='RdYlBu')
        plt.clim([0, np.pi*2])
        plt.colorbar()
        plt.title(title)
        plt.axis('image')
        max = np.max(data)
        min = np.min(data)
    
        print('Max LOS displacement- ', i, ': ', max, ' (m)')
        print('Min LOS displacement- ', i, ': ', min, ' (m)')
        print('One fringe = ', lambd/2, ' (m)')
    plt.show()

    return xx, yy, wrapped, data
    
def makeStartFault(faultstruct):
        
    patchstruct = {}
    triId       = np.array([]).reshape(0,3)
    triCoords   = np.array([]).reshape(0,2)
    nt          = [0]
    keys        = faultstruct.keys()
    
    #For loop over length of faultstruct

    for i in faultstruct.keys():
        Faultstruct = faultstruct[i]
        Faultstrike = Faultstruct['strike']
        dip         = Faultstruct['dip']
        L           = Faultstruct['L']
        W           = Faultstruct['W']
        node        = np.array([[-L/2, 0],[L/2, 0],[L/2, W],[-L/2, W]])        
        tri             = {}
        tri['vertices'] = node
        mesh            = triangle.triangulate(tri)
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
    
    return patchstruct, triCoords, triId, nt

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

if __name__ == '__main__':
    main()
