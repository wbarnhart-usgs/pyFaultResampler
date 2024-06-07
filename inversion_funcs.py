#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 13:18:04 2017

@author: bstressler
"""
from scipy.optimize import least_squares, lsq_linear
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from plot_utils import *
from fault_funcs import *

def inversionFixedRake(patchstruct, plotCoords, triId, invDict):
    ''' Routine to invert for distributed slip with a fixed rake direction
        using Tikhonov regularization and either minimum moment or laplacian smoothing. 
        Regularization parameter is chosen using the approximate jRi method or 
        alternatively with the l-curve method.
    
    Usage: gsmooth, G, Gg, mil, synth = inversionFixedRake(patchstruct, triId, invDict)
    
    Inputs:
        patchstruct: dictionary containing keys 'xfault', 'yfault', 'zfault'
                     which are 3 x n arrays describing the xyz coordinates of
                     triangular dislocations
                     
        plotCoords: rotated triangular dislocation coordinates produced by makeStartFault_multi.py
                   or do_Invert_Remesh.py
                   --> Used for plotting only
        
        triId:  array containing the indices of triangle vertices to reference from triCoords.
                --> Used for plotting and generating laplacian smoothing matrices
        
        invDict: Dictionary containing requisite variables needed to set up inverse
                 problem generated in loadResampData.py
    
    Outputs:
        mil: Model parameters- slip = mil[0:nPatch], ramp parameters = mil[nPatch:len(mil)]
        
        gsmooth,G,Gg: Weighted Green's functions and related matrices
        
        synth: synthetic data predicted by the inverted model
        
    '''
    
    rake          = invDict['rake']
    reg_method    = invDict['reg_method']
    smooth_method = invDict['smooth_method']
    data_type     = invDict['data_type']
    rampg         = invDict['rampg']
    nramp         = invDict['nramp']
    Cdinv         = invDict['Cdinv']
    lambdas       = invDict['lambdas']
    covd2         = invDict['covd2']
    allnp         = invDict['allnp']
    data          = invDict['data']
    plot_flag     = invDict['plot_flag']
    Dnoise        = invDict['Dnoise']
    resampstruct  = invDict['resampstruct']
    faultstruct   = invDict['faultstruct']
    
    nPatch = np.shape(patchstruct['xfault'])
    nPatch = nPatch[0]
    numpts = np.shape(resampstruct['X'])
    numpts = numpts[0]
    
    green  = make_green_meade_tri(patchstruct,resampstruct)
    g1     = green[:,0:nPatch] #dip slip
    g2     = green[:, (nPatch+np.arange(0,nPatch))]#strike slip
    green  = np.cos(rake*np.pi/180)*g1.transpose()+np.sin(rake*np.pi/180)*g2.transpose()

    if nramp > 0:
        tmp    = np.row_stack([green, rampg])
    else:
        tmp    = green
        
    tmp    = tmp.transpose()
    G      = np.dot(Cdinv,tmp)
    D      = np.row_stack([Dnoise.reshape(len(Dnoise),1), sp.zeros([nPatch,1])])
    D      = D.squeeze()
    
    if smooth_method == 'laplacian':
        smooth = triSmooth(triId)
    else:
        smooth = np.eye(nPatch)

    A = np.row_stack([np.zeros([nPatch,1]),-1*np.inf*np.ones([nramp,1])])
    B = np.ones([nPatch+nramp,1])*np.inf
    A = A.squeeze()
    B = B.squeeze()
    

    jRi    = []
    r_norm = []
    m_norm = []
    
    for j in range(0,len(lambdas)):
        alpha                 = lambdas[j]
        gsmooth               = np.row_stack([G, np.column_stack([alpha*smooth, sp.zeros([nPatch,nramp])])])
        out                   = lsq_linear(gsmooth, D, bounds=(A,B), max_iter=1000, verbose=1)
        mil                   = out['x']
        ril                   = out['fun']
        resnorm               = out['cost']
        Gg                    = np.dot(sp.linalg.inv(np.dot(gsmooth.transpose(),gsmooth)),G.transpose())
        N                     = np.dot(G, Gg)
        M                     = np.column_stack([sp.eye(numpts), -N])
        junk                  = ril[0:numpts]**2
        iRi                   = np.sum(junk)/numpts
        covresjRi             = np.dot(M, M.transpose())
        covresiRi             = np.dot(np.dot(M, covd2), M.transpose())
        jRin                  = np.mean(np.diag(covresjRi))
        iRin                  = np.mean(np.diag(covresiRi))
        oro_approx            = iRi-iRin
        jRi.append(oro_approx+jRin)
        r_norm.append(iRi)
        m_norm.append(np.dot(np.transpose(np.dot(smooth, mil[np.arange(0,nPatch)])),np.dot(smooth,mil[np.arange(0,nPatch)])))              
    
    if reg_method == 'jRi':
        id = np.argmin(jRi)
        plt.figure()
        plt.subplot(2,2,1)
        plt.title('L-curve')
        plt.plot(np.sqrt(r_norm),np.sqrt(m_norm),'-o')
        plt.plot(np.sqrt(r_norm[id]),np.sqrt(m_norm[id]),'or')
        plt.xlabel('Data Norm'), plt.ylabel('Model Norm')
        plt.subplot(2,2,2)
        plt.title('jRi curve')
        plt.semilogx(lambdas, jRi, '.-')
        plt.plot(lambdas[id], jRi[id], 'or')
        plt.xlabel('Regularization coefficient'), plt.ylabel('jRi')

    else:
        plt.figure()
        plt.subplot(2,2,1)
        plt.plot(np.sqrt(r_norm),np.sqrt(m_norm),'-o')
        plt.title('Choose L-Curve corner ID')
        plt.xlabel('Data Norm')
        plt.ylabel('Model Norm')
        plt.subplot(2,2,2)
        plt.semilogx(lambdas, jRi, '.-')
        plt.xlabel('Data Norm'), plt.ylabel('Model Norm')
        plt.title('jRi')
        plt.ion(), plt.show()
        plt.pause(0.5)            
        id = int(input('\n \nChoose ID number of corner point, starting from the right \n'))
        id = id-1
        plt.plot(lambdas[id], jRi[id], '.r', markersize=8)
        plt.subplot(2,2,1)
        plt.plot(np.sqrt(r_norm[id]),np.sqrt(m_norm[id]), '.r', markersize=8)
        
    alpha                 = lambdas[id]
    gsmooth               = np.row_stack([G, np.column_stack([alpha*smooth, sp.zeros([nPatch,nramp])])]) 
    out                   = lsq_linear(gsmooth, D, bounds=(A,B), max_iter=1000, verbose=1)
    mil                   = out['x']
    ril                   = out['fun']
    resnorm               = out['cost']
    Gg                    = np.dot(sp.linalg.inv(np.dot(gsmooth.transpose(),gsmooth)),G.transpose())
    synth                 = np.dot(green.transpose(), mil[np.arange(0,nPatch)]);
    slip                  = mil[0:nPatch];
    m0,mw                 = calcMoment(patchstruct, slip)

    #Plot slip model and data residuals
    idx = plotCoords[:,0].nonzero()
    idy = plotCoords[:,1].nonzero()
    plotx = plotCoords[idx,0]
    ploty = plotCoords[idy,1]
    xlim = np.array([np.max(plotx), np.min(plotx)])
    ylim = np.array([np.min(ploty), np.max(ploty)])
    
    plt.subplot(2,1,2)
    plt.tripcolor(plotCoords[:,1], plotCoords[:,0],triangles=triId, facecolors=slip, edgecolors='w', cmap='jet')
    plt.gca().invert_yaxis()
    plt.axis('image'), plt.colorbar()
    plt.xlim(ylim)
    plt.ylim(xlim)
    plt.show(block=False), plt.pause(0.5), plt.title('Modeled Slip')
    plt.xlabel('Distance Along Strike (m)'), plt.ylabel('Down-dip Width (m)')
    plt.pause(0.5)

    if plot_flag == 1:    
        plot_data_resid_subplot(resampstruct['X'], resampstruct['Y'], resampstruct['S'], data, synth, allnp, data_type)
    else: plt.close()
    
    return gsmooth, G, Gg, mil, synth

def inversionFreeRake(patchstruct, plotCoords, triId, invDict):
    ''' Routine to invert for distributed slip allowing rake to vary freely
        using Tikhonov regularization and either minimum moment or laplacian smoothing. 
        Regularization parameter is chosen using the approximate jRi method or 
        alternatively with the l-curve method.
    
    Usage: gsmooth, G, Gg, slip, synth, mil, rake = inversionFixedRake(patchstruct, triId, invDict)
    
    Inputs:
        patchstruct: dictionary containing keys 'xfault', 'yfault', 'zfault'
                     which are 3 x n arrays describing the xyz coordinates of
                     triangular dislocations
                     
        plotCoords: rotated triangular dislocation coordinates produced by makeStartFault_multi.py
                   or do_Invert_Remesh.py
                   --> Used for plotting only
        
        triId:  array containing the indices of triangle vertices to reference from triCoords.
                --> Used for plotting and generating laplacian smoothing matrices
        
        invDict: Dictionary containing requisite variables needed to set up inverse
                 problem generated in loadResampData.py
    
    Outputs:
        mil: Model parameters- strike slip = mil[0:nPatch], dip slip = mil[nPatch:2*nPatch[
                               ramp parameters = mil[2*nPatch:len(mil)]
        
        slip: Magnitude of slip inverted for each dislocation
        
        rake: Rake direction for each dislocation
            
        gsmooth,G,Gg: Weighted Green's functions and related matrices
        
        synth: synthetic data predicted by the inverted model
        
    '''
    reg_method     = invDict['reg_method']
    smooth_method  = invDict['smooth_method']
    data_type      = invDict['data_type']
    rampg          = invDict['rampg']
    nramp          = invDict['nramp']
    Cdinv          = invDict['Cdinv']
    lambdas        = invDict['lambdas']
    covd2          = invDict['covd2']
    ss_constraints = invDict['ss_constraints']
    ds_constraints = invDict['ds_constraints']
    allnp          = invDict['allnp']
    data           = invDict['data']
    plot_flag      = invDict['plot_flag']
    Dnoise         = invDict['Dnoise']
    resampstruct   = invDict['resampstruct']
    faultstruct    = invDict['faultstruct']


    nPatch = np.shape(patchstruct['xfault'])
    nPatch = nPatch[0]
    numpts = np.shape(resampstruct['X'])
    numpts = numpts[0]
    
    green  = make_green_meade_tri(patchstruct,resampstruct)
    green  = green.transpose()
    
    if nramp > 0:
        tmp    = np.row_stack([green, rampg])
    else: 
        tmp    = green
        
    tmp    = tmp.transpose()
    G      = np.dot(Cdinv, tmp)
    D      = np.row_stack([Dnoise.reshape(len(Dnoise),1), sp.zeros([2*nPatch,1])])
    D      = D.squeeze()
  
    #Adjust coefficients to constrain ss/ds slip components of slip
    #Strike slip constraints
    if ss_constraints == 'l':
        # Left lateral slip only
        a1 = 0
        b1 = np.inf
    elif ss_constraints == 'r':
        # Right lateral slip only
        a1 = -1*np.inf
        b1 = 0  
    else:
        # SS--> Unconstrained
        a1 = -1*np.inf
        b1 = np.inf

    # Dip slip constraints
    if ds_constraints == 'n':
        # Normal slip only
        a2 = -1*np.inf
        b2 = 0
    elif ds_constraints == 'r':
        # Reverse slip only
        a2 = 0
        b2 = np.inf
    else:
        # DS--> Unconstrained
        a2 = -1*np.inf
        b2 = np.inf
    
    A      = np.row_stack([a1*np.ones([nPatch,1]), a2*np.ones([nPatch,1]), -1*np.inf*np.ones([nramp,1])]) 
    B      = np.row_stack([b1*np.ones([nPatch,1]), b2*np.ones([nPatch,1]), np.inf*np.ones([nramp,1])]);
    A      = A.squeeze()
    B      = B.squeeze()
    
    if smooth_method == 'laplacian':
        
        smooth = triSmooth(triId)
        smooth = sp.linalg.block_diag(smooth,smooth)
        
    else:
        smooth = np.eye(2*nPatch)
           
    jRi    = []        
    m_norm = []
    r_norm = []
    
    for j in range(0, len(lambdas)):
        
        alpha                 = lambdas[j]
        gsmooth               = np.row_stack([G, np.column_stack([alpha*smooth, sp.zeros([2*nPatch,nramp])])]) 
        out                   = lsq_linear(gsmooth, D, bounds=(A,B), max_iter=1000, verbose=1)
        mil                   = out['x']
        ril                   = out['fun']
        resnorm               = out['cost']
        Gg                    = np.dot(sp.linalg.inv(np.dot(gsmooth.transpose(),gsmooth)),G.transpose())
        N                     = np.dot(G, Gg)
        M                     = np.column_stack([sp.eye(numpts), -N])
        junk                  = ril[0:numpts]**2
        iRi                   = np.sum(junk)/numpts
        covresjRi             = np.dot(M,M.transpose())
        covresiRi             = np.dot(np.dot(M, covd2), M.transpose())
        jRin                  = np.mean(np.diag(covresjRi))
        iRin                  = np.mean(np.diag(covresiRi))
        oro_approx            = iRi-iRin
        jRi.append(oro_approx + jRin)
        r_norm.append(iRi)
        m_norm.append(np.dot(np.transpose(np.dot(smooth, mil[np.arange(0,2*nPatch)])),np.dot(smooth,mil[np.arange(0,2*nPatch)])))              

    if reg_method == 'jRi':
        id = np.argmin(jRi)
        plt.figure()
        plt.subplot(2,2,1)
        plt.title('L-curve')
        plt.plot(np.sqrt(r_norm),np.sqrt(m_norm),'-o')
        plt.plot(np.sqrt(r_norm[id]),np.sqrt(m_norm[id]),'or')
        plt.xlabel('Data Norm')
        plt.ylabel('Model Norm')
        plt.subplot(2,2,2)
        plt.title('jRi curve')
        plt.semilogx(lambdas, jRi, '.-')
        plt.plot(lambdas[id], jRi[id], 'or')
        plt.xlabel('Regularization coefficient'), plt.ylabel('jRi')
        plt.ion(), plt.show(), plt.pause(0.5)
        
    else:
        fig=plt.figure()
        plt.subplot(2,2,1)
        plt.plot(np.sqrt(r_norm),np.sqrt(m_norm),'-o')
        plt.title('Choose L-Curve corner ID')
        plt.xlabel('Data Norm')
        plt.ylabel('Model Norm')
        plt.subplot(2,2,2)
        plt.semilogx(lambdas, jRi, 'o-')
        plt.xlabel('Regularization coefficient'), plt.ylabel('jRi')
        plt.title('jRi')
        plt.ion(), plt.show(), plt.pause(0.5)
        id = int(input('\n \nChoose ID number of corner point, starting from the right \n'))
        id = id - 1
        plt.plot(lambdas[id], jRi[id], '.r', markersize=8)
        plt.subplot(2,2,1)
        plt.plot(np.sqrt(r_norm[id]),np.sqrt(m_norm[id]), '.r', markersize=8)


    alpha                 = lambdas[id]
    gsmooth               = np.row_stack([G, np.column_stack([alpha*smooth, sp.zeros([2*nPatch,nramp])])]) 
    Gg                    = np.dot(sp.linalg.inv(np.dot(gsmooth.transpose(), gsmooth)),G.transpose())
    out                   = lsq_linear(gsmooth, D, bounds=(A,B), max_iter=1000, verbose=1)
    mil                   = out['x']
    ril                   = out['fun']
    resnorm               = out['cost']
    synth                 = np.dot(green.transpose(), mil[0:2*nPatch])
    slip                  = np.sqrt(mil[0:nPatch]**2 + mil[nPatch:2*nPatch]**2)
    m0,mw                 = calcMoment(patchstruct,slip)
    
    ss                    = mil[0:nPatch]
    ds                    = mil[nPatch:2*nPatch]
    rake                  = np.zeros(len(ss))
    
    for i in range(0, len(ss)):
        rake[i] = math.atan2(ds[i], ss[i])*180/np.pi
    
    #Plot final slip model and data residuals
    
    idx = plotCoords[:,0].nonzero()
    idy = plotCoords[:,1].nonzero()
    plotx = plotCoords[idx,0]
    ploty = plotCoords[idy,1]
    xlim = np.array([np.max(plotx), np.min(plotx)])
    ylim = np.array([np.min(ploty), np.max(ploty)])
    
    plt.subplot(2,1,2)
    plt.tripcolor(plotCoords[:,1], plotCoords[:,0], triangles=triId, facecolors=slip, edgecolors='w', cmap='jet')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.axis('image')
    plt.xlim(ylim)
    plt.ylim(xlim)
    plt.ion(), plt.show(block=False), plt.pause(0.5), plt.title('Modeled Slip')
    plt.xlabel('Distance Along Strike (m)'), plt.ylabel('Down-dip Width (m)')

    if plot_flag == 1:
        plot_data_resid_subplot(resampstruct['X'], resampstruct['Y'], resampstruct['S'], data, synth, allnp, data_type)
    else: 
        plt.close()
    
    return gsmooth, G, Gg, slip, synth, mil, rake

def triSmooth(triId):
    '''
    Routine generates a Laplacian smoothing matrix for higher level Tikhonov
    regularization. Smoothing of triangles is not weighted by their area
    since it is assumed that triangle size gradients vary smoothly.
    
    Usage: smooth = triSmooth(triId)
    
    Inputs:
        triId: index terms for triangular dislocations with coordinated defined in
               variable triCoords
   
    Outputs:
        smooth: laplacian smoothing matrix
    
    '''
    i,j    = np.shape(triId)
    smooth = np.eye(i)
    common = []
    
    for k in range(0,i):
        del common
        common = []
        a      = triId[k,:]
        
        for m in range(0,i):
            tmp=[]
            b   = triId[m,:]
            
            for q in a:
                if q in b:
                    tmp.append(q)
            
            if len(tmp) == 2:
                common.append(m)
                
        nneighbor = len(common)
        
        if nneighbor == 1:
            id = common
            smooth[k, id] = -1
        elif nneighbor == 2:
            smooth[k, common] = -.5
        else: 
            smooth[k, common] = -1/3
        
    return smooth
    
def do_invert_Remesh(patchstruct, nt, plotCoords, triCoords, triId, invDict):
    '''  Routine to iteratively invert geodetic observations for fault slip  and downsample a fault 
         model until a "best" model discretization is obtained. Model downsampling is performed by calculating
         smoothing scales for each dislocation following the method of Barnhart & Lohman (2010).
         If the smoothing scale for a given dislocation exceeds the smoothing area, the dislocation
         is broken into 4 equally sized new triangles. The newly downsampled mesh is then refined using 
         the Triangle software to avoid steep gradients in dislocation sizes. The algorithm terminates when 
         all dislocations are appropriately downsampled.
         
         Inputs:
            patchstruct: dictionary containing keys 'xfault', 'yfault', 'zfault'
                         which are 3 x n arrays describing the xyz coordinates of
                         triangular dislocations
                         
            nt: array consisting of the number of triangular dislocations per fault segment.
         
                         
            plotCoords: rotated triangular dislocation coordinates produced by makeStartFault_multi.py
                       or do_Invert_Remesh.py
                   --> Used for plotting only
                   
            triCoords: triangular dislocation coordinates produced by makeStartFault_multi.py
                       or do_Invert_Remesh.py. (Not in geographic reference frame)
                       --> Used for plotting
         
            triId: array containing the indices of triangle vertices to reference from triCoords.
                   --> Used for plotting and generating laplacian smoothing matrices
         
            invDict: Dictionary containing requisite variables needed to set up inverse
                     problem generated in loadResampData.py
        
        Outputs:
            patchstruct: dictionary containing keys 'xfault', 'yfault', 'zfault'
                         which are 3 x n arrays describing the xyz coordinates of
                         triangular dislocations
                         
            mil: Model parameters- strike slip = mil[0:nPatch], dip slip = mil[nPatch:2*nPatch[
                 ramp parameters = mil[2*nPatch:len(mil)]
        
            slip: Magnitude of slip inverted for each dislocation
        
            rake: Rake direction for each dislocation
        
            gsmooth,G,Gg: Weighted Green's functions and related matrices
        
            synth: synthetic data predicted by the inverted model
        
            triCoords, triId: same as inputs but for final mesh generated by this function
            
    '''
    
    OK           = 0
    iter         = 1
    rake_type    = invDict['rake_type']
    faultstruct  = invDict['faultstruct']
    rake         = invDict['rake']
    data_type    = invDict['data_type']
    data         = invDict['data']
    
    while OK < 1:
        xf = patchstruct['xfault']
        nPatch = int(np.size(xf)/3)
        
        if rake_type == 'free':
            gsmooth, G, Gg, slip, synth, mil, rake = inversionFreeRake(patchstruct, plotCoords, triId, invDict)
            R  = np.dot(Gg,G)
            r1 = R[0:nPatch, 0:nPatch]
            r2 = R[nPatch:2*nPatch, nPatch:2*nPatch]
            R  = np.sqrt(r1**2+r2**2)
            
        else:
            gsmooth, G, Gg, mil, synth = inversionFixedRake(patchstruct, plotCoords, triId, invDict)
            slip = mil[0:nPatch]
            R = np.dot(Gg,G)
            R = R[0:nPatch, 0:nPatch]
        
        Newscales, scales, resamp = calcSmoothScales(G, Gg, rake_type, patchstruct, nt, nPatch)
        numResize = np.sum(resamp)
        
        if numResize == 0:
           OK = 1
           print('Current mesh is adequately downsampled ... Plotting final slip model \n')
           resampstruct=invDict['resampstruct']
           plot_data_resid(resampstruct['X'], resampstruct['Y'], resampstruct['S'], data, synth, invDict['allnp'],invDict['data_type'])
        else:
            print(int(numResize), ' triangles need to be downsampled')
            triId = triId.astype(int)
            
            tic = 0
            TriId       = np.array([]).reshape(0,3)
            TriCoords   = np.array([]).reshape(0,2)
            patchstructNew =[]
            NT=[0]
            numtris = 0
            # Downsample Method
            for i in faultstruct.keys():
                start         = nt[tic]
                numtris       = numtris + nt[tic+1]
                id            = range(start, numtris)
                Faultstruct   = faultstruct[i]
                tempId        = triId[id]
                Resamp        =  resamp[id]
                newCoords, newId, PatchstructNew  = downsampleTri(triCoords, tempId, Resamp, Faultstruct)
           
                ##################################################
                ####### Refine newly downsampled mesh ############
                if numResize > 5:
                    tri = {}
                    tri['vertices'] = newCoords
                    mesh            = triangle.triangulate(tri, 'q')
                    tmp             = len(TriCoords)
                    tempId          = mesh['triangles']
                    tempId           = tempId + tmp
                    tempCoords      = mesh['vertices']
                    TriId           = np.row_stack([TriId,tempId])
                    TriCoords       = np.row_stack([TriCoords, tempCoords])
                    PatchstructNew = ver2patchTri(Faultstruct, mesh['vertices'], mesh['triangles'])
                    patchstructNew = patch_append(patchstructNew, PatchstructNew)
                    NT.append(int(np.size(PatchstructNew['xfault'])/3))
                    tic += 1
                else:
                    NT.append(int(np.size(PatchstructNew['xfault'])/3))
                    TriId           = np.row_stack([TriId,newId])
                    TriCoords       = np.row_stack([TriCoords, newCoords])
                    patchstructNew = patch_append(patchstructNew, PatchstructNew)
                    tic += 1
                ###################################################
            
            npatch=int(np.size(patchstructNew['xfault'])/3)
            patchstruct = patchstructNew
            print('\nNumber of old fault patches = ', nPatch)
            print('\nNumber of new fault patches = ', npatch)
            nt = NT
            nt = np.array(nt)
            nt = nt.astype(int)
            
            patch_rot = rotateFinal(faultstruct, patchstruct, nt)
            xf = patch_rot['xfault']
            yf = patch_rot['yfault']
            triId = TriId.astype(int)
            triCoords = TriCoords
            plotCoords = np.zeros([np.max(triId)+1,2])

            for i in range(0, len(triId)):
                for j in range(0,3):
                    k = triId[i,j]
                    plotCoords[k,0]=xf[i,j]
                    plotCoords[k,1]=yf[i,j]
                    
            if nPatch == npatch:
                OK = 1
                print('\n \nCurrent mesh is adequately downsampled... Plotting final slip model \n')
                resampstruct=invDict['resampstruct']
                plot_data_resid_subplot(resampstruct['X'], resampstruct['Y'], resampstruct['S'], data, synth, invDict['allnp'], invDict['data_type'])
                nPatch = npatch
            
            nPatch = npatch
            
    ss = slip*np.cos(rake*np.pi/180)
    ds = slip*np.sin(rake*np.pi/180)
    xc = np.zeros(len(triId))
    yc = np.zeros(len(triId))
    
    if len(faultstruct.keys()) == 1:
        plotCoords = triCoords
        pts = plotCoords[triId]
    else:
        pts = plotCoords[triId]
        idx = plotCoords[:,0].nonzero()
        idy = plotCoords[:,1].nonzero()
        plotx = plotCoords[idx,0]
        ploty = plotCoords[idy,1]
        xlim = np.array([np.max(plotx), np.min(plotx)])
        ylim = np.array([np.min(ploty), np.max(ploty)])

    for i in range(0, len(pts[:,0,0])):
        xc[i] = np.mean(pts[i,:,0])
        yc[i] = np.mean(pts[i,:,1])

    if len(faultstruct.keys()) > 1:
        
        plt.pause(0.001)
        plt.figure()
        plt.subplot(2,1,1)
        plt.tripcolor(plotCoords[:,1], plotCoords[:,0],triangles=triId, edgecolors='w', facecolors=slip, cmap='jet')
        plt.colorbar()
        plt.title('Final Slip Model')
        plt.xlim(ylim)
        plt.ylim(xlim)
        if rake_type == 'free':    
            plt.quiver(yc, xc, ss, ds, scale=10, color='r')
    
        plt.subplot(2,1,2)
        plt.tripcolor(plotCoords[:,1], plotCoords[:,0],triangles=triId, norm = mpl.colors.Normalize(vmin=np.min(Newscales), vmax=np.max(Newscales)),facecolors=Newscales)
        plt.colorbar()
        plt.xlim(ylim)
        plt.ylim(xlim)
        plt.title('Resolution- Smoothing Scales')
        plt.show(block=False)
    
    else: 
        plt.pause(0.001)
        plt.figure()
        plt.subplot(2,1,1)
        plt.tripcolor(triCoords[:,0], triCoords[:,1],triangles=triId, edgecolors='w', facecolors=slip, cmap='jet')
        plt.colorbar(), plt.gca().invert_yaxis()
        plt.title('Final Slip Model')
        plt.axis('image')
        if rake_type == 'free':    
            plt.quiver(yc, xc, ss, ds, scale=10, color='r')
    
        plt.subplot(2,1,2)
        plt.tripcolor(triCoords[:,0], triCoords[:,1],triangles=triId, norm = mpl.colors.Normalize(vmin=np.min(Newscales), vmax=np.max(Newscales)),facecolors=Newscales)
        plt.colorbar(), plt.gca().invert_yaxis()
        plt.axis('image')
        plt.title('Resolution- Smoothing Scales')
        plt.show(block=False)

    return patchstruct, gsmooth, G, Gg, slip, synth, mil, rake, plotCoords, triCoords, triId


def calcSmoothScales(G, Gg, rake_type, patchstruct, nt, nPatch):      
    ''' Routine for calculating smoothing scales for each dislocation. The ith row of
        the model resolution matrix (corresponding to the dislocation in question)
        versus the distance to all other dislocations is fit to a Gaussian curve. The
        smoothing scale is defined as the 1 sigma gaussian width. If the smoothing scale
        is smaller than the size of the dislocation, the dislocation is downsampled
    
    Inputs:
        G, Gg: Green's function-related variables for calulating model resolution matrix (R)
        rake_type: Type of inversion--> 'fixed' or 'free' rake
        patchstruct: dictionary containing keys 'xfault', 'yfault', 'zfault'
                         which are 3 x n arrays describing the xyz coordinates of
                         triangular dislocations                
        nt: array consisting of the number of triangular dislocations per fault segment.
        nPatch: total number of dislocations
        
    Outputs:
        Newscales: smoothing scales
        scales: length scales used for determining is a dislocation is properly sized
        resamp: array of values used to downsample dislocations. 0 = keep dislocation, 1 = downsample 
        
        '''
    
    #Calculate model resolution matrix
    if rake_type == 'free':
        R  = np.dot(Gg,G)
        r1 = R[0:nPatch, 0:nPatch]
        r2 = R[nPatch:2*nPatch, nPatch:2*nPatch]
        R  = np.sqrt(r1**2+r2**2)
    else:
        R = np.dot(Gg,G)
        R = R[0:nPatch, 0:nPatch]

    xf = patchstruct['xfault']
    yf = patchstruct['yfault']
    zf = patchstruct['zfault']
    xc = np.mean(xf, 1)
    yc = np.mean(yf, 1)
    zc = np.mean(zf, 1)
    tmpPatch = 0
    Newscales = []
    Scales   = []
    scales1 = []
    
    # Calculate smoothing scales for each dislocation
    for k in range(0, nPatch):
        dist    = np.sqrt((xc-xc[k])**2 + (yc-yc[k])**2 + (zc-zc[k])**2)
        id_tmp  = np.nonzero(dist)
        scales1.append(np.min(dist[id_tmp]))
        y2      = R[k,:]
        b       = 0
        x2      = dist
        id2     = y2 > 0
        y2      = y2[id2]
        x2      = x2[id2]
        scales2 = np.sqrt(np.sum(x2**2*y2)/np.sum(y2))
        Scales.append(np.nanmax([scales1[k],scales2]))
        
        def gausfun(c, x2, y2, b):
            id2 = np.argwhere(x2==0)
            if len(id2)==0:
                id2 = np.argmin(x2)
                    
            tmp = y2[id2]*np.exp(-1*((x2-b)/c)**2)-y2
            tmp = tmp.squeeze()
            return tmp
        
        out     = least_squares(gausfun,Scales[k], args=(x2, y2, b))
        m       = out['x']
        tmpPatch += 1
        Newscales.append(m[0])
        
    
    '''# Calculate smoothing scales for each dislocation
    for j in range(0, len(nt)-1):
        xc = np.mean(xf[nt[j]:nt[j+1],:], 1)
        yc = np.mean(yf[nt[j]:nt[j+1],:], 1)
        zc = np.mean(zf[nt[j]:nt[j+1],:], 1)
        scales    = []
        scales1   = []
        newscales = []
        
        for k in range(nt[j], nt[j+1]):
            dist    = np.sqrt((xc-xc[k])**2 + (yc-yc[k])**2 + (zc-zc[k])**2)
            id_tmp  = np.nonzero(dist)
            scales1.append(np.min(dist[id_tmp]))
            y2      = R[tmpPatch, nt[j]+0:nt[j+1]]
            b       = 0
            x2      = dist
            id2     = y2 > 0
            y2      = y2[id2]
            x2      = x2[id2]
            scales2 = np.sqrt(np.sum(x2**2*y2)/np.sum(y2))
            scales.append(np.nanmax([scales1[k],scales2]))
            
            def gausfun(c, x2, y2, b):
                id2 = np.argwhere(x2==0)
                if len(id2)==0:
                    id2 = np.argmin(x2)
                        
                tmp = y2[id2]*np.exp(-1*((x2-b)/c)**2)-y2
                tmp = tmp.squeeze()
                return tmp
            
            out     = least_squares(gausfun,scales[k], args=(x2, y2, b))
            m       = out['x']
            tmpPatch += 1
            newscales.append(m[0])
            
        Newscales.append(newscales)
        scales.append(scales)'''
        
    Newscales = np.array(Newscales)
    Scales    = np.array(Scales)
    scales1    = np.array(scales1)
    resamp = np.zeros(len(Scales))
    
    for i in range(0,len(resamp)):
        if Newscales[i]<scales1[i]:
            resamp[i] = 1
    
    return Newscales, scales1, resamp
