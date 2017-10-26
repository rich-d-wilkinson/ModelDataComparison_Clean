from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib
import glob as glob
from cmocean import cm as cm
import numpy as np

def plot_samples(vals, thinby, land_mask, levels=None):
    """
    Assumes we've used a particular thinby value on the gcm grid
    """
    # original GCM grid
    lats_gcm = np.arange(-89.375,89.375,1.25) # Is this right?
    longs_gcm = np.arange(-180,178.75+0.1, 1.25)
    longgrid_gcm, latgrid_gcm = np.meshgrid(longs_gcm, lats_gcm)

    keep_lats= np.arange(0,lats_gcm.size,thinby)
    keep_longs= np.arange(0,longs_gcm.size,thinby)

    # thinned GCM grid
    longgrid_pred = longgrid_gcm[keep_lats,:][:,keep_longs]
    latgrid_pred = latgrid_gcm[keep_lats,:][:,keep_longs]

    # create an array of Falses, change the ocean values to true, then thin
    land_mask_TF = np.zeros(longgrid_gcm.shape, dtype=bool)
    tmp = land_mask_TF.flatten()
    tmp[land_mask-1]=True
    land_mask_TF = tmp.reshape(longgrid_gcm.shape)
    land_mask_TF_pred = land_mask_TF[keep_lats,:][:,keep_longs]

    yplot = np.zeros(land_mask_TF_pred.size)-10000.
    yplot[land_mask_TF_pred.flatten()] = vals # IS THIS RIGHT?
    gcm_grid = yplot.reshape(land_mask_TF_pred.shape)


    mp2 = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
            llcrnrlon=-180,urcrnrlon=180,resolution='c')
    mp2.drawcoastlines()
    mp2.drawparallels(np.arange(-90.,91.,30.))
    mp2.drawmeridians(np.arange(-180.,181.,60.))
    mp2.drawmapboundary(fill_color='white')

    if levels==None:
        levels = np.arange(-10,12,1)

    #mp2.scatter(longgrid_pred.flatten(), latgrid_pred.flatten())
    mp2.contourf(longgrid_pred,latgrid_pred, gcm_grid,15,levels=levels,
        cm = cm.delta)
    mp2.colorbar()
    mp2.fillcontinents('black')
    return(mp2)



def plot_wo_basemap(longgrid, latgrid, ygrid, levels=None, X_obs=None):
    """
    Values in ygrid to be blacked out (continents) should have extreme values
    """
    plt.xlabel('lon')
    plt.ylabel('lat')

    if levels is None:
        levels = np.arange(-10,10,1)

    CS=plt.contourf(longgrid,latgrid,ygrid,15,levels=levels,
    cm = cm.delta, extend='both')

    CS.cmap.set_over('black')
    CS.cmap.set_under('black')

    if X_obs is not None:
        plt.scatter(X_obs[:,0], X_obs[:,1])
    return(CS)

def plot_basemap(longgrid, latgrid, ygrid, levels=None, X_obs=None):
    mp2 = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
            llcrnrlon=-180,urcrnrlon=180,resolution='c')
    mp2.drawcoastlines()
    mp2.drawparallels(np.arange(-90.,91.,30.))
    mp2.drawmeridians(np.arange(-180.,181.,60.))
    mp2.drawmapboundary(fill_color='white')

    if levels==None:
        levels = np.arange(-10,10,1)

    #mp2.scatter(longgrid_pred.flatten(), latgrid_pred.flatten())
    mp2.contourf(longgrid,latgrid, ygrid,15,levels=levels,
        cm = cm.delta)
    mp2.colorbar()
    mp2.fillcontinents('black')

    if X_obs is not None:
        plt.scatter(X_obs[:,0], X_obs[:,1])
    return(mp2)



def plot_gcm(vals, land_mask, X_obs=None, levels=None, basemap=True):
    lats = np.arange(-89.375,89.375,1.25)
    longs = np.arange(-180,178.75+0.1, 1.25)
    longgrid, latgrid = np.meshgrid(longs, lats)
    yplot = np.zeros(longgrid.size)+10000
    yplot[land_mask-1] = vals
    ygrid = yplot.reshape(lats.size,longs.size)
    if basemap:
        CS = plot_basemap(longgrid, latgrid, ygrid, levels=levels, X_obs=X_obs)
    else:
        CS = plot_wo_basemap(longgrid, latgrid, ygrid, levels=levels, X_obs=X_obs)
    return(CS)



def plot_map(longgrid=None, latgrid=None, vals=None, X_obs=None, levels=None, basemap=True):
    if levels==None:
        levels = np.arange(-10,12,1)
    if  vals is not None:
        ygrid = vals.reshape(latgrid.shape)
        if basemap:
            CS = plot_basemap(longgrid, latgrid, ygrid, levels=levels, X_obs=X_obs)
        else:
            CS = plot_wo_basemap(longgrid, latgrid, ygrid, levels=levels, X_obs=X_obs)
    elif basemap:
        mp2 = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90, llcrnrlon=-180,urcrnrlon=180,resolution='c')
        mp2.drawcoastlines()
        mp2.drawparallels(np.arange(-90.,91.,30.))
        mp2.drawmeridians(np.arange(-180.,181.,60.))
        mp2.drawmapboundary(fill_color='white')
        plt.xlabel('lon')
        plt.ylabel('lat')
        mp2.fillcontinents('green')

        if X_obs is not None:
            mp2.scatter(X_obs[:,0], X_obs[:,1])
        return(mp2)
    else:
        print('ERROR: Need basemap to plot just the points')
        return(None)
    return(CS)
"""
    mp2 = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90, llcrnrlon=-180,urcrnrlon=180,resolution='c')
    mp2.drawcoastlines()
    mp2.drawparallels(np.arange(-90.,91.,30.))
    mp2.drawmeridians(np.arange(-180.,181.,60.))
    mp2.drawmapboundary(fill_color='white')


    plt.xlabel('lon')
    plt.ylabel('lat')
    if vals is not None:
        mp2.contourf(longgrid,latgrid,vals.reshape(latgrid.shape),15,levels=levels,
    cm = cm.delta)
        mp2.colorbar()

    mp2.fillcontinents('green')

    if X_obs is not None:
        mp2.scatter(X_obs[:,0], X_obs[:,1])
    return(mp2)
"""

def ThinGrid(gcm_output, land_mask, thinby=2, plot=False):
    """
    We don't want to predict at all locations.
    There are 41184 locations in the GCM grid -
    - too many to want to produce a full covariance matrix for.
    Of these, 27186 are ocean, the others are land, but that is still too many.

    As a simple fix, let's just subset by taking every nth value.
    We will also ignore the land and not predict there.

    land_mask should give the location of all the ocean grid cells.


    This approach reduces the number of grid points by approx 1-1/thinby**2

    """
    # create the GCM grid
    lats_gcm = np.arange(-89.375,89.375,1.25) # Is this right?
    longs_gcm = np.arange(-180,178.75+0.1, 1.25)
    longgrid_gcm, latgrid_gcm = np.meshgrid(longs_gcm, lats_gcm)

    yplot = np.zeros(longgrid_gcm.size)-10000.
    yplot[land_mask-1] = gcm_output # IS THIS RIGHT?
    gcm_grid = yplot.reshape(lats_gcm.size,longs_gcm.size)

    keep_lats= np.arange(0,lats_gcm.size,thinby)
    keep_longs= np.arange(0,longs_gcm.size,thinby)

    longgrid_pred = longgrid_gcm[keep_lats,:][:,keep_longs]
    latgrid_pred = latgrid_gcm[keep_lats,:][:,keep_longs]
    gcm_grid_pred = gcm_grid[keep_lats,:][:,keep_longs]

    # create an array of Falses, change the ocean values to true, then thin
    land_mask_TF = np.zeros(longgrid_gcm.shape, dtype=bool)
    tmp = land_mask_TF.flatten()
    tmp[land_mask-1]=True
    land_mask_TF = tmp.reshape(longgrid_gcm.shape)
    land_mask_TF_pred = land_mask_TF[keep_lats,:][:,keep_longs]

    if plot:
        mp2 = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
        llcrnrlon=-180,urcrnrlon=180,resolution='c')
        mp2.drawcoastlines()
        mp2.drawparallels(np.arange(-90.,91.,30.))
        mp2.drawmeridians(np.arange(-180.,181.,60.))
        mp2.drawmapboundary(fill_color='white')
        #mp2.scatter(longgrid_pred.flatten(), latgrid_pred.flatten())
        mp2.scatter(longgrid_pred.flatten()[land_mask_TF_pred.flatten()], latgrid_pred.flatten()[land_mask_TF_pred.flatten()])
        plt.show()

    # create the X locations for the prediction grid - thinned and with land removed
    X_pred =np.column_stack((longgrid_pred.flatten()[land_mask_TF_pred.flatten()], latgrid_pred.flatten()[land_mask_TF_pred.flatten()]))

    # return the thinned GCM output.
    gcm_grid_pred_S = gcm_grid_pred.flatten()[land_mask_TF_pred.flatten()]
    if gcm_grid_pred_S.min()<-100.:
        print('Error we have not remvoved all the land successfully')
    return X_pred, gcm_grid_pred_S[:,None]
