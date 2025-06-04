from warnings import warn

from matplotlib.colors import LogNorm

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import numpy.ma as ma
import numba as nb

@nb.jit(nopython=True)
def calc_integrated_crosswind(a, b, c, d, xstar_end, nx):
    xstar = np.linspace(d, xstar_end, nx+2)
    xstar = xstar[1:]
    
    fstar = a * (xstar-d)**b * np.exp(-c / (xstar-d))
    mask = ~np.isnan(fstar)
    fstar = fstar[mask]
    xstar = xstar[mask]
    
    return xstar, fstar

@nb.jit(nopython=True)
def calc_scaled_sigystar(ac, bc, cc, xstar):
    sigystar_param = ac * np.sqrt(bc * xstar**2 / (1 + cc * xstar))
    return sigystar_param

@nb.jit(nopython=True)
def z0_calc_psif_xci_fci(z0, zm, ol, oln, h, xstar, fstar):
    if ol > 0 and ol < oln:
        psi_f = -5.3 * zm / ol
    else:
        x = (1 - 19.0 * zm/ol)**0.25
        psi_f = np.log((1 + x**2) / 2.) + 2. * np.log((1 + x) / 2.) - 2. * np.arctan(x) + np.pi/2
    
    x = xstar * zm / (1. - (zm / h)) * (np.log(zm / z0) - psi_f)
    
    if np.log(zm / z0) - psi_f > 0:
        x_ci = x
        f_ci = fstar / zm * (1. - (zm / h)) / (np.log(zm / z0) - psi_f)
    else:
        raise ValueError("np.log(zm / z0) - psi_f must be greater than 0.")
    return x, psi_f, x_ci, f_ci

@nb.jit(nopython=True)
def umean_calc_xci_fci(umean, zm, h, ustar, k, xstar, fstar):
    x = xstar * zm / (1. - zm / h) * (umean / ustar * k)
    if umean / ustar > 0:
        x_ci = x
        f_ci = fstar / zm * (1. - zm / h) / (umean / ustar * k)
    else:
        raise ValueError("umean / ustar must be greater than 0.")
    return x, x_ci, f_ci

@nb.jit(nopython=True)
def z0_calc_peak_loc(xstarmax, zm, z0, h, psi_f):
    return xstarmax * zm / (1. - (zm / h)) * (np.log(zm / z0) - psi_f)

@nb.jit(nopython=True)
def umean_calc_peak_loc(xstarmax, umean, zm, h, ustar, k):
    return xstarmax * zm / (1. - zm / h) * (umean / ustar * k)

@nb.jit(nopython=True)
def calc_fpos(f_ci, sigy, y_pos):
    f_pos = np.empty((len(f_ci), len(y_pos)))
    f_pos[:] = np.nan
    for ix in range(len(f_ci)):
        f_pos[ix,:] = f_ci[ix] * 1 / (np.sqrt(2 * np.pi) * sigy[ix]) * np.exp(-y_pos**2 / ( 2 * sigy[ix]**2))
    
    return f_pos

def FPP(
    zm: float | None = None, 
    z0: float | None = None, 
    umean: float | None = None, 
    h: float | None = None, 
    ol: float | None = None, 
    sigmav: float | None = None, 
    ustar: float | None = None, 
    wind_dir: float | None = None, 
    rs: float | int | list[int] | list[float] | list[int | float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 
    nx: int = 1000, 
    crop: bool = False, 
    fig: bool = False, 
    show_heatmap: bool = False,
):
    """Derive a flux footprint estimate based on the simple parameterisation FFP

    Args:
        zm (float | None): Measurement height above displacement height (i.e. z-d) [m].
        z0 (float | None, optional): Roughness length [m]; enter None if not known . Defaults to None.
        umean (float | None, optional): Mean wind speed at zm [m/s]; enter None if not known.\n
            Either z0 or umean is required. If both are given,
            z0 is selected to calculate the footprint. Defaults to None.
        h (float | None): Boundary layer height [m].
        ol (float | None): Obukhov length [m].
        sigmav (float | None): Standard deviation of lateral velocity fluctuations [ms-1].
        ustar (float | None): Friction velocity [ms-1].
        
        wind_dir (float | None, optional): wind direction in degrees (of 360) for rotation of the footprint. Defaults to None.
        rs (float | int | list[int | float], optional): Percentage of source area for which to provide contours, must be between 10% and 90%.\n
            Can be either a single value (e.g., "80") or a list of values (e.g., "[10, 20, 30]")\n
            Expressed either in percentages ("80") or as fractions of 1 ("0.8").\n
            Default is [10:10:80]. Set to "None" for no output of percentages
        rslayer (bool, optional): Calculate footprint even if zm within roughness sublayer: set rslayer = True\n
            Note that this only gives a rough estimate of the footprint as the model is not\n
            valid within the roughness sublayer.\n
            z0 is needed for estimation of the RS.. Defaults to False (i.e. no footprint for within RS).
        nx (int, optional): Integer scalar defining the number of grid elements of the scaled footprint.\n
            Large nx results in higher spatial resolution and higher computing time.\n
            Default is 1000, nx must be >=600.
        crop (bool, optional): Crop output area to size of the 80% footprint or the largest r given if crop=True. Defaults to False.
        fig (bool, optional): Plot an example figure of the resulting footprint (on the screen): set fig = True. Defaults to False.
        show_heatmap (bool, optional): Plot an example figure of the resulting footprint (on the screen): set show_heatmap = True. Defaults to False.

    Raises:
        ValueError: If any of zm, h, ol, sigmav, ustar is missing.
        ValueError: If z0 or umean is missing.
        ValueError: If zm <= 0.
        ValueError: If z0 is provided and less than 0.
        ValueError: If h <= 10.
        ValueError: If zm > h.
        ValueError: If z0 is provided and zm <= (12.5 * z0).
        ValueError: If sigmav <= 0.
        ValueError: If ustar <= 0.1
        ValueError: If wind_dir is provided and not between 0 and 360.
        ValueError: If nx is less than 600.
        ValueError: If both z0 and umean are provided.
        
    Returns:
        _type_: _description_
    """

    if None in [zm, h, ol, sigmav, ustar]:
        raise ValueError("zm, z0, h, ol, sigmav, ustar must all be provided")
    
    # If passed, can assume all values are valid.
    assert zm
    assert h
    assert ol
    assert ustar
    assert sigmav
    
    if not z0 and not umean:
        raise ValueError("z0 or umean must be provided")
    
    if z0 and umean:
        warn("Both z0 and umean provided. Using z0.", category=UserWarning)
        umean = None
    
    if zm <= 0.:
        raise ValueError("zm (measurement height) must be larger than zero.")
    if z0 and z0 <= 0.:
        raise ValueError("z0 (roughness length) must be larger than zero.")
    if h <= 10.:
        raise ValueError("h (PBL height) must be larger than 10 meters.")
    if zm > h:
        raise ValueError("zm (measurement height) must be smaller than h (PBL height).")
    if z0 and zm <= (12.5 * z0):
        raise ValueError("zm (measurement height) must be larger than 12.5 * z0 (roughness sub-layer).")
    if zm / ol <= -15.5:
        raise ValueError("zm/ol (measurement height/obukhov length) must be larger than -15.5.")
    if sigmav <= 0.:
        raise ValueError("sigmav (standard deviation of crosswind) must be larger than zero.")
    if ustar <= 0.1:
        raise ValueError("ustar (friction velocity) must be at least 0.1.")
    if wind_dir and (wind_dir > 360 or wind_dir < 0):
        raise ValueError("wind_dir (wind direction) must be between 0 and 360 degrees.")
    if nx and nx < 600:
        raise ValueError("nx (number of grid elements) must be at least 600.")
    
    #===========================================================================
    # Check rs
    if type(rs) is int or type(rs) is float:
        if 0.9 < rs < 1 or 90 < rs < 100:
            rs = 0.9
        rs = [rs]
    
    elif isinstance(rs, list):
        if np.max(rs) >= 1:
            rs = [r / 100 for r in rs]
        if np.max(rs) < 1.:
            rs = [r for r in rs if r <= 0.9]
        rs = list(np.sort(rs))
    
    else:
        raise TypeError("rs must be either a single number or a list of numbers.")
    
    # Define outputs
    x_ci = None
    x_ci_max = None
    f_ci = None
    x_2d = None
    y_2d = None
    f_2d = None
    fr = None
    xr = None
    yr = None
    
    #===========================================================================
    # Model parameters
    a = 1.4524
    b = -1.9914
    c = 1.4622
    d = 0.1359
    ac = 2.17 
    bc = 1.66
    cc = 20.0

    xstar_end = 30
    oln = 5000 # Limit to L for neutral scaling
    k = 0.4 # Von Karman

    #===========================================================================
    # Scaled X* for crosswind integrated footprint
    xstar, fstar = calc_integrated_crosswind(a, b, c, d, xstar_end, nx)
    
    # Scaled sig_y*
    sigystar_param = calc_scaled_sigystar(ac, bc, cc, xstar)
    psi_f = None

    #===========================================================================
    # Real scale x and f_ci
    if z0:
        x, psi_f, x_ci, f_ci = z0_calc_psif_xci_fci(z0, zm, ol, oln, h, xstar, fstar)
    if umean:
        x, x_ci, f_ci = umean_calc_xci_fci(umean, zm, h, ustar, k, xstar, fstar)

    # Maximum location of influence (peak location).
    xstarmax = -c / b + d
    if z0:
        x_ci_max = z0_calc_peak_loc(xstarmax, zm, z0, h, psi_f)
    if umean:
        x_ci_max = umean_calc_peak_loc(xstarmax, umean, zm, h, ustar, k)
    
    #Real scale sig_y
    if abs(ol) > oln:
        ol = -1E6
    if ol <= 0:   #convective
        scale_const = 1E-5 * abs(zm / ol)**(-1) + 0.80
    else:  #stable
        scale_const = 1E-5 * abs(zm / ol)**(-1) + 0.55
    if scale_const > 1:
        scale_const = 1.0
    sigy = sigystar_param / scale_const * zm * sigmav / ustar
    sigy[sigy < 0] = np.nan
    
    #Real scale f(x,y)
    if x_ci is None or f_ci is None:
        raise
    
    dx = x_ci[2] - x_ci[1]
    y_pos = np.arange(0, (len(x_ci) / 2.) * dx * 1.5, dx)
    #f_pos = np.full((len(f_ci), len(y_pos)), np.nan)
    f_pos = calc_fpos(f_ci, sigy, y_pos)
    
    #Complete footprint for negative y (symmetrical)
    y_neg = - np.fliplr(y_pos[None, :])[0]
    f_neg = np.fliplr(f_pos)
    y = np.concatenate((y_neg[0:-1], y_pos))
    f = np.concatenate((f_neg[:, :-1].T, f_pos.T)).T

    #Matrices for output
    x_2d = np.tile(x[:,None], (1,len(y)))
    y_2d = np.tile(y.T,(len(x),1))
    f_2d = f
    
    #===========================================================================
    # Derive footprint ellipsoid incorporating R% of the flux, if requested,
    # starting at peak value.
    dy = dx
    if rs:
        clevs = get_contour_levels(f_2d, dx, dy, rs)
        frs = [item[2] for item in clevs]
        xrs = []
        yrs = []
        for ix, fr in enumerate(frs):
            xr,yr = get_contour_vertices(x_2d, y_2d, f_2d, fr)
            if xr is None:
                frs[ix] = np.float64(np.nan)
            xrs.append(xr)
            yrs.append(yr)
    else:
        if crop:
            rs_dummy = 0.8 #crop to 80%
            clevs = get_contour_levels(f_2d, dx, dy, rs_dummy)
            xrs = []
            yrs = []
            xrs,yrs = get_contour_vertices(x_2d, y_2d, f_2d, clevs[0][2])
                
    #===========================================================================
    # Crop domain and footprint to the largest rs value
    if crop:
        # These are formally defined in previous hooks.
        assert xrs
        assert yrs
        
        xrs_crop = [x for x in xrs if x]
        yrs_crop = [x for x in yrs if x]
        if rs:
            dminx = np.floor(min(xrs_crop[-1]))
            dmaxx = np.ceil(max(xrs_crop[-1]))
            dminy = np.floor(min(yrs_crop[-1]))
            dmaxy = np.ceil(max(yrs_crop[-1]))
        else:
            dminx = np.floor(min(xrs_crop))
            dmaxx = np.ceil(max(xrs_crop))
            dminy = np.floor(min(yrs_crop))
            dmaxy = np.ceil(max(yrs_crop))
        jrange = np.where((y_2d[0] >= dminy) & (y_2d[0] <= dmaxy))[0]
        jrange = np.concatenate(([jrange[0]-1], jrange, [jrange[-1]+1]))
        jrange = jrange[np.where((jrange>=0) & (jrange<=y_2d.shape[0]-1))[0]]
        irange = np.where((x_2d[:,0] >= dminx) & (x_2d[:,0] <= dmaxx))[0]
        irange = np.concatenate(([irange[0]-1], irange, [irange[-1]+1]))
        irange = irange[np.where((irange>=0) & (irange<=x_2d.shape[1]-1))[0]]
        jrange = [[it] for it in jrange]
        x_2d = x_2d[irange,jrange]
        y_2d = y_2d[irange,jrange]
        f_2d = f_2d[irange,jrange]
        
    #===========================================================================
    #Rotate 3d footprint if requested
    if wind_dir is not None:			
        wind_dir = wind_dir * np.pi / 180.
        dist = np.sqrt(x_2d**2 + y_2d**2)
        angle = np.arctan2(y_2d, x_2d)
        x_2d = dist * np.sin(wind_dir - angle)
        y_2d = dist * np.cos(wind_dir - angle)

        if rs:
            # Formally defined in previous hooks.
            assert xrs
            assert yrs
            
            for ix, r in enumerate(rs):
                xr_lev = np.array([x for x in xrs[ix] if x is not None])    
                yr_lev = np.array([x for x in yrs[ix] if x is not None])    
                dist = np.sqrt(xr_lev**2 + yr_lev**2)
                angle = np.arctan2(yr_lev,xr_lev)
                xr = dist * np.sin(wind_dir - angle)
                yr = dist * np.cos(wind_dir - angle)
                xrs[ix] = list(xr) 
                yrs[ix] = list(yr) 

    
    output = {"x_ci_max": x_ci_max, "x_ci": x_ci, "f_ci": f_ci,
                "x_2d": x_2d, "y_2d": y_2d, "f_2d": f_2d,
                "rs": rs, "fr": frs, "xr": xrs, "yr": yrs}

    #===========================================================================
    # Plot footprint
    if fig:
        fig_out,ax = plot_footprint(x_2d=x_2d, y_2d=y_2d, fs=f_2d,
                                    show_heatmap=show_heatmap,clevs=frs)
        
    if rs:
        output["rs"] = rs
    return output

def get_contour_levels(f, dx, dy, rs=None):
    """Contour levels of f at percentages of f-integral given by rs"""
    #Check input and resolve to default levels in needed
    if not isinstance(rs, (int, float, list)):
        rs = list(np.linspace(0.10, 0.90, 9))
    if isinstance(rs, (int, float)):
        rs = [rs]

    #Levels
    pclevs = np.empty(len(rs))
    pclevs[:] = np.nan
    ars = np.empty(len(rs))
    ars[:] = np.nan

    sf = np.sort(f, axis=None)[::-1]
    msf = ma.masked_array(sf, mask=(np.isnan(sf) | np.isinf(sf))) #Masked array for handling potential nan
	
    csf = msf.cumsum().filled(np.nan)*dx*dy
    for ix, r in enumerate(rs):
        dcsf = np.abs(csf - r)
        pclevs[ix] = sf[np.nanargmin(dcsf)]
        ars[ix] = csf[np.nanargmin(dcsf)]

    return [(round(r, 3), ar, pclev) for r, ar, pclev in zip(rs, ars, pclevs)]

def get_contour_vertices(x, y, f, lev):
    import matplotlib.pyplot as plt

    cs = plt.contour(x,y, f, [lev])
    plt.close()
    segs = cs.allsegs[0][0]
    xr = [vert[0] for vert in segs]
    yr = [vert[1] for vert in segs]
    #Set contour to None if it's found to reach the physical domain
    if x.min() >= min(segs[:, 0]) or max(segs[:, 0]) >= x.max() or \
        y.min() >= min(segs[:, 1]) or max(segs[:, 1]) >= y.max():
        return [None, None]

    return [xr, yr]   # x,y coords of contour points.	


def plot_footprint(x_2d, y_2d, fs, clevs=None, show_heatmap=True, normalize=None, 
                    colormap=None, line_width=0.5, iso_labels=None):
    '''Plot footprint function and contours if request'''

    # If input is a list of footprints, don't show footprint but only contours,
    # with different colors
    if isinstance(fs, list):
        show_heatmap = False
    else:
        fs = [fs]

    if not colormap: 
        colormap = cm.get_cmap('jet')
    # Define colors for each contour set
    cs = [colormap(ix) for ix in np.linspace(0, 1, len(fs))]

    # Initialize figure
    fig, ax = plt.subplots(figsize=(12, 10))
    # fig.patch.set_facecolor('none')
    # ax.patch.set_facecolor('none')

    if clevs:
        # Temporary patch for pyplot.contour requiring contours to be in ascending orders
        clevs = clevs[::-1]

        # Eliminate contour levels that were set to None
        # (e.g. because they extend beyond the defined domain)
        clevs = [clev for clev in clevs if clev]

        # Plot contour levels of all passed footprints
        # Plot isopleth
        levs = [clev for clev in clevs]
        for f, c in zip(fs, cs):
            cc = [c]*len(levs)
            if show_heatmap:
                cp = ax.contour(x_2d, y_2d, f, levs, colors = 'w', linewidths=line_width)
            else:
                cp = ax.contour(x_2d, y_2d, f, levs, colors = cc, linewidths=line_width)
            # Isopleth Labels
            if iso_labels:
                pers = [str(int(clev[0]*100))+'%' for clev in clevs]
                fmt = {}
                for lvl,s in zip(cp.levels, pers):
                    fmt[lvl] = s
                plt.clabel(cp, cp.levels[:], inline=1, fmt=fmt, fontsize=7) # type: ignore

    # plot footprint heatmap if requested and if only one footprint is passed
    if show_heatmap:
        if normalize == 'log':
            norm = LogNorm()
        else:
            norm = None

        for f in fs:
            pcol = plt.pcolormesh(x_2d, y_2d, f, cmap=colormap, norm=norm)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.gca().set_aspect('equal', 'box')

        cbar = fig.colorbar(pcol, shrink=1.0, format='%.3e')
        cbar.set_label('Flux contribution', color = 'k')
    plt.show()

    return fig, ax
