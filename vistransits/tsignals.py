import os, sys, pdb
import atpy
import ephem
import numpy as np
import scipy.integrate
import tutilities
import matplotlib
import matplotlib.pyplot as plt

G = 6.67428e-11 # gravitational constant in m^3/kg^-1/s^-2
HPLANCK = 6.62607e-34 # planck's constant in J*s
C = 2.99792e8 # speed of light in vacuum in m/s
KB = 1.3806488e-23 # boltzmann constant in J/K
RGAS = 8.314 # gas constant in J/mol/K
RSUN = 6.9551e8 # solar radius in m
RJUP = 7.1492e7 # jupiter radius in m
RSAT = 5.8232e7 # saturn radius in m
RNEP = 2.4622e7 # neptune radius in m
REARTH = 6.371e6 # earth radius in m
MJUP = 1.89852e27 # jupiter mass in kg
MSAT = 5.683e26 # saturn mass in kg
MNEP = 1.024e26 # neptune mass in kg
MEARTH = 5.972e24 # earth mass in kg
AU2M = 1.49598e11 # au to metres conversion factor
MUJUP = 2.22e-3 # jupiter atmosphere mean molecular weight in kg/mole
TR_TABLE = 'exoplanets_transiting.fits' # fits file for known exoplanets that transit


def scatter_plots( download_latest=True, label_top_ranked=0, outfile='', output_format='pdf', \
                   transmission_ref_wav_um=0.6, thermal_ref_wav_um=1.4 ):
    """
    Aim = To produce the following plots:
    1. period versus radius, with marker size indicating transmission amplitude (i.e. change in 
       transit depth corresponding to one atmospheric scale height) and colour indicating 
       equilibrium temperature.
    2. transmission amplitude versus host star magnitude with marker size indicating planet
       mass and colour indicating equilibrium temperature.
    """

    plt.ioff()
    matplotlib.rc( 'text', usetex=True )

    # Convert the wavelengths from microns to metres:
    transmission_ref_wav_m = transmission_ref_wav_um*( 1e-6 )
    thermal_ref_wav_m = thermal_ref_wav_um*( 1e-6 )
    wav_K_m = 2.2e-6
    wav_V_m = 0.55e-6

    figw = 14
    figh = 10

    fig1 = plt.figure( figsize=[ figw, figh ] )
    fig2 = plt.figure( figsize=[ figw, figh ] )
    fig3 = plt.figure( figsize=[ figw, figh ] )
    fig4 = plt.figure( figsize=[ figw, figh ] )

    hbuff = 0.05
    vbuff = 0.05
    xlow = 2.*hbuff
    ylow = 2.*vbuff
    axw = 1-5*hbuff
    axh = 1-2.5*vbuff
    ax1 = fig1.add_axes( [ xlow, ylow, axw, axh ], xscale='log', yscale='linear' )
    ax2 = fig2.add_axes( [ xlow, ylow, axw, axh ], xscale='log', yscale='linear' )
    ax3 = fig3.add_axes( [ xlow, ylow, axw, axh ], xscale='linear', yscale='linear' )
    ax4 = fig4.add_axes( [ xlow, ylow, axw, axh ], xscale='log', yscale='linear' )

    ax1.xaxis.set_major_formatter( matplotlib.ticker.LogFormatterMathtext() )
    ax2.xaxis.set_major_formatter( matplotlib.ticker.LogFormatterMathtext() )
    ax3.xaxis.set_major_formatter( matplotlib.ticker.ScalarFormatter() )
    #ax4.xaxis.set_major_formatter( matplotlib.ticker.ScalarFormatter() )

    sigtype = 'transmission'
    z = get_planet_properties(  sigtype=sigtype, download_latest=download_latest, exclude_kois=False )
    names = z['names']
    ras = z['ras']
    decs = z['decs']
    vmags = z['vmags']
    ksmags = z['ksmags']
    rstars = z['rstars']
    teffs = z['teffs']
    rplanets = z['rplanets']
    mplanets = z['mplanets']
    per = z['periods']
    tpeqs = z['tpeqs']
    Hatms = z['Hatms']
    nplanets_all = len( names )

    ixsc = []
    per_other = []
    rplanets_other = []
    for i in range( nplanets_all ):
        if ( np.isfinite( vmags[i] ) )*( np.isfinite( mplanets[i] ) )*\
           ( np.isfinite( Hatms[i] ) )*( np.isfinite( tpeqs[i] ) )==True:
            ixsc += [ i ]
        else:
            try:
                per_other += [ per[i] ]
                rplanets_other += [ rplanets[i] ]
            except:
                pass
    per_other = np.array( per_other )
    rplanets_other = np.array( rplanets_other )

    ixsc = np.array( ixsc )  
    names = z['names'][ixsc]
    ras = z['ras'][ixsc]
    decs = z['decs'][ixsc]
    vmags = z['vmags'][ixsc]
    ksmags = z['ksmags'][ixsc]
    rstars = z['rstars'][ixsc]
    teffs = z['teffs'][ixsc]
    rplanets = z['rplanets'][ixsc]
    mplanets = z['mplanets'][ixsc]
    per = z['periods'][ixsc]
    tpeqs = z['tpeqs'][ixsc]
    Hatms = z['Hatms'][ixsc]
    nplanets_all = len( names )

    nplanets = len( names )
    ms_min = 8
    ms_max = 7*ms_min
    ms_range = ms_max - ms_min
    ms_min = 6
    ms_max = 5*ms_min
    ms_range34 = ms_max - ms_min
   
    # Transmission signal in terms of transit depth variation:
    n = 3 # number of atmosphere scale heights for transmission signal amplitude
    delta_tr = (1e6)*2*n*( rplanets*RJUP )*Hatms/( ( rstars*RSUN )**2 )

    # Thermal emission signal assuming blackbody radiation:
    bratio = planck( thermal_ref_wav_m, tpeqs )/planck( thermal_ref_wav_m, teffs )

    # Convert the above to the ratio of the measured fluxes:
    RpRs = ( rplanets*RJUP )/( rstars*RSUN )
    fratio = (1e6)*bratio*( RpRs**2 ) # the (1e6) is to put in units of ppm

    # Using the known Ks ( ~2.2microns ) magnitude as a reference,
    # approximate the magnitude in the current wavelength of interest:
    kratio = planck( thermal_ref_wav_m, teffs )/planck( wav_K_m, teffs )
    mag_star = ksmags - 2.5*np.log10( kratio )
    # Note that this assumes the magnitude of the reference star that
    # the magnitudes such as t.KS are defined wrt is approximately the
    # same at wav and 2.2 microns.

    # Convert the approximate magnitude to an unnormalised stellar flux 
    # in the wavelength of interest:
    flux_star_unnorm = 10**( -mag_star / 2.5 )

    # Use the fact that the signal-to-noise is:
    #   signal:noise = f_planet / sqrt( flux_star )
    #                = sqrt( f_star ) * fratio
    # but note that we still have the normalising
    # constant to be taken care of (see next):
    snr_unnorm_th = np.sqrt( flux_star_unnorm ) * fratio

    # Using the known Ks magnitude of the target, estimate the
    # unnormalised signal-to-noise ratio of the change in transit
    # depth that we would measure in the visible and IR separately:
    bratio = planck( transmission_ref_wav_m, teffs )/planck( wav_V_m, teffs )
    mag = vmags - 2.5*np.log10( bratio )
    flux_unnorm = 10**( -mag/2.5 )
    snr_unnorm_tr = np.sqrt( flux_unnorm )*delta_tr

    ixs = np.argsort( snr_unnorm_tr )
    delta_tr_min = delta_tr.min()
    delta_tr_max = delta_tr.max()
    delta_tr_range = float( delta_tr_max - delta_tr_min )
    mplanets_min = mplanets.min()
    mplanets_max = mplanets.max()
    mplanets_range = float( mplanets_max - mplanets_min )
    rplanets_min = rplanets.min()
    rplanets_max = rplanets.max()
    rplanets_range = float( rplanets_max - rplanets_min )
    tpeq_min = tpeqs.min()
    tpeq_max = tpeqs.max()
    tpeq_range = tpeq_max - tpeq_min
    mew = 1
    text_fs = 18
    cmap = matplotlib.cm.jet
    for i in range( nplanets ):
        tpeq_i = ( tpeqs[ixs][i]-tpeq_min )/tpeq_range
        c = cmap( tpeq_i )
        frac = ( delta_tr[ixs][i]-delta_tr_min )/delta_tr_range
        ms = ms_min + frac*ms_range
        ax1.plot( [ per[ixs][i] ], [ rplanets[ixs][i] ], 'o', ms=ms, mfc=c, mec='k', mew=mew, zorder=i+3 )
    ckoi = 0.7*np.ones( 3 )
    ax1.plot( per_other, rplanets_other, 'o', mfc=ckoi, mec=ckoi, ms=4, zorder=0 )

    ixs = np.isfinite( vmags )
    nplanets_vmag_finite = ixs.sum()
    for j in range( nplanets_vmag_finite ):
        tpeq_j = ( tpeqs[ixs][j]-tpeq_min )/tpeq_range
        c = cmap( tpeq_j )
        frac = ( delta_tr[ixs][j]-delta_tr_min )/delta_tr_range
        ms = ms_min + frac*ms_range
        ax2.plot( [ mplanets[ixs][j] ], [ vmags[ixs][j] ], 'o', zorder=2+j, \
                  ms=ms, mfc=c, mec='k', mew=mew  )
        frac = ( rplanets[ixs][j]-rplanets_min )/rplanets_range
        ms = ms_min + frac*ms_range34
        ax3.plot( [ delta_tr[ixs][j] ], [ vmags[ixs][j] ], 'o', ms=ms, mfc=c, mec='k', mew=mew  )    
    ixs = np.isfinite( ksmags )
    nplanets_ksmag_finite = ixs.sum()
    for j in range( nplanets_ksmag_finite ):
        tpeq_j = ( tpeqs[ixs][j]-tpeq_min )/tpeq_range
        c = cmap( tpeq_j )
        frac = ( rplanets[ixs][j]-rplanets_min )/rplanets_range
        ms = ms_min + frac*ms_range34
        ax4.plot( [ fratio[ixs][j] ], [ ksmags[ixs][j] ], 'o', ms=ms, mfc=c, mec='k', mew=mew  )    

    # Label top ranked planets if requested:
    if label_top_ranked>0:
        ixs1 = np.isfinite( vmags )
        ixs2 = np.argsort( snr_unnorm_tr[ixs1] )[::-1]
        for i in range( int( label_top_ranked ) ):
            ax1.text( per[ixs1][ixs2][i], rplanets[ixs1][ixs2][i], names[ixs1][ixs2][i], \
                      fontsize=text_fs, rotation=0., zorder=nplanets+1+i, \
                      horizontalalignment='left', verticalalignment='center' )
            ax2.text( mplanets[ixs1][ixs2][i], vmags[ixs1][ixs2][i], names[ixs1][ixs2][i], \
                      fontsize=text_fs, rotation=0., \
                      horizontalalignment='left', verticalalignment='center' )            
            ax3.text( delta_tr[ixs1][ixs2][i], vmags[ixs1][ixs2][i], names[ixs1][ixs2][i], \
                      fontsize=text_fs, rotation=0., \
                      horizontalalignment='left', verticalalignment='center' )            
        ixs1 = np.isfinite( ksmags )
        ixs2 = np.argsort( snr_unnorm_th[ixs1] )[::-1]
        for i in range( int( label_top_ranked ) ):
            ax4.text( fratio[ixs1][ixs2][i], ksmags[ixs1][ixs2][i], names[ixs1][ixs2][i], \
                      fontsize=text_fs, rotation=0., \
                      horizontalalignment='left', verticalalignment='center' )            

    x1_min = per.min()
    x1_max = per.max()
    x1_range = x1_max - x1_min
    y1_min = rplanets.min()
    y1_max = rplanets.max()
    y1_range = y1_max - y1_min

    x2_min = mplanets.min()
    x2_max = mplanets.max()
    x2_range = x2_max - x2_min
    y2_min = vmags[ixs].min()
    y2_max = vmags[ixs].max()
    y2_range = y2_max - y2_min

    x3_min = delta_tr.min()
    x3_max = delta_tr.max()
    x3_range = x3_max - x3_min
    y3_min = vmags[ixs].min()
    y3_max = vmags[ixs].max()
    y3_range = y3_max - y3_min

    x4_min = fratio.min()
    x4_max = fratio.max()
    x4_range = x4_max - x4_min
    y4_min = ksmags[ixs].min()
    y4_max = ksmags[ixs].max()
    y4_range = y4_max - y4_min

    xl1 = 0.8*x1_min
    xu1 = x1_max+0.1*x1_range
    yl1 = 0.
    yu1 = y1_max+0.1*y1_range
    ax1.set_xlim( [ xl1, xu1 ] )
    ax1.set_ylim( [ yl1, yu1 ] )

    hlw = 2
    hlc = np.array( [ 153., 76., 0. ] )/256.

    xtext = 0.93
    delytext = 0.01
    ax1.axhline( REARTH/RJUP, ls='-', color=hlc, lw=hlw, zorder=1 )
    ytext = ( REARTH/RJUP - yl1 )/( yu1 - yl1 )
    ax1.text( xtext, ytext+delytext, '$R_E$', fontsize=text_fs, color=hlc, zorder=2, transform=ax1.transAxes )

    ax1.axhline( RNEP/RJUP, ls='-', lw=hlw, color=hlc, zorder=1 )
    ytext = ( RNEP/RJUP - yl1 )/( yu1 - yl1 )
    ax1.text( xtext, ytext+delytext, '$R_N$', fontsize=text_fs, color=hlc, zorder=2, transform=ax1.transAxes )

    ax1.axhline( RSAT/RJUP, ls='-', lw=hlw, color=hlc, zorder=1 )
    ytext = ( RSAT/RJUP - yl1 )/( yu1 - yl1 )
    ax1.text( xtext, ytext+delytext, '$R_S$', fontsize=text_fs, color=hlc, zorder=2, transform=ax1.transAxes )

    ax1.axhline( 1., ls='-', lw=hlw, color=hlc, zorder=1 )
    ytext = ( 1. - yl1 )/( yu1 - yl1 )
    ax1.text( xtext, ytext+delytext, '$R_J$', fontsize=text_fs, color=hlc, zorder=2, transform=ax1.transAxes )

    xl2 = x2_min
    xu2 = x2_max+0.1*x2_range
    yl2 = y2_min-0.1*y2_range
    yu2 = y2_max+0.1*y2_range
    ax2.set_xlim( [ xl2, xu2 ] )
    ax2.set_ylim( [ yl2, yu2 ] )

    vlw = 2
    vlc = np.array( [ 153., 76., 0. ] )/256.

    ytext = 0.03
    delxtext = 0.005
    ax2.axvline( MEARTH/MJUP, ls='-', color=vlc, lw=vlw, zorder=1 )
    xtext = ( np.log10( MEARTH/MJUP ) - np.log10( xl2 ) )/( np.log10( xu2 ) - np.log10( xl2 ) )
    ax2.text( xtext+delxtext, ytext, '$M_E$', fontsize=text_fs, color=vlc, zorder=2, transform=ax2.transAxes, \
              horizontalalignment='left', verticalalignment='bottom' )

    ax2.axvline( MNEP/MJUP, ls='-', lw=vlw, color=vlc, zorder=1 )
    xtext = ( np.log10( MNEP/MJUP ) - np.log10( xl2 ) )/( np.log10( xu2 ) - np.log10( xl2 ) )
    ax2.text( xtext+delxtext, ytext, '$M_N$', fontsize=text_fs, color=vlc, zorder=2, transform=ax2.transAxes, \
              horizontalalignment='left', verticalalignment='bottom' )

    ax2.axvline( MSAT/MJUP, ls='-', lw=vlw, color=vlc, zorder=1 )
    xtext = ( np.log10( MSAT/MJUP ) - np.log10( xl2 ) )/( np.log10( xu2 ) - np.log10( xl2 ) )
    ax2.text( xtext+delxtext, ytext, '$M_S$', fontsize=text_fs, color=vlc, zorder=2, transform=ax2.transAxes, \
              horizontalalignment='left', verticalalignment='bottom' )

    ax2.axvline( 1., ls='-', lw=vlw, color=vlc, zorder=1 )
    xtext = ( np.log10( 1. ) - np.log10( xl2 ) )/( np.log10( xu2 ) - np.log10( xl2 ) )
    ax2.text( xtext+delxtext, ytext, '$M_J$', fontsize=text_fs, color=vlc, zorder=2, transform=ax2.transAxes, \
              horizontalalignment='left', verticalalignment='bottom' )

    xl3 = 0.
    xu3 = x3_max+0.1*x3_range
    ax3.set_xlim( [ xl3, xu3 ] )
    ax3.set_ylim( [ y3_min-0.1*y3_range, y3_max+0.1*y3_range ] )
    x3 = np.r_[ xl3:xu3:1j*500 ]

    ixs = np.isfinite( ksmags )
    xl4 = 0.05*fratio[ixs].max()
    xu4 = x4_max+0.1*x4_range
    ax4.set_xlim( [ xl4, xu4 ] )
    yl4 = y4_min-0.1*y4_range
    yu4 = y4_max+0.1*y4_range
    ax4.set_ylim( [ yl4, yu4 ] )
    x4 = np.r_[ np.log10( xl4 ):np.log10( xu4 ):1j*500 ]

    # Transmission signal contours:
    vref = 8.0
    SNref = 5.0
    Aref = 400
    SNs = np.array( [ 5.0, 3.0, 1.0 ] )
    for SN in SNs: 
        y3 = -5*np.log10( ( Aref/x3 )*( SN/SNref ) ) + vref
        ax3.plot( x3, y3, color='k', lw=2, zorder=0 )
        xtext = x3[400]
        ytext = y3[400] + 0.02
        text_str = '$\\textnormal{S/N}$'+'$={{{0:.0f}}}$'.format( SN )
        ax3.text( xtext, ytext, text_str, fontsize=text_fs, rotation=10, zorder=1, \
                  horizontalalignment='center', verticalalignment='bottom' )

    # Emission signal contours:
    ksref = 8.0
    SNref = 5.0
    Aref = 1000
    SNs = np.array( [ 5.0, 3.0, 1.0 ] )
    for SN in SNs: 
        y4 = -5*np.log10( ( Aref/( 10**x4 ) )*( SN/SNref ) ) + ksref
        ax4.plot( 10**x4, y4, color='k', lw=2, zorder=0 )
        xtext = 10**x4[460]
        ytext = y4[460] + 0.02
        text_str = '$\\textnormal{S/N}$'+'$={{{0:.0f}}}$'.format( SN )
        ax4.text( xtext, ytext, text_str, fontsize=text_fs, rotation=27, zorder=1, \
                  horizontalalignment='left', verticalalignment='bottom' )#, transform=ax4.transAxes )


    # Plot marker size indicators outside the axis limits
    # so that they only appear in the legend:
    mskey_x1 = 0.2*x1_min
    mskey_y1 = y1_min-10*y1_range
    mskey_x2 = 0.2*x2_min
    mskey_y2 = y2_min-10*y2_range
    mskey_x3 = 0.2*x3_min
    mskey_y3 = y3_min-10*y3_range
    mskey_x4 = 0.2*x4_min
    mskey_y4 = y4_min-10*y4_range

    frac = ( 100-delta_tr_min )/delta_tr_range
    ms12 = ms_min + frac*ms_range
    frac = ( 1-rplanets_min )/rplanets_range
    ms3jup = ms_min + frac*ms_range34
    frac = ( REARTH/RJUP-rplanets_min )/rplanets_range
    ms3earth = ms_min + frac*ms_range34

    ax1.plot( [ mskey_x1 ], [ mskey_y1 ], 'o', ms=ms12, mec='k', mfc='none', \
              mew=mew, label='$\\textnormal{100ppm}$' )
    ax2.plot( [ mskey_x2 ], [ mskey_y2 ], 'o', ms=ms12, mec='k', mfc='none', \
              mew=mew, label='$\\textnormal{100ppm}$' )
    ax3.plot( [ mskey_x3 ], [ mskey_y3 ], 'o', ms=ms3jup, mec='k', mfc='none', \
              mew=mew, label='$1R_J$' )
    ax3.plot( [ mskey_x3 ], [ mskey_y3 ], 'o', ms=ms3earth, mec='k', mfc='none', \
              mew=mew, label='$1R_E$' )
    ax4.plot( [ mskey_x4 ], [ mskey_y4 ], 'o', ms=ms3jup, mec='k', mfc='none', \
              mew=mew, label='$1R_J$' )
    ax4.plot( [ mskey_x4 ], [ mskey_y4 ], 'o', ms=ms3earth, mec='k', mfc='none', \
              mew=mew, label='$1R_E$' )
    ax1.legend( numpoints=1, fontsize=text_fs )
    ax2.legend( numpoints=1, fontsize=text_fs )
    ax3.legend( numpoints=1, fontsize=text_fs, borderpad=1, loc='lower right' )
    ax4.legend( numpoints=1, fontsize=text_fs, borderpad=1, loc='lower right' )

    tpeq_ticks_low = 0
    while tpeq_ticks_low<tpeq_min:
        tpeq_ticks_low += 50
    tpeq_ticks_upp = tpeq_max + 50
    while tpeq_ticks_upp>tpeq_max:
        tpeq_ticks_upp -= 50
    tpeq_ticks = np.arange( tpeq_ticks_low, tpeq_ticks_upp+50, 500 )
    cb_width = 0.5*hbuff
    cb_height = axh
    side_buffer = 0.8*hbuff
    cb_xlow = xlow + axw + 1.5*side_buffer
    cb_ylow = ylow
    cb_axis1 = fig1.add_axes( [ cb_xlow, cb_ylow, cb_width, cb_height] )
    cb_axis2 = fig2.add_axes( [ cb_xlow, cb_ylow, cb_width, cb_height] )
    cb_axis3 = fig3.add_axes( [ cb_xlow, cb_ylow, cb_width, cb_height] )
    cb_axis4 = fig4.add_axes( [ cb_xlow, cb_ylow, cb_width, cb_height] )
    cb_norm = matplotlib.colors.Normalize( vmin=tpeq_min, vmax=tpeq_max )
    cb1 = matplotlib.colorbar.ColorbarBase( cb_axis1, cmap=cmap, norm=cb_norm, orientation='vertical')
    cb2 = matplotlib.colorbar.ColorbarBase( cb_axis2, cmap=cmap, norm=cb_norm, orientation='vertical')
    cb3 = matplotlib.colorbar.ColorbarBase( cb_axis3, cmap=cmap, norm=cb_norm, orientation='vertical')
    cb4 = matplotlib.colorbar.ColorbarBase( cb_axis4, cmap=cmap, norm=cb_norm, orientation='vertical')
    cb1.solids.set_rasterized( True )
    cb2.solids.set_rasterized( True )
    cb3.solids.set_rasterized( True )
    cb4.solids.set_rasterized( True )
    cb1.solids.set_edgecolor( 'face' )
    cb2.solids.set_edgecolor( 'face' )
    cb3.solids.set_edgecolor( 'face' )
    cb4.solids.set_edgecolor( 'face' )
    cb1.set_ticks( tpeq_ticks )
    cb2.set_ticks( tpeq_ticks )
    cb3.set_ticks( tpeq_ticks )
    cb4.set_ticks( tpeq_ticks )
    
    tick_fs = 28
    ax1.tick_params( labelsize=tick_fs )
    cb_axis1.tick_params( labelsize=tick_fs )
    ax2.tick_params( labelsize=tick_fs )
    cb_axis2.tick_params( labelsize=tick_fs )
    ax3.tick_params( labelsize=tick_fs )
    cb_axis3.tick_params( labelsize=tick_fs )
    ax4.tick_params( labelsize=tick_fs )
    cb_axis4.tick_params( labelsize=tick_fs )

    label_fs = 28
    xlabel_x = xlow + 0.5*axw
    xlabel_y = 0.2*vbuff
    xlabel_str1 = '$\\textnormal{Period (day)}$'
    fig1.text( xlabel_x, xlabel_y, xlabel_str1, fontsize=label_fs, rotation=0., \
               horizontalalignment='center', verticalalignment='bottom' )
    xlabel_str2 = '$M_p \ (M_J)$'
    fig2.text( xlabel_x, xlabel_y, xlabel_str2, fontsize=label_fs, rotation=0., \
               horizontalalignment='center', verticalalignment='bottom' )
    xlabel_str3 = '$\\textnormal{Transmission amplitude (ppm)}$'
    fig3.text( xlabel_x, xlabel_y, xlabel_str3, fontsize=label_fs, rotation=0., \
               horizontalalignment='center', verticalalignment='bottom' )
    xlabel_str4 = '$\\textnormal{Thermal emission amplitude (ppm)}$'
    fig4.text( xlabel_x, xlabel_y, xlabel_str4, fontsize=label_fs, rotation=0., \
               horizontalalignment='center', verticalalignment='bottom' )
    ylabel_x = xlow - hbuff
    ylabel_y = ylow + 0.5*axh
    ylabel_str1 = '$R_p \ (R_J)$'
    fig1.text( ylabel_x, ylabel_y, ylabel_str1, fontsize=label_fs, rotation=90., \
               horizontalalignment='right', verticalalignment='center' )
    ylabel_str2 = '$V \ (\\textnormal{mag})$'
    fig2.text( ylabel_x, ylabel_y, ylabel_str2, fontsize=label_fs, rotation=90., \
               horizontalalignment='right', verticalalignment='center' )
    ylabel_str3 = '$V \ (\\textnormal{mag})$'
    fig3.text( ylabel_x, ylabel_y, ylabel_str3, fontsize=label_fs, rotation=90., \
               horizontalalignment='right', verticalalignment='center' )
    ylabel_str4 = '$K_S \ (\\textnormal{mag})$'
    fig4.text( ylabel_x, ylabel_y, ylabel_str4, fontsize=label_fs, rotation=90., \
               horizontalalignment='right', verticalalignment='center' )

    cblabel_x = cb_xlow #+ 0.5*cb_width
    cblabel_y = cb_ylow - 0.6*vbuff
    cblabel_str = '$T \ (\\textnormal{K})$'
    fig1.text( cblabel_x, cblabel_y, cblabel_str, fontsize=label_fs, rotation=0., \
               horizontalalignment='left', verticalalignment='top' )
    fig2.text( cblabel_x, cblabel_y, cblabel_str, fontsize=label_fs, rotation=0., \
               horizontalalignment='left', verticalalignment='top' )
    fig3.text( cblabel_x, cblabel_y, cblabel_str, fontsize=label_fs, rotation=0., \
               horizontalalignment='left', verticalalignment='top' )
    
    ofigname1 = 'transmission_plot1.{0}'.format( output_format )
    ofigname2 = 'transmission_plot2.{0}'.format( output_format )
    ofigname3 = 'transmission_plot3.{0}'.format( output_format )
    ofigname4 = 'emission_plot1.{0}'.format( output_format )
    fig1.savefig( ofigname1 )
    fig2.savefig( ofigname2 )
    fig3.savefig( ofigname3 )
    fig4.savefig( ofigname4 )
    print '\nSaved:\n{0}\n{1}\n{2}\n{3}'.format( ofigname1, ofigname2, ofigname3, ofigname4 )

    plt.ion()
    matplotlib.rc( 'text', usetex=False )

    return None


def reflection( wav_meas_um=[ 0.55, 0.80 ], wav_ref_um=0.55, obj_ref='HD189733b', \
                outfile='signals_reflection.txt', download_latest=True, \
                include_thermal_contribution=False ):
    """
    Generates a table of properties relevant to eclipse measurements at a specified
    wavelength for all known transiting exoplanets, assuming reflected starlight only.
    Eclipse depths are quoted assuming a geometric albedo of 1. The expected signal
    for lower albedos scales linearly as a function of the geometric albedo, according
    to:  delta_flux = Ag*( ( RpRs/aRs )**2 )

    Perhaps the most useful column of the output table is the one that gives the
    expected signal-to-noise of the eclipse **relative to the signal-to-noise for a
    reference planet at a reference wavelength**. A relative signal-to-noise is used
    due to the unknown normalising constant when working with magnitudes at arbitrary
    wavelengths.
    """
    
    # Convert the wavelengths from microns to metres:
    wav_meas_m = np.array( wav_meas_um )*( 1e-6 )
    wav_meas_m_cuton = wav_meas_m[0]
    wav_meas_m_cutoff = wav_meas_m[1]
    wav_meas_m_cent = np.mean( wav_meas_m )
    wav_ref_m = wav_ref_um*( 1e-6 )

    # Central wavelength of V band in m:
    wav_V_m = 0.55e-6

    # Get table data for planets that we have enough information on:
    t = filter_table( sigtype='reflection', download_latest=download_latest )
    nplanets_all = len( t.NAME )

    # Exclude unconfirmed planets:
    ixsc = []
    for i in range( nplanets_all ):
        if t.NAME[i].find( 'KOI' )<0:
            ixsc += [ i ]
    ixsc = np.array( ixsc )
    nplanets = len( t.NAME[ixsc] )

    # Calculate the equilibrium temperatures for all planets on list:
    tpeq = Teq( t )

    # Calculate other basic quantities:
    RpRs = ( ( t.R * RJUP)/( t.RSTAR * RSUN ) )
    aRs = ( ( t.A * AU2M)/( t.RSTAR * RSUN ) )
    Ag = 1.0 # for reference

    # Convert the above to ratio of planet-to-stellar flux:
    fratio_reflection = Ag*( ( RpRs/aRs )**2. )

    # Calculate thermal emission signals for each planet:
    Bp = np.zeros( nplanets )
    Bs = np.zeros( nplanets )
    for i in range( nplanets ):
        Bp[i] = scipy.integrate.quad( planck, wav_meas_m_cuton, wav_meas_m_cutoff, args=( tpeq[i] ) )[0]
        Bs[i] = scipy.integrate.quad( planck, wav_meas_m_cuton, wav_meas_m_cutoff, args=( t.TEFF[i] ) )[0]
    fratio_thermal = ( RpRs**2. )*( Bp/Bs )
    if include_thermal_contribution==True:
        fratio_total = fratio_reflection + fratio_thermal
    else:
        fratio_total = fratio_reflection
    thermal_frac = fratio_thermal/fratio_reflection
    # NOTE: Be aware that thermal_frac is actually unphysical. It assumes
    # zero Bond albedo to calculate the temperature, but then the reflection
    # signal assumes a geometric albedo of 1. Also, note that the equilibrium
    # temperature assumes uniform redistribution of heat; to convert to
    # zero heat redistribution, multiply fratio_thermal by 8/3.

    # Using the known V ( ~0.55microns ) magnitude as a reference,
    # approximate the magnitude in the current wavelength of interest:
    vratio = planck( wav_meas_m_cent, t.TEFF )/planck( wav_ref_m, t.TEFF )
    mag_star = t.V - 2.5 * np.log10( vratio )

    # Convert the approximate magnitude to an unnormalised stellar flux 
    # in the wavelength of interest:
    flux_star_unnorm = 10**( -mag_star / 2.5 )

    # Use the fact that the signal-to-noise is:
    #   signal:noise = f_planet / sqrt( flux_star )
    #                = sqrt( f_star ) * fratio
    # but note that we still have the normalising
    # constant to be taken care of (see next):
    snr_unnorm = np.sqrt( flux_star_unnorm )*fratio_reflection

    # The signal-to-noise ratio is still not normalised, so we need to repeat
    # the above for another reference star; in fact, we want to compare the
    # signal-to-noise for our target at the measurement wavelength with the
    # signal-to-noise of a reference object at some reference wavelength which
    # may or may not be the same as the measurement wavelength; we choose the
    # reference system and reference wavelength to be 'familiar' signals so
    # that we can get a better feel for the size of our target signal at the
    # wavelength we actually want to measure:
    names = []
    for name in t.NAME:
        names += [ name.replace( ' ', '' ) ]
    obj_ref = obj_ref.replace( ' ', '' )
    names = np.array( names, dtype=str )
    ii = ( names==obj_ref )
    if ii.max()==False:
        print 'Could not match reference {0} to any systems on list'
        print '... using HD189733b instead'
        obj_ref = 'HD189733b'
        ii = ( names==obj_ref )

    # Calculate the fraction drop in flux that will occur for the reference
    # system, independent of wavelength:
    RpRs_ref = ( t.R[ii]*RJUP )/( t.RSTAR[ii]*RSUN )
    aRs_ref = ( t.A[ii]*AU2M )/( t.RSTAR[ii]*RSUN )
    fratio_reflection_ref = Ag*( ( RpRs_ref/aRs_ref )**2. )

    # To convert this into a signal-to-noise, we now need to account for the
    # brightness of the reference system at the reference wavelength. To do
    # this, we use the known V band magnitude and then extrapolate from that 
    # to the flux at the reference wavelength assuming blackbody radiation:
    vratio_ref = planck( wav_ref_m, t.TEFF[ii] ) / planck( wav_V_m, t.TEFF[ii] )
    mag_ref = t.V[ii] - 2.5*np.log10( vratio_ref )
    flux_ref = 10**( -mag_ref/2.5 )

    # Then we calculate the signal to noise of the measurement at the
    # reference wavelength for the reference system:
    snr_ref = np.sqrt( flux_ref )*fratio_reflection_ref

    # Reexpress the signal-to-noise of our target at the measurement wavelength
    # as a scaling of the signal-to-noise of the reference system at the
    # reference wavelength:
    snr_norm = snr_unnorm / snr_ref

    # Rearrange the targets in order of the most promising:
    s = np.argsort( snr_norm[ixsc] )
    s = s[::-1]

    # Open the output file and write the column headings:
    ofile = open( outfile, 'w' )
    header = make_header_reflection( nplanets, wav_meas_m_cuton, wav_meas_m_cutoff, wav_ref_m, obj_ref )
    ofile.write( header )

    # Write the output rows to file and save:
    for j in range( nplanets ):
        i = s[j]
        outstr = make_outstr_reflection( j+1, t.NAME[ixsc][i], t.RA[ixsc][i], t.DEC[ixsc][i], \
                                         t.V[ixsc][i], t.TEFF[ixsc][i], t.RSTAR[ixsc][i], \
                                         t.R[ixsc][i], t.A[ixsc][i], tpeq[ixsc][i], \
                                         RpRs[ixsc][i], fratio_total[ixsc][i], \
                                         snr_norm[ixsc][i], thermal_frac[ixsc][i] )
        ofile.write( outstr )
    ofile.close()
    print 'Saved output in {0}'.format( outfile )
    
    return outfile

    
def thermal( wav_meas_um=2.2, wav_ref_um=2.2, obj_ref='WASP-19 b', \
             outfile='signals_thermal.txt', download_latest=True ):
    """
    Generates a table of properties relevant to eclipse measurements at a specified
    wavelength for all known transiting exoplanets, assuming thermal emission only.
    Basic temperature equilibrium is assumed, with idealised Planck blackbody spectra.
    Perhaps the most useful column of the output table is the one that gives the
    expected signal-to-noise of the eclipse **relative to the signal-to-noise for a
    reference planet at a reference wavelength**. A relative signal-to-noise is used
    due to the unknown normalising constant when working with magnitudes at arbitrary
    wavelengths.
    """

    # Convert the wavelengths from microns to metres:
    wav_meas_m = wav_meas_um*( 1e-6 )
    wav_ref_m = wav_ref_um*( 1e-6 )

    # Central wavelength of K band in m:
    wav_K_m = 2.2e-6

    # Get table data for planets that we have enough information on:
    t = filter_table( sigtype='thermal', download_latest=download_latest )
    nplanets_all = len( t.NAME )

    # Exclude unconfirmed planets:
    ixsc = []
    for i in range( nplanets_all ):
        if t.NAME[i].find( 'KOI' )<0:
            ixsc += [ i ]
    ixsc = np.array( ixsc )
    nplanets = len( t.NAME[ixsc] )

    # Calculate the equilibrium temperatures for all planets on list:
    tpeq = Teq( t )

    # Calculate the radii ratios for each planet:
    RpRs = ( t.R*RJUP )/( t.RSTAR*RSUN )

    # Assuming black body radiation, calculate the ratio between the
    # energy emitted by the planet per m^2 of surface per second,
    # compared to the star:
    bratio = planck( wav_meas_m, tpeq )/planck( wav_meas_m, t.TEFF )

    # Convert the above to the ratio of the measured fluxes:
    fratio = bratio*( RpRs**2 )

    # Using the known Ks ( ~2.2microns ) magnitude as a reference,
    # approximate the magnitude in the current wavelength of interest:
    kratio = planck( wav_meas_m, t.TEFF )/planck( wav_K_m, t.TEFF )
    mag_star = t.KS - 2.5*np.log10( kratio )
    # Note that this assumes the magnitude of the reference star that
    # the magnitudes such as t.KS are defined wrt is approximately the
    # same at wav and 2.2 microns.

    # Convert the approximate magnitude to an unnormalised stellar flux 
    # in the wavelength of interest:
    flux_star_unnorm = 10**( -mag_star / 2.5 )

    # Use the fact that the signal-to-noise is:
    #   signal:noise = f_planet / sqrt( flux_star )
    #                = sqrt( f_star ) * fratio
    # but note that we still have the normalising
    # constant to be taken care of (see next):
    snr_unnorm = np.sqrt( flux_star_unnorm ) * fratio

    # The signal-to-noise ratio is still not normalised, so we need to repeat
    # the above for another reference star; in fact, we want to compare the
    # signal-to-noise for our target at the measurement wavelength with the
    # signal-to-noise of a reference object at some reference wavelength which
    # may or may not be the same as the measurement wavelength; we choose the
    # reference system and reference wavelength to be 'familiar' signals so
    # that we can get a better feel for the size of our target signal at the
    # wavelength we actually want to measure:
    names = []
    for name in t.NAME:
        names += [ name.replace( ' ', '' ) ]
    obj_ref = obj_ref.replace( ' ', '' )
    names = np.array( names, dtype=str )
    ii = ( names==obj_ref )
    bratio_ref = planck( wav_ref_m, tpeq[ii] )/planck( wav_ref_m, t.TEFF[ii] )
    fratio_ref = bratio_ref*( ( t.R[ii]*RJUP )/( t.RSTAR[ii]*RSUN ) )**2
    kratio_ref = planck( wav_ref_m, t.TEFF[ii] )/planck( wav_K_m, t.TEFF[ii] )
    mag_ref = t.KS[ii] - 2.5*np.log10( kratio_ref )
    flux_ref = 10**( -mag_ref/2.5 )

    # Then we calculate the signal to noise of the measurement at the
    # reference wavelength for the reference system:
    snr_ref = np.sqrt( flux_ref )*fratio_ref

    # Reexpress the signal-to-noise of our target at the measurement wavelength
    # as a scaling of the signal-to-noise of the reference system at the
    # reference wavelength:
    snr_norm = snr_unnorm/snr_ref

    # Rearrange the targets in order of the most promising:
    s = np.argsort( snr_norm[ixsc] )
    s = s[::-1]

    # Open the output file and write the column headings:
    ofile = open( outfile, 'w' )
    header = make_header_thermal( nplanets, wav_meas_m, wav_ref_m, obj_ref )
    ofile.write( header )
    
    # Write the output rows to file and save:
    for j in range( nplanets ):
        i = s[j]
        outstr = make_outstr_thermal( j+1, t.NAME[ixsc][i], t.RA[ixsc][i], t.DEC[ixsc][i], \
                                      t.KS[ixsc][i], t.TEFF[ixsc][i], t.RSTAR[ixsc][i], \
                                      t.R[ixsc][i], t.A[ixsc][i], tpeq[ixsc][i], \
                                      fratio[ixsc][i], snr_norm[ixsc][i] )
        ofile.write( outstr )
    ofile.close()
    print 'Saved output in {0}'.format( outfile )
    
    return outfile


def transmission( wav_vis_um=0.7, wav_ir_um=2.2, wav_ref_um=2.2, obj_ref='WASP-19 b', \
                  outfile='signals_transits.txt', download_latest=True ):
    """
    Generates a table of properties relevant to transmission spectroscopy measurements
    at a visible wavelength and an IR wavelength for all known transiting exoplanets.
    Quoted transmission signals are the flux change corresponding to a change in the
    planetary radius of n=1 atmospheric gas scale heights.

    Perhaps the most useful column of the output table is the one that gives the expected
    signal-to-noise of the transmission signal **relative to the signal-to-noise for a
    reference planet at a reference wavelength**. A relative signal-to-noise is used
    due to the unknown normalising constant when working with magnitudes at arbitrary
    wavelengths.
    """

    # Calculate transmission signal as the variation in flux drop
    # caused by a change in the effective planetary radius by n=1
    # atmospheric scale heights; the size of the signal scales
    # approximately linearly with the number of scale heights used,
    # making it simple to extrapolate from the output this produces:
    n = 1

    # Convert the wavelengths from microns to metres:
    wav_vis_m = wav_vis_um*( 1e-6 )
    wav_ir_m = wav_ir_um*( 1e-6 )
    wav_ref_m = wav_ref_um*( 1e-6 )

    # Central wavelengths of V and K bands in m:
    wav_V_m = 0.55e-6
    wav_K_m = 2.2e-6

    z = get_planet_properties(  sigtype='transmission', download_latest=download_latest )
    names = z['names']
    ras = z['ras']
    decs = z['decs']
    vmags = z['vmags']
    ksmags = z['ksmags']
    rstars = z['rstars']
    teffs = z['teffs']
    rplanets = z['rplanets']
    mplanets = z['mplanets']
    tpeqs = z['tpeqs']
    Hatms = z['Hatms']

    # Check to make sure we have both a V and Ks
    # magnitude for the reference star:
    ix = ( names==obj_ref )
    if ( np.isfinite( ksmags[ix] )==False ) or ( np.isfinite( vmags[ix] )==False ):
        print '\n\nPlease select a different reference star for which we have both a V and Ks magnitude\n\n'
        return None

    # Calculate the radii ratios for each planet:
    RpRs = ( rplanets*RJUP )/( rstars*RSUN )

    # Calculate the approximate change in transit depth for a
    # wavelength range where some species in the atmosphere
    # increases the opacity of the planetary limb for an additional
    # 2.5 (i.e. 5/2) scale heights:
    depth_tr = RpRs**2.
    delta_tr = 2*n*( rplanets*RJUP )*Hatms/( ( rstars*RSUN )**2 )

    # Using the known Ks magnitude of the target, estimate the
    # unnormalised signal-to-noise ratio of the change in transit
    # depth that we would measure in the visible and IR separately:
    bratio = planck( wav_vis_m, teffs )/planck( wav_K_m, teffs )
    mag = ksmags - 2.5*np.log10( bratio )
    flux_unnorm = 10**( -mag/2.5 )
    snr_unnorm_vis = np.sqrt( flux_unnorm )*delta_tr

    bratio = planck( wav_ir_m, teffs )/planck( wav_K_m, teffs )
    mag = ksmags - 2.5*np.log10( bratio )
    flux_unnorm = 10**( -mag/2.5 )
    snr_unnorm_ir = np.sqrt( flux_unnorm )*delta_tr

    # Repeat the above using the known V band for any that didn't
    # have known KS magnitudes:
    ixs = ( np.isfinite( ksmags )==False )
    
    bratio = planck( wav_vis_m, teffs[ixs] )/planck( wav_V_m, teffs[ixs] )
    mag = vmags[ixs] - 2.5*np.log10( bratio )
    flux_unnorm = 10**( -mag/2.5 )
    snr_unnorm_vis[ixs] = np.sqrt( flux_unnorm )*delta_tr[ixs]

    bratio = planck( wav_ir_m, teffs[ixs] )/planck( wav_V_m, teffs[ixs] )
    mag = ksmags[ixs] - 2.5*np.log10( bratio )
    flux_unnorm = 10**( -mag/2.5 )
    snr_unnorm_ir[ixs] = np.sqrt( flux_unnorm )*delta_tr[ixs]

    # The signal-to-noise ratio is still not normalised, so we need to repeat
    # the above for another reference star; seeing as the normalising constant
    # It might be useful to put the signal-to-noise in different units, namely,
    # compare the size of the current signal to that of another reference target 
    # at some reference wavelength. Basically repeat the above for the reference:
    names_new = []
    for name in names:
        names_new += [ name.replace( ' ', '' ) ]
    obj_ref = obj_ref.replace( ' ', '' )
    names = np.array( names_new, dtype=str )
    ii = ( names==obj_ref )
    delta_tr_ref = 2*n*( rplanets[ii]*RJUP )*Hatms[ii]/( ( rstars[ii]*RSUN )**2 )
    kratio_ref = planck( wav_ref_m, teffs[ii] )/planck( wav_K_m, teffs[ii] )

    mag_ref_ir = ksmags[ii] - 2.5*np.log10( kratio_ref )
    flux_ref_ir = 10**( -mag_ref_ir/2.5 )
    snr_ref_ir = np.sqrt( flux_ref_ir )*delta_tr_ref

    mag_ref_vis = vmags[ii] - 2.5*np.log10( kratio_ref )
    flux_ref_vis = 10**( -mag_ref_vis/2.5 )
    snr_ref_vis = np.sqrt( flux_ref_vis )*delta_tr_ref

    # Reexpress the signal-to-noise of our target as a scaling of the reference
    # signal-to-noise:
    snr_norm_vis = snr_unnorm_vis/snr_ref_vis
    snr_norm_ir = snr_unnorm_ir/snr_ref_ir

    # Rearrange the confirmed targets in order of the most promising:
    s = np.argsort( snr_norm_vis )
    s = s[::-1]

    # Open the output file and write the column headings:
    nplanets = len( names )
    ofile = open( outfile, 'w' )
    header = make_header_transmission( nplanets, wav_vis_m, wav_ir_m, wav_ref_m, obj_ref, n )
    ofile.write( header )
    
    for j in range( nplanets ):
        i = s[j]
        if np.isfinite( vmags[i] ):
            v = '{0:.1f}'.format( vmags[i] )
        else:
            v = '-'
        if np.isfinite( ksmags[i] ):
            ks = '{0:.1f}'.format( ksmags[i] )
        else:
            ks = '-'
        outstr = make_outstr_transmission( j+1, names[i], ras[i], decs[i], \
                                           v, ks, rstars[i], rplanets[i], tpeqs[i], \
                                           Hatms[i], depth_tr[i], delta_tr[i], \
                                           snr_norm_vis[i], snr_norm_ir[i] )
        ofile.write( outstr )
    ofile.close()
    print 'Saved output in {0}'.format( outfile )
    
    return outfile


def get_planet_properties( sigtype='transmission', download_latest=True, exclude_kois=True ):

    # Make we exclude table rows that do not contain
    # all the necessary properties:
    t = filter_table( sigtype=sigtype, download_latest=download_latest, strict=False )
    nplanets_all = len( t.NAME )

    # Exclude unconfirmed planets:
    if exclude_kois==True:
        ixsc = []
        for i in range( nplanets_all ):
            if t.NAME[i].find( 'KOI' )<0:
                ixsc += [ i ]
        ixsc = np.array( ixsc )
    else:
        ixsc = np.arange( nplanets_all )
    nplanets = len( t.NAME[ixsc] )

    # Calculate the approximate planetary equilibrium temperature:
    tpeq = Teq( t )

    # Calculate the gravitational accelerations at the surface zero-level:
    MPLANET = np.zeros( nplanets )
    for i in range( nplanets ):
        try:
            MPLANET[i] = np.array( t.MASS[ixsc][i], dtype=float )
        except:
            MPLANET[i] = np.array( t.MSINI[ixsc][i], dtype=float )
            print t.NAME[ixsc][i]
    little_g = G*MPLANET*MJUP/( ( t.R[ixsc]*RJUP )**2 )

    # Calculate the atmospheric scale height in metres; note that
    # we use RGAS instead of KB because MUJUP is **per mole**:
    Hatm = RGAS*tpeq[ixsc]/MUJUP/little_g

    z = { 'names':t.NAME[ixsc], 'ras':t.RA[ixsc], 'decs':t.DEC[ixsc], \
          'vmags':t.V[ixsc], 'ksmags':t.KS[ixsc], 'rstars':t.RSTAR[ixsc], \
          'teffs':t.TEFF[ixsc], 'rplanets':t.R[ixsc], 'mplanets':MPLANET, \
          'periods':t.PER[ixsc], 'tpeqs':tpeq[ixsc], 'Hatms':Hatm }

    return z 



def filter_table( sigtype=None, download_latest=True, strict=True ):
    """
    Identify entries from the containing values for all of the
    required properties.
    """

    if ( os.path.isfile( TR_TABLE )==False )+( download_latest==True ):
        tutilities.download_data()
    t = atpy.Table( TR_TABLE )
    t = t.where( np.isfinite( t.RSTAR ) ) # stellar radius
    t = t.where( np.isfinite( t.R ) ) # planetary radius
    t = t.where( np.isfinite( t.A ) ) # semimajor axis
    t = t.where( np.isfinite( t.TEFF ) ) # stellar effective temperature
    if strict==True:
        if sigtype=='thermal':
            t = t.where( np.isfinite( t.KS ) ) # stellar Ks magnitude
        elif sigtype=='transmission':
            t = t.where( np.isfinite( t.KS ) + np.isfinite( t.V ) ) # stellar Ks and/or V magnitude
            try:
                t = t.where( ( np.isfinite( t.MSINI ) * ( t.MSINI>0 ) + \
                             ( np.isfinite( t.MASS ) * ( t.MASS>0 ) ) ) ) # MSINI and MASS available
            except:
                t = t.where( ( np.isfinite( t.MSINI ) * ( t.MSINI>0 ) ) ) # only MSINI available
        elif sigtype=='reflection':
            t = t.where( np.isfinite( t.V ) ) # stellar V magnitude

    return t

def Teq( table ):
    """
    Calculates the equilibrium temperature of the planet, assuming zero
    albedo and homogeneous circulation. Assumes filter_table() has already
    been done to ensure all necessary properties are available for the
    calculation, i.e. t.RSTAR, t.A, t.TSTAR.
    """

    Rstar = table.RSTAR * RSUN
    a = table.A * AU2M
    Tstar = table.TEFF
    Teq = np.sqrt( Rstar / 2. / a ) * Tstar

    return Teq

def planck( wav, temp ):
    """
    Evaluates the Planck function for given values of wavelength
    and temperature. Wavelength should be provided in metres and
    temperature should be provided in Kelvins.
    """
    
    term1 = 2 * HPLANCK * ( C**2. ) / ( wav**5. )
    term2 = np.exp( HPLANCK * C / KB / wav / temp ) - 1
    bbflux = term1 / term2

    return bbflux


def make_header_thermal( nplanets, wav_m, wav_ref_m, obj_ref ):
    """
    Generates a header in a string format that can be written to
    the top of the eclipses output file.
    """

    col1a = 'Rank'.center( 4 )
    col1b = ' '.center( 4 )
    col2a = 'Name'.center( 15 )
    col2b = ' '.center( 15 )
    col3a = 'RA'.center( 11 )
    col3b = ' '.center( 11 )
    col4a = 'Dec'.center( 12 )
    col4b = ' '.center( 12 )
    col5a = 'K '.rjust( 5 )
    col5b = '(mag)'.rjust( 5 )
    col6a = 'Tstar'.center( 6 )
    col6b = '(K)'.center( 6 )
    col7a = 'Rstar'.center( 6 )
    col7b = '(Rsun)'.center( 6 )
    col8a = 'Rp'.center( 6 )
    col8b = '(Rjup)'.center( 6 )
    col9a = 'a'.center( 5 )
    col9b = '(AU)'.center( 5 )
    col10a = 'Tpeq'.rjust( 5 )
    col10b = '(K)'.rjust( 5 )
    col11a = 'Fp/Fs'.rjust( 6 )
    col11b = '(1e-4)'.rjust( 6 )
    col12a = 'S/N'.rjust( 6 )
    col12b = ' '.rjust( 6 )
    colheadingsa = '# {0}{1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11}\n'\
                  .format( col1a, col2a, col3a, col4a, col5a, col6a, \
                          col7a, col8a, col9a, col10a, col11a, col12a )
    colheadingsb = '# {0}{1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11}\n'\
                  .format( col1b, col2b, col3b, col4b, col5b, col6b, \
                          col7b, col8b, col9b, col10b, col11b, col12b )
    nchar = max( [ len( colheadingsa ), len( colheadingsb ) ] )

    header  = '{0}\n'.format( '#'*nchar )
    header += '# Eclipse estimates at {0:.2f} micron arranged in order of increasing\n'.format( (1e6)*wav_m )
    header += '# detectability for {0:d} known transiting exoplanets\n#\n'.format( nplanets )
    header += '# Values for \'K\', \'Tstar\', \'Rstar\', \'Rp\', \'a\' are taken from the literature \n#\n'
    header += '# Other quantities are derived as follows:\n#\n'
    header += '#  \'Tpeq\' is the equilibrium effective temperature of the planet assuming \n'
    header += '#    absorption of all incident star light and uniform redistribution\n'
    header += '#       --->  Tpeq = np.sqrt( Rstar / 2. / a ) * Tstar \n#\n'
    header += '#  \'Fp/Fs\' is the ratio of the planetary dayside flux to the stellar flux\n'
    header += '#       --->  Fp/Fs = ( P(Tplanet)/P(Tstar) ) * ( Rplanet / Rstar )**2 \n'
    header += '#                 where P is the Planck function\n#\n'
    header += '#  \'S/N\' is the signal-to-noise estimated using the known stellar brightness\n'
    header += '#    and expressed relative to the S/N expected for {0} at {1:.2f} micron\n'\
              .format( obj_ref, (1e6)*wav_ref_m )
    header += '#       --->  S/N_ref = Fp_ref / sqrt( Fs_ref )\n'
    header += '#       --->  S/N_targ = Fp / sqrt( Fs ) \n'
    header += '#       --->  S/N = S/N_target / S/N_ref\n#\n'
    header += '{0}\n#\n'.format( '#'*nchar )
    header += colheadingsa
    header += colheadingsb
    header += '{0}{1}\n'.format( '#', '-'*( nchar-1 ) )

    return header
    

def make_header_reflection( nplanets, wav_cuton_m, wav_cutoff_m, wav_ref_m, obj_ref ):
    """
    Generates a header in a string format that can be written to
    the top of the eclipses output file.
    """

    col1a = 'Rank'.center( 4 )
    col1b = ' '.center( 4 )
    col2a = 'Name'.center( 15 )
    col2b = ' '.center( 15 )
    col3a = 'RA'.center( 11 )
    col3b = ' '.center( 11 )
    col4a = 'Dec'.center( 12 )
    col4b = ' '.center( 12 )
    col5a = 'V '.rjust( 5 )
    col5b = '(mag)'.rjust( 5 )
    col6a = 'Tstar'.center( 6 )
    col6b = '(K)'.center( 6 )
    col7a = 'Rstar'.center( 6 )
    col7b = '(Rsun)'.center( 6 )
    col8a = 'Rp'.center( 6 )
    col8b = '(Rjup)'.center( 6 )
    col9a = 'a'.center( 5 )
    col9b = '(AU)'.center( 5 )
    col10a = 'Tpeq'.rjust( 5 )
    col10b = '(K)'.rjust( 5 )
    col11a = 'RpRs'.rjust( 6 )
    col11b = ' '.rjust( 6 )
    col12a = 'Fp/Fs'.rjust( 7 )
    col12b = '(ppm)'.rjust( 7 )
    col13a = 'S/N'.rjust( 5 )
    col13b = ' '.rjust( 5 )
    col14a = 'Thermal'.rjust( 9 )
    col14b = '(percent)'.rjust( 9 )
    colheadingsa = '# {0}{1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13}\n'\
                  .format( col1a, col2a, col3a, col4a, col5a, col6a, col7a, \
                           col8a, col9a, col10a, col11a, col12a, col13a, col14a )
    colheadingsb = '# {0}{1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13}\n'\
                  .format( col1b, col2b, col3b, col4b, col5b, col6b, col7b, \
                           col8b, col9b, col10b, col11b, col12b, col13b, col14b )
    nchar = max( [ len( colheadingsa ), len( colheadingsb ) ] )

    header  = '{0}\n'.format( '#'*nchar )
    header += '# Reflection estimates at {0:.2f}-{1:.2f} micron arranged in order of increasing\n'\
              .format( (1e6)*wav_cuton_m, (1e6)*wav_cutoff_m )
    header += '# detectability for {0:d} known transiting exoplanets\n#\n'.format( nplanets )
    header += '# Values for \'V\', \'Tstar\', \'Rstar\', \'Rp\', \'a\' are taken from the literature \n#\n'
    header += '# Other quantities are derived as follows:\n#\n'
    header += '#  \'Tpeq\' is the equilibrium effective temperature of the planet assuming \n'
    header += '#    absorption of all incident star light and uniform redistribution\n'
    header += '#       --->  Tpeq = np.sqrt( Rstar/2./a )*Tstar \n'
    header += '#  \'Fp/Fs\' is the ratio of the reflected+thermal planetary flux to the total stellar\n'
    header += '#  flux assuming a geometric albedo of 1, i.e. Ag=1, across all wavelengths for the \n'
    header += '#  reflected component, where:\n'
    header += '#       --->  Fp/Fs = Ag*( ( ( Rplanet/Rstar )/( a/Rstar ) )**2 ) \n'
    header += '#  and the thermal contribution is calculated by assuming that the planet radiates as a\n'
    header += '#  blackbody with temperature Tpeq, i.e. assuming Bond albedo of 0 and uniform heat distribution\n#\n'
    header += '#  \'Thermal\' is the fraction of the planetary flux due to thermal emission rather than\n'
    header += '#  reflected light from the planetary atmosphere, i.e. total = thermal + reflected\n#\n'
    header += '#  Note that strictly speaking the above is unphysical, because the equilibrium temperature\n'
    header += '#  is calculated assuming zero albedo, but the reflection is calculated assuming\n'
    header += '#  a geometric albedo that is unity across all wavelengths.\n#\n'
    header += '#  \'S/N\' is the signal-to-noise estimated using the known stellar brightness\n'
    header += '#    and expressed relative to the S/N expected for {0} at {1:.2f} micron\n'\
              .format( obj_ref, (1e6)*wav_ref_m )
    header += '#       --->  S/N_ref = Fp_ref / sqrt( Fs_ref )\n'
    header += '#       --->  S/N_targ = Fp / sqrt( Fs ) \n'
    header += '#       --->  S/N = S/N_target / S/N_ref\n#\n'
    header += '{0}\n#\n'.format( '#'*nchar )
    header += colheadingsa
    header += colheadingsb
    header += '{0}{1}\n'.format( '#', '-'*( nchar-1 ) )

    return header
    

def make_header_transmission( nplanets, wav_vis_m, wav_ir_m, wav_ref_m, obj_ref, n ):
    """
    Generates a header in a string format that can be written to
    the top of the transits output file.
    """

    col1a = 'Rank'.center( 4 )
    col1b = ''.center( 4 )
    col2a = 'Name'.center( 15 )
    col2b = ''.center( 15 )
    col3a = 'RA'.center( 11 )
    col3b = ''.center( 11 )
    col4a = '  Dec'.center( 12 )
    col4b = ''.center( 12 )
    col5a = 'V '.rjust( 5 )
    col5b = '(mag)'.rjust( 5 )
    col6a = 'K '.rjust( 5 )
    col6b = '(mag)'.center( 5 )
    col7a = 'Rstar'.center( 6 )
    col7b = '(Rsun)'.center( 6 )
    col8a = 'Rp'.center( 6 )
    col8b = '(Rjup)'.center( 6 )
    col9a = 'Tpeq'.rjust( 5 )
    col9b = '(K)'.rjust( 5 )
    col10a = 'H '.rjust( 5 )
    col10b = '(km)'.rjust( 5 )
    col11a = 'Depth'.rjust( 6 )
    col11b = '(1e-2)'.rjust( 6 )
    col12a = 'Delta'.rjust( 6 )
    col12b = '(1e-4)'.rjust( 6 )
    col13a = 'S/N '.rjust( 6 )
    col13b = '(vis)'.rjust( 6 )
    col14a = 'S/N '.rjust( 6 )
    col14b = '(IR)'.rjust( 6 )
    colheadingsa = '# {0}{1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13}\n'\
                  .format( col1a, col2a, col3a, col4a, col5a, col6a, col7a, col8a, col9a, col10a, \
                           col11a, col12a, col13a, col14a )
    colheadingsb = '# {0}{1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13}\n'\
                  .format( col1b, col2b, col3b, col4b, col5b, col6b, col7b, col8b, col9b, col10b, \
                           col11b, col12b, col13b, col14b )
    nchar = max( [ len( colheadingsa ), len( colheadingsb ) ] )

    header  = '{0}\n'.format( '#'*nchar )
    header += '# Transit variation estimates at visible ({0:.2f} micron) and IR  ({01:.2f} micron) wavelengths\n'\
              .format( (1e6)*wav_vis_m, (1e6)*wav_ir_m )
    header += '# arranged in order of increasing detectability in the visible wavelength for {0:d} known\n'\
              .format( nplanets )
    header += '# transiting exoplanets\n#\n'
    header += '# SNR is given relative to the approximate signal for {0} expected at {1:.2f} microns \n#\n'\
              .format( obj_ref, (1e6)*wav_ref_m )
    header += '# Values for \'V\', \'K\', \'Rstar\', \'Rp\' are taken from the literature. \n#\n'
    header += '# Other quantities are derived as follows:\n#\n'
    header += '#  \'Tpeq\' is the equilibrium effective temperature of the planet assuming \n'
    header += '#    absorption of all incident star light and uniform redistribution\n'
    header += '#       --->  Tpeq = np.sqrt( Rstar / 2. / a ) * Tstar \n#\n'
    header += '#  \'H\' is the approximate atmospheric scale height \n'
    header += '#       --->   H = Rgas * Teq / mu / g \n'
    header += '#                where Rgas is the gas constant, mu is the atmospheric mean \n'
    header += '#                molecular weight and g is the acceleration due to gravity \n#\n'
    header += '#  \'Depth\' is approximate transit depth\n'
    header += '#       --->   Depth = ( Rplanet / Rstar )**2 \n#\n'
    header += '#  \'Delta\' is the relative signal variation due to a change in planetary \n'
    header += '#     radius of n={0} times the atmospheric scale height \'H\'\n'.format( n )
    header += '#       --->   Delta = 2 * n * Rplanet * H / ( Rstar**2 ) \n#\n'.format( n )
    header += '#  \'S/N\' is the signal-to-noise of the transmission signal estimated using \n'
    header += '#    the known stellar brightness and expressed relative to the S/N expected \n'
    header += '#    for {0} at {1:.2f} micron\n'.format( obj_ref, (1e6)*wav_ref_m )
    header += '#       --->  S/N_ref = Delta_ref / sqrt( F_ref )  \n'
    header += '#       --->  S/N_targ = Delta_targ / sqrt( F_targ )  \n'
    header += '#       --->  S/N = S/N_target / S/N_ref\n#\n'
    header += '{0}\n#\n'.format( '#'*nchar )
    header += colheadingsa
    header += colheadingsb
    header += '{0}{1}\n'.format( '#', '-'*( nchar-1 ) )

    return header

    
def make_outstr_reflection( rank, name, ra, dec, vmag, tstar, rstar, rp, a, tpeq, RpRs, fratio, snr_norm, thermal_frac ):
    """
    Takes quantities that will be written to the eclipses output and formats them nicely.
    """

    name = name.replace( ' ', '' )

    # Convert the RA to hh:mm:ss.s and Dec to dd:mm:ss.s:
    ra_str = str( ephem.hours( ra ) )
    dec_str = str( ephem.degrees( dec ) )


    if ra.replace( ' ','' )=='':
        ra_str = '?'
    elif len( ra_str )==10:
        ra_str = '0{0}'.format( ra_str )

    if dec.replace( ' ','' )=='':
        dec_str = '?'
    else:
        if float( dec )>=0:
            dec_str = '+{0}'.format( dec_str )
        if len( dec_str )==10:
            dec_str = '{0}0{1}'.format( dec_str[0], dec_str[1:] )

    rank_str = '{0}'.format( rank ).rjust( 4 )
    name_str = '{0}'.format( name ).center( 15 )
    ra_str = '{0}'.format( ra_str.replace( ':', ' ' ) ).center( 11 )
    dec_str = '{0}'.format( dec_str.replace( ':', ' ' ) ).rjust( 12 )
    vmag_str = '{0:.1f}'.format( vmag ).rjust( 5 )
    tstar_str = '{0:4d}'.format( int( tstar ) ).center( 6 )
    rstar_str = '{0:.1f}'.format( rstar ).center( 6 )
    rp_str = '{0:.1f}'.format( rp ).center( 6 )
    a_str = '{0:.3f}'.format( a ).center( 5 )
    tpeq_str = '{0:4d}'.format( int( tpeq ) ).center( 6 )
    RpRs_str = '{0:.4f}'.format( RpRs ).center( 5 )    
    fratio_str = '{0:4d}'.format( int( (1e6)*fratio ) ).rjust( 6 )
    snr_str = '{0:.2f}'.format( snr_norm ).rjust( 6 )
    thermal_str = '{0:.2f}'.format( (1e2)*thermal_frac ).rjust( 7 )
    outstr = '  {0}{1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13}\n'\
             .format( rank_str, \
                      name_str, \
                      ra_str, \
                      dec_str, \
                      vmag_str, \
                      tstar_str, \
                      rstar_str, \
                      rp_str, \
                      a_str, \
                      tpeq_str, \
                      RpRs_str, \
                      fratio_str, \
                      snr_str, \
                      thermal_str )

    return outstr


def make_outstr_thermal( rank, name, ra, dec, kmag, tstar, rstar, rp, a, tpeq, fratio, snr_norm  ):
    """
    Takes quantities that will be written to the eclipses output and formats them nicely.
    """

    name = name.replace( ' ', '' )

    # Convert the RA to hh:mm:ss.s and Dec to dd:mm:ss.s:
    ra_str = str( ephem.hours( ra ) )
    dec_str = str( ephem.degrees( dec ) )


    if ra.replace( ' ','' )=='':
        ra_str = '?'
    elif len( ra_str )==10:
        ra_str = '0{0}'.format( ra_str )

    if dec.replace( ' ','' )=='':
        dec_str = '?'
    else:
        if float( dec )>=0:
            dec_str = '+{0}'.format( dec_str )
        if len( dec_str )==10:
            dec_str = '{0}0{1}'.format( dec_str[0], dec_str[1:] )

    rank_str = '{0}'.format( rank ).rjust( 4 )
    name_str = '{0}'.format( name ).center( 15 )
    ra_str = '{0}'.format( ra_str.replace( ':', ' ' ) ).center( 11 )
    dec_str = '{0}'.format( dec_str.replace( ':', ' ' ) ).rjust( 12 )
    kmag_str = '{0:.1f}'.format( kmag ).rjust( 5 )
    tstar_str = '{0:4d}'.format( int( tstar ) ).center( 6 )
    rstar_str = '{0:.1f}'.format( rstar ).center( 6 )
    rp_str = '{0:.1f}'.format( rp ).center( 6 )
    a_str = '{0:.3f}'.format( a ).center( 5 )
    tpeq_str = '{0:4d}'.format( int( tpeq ) ).center( 5 )
    fratio_str = '{0:.4f}'.format( (1e4)*fratio ).rjust( 6 )
    snr_str = '{0:.2f}'.format( snr_norm ).rjust( 6 )
    outstr = '  {0}{1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11}\n'\
             .format( rank_str, \
                      name_str, \
                      ra_str, \
                      dec_str, \
                      kmag_str, \
                      tstar_str, \
                      rstar_str, \
                      rp_str, \
                      a_str, \
                      tpeq_str, \
                      fratio_str, \
                      snr_str )

    return outstr


def make_outstr_transmission( rank, name, ra, dec, vmag, kmag, rstar, rp, tpeq, hatm, \
                              depth_tr, delta_tr, snr_norm_vis, snr_norm_ir  ):
    """
    Takes quantities that will be written to the transits output and formats them nicely.
    """

    name = name.replace( ' ', '' )

    # Convert the RA to hh:mm:ss.s and Dec to dd:mm:ss.s:
    ra_str = str( ephem.hours( ra ) )
    dec_str = str( ephem.degrees( dec ) )

    if float( dec )>=0:
        dec_str = '+{0}'.format( dec_str )

    if len( ra_str )==10:
        ra_str = '0{0}'.format( ra_str )
    if len( dec_str )==10:
        dec_str = '{0}0{1}'.format( dec_str[0], dec_str[1:] )
    
    rank_str = '{0}'.format( rank ).rjust( 4 )
    name_str = '{0}'.format( name ).center( 15 )
    ra_str = '{0}'.format( ra_str.replace( ':', ' ' ) ).center( 11 )
    dec_str = '{0}'.format( dec_str.replace( ':', ' ' ) ).rjust( 12 )
    vmag_str = '{0}'.format( vmag ).rjust( 5 )
    kmag_str = '{0}'.format( kmag ).rjust( 5 )
    rstar_str = '{0:.1f}'.format( rstar ).center( 6 )
    rp_str = '{0:.1f}'.format( rp ).center( 6 )
    tpeq_str = '{0:4d}'.format( int( tpeq ) ).center( 5 )
    if np.isfinite( hatm ):
        hatm_str = '{0:d}'.format( int( hatm / (1e3) ) ).rjust( 5 )
    else:
        hatm_str = 'NaN'
    depth_tr_str = '{0:.2f}'.format( (1e2)*depth_tr ).center( 6 )
    delta_tr_str = '{0:.2f}'.format( (1e4)*delta_tr ).center( 6 )
    snr_vis_str = '{0:.2f}'.format( snr_norm_vis ).rjust( 6 )
    snr_ir_str = '{0:.2f}'.format( snr_norm_ir ).rjust( 6 )
    outstr = '  {0}{1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13}\n'\
             .format( rank_str, \
                      name_str, \
                      ra_str, \
                      dec_str, \
                      vmag_str, \
                      kmag_str, \
                      rstar_str, \
                      rp_str, \
                      tpeq_str, \
                      hatm_str, \
                      depth_tr_str, \
                      delta_tr_str, \
                      snr_vis_str, \
                      snr_ir_str )

    return outstr


    
