import os, sys, pdb
import atpy
import ephem
import numpy as np
import scipy.integrate
import tutilities

G = 6.67428e-11 # gravitational constant in m^3/kg^-1/s^-2
HPLANCK = 6.62607e-34 # planck's constant in J*s
C = 2.99792e8 # speed of light in vacuum in m/s
KB = 1.3806488e-23 # boltzmann constant in J/K
RGAS = 8.314 # gas constant in J/mol/K
RSUN = 6.9551e8 # solar radius in m
RJUP = 7.1492e7 # jupiter radius in m
MJUP = 1.89852e27 # jupiter mass in kg
AU2M = 1.49598e11 # au to metres conversion factor
MUJUP = 2.22e-3 # jupiter atmosphere mean molecular weight in kg/mole
TR_TABLE = 'exoplanets_transiting.fits' # fits file for known exoplanets that transit


def reflection( wav_meas_um=[ 0.55, 0.80 ], wav_ref_um=0.55, obj_ref='HD189733b', \
                outfile='signals_reflection.txt', download_latest=True ):
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
    wavelengths. Note that we 
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
    nplanets = len( t.NAME )

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
    fratio_total = fratio_reflection + fratio_thermal
    thermal_frac = fratio_thermal/fratio_reflection

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
    s = np.argsort( snr_norm )
    s = s[::-1]

    # Open the output file and write the column headings:
    ofile = open( outfile, 'w' )
    header = make_header_reflection( nplanets, wav_meas_m_cuton, wav_meas_m_cutoff, wav_ref_m, obj_ref )
    ofile.write( header )

    # Write the output rows to file and save:
    for j in range( nplanets ):
        i = s[j]
        outstr = make_outstr_reflection( j+1, t.NAME[i], t.RA[i], t.DEC[i], t.V[i], \
                                         t.TEFF[i], t.RSTAR[i], t.R[i], t.A[i], tpeq[i], \
                                         RpRs[i], fratio_total[i], snr_norm[i], \
                                         thermal_frac[i] )
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
    nplanets = len( t.NAME )

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
    s = np.argsort( snr_norm )
    s = s[::-1]

    # Open the output file and write the column headings:
    ofile = open( outfile, 'w' )
    header = make_header_thermal( nplanets, wav_meas_m, wav_ref_m, obj_ref )
    ofile.write( header )
    
    # Write the output rows to file and save:
    for j in range( nplanets ):
        i = s[j]
        outstr = make_outstr_thermal( j+1, t.NAME[i], t.RA[i], t.DEC[i], t.KS[i], \
                                      t.TEFF[i], t.RSTAR[i], t.R[i], t.A[i], tpeq[i], \
                                      fratio[i], snr_norm[i] )
        ofile.write( outstr )
    ofile.close()
    print 'Saved output in {0}'.format( outfile )
    
    return outfile


def transmission( wav_vis_um=0.7, wav_ir_um=2.2, wav_ref_um=2.2, obj_ref='WASP-19 b', \
                  outfile='signals_transits.txt', download_latest=True ):
    """
    Generates a table of properties relevant to transmission spectroscopy measurements
    at a specified wavelength for all known transiting exoplanets. 
    Eclipse depths are quoted assuming a geometric albedo of 1. The expected signal
    for lower albedos scales linearly as a function of the geometric albedo, according
    to:  delta_flux = Ag*( ( RpRs/aRs )**2 )

    Perhaps the most useful column of the output table is the one that gives the
    expected signal-to-noise of the eclipse **relative to the signal-to-noise for a
    reference planet at a reference wavelength**. A relative signal-to-noise is used
    due to the unknown normalising constant when working with magnitudes at arbitrary
    wavelengths. Note that we 
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

    # Make we exclude table rows that do not contain
    # all the necessary properties:
    t = filter_table( sigtype='transmission', download_latest=download_latest )
    nplanets = len( t.NAME )

    # First check to make sure we have both a V and Ks
    # magnitude for the reference star:
    ix = ( t.NAME==obj_ref )
    if ( np.isfinite( t.KS[ix] )==False ) or ( np.isfinite( t.V[ix] )==False ):
        print '\n\nPlease select a different reference star for which we have both a V and Ks magnitude\n\n'
        return None


    # Calculate the approximate planetary equilibrium temperature:
    tpeq = Teq( t )

    # Calculate the radii ratios for each planet:
    RpRs = ( t.R*RJUP )/( t.RSTAR*RSUN )

    # Calculate the gravitaional accelerations at the surface zero-level:
    MPLANET = np.zeros( nplanets )
    for i in range( nplanets ):
        try:
            MPLANET[i] = np.array( t.MASS[i], dtype=float )
        except:
            MPLANET[i] = np.array( t.MSINI[i], dtype=float )
            print t.NAME[i]
    little_g = G*MPLANET*MJUP/( ( t.R*RJUP )**2 )

    # Calculate the atmospheric scale height in metres; note that
    # we use RGAS instead of KB because MUJUP is **per mole**:
    Hatm = RGAS*tpeq/MUJUP/little_g

    # Calculate the approximate change in transit depth for a
    # wavelength range where some species in the atmosphere
    # increases the opacity of the planetary limb for an additional
    # 2.5 (i.e. 5/2) scale heights:
    depth_tr = RpRs**2.
    delta_tr = 2*n*( t.R*RJUP )*Hatm/( ( t.RSTAR*RSUN )**2 )

    # Using the known Ks magnitude of the target, estimate the
    # unnormalised signal-to-noise ratio of the change in transit
    # depth that we would measure in the visible and IR separately:
    bratio = planck( wav_vis_m, t.TEFF )/planck( wav_K_m, t.TEFF )
    mag = t.KS - 2.5*np.log10( bratio )
    flux_unnorm = 10**( -mag/2.5 )
    snr_unnorm_vis = np.sqrt( flux_unnorm )*delta_tr

    bratio = planck( wav_ir_m, t.TEFF )/planck( wav_K_m, t.TEFF )
    mag = t.KS - 2.5*np.log10( bratio )
    flux_unnorm = 10**( -mag/2.5 )
    snr_unnorm_ir = np.sqrt( flux_unnorm )*delta_tr

    # Repeat the above using the known V band for any that didn't
    # have known KS magnitudes:
    ixs = ( np.isfinite( t.KS )==False )
    
    bratio = planck( wav_vis_m, t.TEFF[ixs] )/planck( wav_V_m, t.TEFF[ixs] )
    mag = t.V[ixs] - 2.5*np.log10( bratio )
    flux_unnorm = 10**( -mag/2.5 )
    snr_unnorm_vis[ixs] = np.sqrt( flux_unnorm )*delta_tr[ixs]

    bratio = planck( wav_ir_m, t.TEFF[ixs] )/planck( wav_V_m, t.TEFF[ixs] )
    mag = t.KS[ixs] - 2.5*np.log10( bratio )
    flux_unnorm = 10**( -mag/2.5 )
    snr_unnorm_ir[ixs] = np.sqrt( flux_unnorm )*delta_tr[ixs]

    # The signal-to-noise ratio is still not normalised, so we need to repeat
    # the above for another reference star; seeing as the normalising constant
    # It might be useful to put the signal-to-noise in different units, namely,
    # compare the size of the current signal to that of another reference target 
    # at some reference wavelength. Basically repeat the above for the reference:
    names = []
    for name in t.NAME:
        names += [ name.replace( ' ', '' ) ]
    names = np.array( names, dtype=str )
    ii = ( names==obj_ref )
    delta_tr_ref = 2*n*( t.R[ii]*RJUP )*Hatm[ii]/( ( t.RSTAR[ii]*RSUN )**2 )
    kratio_ref = planck( wav_ref_m, t.TEFF[ii] )/planck( wav_K_m, t.TEFF[ii] )

    mag_ref_ir = t.KS[ii] - 2.5*np.log10( kratio_ref )
    flux_ref_ir = 10**( -mag_ref_ir/2.5 )
    snr_ref_ir = np.sqrt( flux_ref_ir )*delta_tr_ref

    mag_ref_vis = t.V[ii] - 2.5*np.log10( kratio_ref )
    flux_ref_vis = 10**( -mag_ref_vis/2.5 )
    snr_ref_vis = np.sqrt( flux_ref_vis )*delta_tr_ref

    # Reexpress the signal-to-noise of our target as a scaling of the reference
    # signal-to-noise:
    snr_norm_vis = snr_unnorm_vis/snr_ref_vis
    snr_norm_ir = snr_unnorm_ir/snr_ref_ir

    # Rearrange the targets in order of the most promising:
    s = np.argsort( snr_norm_vis )
    s = s[::-1]

    # Open the output file and write the column headings:
    ofile = open( outfile, 'w' )
    header = make_header_transmission( nplanets, wav_vis, wav_ir, wav_ref, obj_ref, n )
    ofile.write( header )
    
    for j in range( nplanets ):
        i = s[j]
        if np.isfinite( t.V[i] ):
            v = '{0:.1f}'.format( t.V[i] )
        else:
            v = '-'
        if np.isfinite( t.KS[i] ):
            ks = '{0:.1f}'.format( t.KS[i] )
        else:
            ks = '-'
        outstr = make_outstr_transmission( j+1, t.NAME[i], t.RA[i], t.DEC[i], v, ks, \
                                           t.RSTAR[i], t.R[i], tpeq[i], \
                                           Hatm[i], depth_tr[i], delta_tr[i], \
                                           snr_norm_vis[i], snr_norm_ir[i] )
        ofile.write( outstr )
    ofile.close()
    print 'Saved output in {0}'.format( outfile )
    
    return outfile


def filter_table( sigtype=None, download_latest=True ):
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


def make_header_thermal( nplanets, wav, wav_ref, obj_ref ):
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
    header += '# Eclipse estimates at {0:.2f} micron arranged in order of increasing\n'.format( (1e6)*wav )
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
    header += '#    and expressed relative to the S/N expected for {0} at {1:.2f} micron\n'.format( obj_ref, (1e6)*wav_ref )
    header += '#       --->  S/N_ref = Fp_ref / sqrt( Fs_ref )\n'
    header += '#       --->  S/N_targ = Fp / sqrt( Fs ) \n'
    header += '#       --->  S/N = S/N_target / S/N_ref\n#\n'
    header += '{0}\n#\n'.format( '#'*nchar )
    header += colheadingsa
    header += colheadingsb
    header += '{0}{1}\n'.format( '#', '-'*( nchar-1 ) )

    return header
    

def make_header_reflection( nplanets, wav_cuton, wav_cutoff, wav_ref, obj_ref ):
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
              .format( (1e6)*wav_cuton, (1e6)*wav_cutoff )
    header += '# detectability for {0:d} known transiting exoplanets\n#\n'.format( nplanets )
    header += '# Values for \'V\', \'Tstar\', \'Rstar\', \'Rp\', \'a\' are taken from the literature \n#\n'
    header += '# Other quantities are derived as follows:\n#\n'
    header += '#  \'Tpeq\' is the equilibrium effective temperature of the planet assuming \n'
    header += '#    absorption of all incident star light and uniform redistribution\n'
    header += '#       --->  Tpeq = np.sqrt( Rstar/2./a )*Tstar \n#\n'
    header += '#  \'Fp/Fs\' is the ratio of the reflected+thermal planetary flux to the total stellar\n'
    header += '#  flux assuming a geometric albedo of 1, i.e. Ag=1, where:\n'
    header += '#       --->  Fp/Fs = Ag*( ( ( Rplanet/Rstar )/( a/Rstar ) )**2 ) \n#\n'
    header += '#  \'Thermal\' is the fraction of the planetary flux due to thermal emission rather than\n'
    header += '#  reflected light from the planetary atmosphere\n#\n'
    header += '#  assuming a geometric albedo of 1, i.e. Ag=1, where:\n'
    header += '#       --->  Fp/Fs = Ag*( ( ( Rplanet/Rstar )/( a/Rstar ) )**2 ) \n#\n'
    header += '#  \'S/N\' is the signal-to-noise estimated using the known stellar brightness\n'
    header += '#    and expressed relative to the S/N expected for {0} at {1:.2f} micron\n'.format( obj_ref, (1e6)*wav_ref )
    header += '#       --->  S/N_ref = Fp_ref / sqrt( Fs_ref )\n'
    header += '#       --->  S/N_targ = Fp / sqrt( Fs ) \n'
    header += '#       --->  S/N = S/N_target / S/N_ref\n#\n'
    header += '{0}\n#\n'.format( '#'*nchar )
    header += colheadingsa
    header += colheadingsb
    header += '{0}{1}\n'.format( '#', '-'*( nchar-1 ) )

    return header
    

def make_header_transmission( nplanets, wav_vis, wav_ir, wav_ref, obj_ref, n ):
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
              .format( (1e6)*wav_vis, (1e6)*wav_ir )
    header += '# arranged in order of increasing detectability in the visible wavelength for {0:d} known\n'\
              .format( nplanets )
    header += '# transiting exoplanets\n#\n'
    header += '# SNR is given relative to the approximate signal for {0} expected at {1:.2f} microns \n#\n'\
              .format( obj_ref, (1e6)*wav_ref )
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
    header += '#    for {0} at {1:.2f} micron\n'.format( obj_ref, (1e6)*wav_ref )
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
    fratio_str = '{0:.2f}'.format( (1e4)*fratio ).rjust( 6 )
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
    hatm_str = '{0:d}'.format( int( hatm / (1e3) ) ).rjust( 5 )
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


    
