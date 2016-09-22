import os, pdb
import numpy as np
import atpy

ALL_CSV = 'exoplanets.csv' # csv file for all known exoplanets
ALL_FITS = 'exoplanets_all.fits' # fits file for all known exoplanets
ALL_IPAC = 'exoplanets_all.ipac' # ipac file for all known exoplanets
TR_FITS = 'exoplanets_transiting.fits' # fits file for known exoplanets that transit
TR_IPAC = 'exoplanets_transiting.ipac' # ipac file for known exoplanets that transit 


def download_data():
    """
    Uses wget to download a csv ascii file of planetary properties from
    exoplanets.org then saves the table in fits and ipac formats. Output
    files are first generated for all planets and then again for the
    transiting planets only.
    """
    
    # Use wget to download the data:
    cmd = 'wget http://exoplanets.org/csv-files/exoplanets.csv'
    if os.path.exists( ALL_CSV ):
      os.remove( ALL_CSV )
    os.system( cmd )
    
    # Read in using atpy:
    exo_dat = atpy.Table( ALL_CSV, type='ascii', delimiter=',', data_start=1, \
                          fill_values=( '', 'nan', 'RSTAR', 'TT', 'T14', 'TEFF', 'A', 'R', 'KS', 'PER', 'MSINI', 'V' ) )
    
    # Save table for all planets in fits and ipac files:
    exo_dat.write( ALL_FITS, overwrite=True )
    exo_dat.write( ALL_IPAC, overwrite=True )
    exo_dat.describe()
    
    # Save table for transiting planets only in fits and ipac files:
    
    transit_dat = exo_dat.where( ( exo_dat.TRANSIT=='1' ) + ( exo_dat.TRANSIT==1 ) )
    transit_dat = transit_dat.where( np.isfinite( transit_dat.TT ) )
    transit_dat = transit_dat.where( np.isfinite( transit_dat.T14 ) )  
    transit_dat = transit_dat.where( np.isfinite( transit_dat.PER ) )  
    transit_dat.write( TR_FITS, overwrite=True )
    transit_dat.write( TR_IPAC, overwrite=True )
    transit_dat.describe()
    
    return None


def table_append( t, extra_planets ):
    """
    """
    keys = [ 'NAME', 'RA_STRING', 'DEC_STRING', 'V', 'KS', 'RSTAR', 'TEFF', 'A', 'PER', \
             'MASS', 'MSINI', 'R', 'TT', 'T14' ]
    nentries = len( keys )
    entries = []
    for j in range( nentries ):
        entries += [ [] ]
    nextra = len( extra_planets )
    for i in range( nextra ):
        for j in range( nentries ):
            entries[j] += [ extra_planets[i][keys[j]] ]
    # Add RA and DEC in decimal format:
    keys = keys + [ 'RA', 'DEC' ]
    entries += [ [], [] ]
    nentries += 2
    for i in range( nextra ):
        for j in range( nentries ):
            if keys[j]=='RA_STRING':
                RAstr = extra_planets[i]['RA_STRING']
                while RAstr[0]==' ': # strip any leading whitespace
                    RAstr = RAstr[1:]
                ix1 = RAstr.find( ':' )
                if ix1>=0:
                    ix2 = ix1 + RAstr[ix1+1:].find( ':' )
                else:
                    ix1 = RAstr.find( ' ' )
                    ix2 = ix1 + RAstr[ix1+1:].find( ' ' )
                RAhr = float( RAstr[:ix1] )
                RAmin = float( RAstr[ix1+1:ix2+1] )
                RAsec = float( RAstr[ix2+2:] )
                RAdec = RAhr + RAmin/60. + RAsec/60./60.
                entries[-2] += [ RAdec ]
            elif keys[j]=='DEC_STRING':
                DECstr = extra_planets[i]['DEC_STRING']
                while DECstr[0]==' ': # strip any leading whitespace
                    DECstr = DECstr[1:]
                ix1 = DECstr.find( ':' )
                if ix1>=0:
                    ix2 = ix1 + DECstr[ix1+1:].find( ':' )
                else:
                    ix1 = DECstr.find( ' ' )
                    ix2 = ix1 + DECstr[ix1+1:].find( ' ' )
                DECdeg = float( DECstr[:ix1] )
                DECmin = float( DECstr[ix1+1:ix2+1] )
                DECsec = float( DECstr[ix2+2:] )
                DECdec = DECdeg + DECmin/60. + DECsec/60./60.
                entries[-1] += [ DECdec ]
    textra = atpy.Table()
    for j in range( nentries ):
        textra.add_column( keys[j], np.array( entries[j], dtype=t[keys[j]].dtype ) )
    
    for key in t.keys():
        if key in textra.keys():
            continue
        else:
            textra.add_column( key, np.empty( nextra, dtype=t[key].dtype ) )
    
    textra_sorted = atpy.Table()
    for key in t.keys():
        textra_sorted.add_column( key, textra[key] )
    t.append( textra_sorted )
    return t


