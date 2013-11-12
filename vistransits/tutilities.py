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
