import numpy as np
import pdb
import os
import ephem
import atpy
import pytz
import datetime
import tutilities


EPH_FILE = 'exoplanets-org-ephem.txt'


def calc_visible( observatory, date_start, date_end, sigtype='transits', \
                  ofilename_byplanet='default', ofilename_chronolog='default',
                  sun_alt_max=-6, sun_alt_twil=-12, sun_alt_dark=-18, \
                  moon_alt_set=-6, target_elev_min=25, oot_deltdur=0.5, \
                  tr_signals='signals_transits.txt', thermal_signals='signals_thermal.txt', \
                  reflect_signals='signals_reflection.txt' ,\
                  exclude_unranked=False, max_rank=None ):
    """
    Calculates the visible transits for a list of targets at a given observatory within
    a specified time window. Saves them in an output file with name of the form:

      >observatory-name<_transits.txt  or  >observatory-name<_eclipses.txt

    depending on whether sigtype='transits' (default) or sigtype='thermal' or sigtype='reflection'.

    INPUTS
      **observatory - String matching one of the observatory string identifiers used in
          the observatories() routine ( eg. 'LaPalma', 'Paranal', 'MaunaKea' ).
      **date_start - String in the format 'YYYY/MM/DD' giving the start date.
      **date_end - String in the format 'YYYY/MM/DD' giving the end date.
      **sigtype - 'transits', 'reflection' or 'thermal'
      **ofilename_byplanet - Name of first output file that gets generated, divided
          between the different planets.
      **ofilename_chronolog - Name of the second output file that gets generated, with
          transits/eclipses ordered chronologically.
      **sun_alt_max - Maximum altitude (i.e. angle in degrees from the horizon) that
          the Sun can be before the sky is too bright to observe.
      **sun_alt_twil - Sun altitude marking the transition from bright time to twilight.
      **sun_alt_dark - Sun altitude marking the transition from twilight to dark time.
      **moon_alt_set - Altitude at which the Moon is considered to have 'set'.
      **target_elev_min - Minimum altitude of target for which is is observable.
      **oot_deltdur - Float giving the fraction of the transit duration either side of
          ingress/egress requested for sampling the out-of-transit baseline.
      **tr_signals - Input file containing the estimated transmission signals.
      **thermal_signals - Input file containing the estimated thermal emission signals.
      **reflect_signals - Input file containing the estimated reflection signals.
      **exclude_unranked - If set to True, transit/eclipse predictions will ony be made
          for those planets where a transmission/emission signal estimate was possible.
      **max_rank - Lowest ranked signal that output will be printed for; can be set
          to None if no such constraint is to be applied.
      
    OUTPUT
      Output is printed to the files specified by the ofilename_byplanet and
      ofilename_chronolog keyword arguments.
    """

    # Convert the minimum target altitude to a maximum zenith angle:
    zenith_max = 90 - target_elev_min

    # Read in the basic target information for all transiting exoplanets:
    targets_all, vmags_all, ras_all, decs_all, ttrs_all, pers_all, durs_all = read_eph( EPH_FILE )
    ntargets_all = len( targets_all )

    # Read in the rankings for transit and eclipse signals:
    if sigtype=='transits':
        targets, ranks, ttrs, pers, durs = transit_ranks( tr_signals, targets_all, \
                                                          ttrs_all, pers_all, durs_all )
    elif sigtype=='reflection':
        targets, ranks, ttrs, pers, durs = eclipse_ranks( reflect_signals, targets_all, \
                                                          ttrs_all, pers_all, durs_all )
    elif sigtype=='thermal':
        targets, ranks, ttrs, pers, durs = eclipse_ranks( thermal_signals, targets_all, \
                                                          ttrs_all, pers_all, durs_all )
    else:
        pdb.set_trace() # sigtype not recognised
    ntargets = len( targets )

    # Create the databases that will be used by pyephem for calculating
    # ephemerides for each object:
    dbs = []
    ras = []
    decs = []
    vmags = []
    for i in range( ntargets ):
        # Ensure that the target is a signal of interest:
        for j in range( ntargets_all ):
            if targets[i]==targets_all[j]:
                db_str = '{targ},f|S,{ra},{dec},{vmag}'.format( targ=targets[i], \
                                                                ra=ras_all[j], \
                                                                dec=decs_all[j], \
                                                                vmag=vmags_all[j] )
                ras += [ ras_all[j] ]
                decs += [ decs_all[j] ]
                vmags += [ vmags_all[j] ]
                dbs += [ db_str ]
                break
            elif ( i==ntargets-1 )*( j==ntargets_all-1 ):
                pdb.set_trace() # not matched to any targets in database
                
    # Open the output file and write a header:
    if ofilename_byplanet=='default':
        if sigtype=='transits':
            ofilename_byplanet = '{0}_transits_byplanet.txt'.format( observatory )
            ofilename_chronolog = '{0}_transits_chronolog.txt'.format( observatory )
        elif sigtype=='thermal':
            ofilename_byplanet = '{0}_thermal_byplanet.txt'.format( observatory )
            ofilename_chronolog = '{0}_thermal_chronolog.txt'.format( observatory )            
        elif sigtype=='reflection':
            ofilename_byplanet = '{0}_reflection_byplanet.txt'.format( observatory )
            ofilename_chronolog = '{0}_reflection_chronolog.txt'.format( observatory )            
        else:
            pdb.set_trace() #this shouldn't happen

    # Create the strings that will be used for column headers:
    colheadingsa_bp, colheadingsb_bp = make_colheadings( 'byplanet' )
    colheadingsa_ch, colheadingsb_ch = make_colheadings( 'chronolog' )
    nchar_bp = np.max( [ len( colheadingsa_bp ), len( colheadingsb_bp ) ] )
    nchar_ch = np.max( [ len( colheadingsa_ch ), len( colheadingsb_ch ) ] )

    ofile_bp = open( ofilename_byplanet, 'w' )
    ofile_bp.write( '{0}\n#\n'.format( '#'*nchar_bp ) )

    ofile_ch = open( ofilename_chronolog, 'w' )
    ofile_ch.write( '{0}\n#\n#'.format( '#'*nchar_ch ) )
        
    if sigtype=='transits':
        sigtype_lower_singular = 'transit'
        sigtype_upper_singular = 'Transit'
    elif sigtype=='thermal':
        sigtype_lower_singular = 'thermal'
        sigtype_upper_singular = 'Thermal'
    elif sigtype=='reflection':
        sigtype_lower_singular = 'reflection'
        sigtype_upper_singular = 'Reflection'


    ofile_bp.write( '# Visible {0} targets from {1} between {2} and {3}, arranged by planet\n#\n'\
                 .format( sigtype_lower_singular, observatory, date_start, date_end ) )
    ofile_ch.write( '# Visible {0} targets from {1} between {2} and {3}, arranged in chronological order\n#\n'\
                 .format( sigtype_lower_singular, observatory, date_start, date_end ) )
    if max_rank!=None:
        header_str = '# Only considered the top ranked {0} signals and of these only those with {1}s\n'\
                     .format( max_rank, sigtype_lower_singular )
        header_str += '# where the mid-time occurs at a zenith angle <{0}deg\n#\n'\
                      .format( zenith_max )
    else:
        header_str = '# Considered all published transiting planets but output only printed where the\n'
        header_str += '# {0} mid-time occurs at a zenith angle <{0}deg\n#\n'.format( zenith_max )
    if oot_deltdur>0:
        header_str += '# Accounts for {0:.2f} transit durations before and after the transit to sample\n'\
                      .format( oot_deltdur )
        header_str += '# the out-of-transit baseline flux level\n#\n'
    else:
        header_str += '# No allowance made for time required to sample the out-of-transit baseline flux\n'
        header_str += '# before and after the transit\n#\n'
    header_str += '# The sun elevation angle has been divided into the following ranges:\n#\n'
    header_str += '#   1. Maximum acceptable = {0}deg\n'.format( sun_alt_max )
    header_str += '#   2. Dawn/Dusk = [ {0}deg to {1}deg ]\n'.format( sun_alt_twil, sun_alt_max )
    header_str += '#   3. Twilight = [ {0}deg to {1}deg ]\n'.format( sun_alt_dark, sun_alt_twil )
    header_str += '#   4. Dark < {0}deg\n#\n'.format( sun_alt_dark )
    header_str +=  '# {0} signals are described by a string with format {{type}}-{{details}}, where:\n#\n'\
                  .format( sigtype_upper_singular )
    header_str += '#   {{type}} refers to whether or not we get all of the {0} with the sun below its\n'\
                  .format( sigtype_lower_singular )
    header_str += '#   maximum acceptable value:\n'
    header_str += '#     - \'full\' if the sun elevation is less than the maximum acceptable value for the\n'
    header_str += '#       entire duration of the {0} signal\n'.format( sigtype_lower_singular )
    header_str += '#     - \'partial\' if the sun elevation is less than the maximum acceptable value at\n'
    header_str += '#       the {0} mid-time, but not for the entire duration of the {0} signal\n#\n'\
                  .format( sigtype_lower_singular )
    if oot_deltdur==0:
        header_str += '#   {details} can have the following values:\n'
    else:
        header_str += '#   {{details}} refer to the full span of the observations including the {0:.2f} transit\n'\
                      .format( oot_deltdur )
        header_str += '#   durations before and after the {0} signal itself for sampling the baseline flux:\n'\
                      .format( sigtype_lower_singular )
    header_str += '#     - \'all_in_darktime\' if the sun elevation is within the dark range for the entire\n'
    header_str += '#     duration of the observations\n'
    header_str += '#     - \'dark_to_twilight\' if the sun elevation is within the dark range at the start\n'
    header_str += '#       of the observations and within the twilight range at the end of the observations\n'
    header_str += '#     - \'twilight_to_dark\' if the sun elevation is within the twilight range at the \n'
    header_str += '#       start of the observations and within the dark range at the end of the observations\n'
    header_str += '#     - \'start_at_dusk\' if the sun elevation is within the twilight range at the start\n'
    header_str += '#       of the observations\n'
    header_str += '#     - \'end_at_dawn\' if the sun elevation is within the twilight range at the end of\n'
    header_str += '#       the observations\n'
    if oot_deltdur>0:
        header_str += '#     - \'partial_oot\' if the sun elevation is above the maximum acceptable value at some\n'
        header_str += '#       time during either the pre- or post-{0} signal baseline flux observations\n'\
                      .format( sigtype_lower_singular )
    header_str += '#     - \'miss_ingress\' if the sun elevation is above the maximum acceptable value at\n'
    header_str += '#       ingress but below this value at egress\n'
    header_str += '#     - \'miss_ingress\' if the sun elevation is below the maximum acceptable value at\n'
    header_str += '#       ingress but above this value at egress\n'
    header_str += '#     - \'only_middle\' if the sun elevation is above the maximum acceptable value at\n'
    header_str += '#       value at both ingress and egress, but descends below this value at some point\n'
    header_str += '#       during the {0} signal\n#\n'.format( sigtype_lower_singular )
    ofile_bp.write( header_str )
    ofile_ch.write( header_str )

    # Create the observatory and timezone objects:
    obs, tz = setup_observatory( observatory )
    if ( obs==None ) and ( tz==None ):
        ofile_bp.close()
        os.remove( ofilename_byplanet )
        return None

    # Generate instances of the Sun and Moon:
    sun = ephem.Sun()
    moon = ephem.Moon()

    # Convert start and end dates of observing period
    # to pyephem float objects:
    date_start = ephem.Date( date_start )
    date_end = ephem.Date( date_end )

    # Go through the targets one-at-a-time, checking for visible transits:
    mjds = []
    date_floats = []
    outstrs_ch = []
    print '\nCalculating visible transits for:'
    for i in range( ntargets ):

        print '  ... target {0:d} of {1:d} --> {2} '\
              .format( i+1, ntargets, targets[i] )
        first_i = True
        per_i = pers[i]
        dur_i = durs[i]/24.
        
        # Initiate the ephem object for the target:
        target_i = ephem.readdb( dbs[i] )
        if target_i.name!=targets[i]:
            pdb.set_trace()

        # Check to see if the current target's signal has been ranked,
        # and if it has, make sure that it was ranked highly enough,
        # otherwise we skip to the next target straight away:
        nranked = len( targets )
        for j in range( nranked ):
            if targets[j]==target_i.name:
                rank_i = int( ranks[j] )
                if ( max_rank!=None ):
                    if ( max_rank != -1 ) * ( rank_i>max_rank ):
                        include = False
                    else:
                        include = True
                unranked = False
                break
            else:
                if j==nranked-1:
                    unranked = True

        # See if we want to skip the current target for some reason:
        if unranked==True:
            if exclude_unranked==True:
                continue
        else:
            if include==False:
                continue
                    
        # Convert the JD transit time to a pyephem date:
        ttr_i = jd2pyephemdate( ttrs[i] )

        # If we're wanting eclipse information, approximate the eclipse
        # time by subtracting half a period (the more eccentric the orbit,
        # the worse this approximation will be):
        if ( sigtype=='thermal' )+( sigtype=='reflection' ):
            ttr_i -= 0.5 * per_i
        elif sigtype!='transits':
            pdb.set_trace() # if stopped here, make sure sigtype is valid

        #if target_i.name=='WASP-43b':
        #    pdb.set_trace()

        # Find the transit with mid-time occurring immediately
        # before the observing run:
        if ttr_i>date_start:
            while ttr_i>date_start:
                ttr_i -= per_i
        else:
            while ttr_i<date_start:
                ttr_i += per_i
            ttr_i -= per_i

        # If it's not even partially visible, start at the next transit: 
        if ( ttr_i + 0.5*dur_i )<date_start:
            ttr_i += per_i

        # Now that we have a starting time, loop over successive transits
        # until we reach the one with mid-time immediately before the end
        # of the observing window:
        while ttr_i<date_end:

            # Set the UT date of the current transit within the
            # observatory object:
            obs.date = ttr_i

            # Update the target ephemerides and calculate its
            # elevation in the sky:
            target_i.compute( obs )
            target_i_alt_midtime = np.rad2deg( float( target_i.alt ) )
            # If the target is not above the minimum elevation, bump up to
            # the next transit and go back to the top of the loop:
            if target_i_alt_midtime<target_elev_min:
                ttr_i += per_i
                continue

            # Given the altitude, calculate the approximate airmass:
            zenith_i_midtime = 90 - target_i_alt_midtime
            airmass = calc_airmass( zenith_i_midtime )
            
            # Update the Sun and Moon ephemerides and work out
            # their elevations with respect to the horizon:
            sun.compute( obs )
            sun_alt_midtime = np.rad2deg( float( sun.alt ) )
            moon.compute( obs )
            moon_alt_midtime = np.rad2deg( float( moon.alt ) )
            # If the Sun is above the maximum elevation limit, bump up to
            # the next transit and go back to the top of the while loop:
            if sun_alt_midtime>sun_alt_max:
                ttr_i += per_i
                continue

            # Get the Moon phase as a percentage of the illuminated face:
            moonphase = '{0:d}'.format( int( np.round( moon.phase ) ) )
            # Get the target-Moon angular separation:
            moondist = int( np.round( np.rad2deg( ephem.separation( ( target_i.az, target_i.alt ), \
                                                                      ( moon.az, moon.alt ) ) ) ) )
            moondist = '{0:d}'.format( moondist )

            # If we make it to here we will consider the transit potentially
            # observable. Next, evaluate whether or not its detectability rank
            # meets our requirements:
            if first_i==True:

                # Write the header for the current object to the output file:
                nchar = max( [ len( colheadingsa_bp ), len( colheadingsb_bp ) ] )
                ofile_bp.write( '\n\n{0}\n#\n'.format( '#'*( nchar_bp ) ) )
                if unranked==True:
                    if sigtype=='transits':
                        header_str = '#  {0}   -->   not enough information to rank primary transit signal \n#\n'\
                                     .format( targets[i]  )
                    elif ( sigtype=='thermal' )+( sigtype=='reflection' ):
                        header_str = '#  {0}   -->   not enough information to rank secondary eclipse signal \n#\n'\
                                     .format( targets[i]  )
                else:
                    if sigtype=='transits':
                        header_str = '#  {0}   -->   primary transit signal ranked {1} out of {2} \n#\n'\
                                     .format( targets[i], str( rank_i ), str( nranked ) )
                    elif ( sigtype=='thermal' )+( sigtype=='reflection' ):
                        header_str = '#  {0}   -->   secondary eclipse signal ranked {1} out of {2} \n#\n'\
                                     .format( targets[i], str( rank_i ), str( nranked ) )
                header_str += '#  RA (hh mm ss.s) Dec (dd mm ss.s)\n'
                ra_str = ras[i].replace( ':', ' ' )
                dec_str = decs[i].replace( ':', ' ' )
                header_str += '#  {0} {1}\n#\n'.format( ra_str, dec_str )
                header_str += colheadingsa_bp
                header_str += colheadingsb_bp

                ofile_bp.write( header_str )
                ofile_bp.write( '{0}{1}\n'.format( '#', '-'*( nchar_bp-1 ) ) )
                
                first_i = False
                
            # Now we want to work out some more details about
            # what kind of transit it will be.

            # Determine the Sun, Moon and target elevations
            # at the start of the observations:
            obs.date = ttr_i - dur_i*( 0.5 + oot_deltdur )
            sun.compute( obs )
            sun_alt_start = np.rad2deg( float( sun.alt ) )
            moon.compute( obs )
            moon_alt_start = np.rad2deg( float( moon.alt ) )
            target_i.compute( obs )
            target_i_alt_start = np.rad2deg( float( target_i.alt ) )            

            # Do the same for the end of the observations:
            obs.date = ttr_i + dur_i*( 0.5 + oot_deltdur )
            sun.compute( obs )
            sun_alt_end = np.rad2deg( float( sun.alt ) )
            moon.compute( obs )
            moon_alt_end = np.rad2deg( float( moon.alt ) )
            target_i.compute( obs )
            target_i_alt_end = np.rad2deg( float( target_i.alt ) )

            # Determine the Sun, Moon and target elevations
            # at ingress:
            obs.date = ttr_i-0.5*dur_i
            sun.compute( obs )
            sun_alt_ingress = np.rad2deg( float( sun.alt ) )
            moon.compute( obs )
            moon_alt_ingress = np.rad2deg( float( moon.alt ) )
            target_i.compute( obs )
            target_i_alt_ingress = np.rad2deg( float( target_i.alt ) )            

            # Do the same for egress:
            obs.date = ttr_i+0.5*dur_i
            sun.compute( obs )
            sun_alt_egress = np.rad2deg( float( sun.alt ) )
            moon.compute( obs )
            moon_alt_egress = np.rad2deg( float( moon.alt ) )
            target_i.compute( obs )
            target_i_alt_egress = np.rad2deg( float( target_i.alt ) )

            # First category of transits are those where the entire transit and
            # requested out-of-transit baselines either side of ingress and egress
            # occur while the Sun is below the maximum acceptable altitude:
            if ( sun_alt_start<sun_alt_max )*( sun_alt_end<sun_alt_max ):

                # Case 1 = The Sun is always below our dark night time threshold:
                if ( sun_alt_start<sun_alt_dark )*( sun_alt_end<sun_alt_dark ):
                    trtype = 'full-all_in_darktime'

                # Case 2 = The Sun remains within the twilight range for the entire transit:
                elif ( sun_alt_start>sun_alt_dark )*( sun_alt_start<sun_alt_twil )*\
                     ( sun_alt_end>sun_alt_dark )*( sun_alt_end<sun_alt_twil ):
                    trtype = 'full-all_in_twilight'
                    
                # Case 3 = The Sun starts low enough to be classified as twilight,
                # but not low enough to be classified dark, and by the end of the
                # transit has descended low enough to be considered dark:         
                elif ( sun_alt_start<sun_alt_twil )*( sun_alt_start>sun_alt_dark )* \
                     ( sun_alt_end<sun_alt_dark ):
                    trtype = 'full-twilight_to_dark'

                # Case 4 = The Sun starts low enough to be considered at dark, but
                # by the end of the transit it has ascended enough that it is no
                # longer considered dark but twilight instead:
                elif ( sun_alt_end<sun_alt_twil )*( sun_alt_end>sun_alt_dark)* \
                     ( sun_alt_start<sun_alt_dark ):
                    trtype = 'full-dark_to_twilight'

                # Case 5 = At the start of the transit, the Sun is at an altitude
                # somewhere between its maximum acceptable value and the twilight
                # value, so we say the transit starts at 'dusk':
                elif ( sun_alt_start<sun_alt_max )*( sun_alt_start>sun_alt_twil ):
                    trtype = 'full-start_at_dusk'

                # Case 6 = At the end of the transit, the Sun is at an altitude
                # somewhere between its maximum acceptable value and the twilight
                # value, so we say the transit ends at 'dawn':
                elif ( sun_alt_end<sun_alt_max )*( sun_alt_end>sun_alt_twil ):
                    trtype = 'full-end_at_dawn'

                # Don't think there should be any other cases?
                else:
                    pdb.set_trace() #if this happens, need to work out what the Case should be
                    print 'backstop'
                    pdb.set_trace()

            # The following variations describe cases where we do not get the
            # full transit plus out-of-transit baseline:

            elif ( sun_alt_ingress<sun_alt_max )*( sun_alt_egress<sun_alt_max ):

                # Case 6 = The sky darkness was within our acceptable range for the
                # full duration of the transit but we don't get the full out-of-transit
                # baseline either before or after the transit within this range:
                trtype = 'full-partial_oot'

            elif ( sun_alt_ingress>sun_alt_max )*( sun_alt_end<sun_alt_max ):

                # Case 7 = At the start of the transit, the Sun was above our minimum
                # acceptable altitude, but by the time of mid-transit it had descended
                # below this level and remained acceptable until the end of the transit:
                trtype = 'partial-miss_ingress'

            elif ( sun_alt_start<sun_alt_max )*( sun_alt_egress>sun_alt_max ):

                # Case 8 = At the start of the transit, the Sun was above our minimum
                # acceptable altitude, but by the time of mid-transit it had descended
                # below this level and remained acceptable until the end of the transit:
                trtype = 'partial-miss_egress'

            else:
                # Case 9 = In the case of a transit lasting longer than the entire night,
                # the Sun altitude meets the minimum requirement at the transit mid-time,
                # but it's at an unacceptable altitude for both the start and end:
                trtype = 'partial-only_middle'

            # Now work out if the Moon is above or below the horizon:
            if ( moon_alt_start<moon_alt_set )*( moon_alt_end<moon_alt_set ):
                moonpos = 'moon-down'
                moonphase = '-'
                moondist = '-'
            else:
                if ( moon_alt_start>0 )*( moon_alt_end>0 ):
                    moonpos = 'moon-up'
                elif ( moon_alt_start<0 )*( moon_alt_end>moon_alt_set ):
                    moonpos = 'moon-rising'
                elif ( moon_alt_start>moon_alt_set )*( moon_alt_end<0 ):
                    moonpos = 'moon-setting'
                else:
                    pdb.set_trace() #this shouldn't happen

            # Determine the start and end times of transit in UT: 
            utc_tstart_tuple = ephem.date( ttr_i-0.5*dur_i ).tuple()
            utc_tstart_dt = datetime.datetime( int(utc_tstart_tuple[0]), \
                                               int(utc_tstart_tuple[1]), \
                                               int(utc_tstart_tuple[2]), \
                                               int(utc_tstart_tuple[3]), \
                                               int(utc_tstart_tuple[4]), \
                                               int(utc_tstart_tuple[5]), \
                                               tzinfo=pytz.utc )

            utc_tend_tuple = ephem.date( ttr_i+0.5*dur_i ).tuple()
            utc_tend_dt = datetime.datetime( int(utc_tend_tuple[0]), \
                                             int(utc_tend_tuple[1]), \
                                             int(utc_tend_tuple[2]), \
                                             int(utc_tend_tuple[3]), \
                                             int(utc_tend_tuple[4]), \
                                             int(utc_tend_tuple[5]), \
                                             tzinfo=pytz.utc )
            
            # Express the transit mid-time as a Modified Julian Date:
            mjd = ephem.julian_date( ttr_i ) - 2400000.5

            # Prepare and write output line to file:
            outstr_bp = make_outstr_bp( mjd, utc_tstart_dt, utc_tend_dt, zenith_i_midtime, \
                                        airmass, trtype, moonpos, moondist, moonphase )
            ofile_bp.write( outstr_bp )

            # Also save the lines to be written to other chronological output file later:
            mjds += [ mjd ]
            date_floats += [ ephem.Date( utc_tstart_dt )+1.0 ] # number of days since midday on 1 Jan 1900 

            outstr_ch = make_outstr_ch( targets[i], mjd, utc_tstart_dt, utc_tend_dt, zenith_i_midtime, \
                                        airmass, trtype, moonpos, moondist, moonphase )
            outstrs_ch += [ outstr_ch ]

            # Tick over to the next transit for the next loop:
            ttr_i += per_i

    # Now that we've identified all of the transits, sort them into
    # chronological order and write this information to output:
    header_str = '#\n#\n{0}\n'.format( '#'*nchar_ch )
    header_str += colheadingsa_ch
    header_str += colheadingsb_ch
    ofile_ch.write( header_str )
    ofile_ch.write( '{0}{1}\n'.format( '#', '-'*( nchar_ch-1 ) ) )
    mjds = np.array( mjds )
    date_floats = np.array( date_floats )
    ixs = np.argsort( mjds )
    for i in range( len( mjds ) ):
        j = ixs[i]
        df = np.floor( date_floats[j] )
        if i!=0:
            df_prev = np.floor( date_floats[ixs[i-1]] )
            if df-df_prev>=1.0:
                ofile_ch.write( '#{0}\n'.format( '-'*( nchar_ch-1 ) ) )
        ofile_ch.write( outstrs_ch[j] )
    ofile_ch.write( '{0}{1}\n'.format( '#', '-'*( nchar-1 ) ) )

    # Save the output files and finish:
    ofile_bp.close()
    ofile_ch.close()
    print '\nSaved output in:'
    print '  %s' % ofilename_byplanet
    print '  %s' % ofilename_chronolog

    return ofilename_byplanet, ofilename_chronolog


def make_eph( exclude_unconfirmed=True ):
    """
    Generates the ephemerides file for all of the transiting exoplanets in
    the exoplanets.org database.
    """

    # Get table data:
    tr_file = 'exoplanets_transiting.fits'
    if os.path.isfile( tr_file )==False:
        tutilities.download_data()
    t = atpy.Table( tr_file )

    # Open and prepare file for output writing to:
    eph_file_w = open( EPH_FILE, 'w' )
    header_str =  '#  Transiting planet positions and epochs \n'
    header_str += '#  Generated from exoplanet.org data \n'
    header_str += '#  Comment out those not needed \n\n'
    header_str += '#  COLUMNS: \n'
    header_str += '#  Name, Vmag, RA, Dec, Epoch(HJD), Period(days), Duration(hrs) \n\n'
    eph_file_w.write( header_str )

    # Go through each of the planets alphabetically and extract
    # the necessary information:
    q = np.argsort( t.NAME )
    for i in range( t.NAME.size ):
        if t.NAME[ q[i] ].find( 'KOI' )>=0:
            confirmed = False
        else:
            confirmed = True
        if ( confirmed==False )*( exclude_unconfirmed==True ):
            continue
        ostr = '{0:15s}  {1:.1f}  {2:s}  {3:s}  {4:15.7f}  {5:13.8f}  {6:8.4f} \n'\
               .format( t.NAME[ q[i] ].replace(' ',''), t.V[ q[i] ], t.RA_STRING[ q[i] ], \
                       t.DEC_STRING[ q[i] ], t.TT[ q[i] ], t.PER[ q[i] ], t.T14[ q[i] ]*24. )
        eph_file_w.write( ostr )
    eph_file_w.close()
    print '\n\nSaved output in {0}'.format( EPH_FILE )

    return None


def setup_observatory( obs ):
    """
    Initialises a pyephem Observer() object using the details
    of a predefined observatory specified by as a string by
    the input argument obs. Currently recognised observatories:
       'Paranal', 'LaSilla', 'MaunaKea', 'SidingSpring',
       'KittPeak', 'CalarAlto', 'Gemini-N', 'Gemini-S'
    """

    obs_obj = ephem.Observer()

    # Pre-defined observatory with string identifier:
    if type( obs )==str:
        obs_db = observatories()
        try:
            obs_dict = obs_db[ obs ]
            obs_obj.lat = obs_dict['lat']
            obs_obj.long = obs_dict['long']
            obs_obj.elevation = obs_dict['altitude-metres']
            timezone = obs_dict['timezone']
        except:
            print '\n\nObservatory string does not match any in database!'
            print 'Currently available observatories are:'
            for i in obs_db.keys():
                print '  {0}'.format( i )
            obs_obj = None
            timezone = None

    # Custom-defined observatory as dictionary:
    else:
        obs_obj.lat = obs['lat']
        obs_obj.long = obs['long']
        try:
            obs_obj.elevation = obs['altitude-metres']
        except:
            print 'No elevation provided - assuming sea level'
            obs_obj.elevation = 0.
        try:
            timezone = obs['timezone']
        except:
            timezone = None

    return obs_obj, timezone


def observatories():
    """
    Returns a dictionary of dictionaries, each of which contain
    information about the coordinates, elevation and time zone
    of a different observatory.

     NOTES:
      - Elevations are measured in metres.
      - Positive latitudes are North, negative latitudes are South
      - Positive longitudes are East, negative longitudes are West    
      - To find a new timezone look up the
        country abbreviation in the ISO 3166-1
        catalogue (eg. on Wikipedia) then enter:
          eg. pytz.country_timezones['es']
        to see which timezones are available.
    """

    obs_db = {}

    obs_db['PWT-Oxford'] = { 'long':'-01:15:00', \
                             'lat':'+51:45:00', \
                             'altitude-metres':130.0, \
                             'timezone':'Europe/London' }

    obs_db['LaPalma'] = { 'lat':'+28:45:00', \
                          'long':'-17:53:00', \
                          'altitude-metres':2326, \
                          'timezone':'Atlantic/Canary' }
    
    obs_db['Paranal'] = { 'lat':'-24:37:00', \
                          'long':'-70:24:00', \
                          'altitude-metres':2635, \
                          'timezone':'America/Santiago' }

    obs_db['LaSilla'] = { 'lat':'-29:15:00', \
                          'long':'-70:44:00', \
                          'altitude-metres':2380, \
                          'timezone':'America/Santiago' }

    obs_db['CerraTololo'] = { 'lat':'-31:10:10.8', \
                          'long':'-70:48:23.5', \
                          'altitude-metres':2207, \
                          'timezone':'America/Santiago' }

    obs_db['MaunaKea'] = { 'lat':'+19:50:00', \
                           'long':'-155:28:00', \
                           'altitude-metres':4190, \
                           'timezone':'Pacific/Honolulu' }
    
    obs_db['SidingSpring'] = { 'lat':'-31:16:00', \
                               'long':'+149:04:00', \
                               'altitude-metres':1149, \
                               'timezone':'Australia/Sydney' }
     
    obs_db['KittPeak'] = { 'lat':'+31:58:00', \
                           'long':'-111:36:00', \
                           'altitude-metres':2096, \
                           'timezone':'America/Phoenix' }

    obs_db['CalarAlto'] = { 'lat':'+37:13:25', \
                            'long':'-2:32:47', \
                            'altitude-metres':2168, \
                            'timezone':'Europe/Madrid' }
     
    obs_db['Gemini-N'] = { 'lat':'+19:49:26', \
                           'long':'-155:28:09', \
                           'altitude-metres':4213, \
                           'timezone':'Pacific/Honolulu' }

    obs_db['Gemini-S'] = { 'lat':'-30:14:27', \
                           'long':'-70:44:12', \
                           'altitude-metres':2722, \
                           'timezone':'America/Santiago' }

    obs_db['HauteProvence'] = { 'lat':'+43:55:51', \
                            'long':'5:42:48', \
                            'altitude-metres':650, \
                            'timezone':'Europe/Paris' }
    return obs_db


def jd2pyephemdate( jd ):
    """
    Converts a Julian date to a pyephem date object by first
    subtracting the difference between the pyephem zero date
    and the JD zero date.
    """
    pyephem_zero = ephem.julian_date( ephem.Date( 0 ) )
    return  ephem.Date( jd - pyephem_zero )


def calc_airmass( zenith_angle ):
    """
    Takes the angle between zenith and the target
    and returns the airmass.
    """
    airmass = 1.0 / np.cos( np.deg2rad( zenith_angle ) )
    return airmass


def make_colheadings( output_type ):
    """
    Creates strings to be used for output file column headings.
    """
    col0a = 'Target '.rjust( 11 )
    col0b = ''.rjust( 11 )
    col1a = 'Epoch'.center( 8 )
    col1b = '(MJD)'.center( 8 )
    col2a = 'Time Start'.center( 19 )
    col2b = '(UT)'.center( 19 )
    col3a = 'Time End'.center( 19 )
    col3b = '(UT)'.center( 19 )
    col4a = 'Zenith'.center( 6 )
    col4b = '(deg)'.center( 6 )
    col5a = 'Airm'.center( 4 )
    col5b = ''.center( 4 )
    col6a = 'Transit-type'.center( 21 )
    col6b = ''.center( 21 )
    col7a = 'Moon-type'.center( 12 )
    col7b = ''.center( 12 )
    col8a = 'Moon-dist'.center( 9 )
    col8b = '(deg)'.center( 9 )
    col9a = 'Moon-phase'.center( 10 )
    col9b = '(percent)'.center( 10 )
    if output_type=='byplanet':
        colheadingsa = '#{0}  {1}  {2}  {3} {4} {5} {6} {7} {8}\n'\
                       .format( col1a, col2a, col3a, col4a, col5a, \
                                col6a, col7a, col8a, col9a )
        colheadingsb = '#{0}  {1}  {2}  {3} {4} {5} {6} {7} {8}\n'\
                       .format( col1b, col2b, col3b, col4b, col5b, \
                                col6b, col7b, col8b, col9b )
    elif output_type=='chronolog':
        colheadingsa = '#{0} {1}  {2}  {3}  {4} {5} {6} {7} {8} {9}\n'\
                       .format( col0a, col1a, col2a, col3a, col4a, col5a, \
                                col6a, col7a, col8a, col9a )
        colheadingsb = '#{0} {1}  {2}  {3}  {4} {5} {6} {7} {8} {9}\n'\
                       .format( col0b, col1b, col2b, col3b, col4b, col5b, \
                                col6b, col7b, col8b, col9b )
    return colheadingsa, colheadingsb


def make_outstr_bp( mjd, utc_tstart_dt, utc_tend_dt, zenith_midtime, airmass, \
                    trtype, moonpos, moondist, moonphase ):
    """
    Takes quantities that will be written to output and formats them nicely.
    """

    mjd_str = '{0:.2f}'.format( mjd ).center( 8 )
    
    utc_tstart_str = '{0:04d}:{1:02d}:{2:02d}:{3:02d}:{4:02d}:{5:02d}'\
                       .format( utc_tstart_dt.year, \
                                utc_tstart_dt.month, \
                                utc_tstart_dt.day, \
                                utc_tstart_dt.hour, \
                                utc_tstart_dt.minute, \
                                utc_tstart_dt.second )
    utc_tstart_str = utc_tstart_str.center( 19 )
    
    utc_tend_str = '{0:04d}:{1:02d}:{2:02d}:{3:02d}:{4:02d}:{5:02d}'\
                     .format( utc_tend_dt.year, \
                              utc_tend_dt.month, \
                              utc_tend_dt.day, \
                              utc_tend_dt.hour, \
                              utc_tend_dt.minute, \
                              utc_tend_dt.second )
    utc_tend_str = utc_tend_str.center( 19 )

    zenith_str = '{0:d}'.format( int( np.round( zenith_midtime ) ) ).center( 6 )
    airmass_str = '{0:.2f}'.format( airmass ).center( 4 )
    trtype_str = trtype.center( 21 )
    moonpos_str = moonpos.center( 12 )
    moondist_str = moondist.center( 9 )
    moonphase_str = moonphase.center( 10 )
    outstr = ' {0}  {1}  {2}  {3} {4} {5} {6} {7} {8}\n'\
             .format( mjd_str, \
                      utc_tstart_str, \
                      utc_tend_str, \
                      zenith_str, \
                      airmass_str, \
                      trtype_str, \
                      moonpos_str, \
                      moondist_str, \
                      moonphase_str )
    
    return outstr 

def make_outstr_ch( target, mjd, utc_tstart_dt, utc_tend_dt, zenith_midtime, airmass, \
                    trtype, moonpos, moondist, moonphase ):
    """
    Takes quantities that will be written to output and formats them nicely.
    """

    target_str = target.replace( ' ', '' ).rjust( 11 )
    mjd_str = '{0:.2f}'.format( mjd ).center( 8 )
    
    utc_tstart_str = '{0:04d}:{1:02d}:{2:02d}:{3:02d}:{4:02d}:{5:02d}'\
                       .format( utc_tstart_dt.year, \
                                utc_tstart_dt.month, \
                                utc_tstart_dt.day, \
                                utc_tstart_dt.hour, \
                                utc_tstart_dt.minute, \
                                utc_tstart_dt.second )
    utc_tstart_str = utc_tstart_str.center( 19 )
    
    utc_tend_str = '{0:04d}:{1:02d}:{2:02d}:{3:02d}:{4:02d}:{5:02d}'\
                     .format( utc_tend_dt.year, \
                              utc_tend_dt.month, \
                              utc_tend_dt.day, \
                              utc_tend_dt.hour, \
                              utc_tend_dt.minute, \
                              utc_tend_dt.second )
    utc_tend_str = utc_tend_str.center( 19 )

    zenith_str = '{0:d}'.format( int( np.round( zenith_midtime ) ) ).center( 6 )
    airmass_str = '{0:.2f}'.format( airmass ).center( 4 )
    trtype_str = trtype.center( 21 )
    moonpos_str = moonpos.center( 12 )
    moondist_str = moondist.center( 9 )
    moonphase_str = moonphase.center( 10 )
    outstr = '{0}  {1}  {2}  {3}  {4} {5} {6} {7} {8} {9}\n'\
             .format( target_str, \
                      mjd_str, \
                      utc_tstart_str, \
                      utc_tend_str, \
                      zenith_str, \
                      airmass_str, \
                      trtype_str, \
                      moonpos_str, \
                      moondist_str, \
                      moonphase_str )
    
    return outstr 


def read_eph( eph_file ):

    ifile = open( eph_file, 'r' )
    targets = []
    vmags = []
    ras = []
    decs = []
    ttrs = []
    pers = []
    durs = []
    for line in ifile:
        if ( line[0]=='#' ) or ( line[0]=='\n' ) or ( line[0]=='' ):
            continue
        else:
            entries = line.split()
            if len( entries )==7:
                targets += [ str( line.split()[0] ) ]
                vmags   += [ float( line.split()[1] ) ]
                ras     += [ str( line.split()[2] ) ]            
                decs    += [ str( line.split()[3] ) ]
                ttrs    += [ float( line.split()[4] ) ]
                pers    += [ float( line.split()[5] ) ]
                durs    += [ float( line.split()[6] ) ]
            else:
                pdb.set_trace() # something not right about current entry
    ifile.close()

    return targets, vmags, ras, decs, ttrs, pers, durs


def eclipse_ranks( ec_signals, targets_all, ttrs_all, pers_all, durs_all ):
    """
    **ec_signals - ASCII file containing a ranked list of the signals, with
    columns corresponding to:
     Rank, Name, RA, Dec, Kmag, Tstar, Rstar, Rp, a, Tpeq, Fp/Fs, S/N
    
    **targets_all - List of strings containing the database names for each
    planet considered to be a suitable target
    **pers_all - List containing the orbital periods for the corresponding
    targets listed in tagets_all
    **durs_all - List containing the eclipse duractions for the corresponding
    targets listed in tagets_all
    """
    ntargets_all = len( targets_all )
    ec_file = open( ec_signals, 'r' )
    targets_ec = []
    ranks_ec = []
    ttrs = []
    pers = []
    durs = []
    ec_file.seek( 0 )
    for line in ec_file:
        if line[0]=='#':
            continue
        ranks_ec += [ line.split()[0] ]
        target_ec = line.split()[1]
        targets_ec += [ target_ec ]
        for i in range( ntargets_all ):
            if targets_all[i]==target_ec:
                ttrs += [ ttrs_all[i] ]
                pers += [ pers_all[i] ]
                durs += [ durs_all[i] ]
                break
            elif i==ntargets_all-1:
                print target_ec
                pdb.set_trace()
    ec_file.close()
    
    return targets_ec, ranks_ec, ttrs, pers, durs


def transit_ranks( tr_signals, targets_all, ttrs_all, pers_all, durs_all ):
    ntargets_all = len( targets_all )
    tr_file = open( tr_signals, 'r' )
    targets_tr = []
    ranks_tr = []
    ttrs = []
    pers = []
    durs = []
    tr_file.seek( 0 )
    for line in tr_file:
        if line[0]=='#':
            continue
        ranks_tr += [ line.split()[0] ]
        target_tr = line.split()[1]
        targets_tr += [ target_tr ]
        for i in range( ntargets_all ):
            if targets_all[i]==target_tr:
                ttrs += [ ttrs_all[i] ] 
                pers += [ pers_all[i] ] 
                durs += [ durs_all[i] ]
                break
            elif i==ntargets_all-1:
                print target_tr
                pdb.set_trace()
    tr_file.close()

    return targets_tr, ranks_tr, ttrs, pers, durs


