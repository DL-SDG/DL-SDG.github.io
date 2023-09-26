#!/usr/bin/env python3
"""Usage:
    history_dlm_to_config.py [--in <histin> --out <conout> --mscale <mscale> --lscale <lscale> --tscale <tscale> --frame <frame> --key <key>]

Converts a frame from DL_MESO_DPD HISTORY file into DL_MESO_DPD/DL_POLY CONFIG 
format to initialise new calculation

Options:
    --in <histin>       Name of DL_MESO_DPD-format HISTORY file to read in and
                        convert [default: HISTORY]
    --out <conout>      Name of DL_MESO_DPD/DL_POLY-format CONFIG file to write [default: CONFIG]
    --mscale <mscale>   DPD mass scale in daltons or unified atomic mass units [default: 1.0]
    --lscale <lscale>   DPD length scale in Angstroms [default: 1.0]
    --tscale <tscale>   DPD time scale in picoseconds [default: 1.0]
    --frame <frame>     DL_MESO_DPD HISTORY file frame number to use for DL_MESO_DPD/
                        DL_POLY-formatted CONFIG file (value of 0 here will use last
                        available frame) [default: 0]
    --key <key>         Particle data level key to use in DL_MESO_DPD/DL_POLY-formatted CONFIG
                        file (0 = positions, 1 = positions and velocities, 2 = positions,
                        velocities and forces) - maximum value depends on available data in
                        HISTORY file [default: 0]

michael.seaton@stfc.ac.uk, 19/10/22
"""
from docopt import docopt
from tqdm import tqdm
import dlmhistoryread as dlm
import sys

def config_write(fw, text, levcfg, dimx, dimy, dimz, particledata, particleprop, speciesprop, lscale, vscale, fscale):

    # writes particle data to DL_MESO_DPD/DL_POLY CONFIG file (assuming it is already open)
    
    nbeads = len(particledata)
    
    # start with simulation title at top of file

    fw.write(text[:80]+"\n")

    # write CONFIG data key, boundary condition key (not used by DL_MESO_DPD) 
    # and total number of particles (also not directly used by DL_MESO_DPD)

    fw.write('{0:10d} {1:10d} {2:10d}\n'.format(levcfg, 2, nbeads))

    # write simulation box size (rescaling with lengthscale if required)
    
    fw.write('{0:20.10f}{1:20.10f}{2:20.10f}\n'.format(dimx*lscale, 0.0, 0.0))
    fw.write('{0:20.10f}{1:20.10f}{2:20.10f}\n'.format(0.0, dimy*lscale, 0.0))
    fw.write('{0:20.10f}{1:20.10f}{2:20.10f}\n'.format(0.0, 0.0, dimz*lscale))

    # run through particles and write record for each to file, including
    # species name and global index for each particle

    for i in range(nbeads):
        glob = particledata[i][0]
        spec = particleprop[glob-1][1] - 1
        name = speciesprop[spec][0]
        fw.write('{0:8s}{1:10d}\n'.format(name, glob))
        fw.write('{0:20.10f}{1:20.10f}{2:20.10f}\n'.format(particledata[i][1]*lscale, particledata[i][2]*lscale, particledata[i][3]*lscale))
        if(levcfg>0):
            fw.write('{0:20.10f}{1:20.10f}{2:20.10f}\n'.format(particledata[i][4]*vscale, particledata[i][5]*vscale, particledata[i][6]*vscale))
        if(levcfg>1):
            fw.write('{0:20.10f}{1:20.10f}{2:20.10f}\n'.format(particledata[i][7]*fscale, particledata[i][8]*fscale, particledata[i][9]*fscale))



if __name__ == '__main__':
    # first check command-line arguments, including timestep size (in
    # DPD units) and mass, length and time scales of DPD simulation

    args = docopt(__doc__)
    histin = args["--in"]
    conout = args["--out"]
    mscale = float(args["--mscale"])
    lscale = float(args["--lscale"])
    tscale = float(args["--tscale"])
    frame = int(args["--frame"])
    levcfg = int(args["--key"])

    # read very beginning of DL_MESO_DPD HISTORY file to determine endianness,
    # sizes of number types, filesize, number of frames and last timestep number
    
    bo, ri, rd, intsize, longintsize, realsize, filesize, numframe, nsteplast = dlm.read_prepare(histin)

    # if not specified last frame to use in command-line argument or value for
    # first and/or last frames are too small/large, reset to last available frame
    # and tell user how many frames are available in HISTORY file

    print('Number of frames available in supplied {0:s} file: {1:d}'.format(histin, numframe))
    if numframe<1:
        sys.exit("No trajectory data available in "+histin+" file to write to CONFIG file")

    if frame<1 or frame>numframe:
        frame = numframe

    print('Selecting frame {0:d} to write to CONFIG file'.format(frame))
    frame = frame - 1

    # get scaling factors for velocities and forces from those for lengths, times and masses

    vscale = lscale / tscale
    fscale = mscale * vscale / tscale

    # get information from DL_MESO_DPD HISTORY file header, including species,
    # molecule, particle and bond data, and data level
    
    nsyst, nusyst, timefirst, timelast, text, keytrj, surfaceprop, speciesprop, moleculeprop, particleprop, bondtable, framesize, headerpos = dlm.read_header(histin, bo, ri, rd, intsize, longintsize, realsize, numframe)
    
    # check data level in HISTORY file is compatible with selection for CONFIG
    # file: if not, revert to highest available level in HISTORY file and warn
    # user of that fact

    if levcfg > keytrj:
        print('WARNING: particle data level in {0:s} insufficiently high for requested level in CONFIG file ({1:d})'.format(histin, levcfg))
        levcfg = keytrj
        print('         reverting to {0:d} (particle {1:s})'.format(keytrj, "positions and velocities" if keytrj==1 else "positions"))

    # open CONFIG file
    
    fw = open(conout, "w")

    # get all available information from selected DL_MESO_DPD HISTORY frame
    time, dimx, dimy, dimz, shrdx, shrdy, shrdz, particledata = dlm.read_frame(histin, ri, rd, frame, framesize, headerpos, keytrj)
    # write frame data to CONFIG file
    config_write(fw, text, levcfg, dimx, dimy, dimz, particledata, particleprop, speciesprop, lscale, vscale, fscale)

    # close CONFIG file
    fw.close()

