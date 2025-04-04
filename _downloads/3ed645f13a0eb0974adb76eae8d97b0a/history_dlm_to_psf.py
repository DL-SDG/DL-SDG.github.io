#!/usr/bin/env python3
"""Usage:
    history_dlm_to_psf.py [--in <histin> --out <histout> --mscale <mscale>]

Converts header of DL_MESO_DPD HISTORY file into PSF (protein structure file) format for use with DCD trajectory files

Options:
    --in <histin>       Name of DL_MESO_DPD-format HISTORY file to read in and
                        convert [default: HISTORY]
    --out <psfout>      Name of PSF file to write [default: TRAJOUT.psf]
    --mscale <mscale>   DPD mass scale in daltons or unified atomic mass units [default: 1.0]

michael.seaton@stfc.ac.uk, 26/08/21
"""
from docopt import docopt
from tqdm import tqdm
import dlmhistoryread as dlm
import numpy as np


if __name__ == '__main__':
    # first check command-line arguments, including timestep size (in
    # DPD units) and mass, length and time scales of DPD simulation

    args = docopt(__doc__)
    histin = args["--in"]
    psfout = args["--out"]
    mscale = float(args["--mscale"])

    # read very beginning of DL_MESO_DPD HISTORY file to determine endianness,
    # sizes of number types, filesize, number of frames and last timestep number
    
    bo, ri, rd, intsize, longintsize, realsize, filesize, numframe, nsteplast = dlm.read_prepare(histin)

    nsyst, nusyst, timefirst, timelast, text, keytrj, surfaceprop, speciesprop, moleculeprop, particleprop, bondtable, framesize, headerpos = dlm.read_header(histin, bo, ri, rd, intsize, longintsize, realsize, numframe)
    
    # start with PSF file: open and write first line and then headers for simulation
    
    fw = open(psfout, "w")
    fw.write("PSF EXT\n\n")
    fw.write("{0:10d} !NTITLE\n".format(2))
    fw.write(text+"\n")
    fw.write("Generated by history_dlm_to_psf.py for DL_MESO_DPD (author: M A Seaton)         \n\n")

    # write particle properties
    
    fw.write("{0:10d} !NATOM\n".format(nsyst))
    
    for i in range(nsyst):
        spec = particleprop[i][1]-1
        mole = particleprop[i][2]-1
        molnum = particleprop[i][3]
        segid = 'M' if mole>=0 else 'S'
        res = moleculeprop[mole] if mole>=0 else 'NOTMOLE '
        move = 1 if speciesprop[spec][4] else 0
        fw.write("{0:10d} {1:8s} {2:8d} {3:8s} {4:8d} {5:8d} {6:8s} {7:14.6f}{8:14.6f}{9:8d}\n".format(i+1, segid, particleprop[i][3], res, molnum, spec+1, speciesprop[spec][0], speciesprop[spec][2], speciesprop[spec][1]*mscale, move))
    
    fw.write("\n")
    
    # if available, write bonds to file (four pairs per line)
    
    if len(bondtable)>0:
        fw.write("{0:10d} !NBOND\n".format(len(bondtable)))
        for i in range(len(bondtable)):
            fw.write("{0:10d} {1:10d} ".format(bondtable[i][0], bondtable[i][1]))
            if (i+1) % 4 == 0 or i == len(bondtable):
                fw.write("\n")
        fw.write("\n")
    
    # close file
    
    fw.close()
