#!/usr/bin/env python3
"""Usage:
    history_dlm_to_history_dlp.py [--con <con> --in <histin> --out <histout> --dt <dt> --mscale <mscale> --lscale <lscale> --tscale <tscale> --first <first> --last <last> --step <step>]

Converts DL_MESO_DPD HISTORY file into DL_POLY_4 HISTORY format

Options:
    --con <con>         Use CONTROL-format file named <con> to find timestep size
                        [default: CONTROL]
    --in <histin>       Name of DL_MESO_DPD-format HISTORY file to read in and
                        convert [default: HISTORY]
    --out <histout>     Name of DL_POLY_4-format HISTORY file to write [default: HISTORY.dlp]
    --dt <dt>           Timestep size used in simulations, needed if no CONTROL
                        file present from which to read value [default: 0.0]
    --mscale <mscale>   DPD mass scale in daltons or unified atomic mass units [default: 1.0]
    --lscale <lscale>   DPD length scale in Angstroms [default: 1.0]
    --tscale <tscale>   DPD time scale in picoseconds [default: 1.0]
    --first <first>     Starting DL_MESO_DPD HISTORY file frame number for inclusion
                        in DL_POLY_4-formatted HISTORY file [default: 1]
    --last <last>       Finishing DL_MESO_DPD HISTORY file frame number for inclusion
                        in DL_POLY_4-formatted HISTORY file (value of 0 here will use last
                        available frame) [default: 0]
    --step <step>       Incrementing number of frames in DL_MESO_DPD HISTORY between
                        frames in DL_POLY_4-formatted HISTORY file [default: 1]

michael.seaton@stfc.ac.uk, 15/08/21
"""
from docopt import docopt
from tqdm import tqdm
import dlmhistoryread as dlm
import numpy as np
import math
import os
import sys

def images(rr, box, inv_box, dLx, dLy, dLz, shrx, shry, shrz):
    """Calculates minimum change in particle positions based on periodic boundary conditions"""
    G, Gn, rrn = np.zeros(3), np.zeros(3), np.zeros(3)
    cor = 0.0

    # check for lees-edwards shearing boundaries first
    if shrx:
        cor = round(rr[0]*inv_box[0][0])
        rr[1] = rr[1] - cor * dLy
        rr[2] = rr[2] - cor * dLz
    elif shry:
        cor = round(rr[1]*inv_box[1][1])
        rr[0] = rr[0] - cor * dLx
        rr[2] = rr[2] - cor * dLz
    elif shrz:
        cor = round(rr[2]*inv_box[2][2])
        rr[0] = rr[0] - cor * dLx
        rr[1] - rr[1] - cor * dLy
    
    # apply periodic minimum image

    G = inv_box @ rr
    Ground = np.empty_like(G)
    np.round(G, 0, Ground)
    Gn = G - Ground
    rrn = box @ Gn

    return rrn


def history_dlp_write_frame(fw, nstep, keytrj, dt, time, particledata, particleprop, speciesprop, rsd, tscale, lscale, mscale, vscale, fscale):

    # writes frame to DL_POLY_4 HISTORY file (assuming it is already open)
    
    nbeads = len(particledata)
    
    # write header for frame with timestep number, number of particles, trajectory
    # key, boundary condition key (assuming orthorhombic boxes - only option in DL_MESO_DPD),
    # timestep size, time of frame and simulation box size
    
    fw.write('timestep{0:10d}{1:10d}{2:2d}{3:2d}{4:20.6f}{5:20.6f}\n'.format(nstep, nbeads, keytrj, 2, dt*tscale, time*tscale))
    fw.write('{0:20.10f}{1:20.10f}{2:20.10f}            \n'.format(dimx*lscale, 0.0, 0.0))
    fw.write('{0:20.10f}{1:20.10f}{2:20.10f}            \n'.format(0.0, dimy*lscale, 0.0))
    fw.write('{0:20.10f}{1:20.10f}{2:20.10f}            \n'.format(0.0, 0.0, dimz*lscale))

    # run through particles and write record for each to file, working out
    # species name, particle mass and charge, including root-squared-displacement
    # and all particle data based on values available from DL_MESO_DPD HISTORY file

    for i in range(nbeads):
        glob = particledata[i][0]
        spec = particleprop[glob-1][1] - 1
        name = speciesprop[spec][0]
        mass = speciesprop[spec][1]*mscale
        qi = speciesprop[spec][3]
        fw.write('{0:8s}{1:10d}{2:12.6f}{3:12.6f}{4:12.6f}                  \n'.format(name, glob, mass, qi, rsd[i]*lscale))
        fw.write('{0:20.10f}{1:20.10f}{2:20.10f}            \n'.format(particledata[i][1]*lscale, particledata[i][2]*lscale, particledata[i][3]*lscale))
        if(keytrj>0):
            fw.write('{0:20.10f}{1:20.10f}{2:20.10f}            \n'.format(particledata[i][4]*vscale, particledata[i][5]*vscale, particledata[i][6]*vscale))
        if(keytrj>1):
            fw.write('{0:20.10f}{1:20.10f}{2:20.10f}            \n'.format(particledata[i][7]*fscale, particledata[i][8]*fscale, particledata[i][9]*fscale))



if __name__ == '__main__':
    # first check command-line arguments, including timestep size (in
    # DPD units) and mass, length and time scales of DPD simulation

    args = docopt(__doc__)
    dt = float(args["--dt"])
    con = args["--con"]
    histin = args["--in"]
    histout = args["--out"]
    mscale = float(args["--mscale"])
    lscale = float(args["--lscale"])
    tscale = float(args["--tscale"])
    first = int(args["--first"])
    last = int(args["--last"])
    step = int(args["--step"])

    # if no timestep size found in command-line, try looking for a CONTROL
    # file and read the value from there

    if(dt<=0.0 and os.path.isfile(con)):
        fin = open(con,"r")
        for line in fin:
            words = line.replace(',',' ').replace('\t',' ').lower().split()
            if(len(words)>0):
                if(words[0].startswith('timestep')):
                    dt = float(words[1])

    # if no luck with CONTROL file, ask user directly for timestep size

    while (dt<=0.0):
        test_text = input ("Enter the timestep size used in simulation: ")
        dt = float(test_text)

    # read very beginning of DL_MESO_DPD HISTORY file to determine endianness,
    # sizes of number types, filesize, number of frames and last timestep number
    
    bo, ri, rd, intsize, longintsize, realsize, filesize, numframe, nsteplast = dlm.read_prepare(histin)

    # if not specified last frame to use in command-line argument or value for
    # first and/or last frames are too small/large, reset to defaults:
    # also work out how many frames are going into DL_POLY-style HISTORY file

    if numframe<1:
        sys.exit("No trajectory data available in "+histin+" file to write to DL_POLY-style HISTORY file")

    if first>numframe:
        first = numframe

    first = first - 1

    if last==0 or last<first:
        last = numframe

    numframes = (last - first - 1) // step + 1

    # get scaling factors for velocities and forces

    vscale = lscale / tscale
    fscale = mscale * vscale / tscale

    # get information from DL_MESO_DPD HISTORY file header, including species,
    # molecule, particle and bond data
    
    nsyst, nusyst, timefirst, timelast, text, keytrj, surfaceprop, speciesprop, moleculeprop, particleprop, bondtable, framesize, headerpos = dlm.read_header(histin, bo, ri, rd, intsize, longintsize, realsize, numframe)
    
    # find timestep number for first frame
    
    stepdiff = math.floor((timelast-timefirst)/dt+0.5)
    nstepfirst = nsteplast - stepdiff

    # open and write header for DL_POLY HISTORY file
    
    fw = open(histout, "w")
    fw.write(text[:72]+"\n")
    fw.write('{0:10d}{1:10d}{2:10d}{3:21d}{4:21d}\n'.format(keytrj, 2, nsyst, numframes, numframes*nsyst*(keytrj+2)+4*numframes+2))

    # prepare arrays for particle positions and distances between frames

    xyz0 = np.zeros((nsyst, 3))
    dxyz = np.zeros(3)
    adisp = np.zeros((nsyst, 3))

    # setup flags for lees-edwards boundary conditions
    # (no need to directly account for reflecting walls)

    lex = surfaceprop[0]==1
    ley = surfaceprop[1]==1
    lez = surfaceprop[2]==1

    # major loop through all required frames in DL_MESO_DPD HISTORY file
    
    for frame in tqdm(range(first, last, step)):
        # get all available information from DL_MESO_DPD HISTORY frame
        time, dimx, dimy, dimz, shrdx, shrdy, shrdz, particledata = dlm.read_frame(histin, ri, rd, frame, framesize, headerpos, keytrj)
        L = np.asfarray([dimx, dimy, dimz], np.double)
        box = L * np.eye(3)
        inv_box = np.linalg.pinv(box)
        # get hold of root-squared-displacements for current frame 
        # relative to first selected frame  
        rsd = []
        for i in range(nsyst):
            if frame==first:
                rsdvalue = 0.0
            else:
                dxyz = particledata[i][1:4] - xyz0[i][0:3]
                dxyz = images(dxyz, box, inv_box, shrdx, shrdy, shrdz, lex, ley, lez)
                adisp[i] = adisp[i] + dxyz
                rsdvalue = math.sqrt(adisp[i][0]*adisp[i][0] + adisp[i][1]*adisp[i][1] + adisp[i][2]*adisp[i][2])
            rsd.append(rsdvalue)
            xyz0[i] = particledata[i][1:4].copy()
        # calculate timestep from specified time
        nstep = nstepfirst + math.floor((time-timefirst)/dt+0.5)
        # write frame data to DL_POLY HISTORY file
        history_dlp_write_frame(fw, nstep, keytrj, dt, time, particledata, particleprop, speciesprop, rsd, tscale, lscale, mscale, vscale, fscale)
        
    fw.close()

