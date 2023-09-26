#!/usr/bin/env python3
"""Usage:
    history_dlm_to_msd.py [--in <histin> --out <msdout> --lscale <lscale> --tscale <tscale> --first <first> --last <last> --step <step> --block <block>]

Calculates mean-squared displacements (MSDs) from DL_MESO_DPD HISTORY file 
for all particle species, estimates self-diffusivities from MSDs

Options:
    --in <histin>       Name of DL_MESO_DPD-format HISTORY file to read in and
                        convert [default: HISTORY]
    --out <msdout>      Name of file to write for MSD data [default: MSDDAT]
    --lscale <lscale>   DPD length scale in Angstroms [default: 1.0]
    --tscale <tscale>   DPD time scale in picoseconds [default: 1.0]
    --first <first>     Starting DL_MESO_DPD HISTORY file frame number for inclusion
                        in MSD calculations [default: 1]
    --last <last>       Finishing DL_MESO_DPD HISTORY file frame number for inclusion
                        in MSD calculations (value of 0 here will use last available 
                        frame) [default: 0]
    --step <step>       Incrementing number of frames in DL_MESO_DPD HISTORY between
                        frames used for MSD calculations [default: 1]
    --block <block>     Number of frames to block-average MSDs when calculating
                        self-diffusivities [default: 10]

michael.seaton@stfc.ac.uk, 12/10/22
"""
from docopt import docopt
from tqdm import tqdm
import dlmhistoryread as dlm
import numpy as np
from numba import jit, float64, int64
import math
import sys

@jit(float64(float64[:]), nopython=True)
def squarenorm_numba(r):
    """Calculates square of scalar distance from vector (using Numba JIT compiler to speed this up)"""
    rn = 0.0
    for ri in r:
        rn += ri * ri
    return rn


@jit(nopython=True)
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

if __name__ == '__main__':
    # first check command-line arguments

    args = docopt(__doc__)
    histin = args["--in"]
    msdout = args["--out"]
    lscale = float(args["--lscale"])
    tscale = float(args["--tscale"])
    first = int(args["--first"])
    last = int(args["--last"])
    step = int(args["--step"])
    block = int(args["--block"])
    
    # read very beginning of DL_MESO_DPD HISTORY file to determine endianness,
    # sizes of number types, filesize, number of frames and last timestep number
    
    bo, ri, rd, intsize, longintsize, realsize, filesize, numframe, nsteplast = dlm.read_prepare(histin)

    # if not specified last frame to use in command-line argument or value for
    # first and/or last frames are too small/large, reset to defaults:
    # also work out how many frames are going to be used

    if numframe<1:
        sys.exit("No trajectory data available in "+histin+" file to calculate MSDs")

    if first>numframe:
        first = numframe

    first = first - 1

    if last==0 or last<first:
        last = numframe

    numframes = (last - first - 1) // step + 1

    # work out how many block-averaging samples will be used based on
    # selected number of frames: if too few, set to total number of frames
    # and assume a single diffusivity value will be calculated

    if block>numframes:
        block = numframes
    numblock = numframes - block + 1

    # get information from DL_MESO_DPD HISTORY file header, including species,
    # molecule, particle and bond data
    
    nsyst, nusyst, timefirst, timelast, text, keytrj, surfaceprop, speciesprop, moleculeprop, particleprop, bondtable, framesize, headerpos = dlm.read_header(histin, bo, ri, rd, intsize, longintsize, realsize, numframe)

    # create lists/arrays of particle species (previously sorted by global indices),
    # find numbers of each kind and work out total number of non-frozen particles

    numspecies = len(speciesprop)
    nsystfree = 0

    beadspecies = [x[1] for x in particleprop]
    nspec = np.zeros(numspecies)
    for i in range(numspecies):
        nspec[i] = sum(x==i+1 for x in beadspecies)
        nsystfree = nsystfree + (nspec[i] if speciesprop[i][4]==False else 0)

    # set up arrays to hold positions of particles for previous frame,
    # displacement for each particle, accumulated displacements of all 
    # particles, MSDs for each and all species and times for each frame 
 
    xyz0 = np.zeros((nsyst, 3))
    dxyz = np.zeros(3)
    adisp = np.zeros((nsyst, 3))
    msd = np.zeros((numspecies, numframes))
    msdall = np.zeros(numframes)
    msdtime = []

    # setup flags for lees-edwards boundary conditions
    # (no need to directly account for reflecting walls)

    lex = surfaceprop[0]==1
    ley = surfaceprop[1]==1
    lez = surfaceprop[2]==1

    # major loop through all required frames in DL_MESO_DPD HISTORY file
    
    print('Collecting data from trajectories supplied in {0:s} file'.format(histin))

    for frame in tqdm(range(first, last, step)):
        # get all available information from DL_MESO_DPD HISTORY frame
        time, dimx, dimy, dimz, shrdx, shrdy, shrdz, particledata = dlm.read_frame(histin, ri, rd, frame, framesize, headerpos, keytrj)
        # set up box size for periodic boundary condition
        L = np.asfarray([dimx, dimy, dimz], np.double)
        box = L * np.eye(3)
        inv_box = np.linalg.pinv(box) 

        # determine data point number for MSD data (based on number of times in list)

        timeframe = len(msdtime)

        # append current time to list

        msdtime.append(time)

        # calculate and accumulate displacements since previous frame for all particles
        # (already set to zero for first frame), including any periodic boundary shifts,
        # store current particle positions for next frame, calculate squared displacements
        # since first frame and assign to arrays for each species

        if frame > first:
            for i in range(nsyst):
                dxyz = particledata[i][1:4] - xyz0[i][0:3]
                dxyz = images(dxyz, box, inv_box, shrdx, shrdy, shrdz, lex, ley, lez)
                adisp[i] = adisp[i] + dxyz
                xyz0[i] = particledata[i][1:4].copy()
                dispsq = squarenorm_numba(adisp[i])
                msd[beadspecies[i]-1][timeframe] = msd[beadspecies[i]-1][timeframe] + dispsq
                msdall[timeframe] = msdall[timeframe] + dispsq
        else:
            for i in range(nsyst):
                xyz0[i] = particledata[i][1:4].copy()

    # open MSDDAT file and write header at top

    fw = open(msdout, "w")
    fw.write(text+"\n")
    fw.write('{0:10d}{1:10d}\n\n'.format(numframes, numspecies))

    # work out individual and all (free-moving) species mean squared displacements
    # before writing each of them (individual ones first) to MSDDAT file

    time0 = msdtime[0]

    for i in range(numspecies):
        fw.write('\n{0:8s}\n'.format(speciesprop[i][0]))
        msd[i] = msd[i] / nspec[i]
        for j in range(numframes):
            fw.write('{0:14.6e}{1:14.6e}{2:14.6e}\n'.format((msdtime[j]-time0)*tscale, msd[i][j]*lscale*lscale, math.sqrt(msd[i][j])*lscale))
        fw.write('\n')
    
    fw.write('\nall species\n') 
    msdall = msdall / nsystfree
    for j in range(numframes):
        fw.write('{0:14.6e}{1:14.6e}{2:14.6e}\n'.format((msdtime[j]-time0)*tscale, msdall[j]*lscale*lscale, math.sqrt(msdall[j])*lscale))    

    fw.close()
    print('Written MSD data from {0:d} trajectory frames to {1:s} file'.format(numframes, msdout))

    # find gradients of MSDs (for each species and all free-moving ones)
    # over blocks of frames and calculate self-diffusivities

    if numframes>1:
        dtime = msdtime[1] - msdtime[0]
        factor = 1.0 / (6.0 * dtime * float(block-1))
        for i in range(numspecies):
            avediff = 0.0
            sddiff = 0.0
            for j in range(numblock):
                diffuse = factor * (msd[i][j+block-1] - msd[i][j])
                sclnv2 = 1.0 / float(j+1)
                sclnv1 = float(j) * sclnv2
                sddiff = sclnv1 * (sddiff + sclnv2 * (diffuse - avediff) * (diffuse - avediff))
                avediff = sclnv1 * avediff + sclnv2 * diffuse
            if numblock==1:
                if tscale==1.0 and lscale==1.0:
                    print('Self-diffusivity of {0:8s}    = {1:13.6e}'.format(speciesprop[i][0], avediff))
                else:
                    print('Self-diffusivity of {0:8s}    = {1:13.6e} m^2/s'.format(speciesprop[i][0], 1.0e-8*avediff*lscale*lscale/tscale))
            else:
                if tscale==1.0 and lscale==1.0:
                    print('Self-diffusivity of {0:8s}    = {1:13.6e} +/- {2:13.6e}'.format(speciesprop[i][0], avediff, sddiff))
                else:
                    print('Self-diffusivity of {0:8s}    = {1:13.6e} +/- {2:13.6e} m^2/s'.format(speciesprop[i][0], 1.0e-8*avediff*lscale*lscale/tscale, 1.0e-8*sddiff*lscale*lscale/tscale))
        avediff = 0.0
        sddiff = 0.0
        for j in range(numblock):
            diffuse = factor * (msdall[j+block-1] - msdall[j])
            sclnv2 = 1.0 / float(j+1)
            sclnv1 = float(j) * sclnv2
            sddiff = sclnv1 * (sddiff + sclnv2 * (diffuse - avediff) * (diffuse - avediff))
            avediff = sclnv1 * avediff + sclnv2 * diffuse
        if numblock==1:
            if tscale==1.0 and lscale==1.0:
                print('Self-diffusivity of all species = {0:13.6e}'.format(avediff))
            else:
                print('Self-diffusivity of all species = {0:13.6e} m^2/s'.format(1.0e-8*avediff*lscale*lscale/tscale))
        else:
            if tscale==1.0 and lscale==1.0:
                print('Self-diffusivity of all species = {0:13.6e} +/- {1:13.6e}'.format(avediff, sddiff))
            else:
                print('Self-diffusivity of all species = {0:13.6e} +/- {1:13.6e} m^2/s'.format(1.0e-8*avediff*lscale*lscale/tscale, 1.0e-8*sddiff*lscale*lscale/tscale))



