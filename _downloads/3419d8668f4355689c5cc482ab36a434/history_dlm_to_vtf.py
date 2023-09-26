#!/usr/bin/env python3
"""Usage:
    history_dlm_to_vtf.py [--in <histin> --out <vtfout> --mscale <mscale> --lscale <lscale> --first <first> --last <last> --step <step> --separate]

Converts DL_MESO_DPD HISTORY file into VMD trajectory (VTF) file or structure (VSF) and coordinate (VCF) files

Options:
    --in <histin>       Name of DL_MESO_DPD-format HISTORY file to read in and
                        convert [default: HISTORY]
    --out <vtfout>      Name of VTF format file(s) to write [default: traject.vtf]
    --mscale <mscale>   DPD mass scale in daltons or unified atomic mass units [default: 1.0]
    --lscale <lscale>   DPD length scale in Angstroms [default: 1.0]
    --first <first>     Starting DL_MESO_DPD HISTORY file frame number for inclusion
                        in VTF/VCF file [default: 1]
    --last <last>       Finishing DL_MESO_DPD HISTORY file frame number for inclusion
                        in VTF/VCF file (value of 0 here will use last available frame)
                        [default: 0]
    --step <step>       Incrementing number of frames in DL_MESO_DPD HISTORY between
                        frames in VTF/VCF file [default: 1]
    --separate          Write separate files for structure (traject.vsf) and coordinates
                        (traject.vcf), using user-specified name in place of 'traject' if required

michael.seaton@stfc.ac.uk, 15/08/21
"""
from docopt import docopt
from tqdm import tqdm
import dlmhistoryread as dlm
import numpy as np
from statistics import mode
import sys


if __name__ == '__main__':
    # first check command-line arguments

    args = docopt(__doc__)
    histin = args["--in"]
    vtfout = args["--out"]
    mscale = float(args["--mscale"])
    lscale = float(args["--lscale"])
    first = int(args["--first"])
    last = int(args["--last"])
    step = int(args["--step"])
    separate = args["--separate"]
    
    # read very beginning of DL_MESO_DPD HISTORY file to determine endianness,
    # sizes of number types, filesize, number of frames and last timestep number
    
    bo, ri, rd, intsize, longintsize, realsize, filesize, numframe, nsteplast = dlm.read_prepare(histin)

    # if not specified last frame to use in command-line argument or value for
    # first and/or last frames are too small/large, reset to defaults:
    # also work out how many frames are going into VTF file

    if numframe<1:
        sys.exit("No trajectory data available in "+histin+" file to write to VTF file")

    if first>numframe:
        first = numframe

    first = first - 1

    if last==0 or last<first:
        last = numframe

    numframes = (last - first - 1) // step + 1

    # get information from DL_MESO_DPD HISTORY file header, including species,
    # molecule, particle and bond data
    
    nsyst, nusyst, timefirst, timelast, text, keytrj, surfaceprop, speciesprop, moleculeprop, particleprop, bondtable, framesize, headerpos = dlm.read_header(histin, bo, ri, rd, intsize, longintsize, realsize, numframe)

    # determine filename to write either full VTF file or VSF structure file
    # based on user input (if given filename does not include '.vtf', add
    # required extension)
    
    filename = vtfout
    if (vtfout[-4:]!='.vtf'):
        filename += '.vtf'
    
    if separate:
        filename = filename.replace('.vtf', '.vsf')
    
    # open and write either beginning of VTF file or VSF file
    
    fw = open(filename, "w")
    
    # write particle species to file, using simulation-dependent options
    
    # option 1: only one particle species available and no molecules
    
    if len(speciesprop)==1 and len(moleculeprop)==0:
        fw.write('atom 0:{0:d}    radius {1:10.6f} mass {2:10.6f} charge {3:10.6f} name {4:8s}\n'.format(nsyst-1, speciesprop[0][2]*lscale, speciesprop[0][1]*mscale, speciesprop[0][3], speciesprop[0][0]))
    else:
    # option 2: search for most common species among particles (preferably *not* in molecules)
    #           and use as default species, specifying particles of other species and
    #           those in molecules explicitly
        speclist = [x[1] for x in particleprop]
        if nusyst > 0:
            common_spec = mode(speclist[0:nusyst])
        else:
            common_spec = mode(speclist[0:nsyst])
        fw.write('atom default    radius {0:10.6f} mass {1:10.6f} charge {2:10.6f} name {3:8s}\n'.format(speciesprop[common_spec-1][2]*lscale, speciesprop[common_spec-1][1]*mscale, speciesprop[common_spec-1][3], speciesprop[common_spec-1][0]))
        for i in range(nusyst):
            if speclist[i]!=common_spec or i==nusyst-1:
                spec = speclist[i] - 1
                fw.write('atom {0:10d}    radius {1:10.6f} mass {2:10.6f} charge {3:10.6f} name {4:8s}\n'.format(i, speciesprop[spec][2]*lscale, speciesprop[spec][1]*mscale, speciesprop[spec][3], speciesprop[spec][0]))
        for i in range(nusyst, nsyst):
            spec = particleprop[i][1]-1
            moletype = particleprop[i][2]-1
            molenum = particleprop[i][3]
            fw.write('atom {0:10d}    radius {1:10.6f} mass {2:10.6f} charge {3:10.6f} name {4:8s} resid {5:d} resname {6:8s}\n'.format(i, speciesprop[spec][2]*lscale, speciesprop[spec][1]*mscale, speciesprop[spec][3], speciesprop[spec][0], molenum, moleculeprop[moletype]))
            
    # write bond tables to file (if available)
    
    if len(bondtable)>0:
        fw.write('\n')
        for i in range(len(bondtable)):
            fw.write('bond {0:10d}:{1:10d}\n'.format(bondtable[i][0]-1, bondtable[i][1]-1))
    
    # major loop through all required frames in DL_MESO_DPD HISTORY file
    
    dimx0 = dimy0 = dimz0 = 0.0
    
    for frame in tqdm(range(first, last, step)):
        # get all available information from DL_MESO_DPD HISTORY frame
        time, dimx, dimy, dimz, shrdx, shrdy, shrdz, particledata = dlm.read_frame(histin, ri, rd, frame, framesize, headerpos, keytrj)
        # if writing separate structure and coordinate files, write first dimensions to structure
        # file before closing, then open coordinate file
        if frame==first and separate:
            fw.write('\n')
            fw.write('pbc {0:12.6f} {1:12.6f} {2:12.6f} 90 90 90\n'.format(dimx*lscale, dimy*lscale, dimz*lscale))
            fw.close()
            filename = filename.replace('.vsf', '.vcf')
            fw = open(filename, "w")
        # write header for trajectory frame to VTF or VCF file, including dimensions if on first
        # frame or if volume has changed
        fw.write('\n')
        fw.write('timestep indexed\n')
        if dimx!=dimx0 or dimy!=dimy0 or dimz!=dimz0:
            fw.write('pbc {0:12.6f} {1:12.6f} {2:12.6f} 90 90 90\n'.format(dimx*lscale, dimy*lscale, dimz*lscale))
            dimx0 = dimx
            dimy0 = dimy
            dimz0 = dimz
            halfx = 0.5 * dimx
            halfy = 0.5 * dimy
            halfz = 0.5 * dimz
        # write frame data to VTF or VCF file
        for i in range(nsyst):
            fw.write('{0:10d} {1:12.6f} {2:12.6f} {3:12.6f}\n'.format(i, (particledata[i][1]+halfx)*lscale, (particledata[i][2]+halfy)*lscale, (particledata[i][3]+halfz)*lscale))

    fw.close()
