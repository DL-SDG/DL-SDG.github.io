#!/usr/bin/env python3
"""Usage:
    history_dlm_to_history_dlp.py [--con <con> --in <histin> --out <histout> --dt <dt> --mscale <mscale> --lscale <lscale> --tscale <tscale> --first <first> --last <last> --step <step>]

Converts DL_MESO_DPD HISTORY file into DCD format, based on dlp2dcd.py3 (January 2018) by Dr. Andrey Brukhno

Options:
    --con <con>         Use CONTROL-format file named <con> to find timestep size
                        [default: CONTROL]
    --in <histin>       Name of DL_MESO_DPD-format HISTORY file to read in and
                        convert [default: HISTORY]
    --out <dcdout>      Name of DCD file(s) to write [default: TRAJOUT.dcd]
    --dt <dt>           Timestep size used in simulations, needed if no CONTROL
                        file present from which to read value [default: 0.0]
    --mscale <mscale>   DPD mass scale in daltons or unified atomic mass units [default: 1.0]
    --lscale <lscale>   DPD length scale in Angstroms [default: 1.0]
    --tscale <tscale>   DPD time scale in picoseconds [default: 1.0]
    --first <first>     Starting DL_MESO_DPD HISTORY file frame number for inclusion
                        in DCD file [default: 1]
    --last <last>       Finishing DL_MESO_DPD HISTORY file frame number for inclusion
                        in DCD file (value of 0 here will use last available frame) [default: 0]
    --step <step>       Incrementing number of frames in DL_MESO_DPD HISTORY between
                        frames in DCD file [default: 1]

michael.seaton@stfc.ac.uk, 20/08/21
"""
from docopt import docopt
from tqdm import tqdm
import dlmhistoryread as dlm
import numpy as np
import struct
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


def fort_write_bin(fout, data, format_string, byte_order):

    # write data to binary file as a Fortran-style record (assuming file is already open):
    # based on equivalent fort_write_bin function in dlp2dcd.py3
    
    # prepare format string (replacing commas with spaces) and find size in bytes
    
    format_string = format_string.replace(',',' ')
    recl = int(struct.calcsize(format_string))

    # prepare binary writers based on endianness (byte order)
    
    if(byte_order == 'little'):
        fmt0 = '<'
        ifmt = '<i'
    else:
        fmt0 = '>'
        ifmt = '>i'

    # prepare to write a bytes object of required size (determined from format string)
    
    fout.write(struct.pack(ifmt,recl))

    i = 0
    fmt = fmt0
    for char in format_string:
        if char == ' ' :
            if len(fmt) > 1 :
                if hasattr(data,"__len__") : # array-like
                    if hasattr(data[i],"__iter__") : # an actual list or array
                        fout.write(struct.pack(fmt,*data[i]))
                    else : # a scalar or a string
                        fout.write(struct.pack(fmt,data[i]))
                else : # a scalar
                    fout.write(struct.pack(fmt,data))

                i += 1
            fmt = fmt0
        else :
            fmt += char

    fout.write(struct.pack(ifmt,recl))


def dcd_write_header(fout, nsyst, rec1, rec2, byte_order):

    # write two-record header for binary DCD file (assuming file is already open):
    # based on equivalent dcd_write_header function in dlp2dcd.py3
    
    # write record 1 with numbers of frames, first frame timestep number, frequency of
    # saving frames, total number of timesteps, timestep size, presence of crystal information
    # and CHARMM version number
    
    fort_write_bin(fout, rec1,'4B 9i f 10i ', byte_order)

    # prepare formatting for record 2 (based on its length) and write to file
    
    fmt='i '
    for i in range(rec2[0]):
        fmt +='80B '
        
    fort_write_bin(fout, rec2, fmt, byte_order)

    # write total number of particles to file
    
    fort_write_bin(fout, nsyst, 'i ', byte_order)


def dcd_write_frame(fout, nsyst, cell, xyz, byte_order):

    # write trajectory frame to binary DCD file (assuming file is already open):
    # based on equivalent dcd_write_frame function in dlp2dcd.py3
    
    # write simulation cell data for current trajectory frame
    
    fort_write_bin(fout,tuple([cell,]),'6d ',byte_order)

    # prepare formatting for writing particle positions
    ffmt = '{0:d}f '.format(nsyst)
    
    # write particle positions to file: x-components, y-components, z-components
    
    fort_write_bin(fout, tuple([xyz[0],]), ffmt, byte_order)
    fort_write_bin(fout, tuple([xyz[1],]), ffmt, byte_order)
    fort_write_bin(fout, tuple([xyz[2],]), ffmt, byte_order)




if __name__ == '__main__':
    # first check command-line arguments, including timestep size (in
    # DPD units) and mass, length and time scales of DPD simulation

    args = docopt(__doc__)
    dt = float(args["--dt"])
    con = args["--con"]
    histin = args["--in"]
    dcdout = args["--out"]
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
        sys.exit("No trajectory data available in "+histin+" file to write to DCD file")

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
    
    filename = dcdout
    if (dcdout[-4:]=='.dcd'):
        filename = filename[:-4]
    
    # find timestep number for first frame
    
    stepdiff = math.floor((timelast-timefirst)/dt+0.5)
    nstepfirst = nsteplast - stepdiff
    nstepfreq = stepdiff // (numframe - 1)

    # open and write particle specifications to DCD-ATOM file: name, global index number,
    # mass, charge, initial root-squared-displacement (assumed to be zero)
    
    fw = open(filename+"-ATOM", "w")
    for i in range(nsyst):
        spec = particleprop[i][1]-1
        fw.write("{0:8s}     {1:10d}     {2:12.6f}     {3:12.6f}     {4:12.6f}     \n".format(speciesprop[spec][0], particleprop[i][0], speciesprop[spec][1]*mscale, speciesprop[spec][2], 0.0))
    fw.close()
    
    # open and write header for DCD file
    
    fw = open(filename+".dcd", "wb")
    rec1 = ['CORD'.encode('utf-8'),[0,0,0,0,0,0,0,0,0],0.0,[0,0,0,0,0,0,0,0,0,0]]
    rec1[1][0] = numframes                                              # number of frames in DCD file
    rec1[1][1] = nstepfirst + first * nstepfreq                         # timestep number for first frame
    rec1[1][2] = nstepfreq * step                                       # number of timesteps between frames
    rec1[1][3] = nstepfirst + nstepfreq * (first + numframes * step)    # timestep number at last frame
    rec1[2] = 1000.0*tscale*dt/48.8882099                               # timestep size in AKMA-units
    rec1[3][0] = 2                                                      # periodic boundary key (fixed as orthorhombic)
    rec1[3][9] = 410                                                    # CHARMM version number
    
    remark = "REMARK Created with history_dlm_to_dcd.py for DL_MESO_DPD (author: M A Seaton)"
    if len(remark) < 80:
        remark = remark.ljust(80)
        
    rec2 = [2,text[:80].encode('utf-8'),remark[:80].encode('utf-8')]
    
    dcd_write_header(fw, nsyst, rec1, rec2, bo)
    
    # prepare arrays for particle positions and distances between frames

    xyz0 = np.zeros((nsyst, 3))
    dxyz = np.zeros(3)
    adisp = np.zeros((nsyst, 3))

    # setup flags for lees-edwards boundary conditions
    # (no need to directly account for reflecting walls)

    lex = surfaceprop[0]==1
    ley = surfaceprop[1]==1
    lez = surfaceprop[2]==1

    # open file for root-squared-displacements
    
    fwd = open(filename+"-DISP", "w")

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
        # prepare cell size and particle positions for writing to DCD file
        cell = [dimx*lscale, 0.0, dimy*lscale, 0.0, 0.0, dimz*lscale]
        xyz = np.array([elem for singleList in particledata for elem in singleList]).reshape(nsyst,4)
        xyz = xyz[:,1:].transpose()
        # write frame data to DCD file
        dcd_write_frame(fw, nsyst, cell, xyz, bo)
        # write displacements to file
        fwd.write("#t {0:12.6f} {1:10d}\n".format(time*tscale, nstep))
        for i in range(nsyst):
            fwd.write("{0:12.6f}\n".format(rsd[i]*lscale))

    fw.close()
    fwd.close()
