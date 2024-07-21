#!/usr/bin/env python3
"""DL_MESO_DPD HISTORY file readers

Module to read DL_MESO_DPD HISTORY files: preparatory information, header and individual trajectory frames

michael.seaton@stfc.ac.uk, 04/10/22
"""

import sys
import numpy as np
import struct
import os

def read_prepare(filename):
    """Reads first few values in DL_MESO_DPD HISTORY file to find essential information for reading further"""
    
    # inputs:
    #   filename        name of HISTORY file to start reading
    # outputs:
    #   bo              byte order for reading HISTORY file
    #   ri              binary reader for integers
    #   rd              binary reader for (single or double precision) real numbers
    #   intsize         size of integer in HISTORY file (in bytes)
    #   longintsize     size of long integer in HISTORY file (in bytes)
    #   realsize        size of real numbers in HISTORY file (in bytes)
    #   filesize        size of HISTORY file in bytes
    #   numframe        number of trajectory frames in HISTORY file
    #   nsteplast       timestep for last trajectory frame in HISTORY file

    # check current endianness and prepare binary readers accordingly

    bo = sys.byteorder
    if(bo == 'big'):
        ri = ">i"
        rd = ">"
    else:
        ri = "<i"
        rd = "<"

    intsize = 4
    longintsize = 8

    filesize = 0
    numframe = 0
    nsteplast = 0

    # open DL_MESO_DPD HISTORY file and check endianness (swap if necessary)

    try:
        fr = open(filename, "rb")
        endcheck = (int.from_bytes(fr.read(intsize), byteorder=bo) == 1)

        if(endcheck==False):
            if bo=='big':
                bo = 'little'
                ri = "<i"
                rd = "<"
            else:
                bo = 'big'
                ri = ">i"
                rd = ">"
            fr.seek(0, 0)
            endcheck = (int.from_bytes(fr.read(intsize), byteorder=bo) == 1)
            if endcheck==False: 
                sys.exit("ERROR: Cannot read HISTORY file")

    # obtain information on real number sizes, projected size of HISTORY file,
    # number of available trajectory frames and timestep number for last frame

        realsize = int.from_bytes(fr.read(intsize), byteorder = bo)
        filesize = int.from_bytes(fr.read(longintsize), byteorder = bo)
        numframe = int.from_bytes(fr.read(intsize), byteorder = bo)
        nsteplast = int.from_bytes(fr.read(intsize), byteorder = bo)
    
    # check size of real numbers and set up binary reader accordingly
    
        if realsize==4:
            rd += "f"
        else:
            rd += "d"
    
    # close HISTORY file
    
        fr.close()
    
    except OSError:
        print("ERROR: Cannot open HISTORY file")
    
    return bo, ri, rd, intsize, longintsize, realsize, filesize, numframe, nsteplast

def read_header(filename, bo, ri, rd, intsize, longintsize, realsize, numframe):
    """Reads DL_MESO_DPD HISTORY file header to find information about simulation"""
    
    # inputs:
    #   filename        name of HISTORY file to start reading
    #   bo              byte order for reading HISTORY file
    #   ri              binary reader for integers
    #   rd              binary reader for real numbers (single or double precision)
    #   intsize         size of integer in HISTORY file (in bytes)
    #   longintsize     size of long integer in HISTORY file (in bytes)
    #   realsize        size of real numbers in HISTORY file (in bytes)
    #   numframe        number of trajectory frames in HISTORY file
    # outputs:
    #   nsyst           total number of particles in simulation box
    #   nusyst          number of particles not involved in molecules in simulation box
    #   timefirst       time at first available trajectory frame in HISTORY file
    #   timelast        time at last available trajectory frame in HISTORY file
    #   text            name of simulation as given in HISTORY file
    #   keytrj          trajectory key: level of information available per particle (0 = positions,
    #                   1 = positions and velocities, 2 = positions, velocities and forces)
    #   surfaceprop     information about boundary conditions at box boundaries, given orthogonally
    #                   to x-, y- and z-axes (0 = periodic, 1 = shear, 2 = specular reflection, 
    #                   3 = bounceback reflection)
    #   speciesprop     information about all available species: name, mass, radius, charge,
    #                   frozen property for each species
    #   moleculeprop    information about all available molecule types: name for each molecule type
    #   particleprop    information about all available particles: global particle ID,
    #                   species number, molecule type, molecule number for each particle
    #   bondtable       bond connectivity table: each entry consists of global particle IDs for
    #                   pair of particles bonded together
    #   framesize       total size of a single trajectory frame in bytes
    #   headerpos       position in HISTORY file (in bytes) where trajectory data starts

    # open DL_MESO_DPD HISTORY file and skip past first few values
    
    fr = open(filename, "rb")
    fr.seek(4*intsize+longintsize, 0)
    
    # read simulation name

    text = fr.read(80).decode('ascii')

    # read numbers of species, molecule types, particles not in molecules,
    # total number of particles, number of bonds, trajectory key and
    # surface indicators in x, y and z

    numspe, nmoldef, nusyst, nsyst, numbonds, keytrj, srfx, srfy, srfz = np.fromfile(fr, dtype = np.dtype(ri), count = 9)

    # read particle species properties: name, mass, radius, charge and
    # frozen property for each species

    speciesprop = []
    for i in range(numspe):
        namspe = fr.read(8).decode('ascii').strip()
        mass, rc, qi = np.fromfile(fr, dtype = np.dtype(rd), count = 3)
        lfrzn = (int.from_bytes(fr.read(intsize), byteorder = bo) > 0)
        speciesprop.append([namspe, mass, rc, qi, lfrzn])

    # read molecule property: names of molecule types

    moleculeprop = []
    for i in range(nmoldef):
        moleculeprop.append(fr.read(8).decode('ascii').strip())

    # now read properties of individual particles, identifying global ID
    # numbers, species types, molecule types and molecule numbers

    particleprop = []
    for i in range(nsyst):
        glob, spec, mole, chain = np.fromfile(fr, dtype = np.dtype(ri), count = 4)
        particleprop.append([glob, spec, mole, chain])

    # sort particle properties based on global ID numbers

    particleprop = sorted(particleprop, key = lambda x: x[0])

    # read table of bonds

    bondtable = []
    for i in range(numbonds):
        bond1, bond2 = np.fromfile(fr, dtype = np.dtype(ri), count = 2)
        bondtable.append([min(bond1, bond2), max(bond1, bond2)])

    # put together surface indicators as a surface property

    surfaceprop = [srfx, srfy, srfz]

    # note current location in HISTORY file: used to skip past header when
    # reading trajectory frames
    
    headerpos = fr.tell()

    # find times for first and last timeframes (can be used to work
    # out timestep of first frame)

    framesize = 7*realsize+intsize+nsyst*intsize+nsyst*3*(keytrj+1)*realsize
    timefirst = np.fromfile(fr, dtype = np.dtype(rd), count = 1)
    fr.seek(headerpos+(numframe-1)*framesize, 0)
    timelast = np.fromfile(fr, dtype = np.dtype(rd), count = 1)
    
    fr.close()
    
    return nsyst, nusyst, timefirst, timelast, text, keytrj, surfaceprop, speciesprop, moleculeprop, particleprop, bondtable, framesize, headerpos

def read_frame(filename, ri, rd, framenum, framesize, headerpos, keytrj):
    """Reads trajectory frame from DL_MESO_DPD HISTORY file"""
    
    # inputs:
    #   filename        name of HISTORY file to start reading
    #   ri              binary reader for integers
    #   rd              binary reader for real numbers
    #   framenum        frame number to read in HISTORY file (first frame is 0)
    #   framesize       total size of a single trajectory frame in bytes
    #   headerpos       position in HISTORY file (in bytes) where trajectory data starts
    #   keytrj          trajectory key: level of information available per particle (0 = positions,
    #                   1 = positions and velocities, 2 = positions, velocities and forces)
    # outputs:
    #   time            time at current trajectory frame in HISTORY file
    #   dimx            length of simulation box in x-direction
    #   dimy            length of simulation box in y-direction
    #   dimz            length of simulation box in z-direction
    #   shrdx           shear-based displacement of periodic boundary in x-direction
    #   shrdy           shear-based displacement of periodic boundary in y-direction
    #   shrdz           shear-based displacement of periodic boundary in z-direction
    #   particledata    particle data read from current trajectory frame in HISTORY file:
    #                   global particle ID, position (x, y, z), velocity (vx, vy, vz) if available,
    #                   force (fx, fy, fz) if available for each particle (sorted by global ID)
    
    # open HISTORY file and find location of required frame
    
    fr = open(filename, "rb")
    fr.seek(headerpos+framenum*framesize, 0)
    
    # read in trajectory frame, starting with header with time, number of
    # particles, box dimensions and lees-edwards shearing displacement

    time = float(np.fromfile(fr, dtype = np.dtype(rd), count = 1))
    nbeads = int(np.fromfile(fr, dtype = np.dtype(ri), count = 1))
    dimx, dimy, dimz, shrdx, shrdy, shrdz = np.fromfile(fr, dtype = np.dtype(rd), count = 6)

    # now read global indices of particles in trajectory frame
    # to prepare for sorting data based on global ID numbers

    gloindex = np.fromfile(fr, dtype = np.dtype(ri), count = nbeads)
    
    # read data for each particle, put into arrays and sort by global ID

    particledata = []
    for i in range(nbeads):
        partdata = gloindex[i:i+1].tolist()
        framedata = np.fromfile(fr, dtype = np.dtype(rd), count = (keytrj+1)*3)
        partdata += tuple(framedata)
        particledata.append(partdata)
        
    particledata = sorted(particledata, key = lambda x: x[0])
    
    fr.close()
    
    return time, dimx, dimy, dimz, shrdx, shrdy, shrdz, particledata

