#!/usr/bin/env python3
"""Usage:
    flory_huggins.py [--rho <rho> --Aii <Aii> --Aijmin <Aijmin> --Aijmax <Aijmax> --dA <dA> --dx <dx> --L <L> --W <W> --print-to-screen --nproc <nproc>]

Carries out DL_MESO_DPD calculations to determine relationship between
Flory-Huggins chi parameters and conservative force parameters, analyse
data and plot results

Options:
    --rho <rho>         Particle density [default: 3.0]
    --Aii <Aii>         Conservative force parameter for like-like particle
                        interactions [default: 25.0]
    --Aijmin <Aijmin>   Mininum conservative force parameter between particle
                        species [default: 33.0]
    --Aijmax <Aijmax>   Maximum conservative force parameter between particle
                        species [default: 43.0]
    --dA <dA>           Steps between values of conservative force parameter
                        between particle species for each run [default: 1.0]
    --dx <dx>           Bin size for concentration profile in x-direction
                        [default: 0.1]
    --L <L>             Length of box in x-direction [default: 20.0]
    --W <W>             Width of box in y- and z-directions [default: 8.0]
    --print-to-screen   Re-direct simulation outputs to screen
    --nproc <nproc>     Number of processor cores to run DL_MESO_DPD calculations [default: 1]

michael.seaton@stfc.ac.uk, 08/10/20
"""
from docopt import docopt
from tqdm import tqdm
import sys
import numpy as np
import struct
import statistics
import math
import os

args = docopt(__doc__)
rho = float(args["--rho"])
Aii = float(args["--Aii"])
Aijmin = float(args["--Aijmin"])
Aijmax = float(args["--Aijmax"])
dA = float(args["--dA"])
dx = float(args["--dx"])
L = float(args["--L"])
W = float(args["--W"])
printtoscreen = args["--print-to-screen"]
nproc = int(args["--nproc"])

bo = sys.byteorder
if(bo == 'big'):
    ri = ">i"
    rd = ">d"
else:
    ri = "<i"
    rd = "<d"

if nproc > 1 :
    invoke = "mpirun -np {0:d} ./dpd.exe".format(nproc)
else :
    invoke = "./dpd.exe"

intsize = 4
longintsize = 8

gam = 4.5
Npart = int(round(0.5*rho*L*W*W))
if Npart%2 == 1:
    Npart += 1

factors = []
for i in range(2,Npart):
    if Npart%i == 0:
        factors.append(i)

boxlist = []
errors = []
for i in range(len(factors)):
    Nxtry = Npart//(factors[i]*factors[i])
    Nparttry = Nxtry * factors[i] * factors[i]
    if Npart==Nparttry:
        error = (2.*Nxtry/L - factors[i]/W) ** 2
        listitem = [Nxtry, factors[i]]
        boxlist.append(listitem)
        errors.append(error)

minerror = errors.index(min(errors))
Nx = boxlist[minerror][0]
Ny = boxlist[minerror][1]

print("Box size: {0:.4f} by {1:.4f} by {2:.4f}, density: {3:.4f},".format(L, W, W, rho))
print("Number of particles: {0:d} ({1:d} by {2:d} by {3:d})".format(2*Npart, 2*Nx, Ny, Ny))

# create CONTROL file - same for all simulations

sc = "DL_MESO Flory-Huggins chi-parameter determination\n\n"
sc += "volume {0:.4f} {1:.4f} {2:.4f}\n".format(L, W, W)
sc += "temperature 1.0\n"
sc += "cutoff 1.0\n\n"
sc += "timestep 0.01\n"
sc += "steps 70000\n"
sc += "equilibration steps 20000\n"
sc += "trajectory 20000 100\n"
sc += "stack size 100\n"
sc += "print every 100\n"
sc += "job time 3600.0\n"
sc += "close time 200.0\n\n"
sc += "ensemble nvt mdvv\n\n"

if printtoscreen:
    sc += "l_scr\n\n"

sc += "finish\n"

open("CONTROL", "w").write(sc)
print("CONTROL file saved.")

# create CONFIG file

wx = 0.5*L/Nx
wy = W/Ny
hx = 0.5*L
hy = 0.5*W
cf = "DL_MESO Flory-Huggins chi-parameter determination\n"
cf += "0\t1\t{0:d}\n".format(2*Npart)
cf += "{0:16.10f}{1:16.10f}{2:16.10f}\n".format(L, 0.0, 0.0)
cf += "{0:16.10f}{1:16.10f}{2:16.10f}\n".format(0.0, W, 0.0)
cf += "{0:16.10f}{1:16.10f}{2:16.10f}\n".format(0.0, 0.0, W)
for i in range(Npart):
    cf += "{0:s}        {1:d}\n".format("A", i+1)
    iz = i%Ny
    ix = i//(Ny*Ny)
    iy = (i%(Ny*Ny))//Ny
    xx = (ix+0.5)*wx-hx
    yy = (iy+0.5)*wy-hy
    zz = (iz+0.5)*wy-hy
    cf += "{0:16.10f}{1:16.10f}{2:16.10f}\n".format(xx, yy, zz)
for i in range(Npart):
    cf += "{0:s}        {1:d}\n".format("B", Npart+i+1)
    iz = i%Ny
    ix = i//(Ny*Ny)
    iy = (i%(Ny*Ny))//Ny
    xx = (ix+0.5)*wx
    yy = (iy+0.5)*wy-hy
    zz = (iz+0.5)*wy-hy
    cf += "{0:16.10f}{1:16.10f}{2:16.10f}\n".format(xx, yy, zz)

open("CONFIG", "w").write(cf)
print("CONFIG file saved.")

# open file for recording concentration profiles and chi values

filename = 'floryhuggins-rho-{0:.3f}.dat'.format(rho)
fw = open(filename, "a+")

# loop through unlike interaction parameters

print(Aijmin, Aijmax, dA)
print(np.arange(Aijmin, Aijmax, dA))

for Aij in np.arange(Aijmin, Aijmax+0.5*dA, dA):

# create FIELD file

    sf = "DL_MESO Flory-Huggins chi-parameter determination\n\n"
    sf += "SPECIES 2\n"
    sf += "A 1.0 0.0 {0:d} 0\n".format(Npart)
    sf += "B 1.0 0.0 {0:d} 0\n\n".format(Npart)
    sf +="INTERACTIONS 3\n"
    sf += "A A dpd  {0:.3f} 1.0 {1:.3f}\n".format(Aii, gam)
    sf += "A B dpd  {0:.3f} 1.0 {1:.3f}\n".format(Aij, gam)
    sf += "B B dpd  {0:.3f} 1.0 {1:.3f}\n\n".format(Aii, gam)
    sf += "CLOSE\n"

    open("FIELD", "w").write(sf)
    print("FIELD file saved for Aij = {0:f}.".format(Aij))

# run DL_MESO_DPD

    print("Starting DL_MESO_DPD run for Aii = {0:f}, Aij = {1:f}".format(Aii, Aij))
    os.system(invoke)
    print("Completed DL_MESO_DPD run for Aii = {0:f}, Aij = {1:f}".format(Aii, Aij))

# open HISTORY file and check endianness (swap if necessary)

    fr = open("HISTORY", "rb")
    endcheck = (int.from_bytes(fr.read(intsize), byteorder=bo) == 1)

    if(endcheck==False):
        if bo=='big':
            bo = 'little'
            ri = "<i"
            rd = "<d"
        else:
            bo = 'big'
            ri = ">i"
            rd = ">d"
        fr.seek(0, 0)
        endcheck = (int.from_bytes(fr.read(intsize), byteorder=bo) == 1)
        assert (endcheck==True), "Cannot read HISTORY file"

    doublesize = int.from_bytes(fr.read(intsize), byteorder = bo)
    filesize = int.from_bytes(fr.read(longintsize), byteorder = bo)
    numframe = int.from_bytes(fr.read(intsize), byteorder = bo)
    nstep = int.from_bytes(fr.read(intsize), byteorder = bo)

    text = fr.read(80).decode('ascii')

    numspe, nmoldef, nusyst, nsyst, numbonds, keytrj, srfx, srfy, srfz = np.fromfile(fr, dtype = np.dtype(ri), count = 9)

    namspe = []
    amass = []
    bbb = []
    chge = []
    lfrzn = []
    for i in range(numspe):
        namspe.append(fr.read(8).decode('ascii').strip())
        mass, rc, qi = np.fromfile(fr, dtype = np.dtype(rd), count = 3)
        amass.append(mass)
        bbb.append(rc)
        chge.append(qi)
        lfrzn.append(int.from_bytes(fr.read(intsize), byteorder = bo))

    nammol = []
    for i in range(nmoldef):
        nammol.append(fr.read(8).decode('ascii'))

    partproperties = []
    for i in range(nsyst):
        glob, spec, mole, chain = np.fromfile(fr, dtype = np.dtype(ri), count = 4)
        partproperties.append([glob, spec])

    partproperties = sorted(partproperties, key = lambda x: x[0])

# skip past bonds - not needed here

    fr.seek(2*numbonds*intsize, 1)

# read in trajectory frames

    for frame in tqdm(range(numframe)):
        time = np.fromfile(fr, dtype = np.dtype(rd), count = 1)
        nbeads = np.fromfile(fr, dtype = np.dtype(ri), count = 1)
        dimx, dimy, dimz, shrdx, shrdy, shrdz = np.fromfile(fr, dtype = np.dtype(rd), count = 6)
    
        if frame == 0:
            nx = int(dimx/dx)
            dx = dimx/nx
            popall = np.zeros(nx)
            popspec = np.zeros(nx)
    
        gloindex = np.fromfile(fr, dtype = np.dtype(ri), count = nsyst)
        specglobal = []
        for i in range(nsyst):
            specglobal.append(partproperties[gloindex[i]-1][1]==1)
        for i in range(nsyst):
            framedata = np.fromfile(fr, dtype = np.dtype(rd), count = (keytrj+1)*3)
            xpos = int((framedata[0] + 0.5*dimx)/dx)
            popall[xpos] += 1
            if(specglobal[i] == True):
                popspec[xpos] += 1
            

    volfrac = popspec / popall
    minchi = int(0.15*nx)
    maxchi = int(0.35*nx)
    meanvolfrac = statistics.mean(volfrac[minchi:maxchi])
    stdvolfrac = statistics.stdev(volfrac[minchi:maxchi])
    chi = math.log((1.0-meanvolfrac)/meanvolfrac)/(1.0-2.0*meanvolfrac)
    chimax = math.log((1.0-meanvolfrac-stdvolfrac)/(meanvolfrac+stdvolfrac))/(1.0-2.0*(meanvolfrac+stdvolfrac))
    chimin = math.log((1.0-meanvolfrac+stdvolfrac)/(meanvolfrac-stdvolfrac))/(1.0-2.0*(meanvolfrac-stdvolfrac))
    chierr = max(abs(chimax-chi), abs(chi-chimin))

    fw.write("{0:f},{1:f},{2:f},{3:f},{4:d}\n".format(Aii, Aij, chi, chierr, nx))
    for i in range(nx):
        fw.write("{0:11.7f}      {1:11.7f}\n".format((i+0.5)*dx, volfrac[i]))

    print("Written data for Aii = {0:f}, Aij = {1:f} to {2:s} - chi = {3:f} +/- {4:f}".format(Aii, Aij, filename, chi, chierr))
