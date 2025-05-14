#!/usr/bin/env python3
"""Usage:
    flory_huggins.py [--rho <rho>] [--Aii <Aii>] [--Aijmin <Aijmin>] 
                     [--Aijmax <Aijmax>] [--dA <dA>] [--dz <dz>] 
                     [--L <L>] [--W <W>] [--zden] [--nproc <nproc>]
                     [--dlmeso <dlmeso>]

Carries out DL_MESO_DPD calculations to determine relationship between
Flory-Huggins chi parameters and conservative force parameters, analyse
data and plot results

Options:
    -h, --help          Display this message
    --rho <rho>         Particle density [default: 3.0]
    --Aii <Aii>         Conservative force parameter for like-like particle
                        interactions [default: 25.0]
    --Aijmin <Aijmin>   Mininum conservative force parameter between particle
                        species [default: 33.0]
    --Aijmax <Aijmax>   Maximum conservative force parameter between particle
                        species [default: 43.0]
    --dA <dA>           Steps between values of conservative force parameter
                        between particle species for each run [default: 1.0]
    --dz <dz>           Bin size for concentration profile in z-direction
                        [default: 0.1]
    --L <L>             Length of box in x-direction [default: 20.0]
    --W <W>             Width of box in y- and z-directions [default: 8.0]
    --zden              Get DL_MESO_DPD to sample density profile along
                        z-axis automatically instead of converting it from
                        a HISTORY (trajectory) file: only works for DL_MESO
                        version 2.8 onwards
    --nproc <nproc>     Number of processor cores to run DL_MESO_DPD
                        calculations [default: 1]
    --dlmeso <dlmeso>   Location of DL_MESO_DPD executable [default: ./dpd.exe]

michael.seaton@stfc.ac.uk, 08/05/25
"""
from docopt import docopt
from tqdm import tqdm
from pathlib import Path
import subprocess
import shlex
import sys
import numpy as np
import statistics
import math
import os

def read_prepare(filename):
    """Scans DL_MESO_DPD OUTPUT file to find essential information for reading further"""
    
    # inputs:
    #   filename        name of OUTPUT file to start reading
    # outputs:
    #   numlines        total number of lines in OUTPUT file
    #   startrun        line number where first timestep in run is recorded (return -1 if does not exist)
    #   startstep       number of first timestep in run (return -1 if does not exist)
    #   termstep        number of timestep at which calculation has been terminated (returns 0 if does not exist)
    #   numstep         number of available timesteps in run (return 0 if cannot find any)
    #   terminate       flag indicating if calculation has terminated properly
    #   datanames       names of columns for data for each timestep

    # open OUTPUT file and split into lines

    try:
        with open(filename) as file:
            content = file.read().splitlines()

        numlines = len(content)
        startrun = -1
        startstep = -1
        numstep = 0
        termstep = 0
        datanames = []
        terminate = False

    # scan through lines to find first line with run results
    # (indicated as a series of dashes)

        if numlines>0:
            for line in range(numlines):
                if "---" in content[line]:
                    startrun = line
                    break

    # look at next line to find column names - all but first one
    # are names of properties being printed - and data line afterwards
    # to find number of first timestep in run

        if startrun>-1 and startrun+3<numlines:
            names = content[startrun+1].split()
            datanames = names[1:]
            words = content[startrun+3].split()
            if len(words)>1:
                startstep = int(words[0])

    # now scan through available data lines to see how many timesteps
    # are available in file - checks for lines with dashes and sees 
    # if third line after also contains dashes, which should skip
    # over any blank lines and column headers - and stop if 
    # "run terminating" is indicated (to avoid reading in averaged
    # properties afterwards)

        if startstep>-1:
            for line in range(startrun, numlines):
                if "---" in content[line] and line+3<numlines:
                    if "---" in content[line+3]:
                        numstep += 1
                elif "run terminating" in content[line]:
                    terminate = True
                    if line+3<=numlines:
                        words = content[line+3].split()
                        termstep = int(words[4])
                    break

    except FileNotFoundError:
        print("ERROR: Cannot open OUTPUT file")
    
    return numlines, startrun, startstep, termstep, numstep, terminate, datanames

def read_run(filename,startrun,terminate):
    """Reads statistical properties written to OUTPUT file during DL_MESO_DPD calculation, including any averaged values"""

    # inputs:
    #   filename        name of OUTPUT file to read
    #   startrun        line number where simulation run starts in OUTPUT file
    #   terminate       flag indicating if simulation terminated properly (and averaged values are available)
    # outputs:
    #   rundata         statistical properties read from OUTPUT file during DL_MESO_DPD calculation:
    #                   each entry includes timestep number, calculation walltime, instantaneous
    #                   values for each property, rolling average values for each property
    #   averages        averaged values for each property, including pressure tensors (conservative, dissipative, 
    #                   random and kinetic contributions, plus overall values)
    #   fluctuations    fluctuations (standard deviations) for each property, including pressure tensors
    #                   (conservative, dissipative, random and kinetic contributions, plus overall values)
    #   datanames       names of data for averaged values and fluctuations (including pressure tensors)

    with open(filename) as file:
        content = file.read().splitlines()

    numlines = len(content)
    rundata = []
    avelines = 0

    # go through all lines in OUTPUT file where data is available,
    # and find all available timesteps - indicated by two lines of
    # dashes separated by two lines of numbers - then read in data
    # and add to list (also work out where data ends and any averages
    # can be found)
 
    for line in range(startrun,numlines):
        if "---" in content[line] and line+3<numlines:
            if "---" in content[line+3]:
                words = content[line+1].split()
                timestep = int(words[0])
                instantdata = list(map(float, words[1:]))
                words = content[line+2].split()
                walltime = float(words[0])
                runningdata = list(map(float, words[1:]))
                data = [timestep, walltime]
                data += instantdata
                data += runningdata
                rundata.append(data)
        elif "run terminating" in content[line]:
            avelines = line
            break

    rundata = np.array(rundata)
    
    # if available, look for final averages and fluctuations for
    # properties and pressure tensors and read these into lists

    averages = []
    fluctuations = []
    datanames = []
    totaltensor = False
    numtensor = 0

    if terminate:
        for line in range(avelines,numlines):
            if "---" in content[line] and line+4<numlines:
                if "---" in content[line+2]:
                    names = content[line+1].split()
                    datanames = names[1:]
                    words = content[line+3].split()
                    data = list(map(float, words))
                    averages.extend(data)
                    words = content[line+4].split()
                    data = list(map(float, words))
                    fluctuations.extend(data)
            elif "average conservative" in content[line] or "average dissipative" in content[line] or "average random" in content[line] or "average kinetic" in content[line] and line+4<numlines:
                words = content[line+2].split()
                data = list(map(float, words))
                averages.extend(data[0:3])
                fluctuations.extend(data[3:6])
                words = content[line+3].split()
                data = list(map(float, words))
                averages.extend(data[0:3])
                fluctuations.extend(data[3:6])
                words = content[line+4].split()
                data = list(map(float, words))
                averages.extend(data[0:3])
                fluctuations.extend(data[3:6])
                numtensor += 1
                if "conservative" in content[line]:
                    datanames += ['p_xx^c','p_xy^c','p_xz^c','p_yx^c','p_yy^c','p_yz^c','p_zx^c','p_zy^c','p_zz^c']
                elif "dissipative" in content[line]:
                    datanames += ['p_xx^d','p_xy^d','p_xz^d','p_yx^d','p_yy^d','p_yz^d','p_zx^d','p_zy^d','p_zz^d']
                elif "random" in content[line]:
                    datanames += ['p_xx^r','p_xy^r','p_xz^r','p_yx^r','p_yy^r','p_yz^r','p_zx^r','p_zy^r','p_zz^r']
                elif "kinetic" in content[line]:
                    datanames += ['p_xx^k','p_xy^k','p_xz^k','p_yx^k','p_yy^k','p_yz^k','p_zx^k','p_zy^k','p_zz^k']
            elif "average overall" in content[line] and line+4<numlines:
                totaltensor = True
                words = content[line+2].split()
                data = list(map(float, words))
                averages.extend(data[0:3])
                fluctuations.extend(data[3:6])
                words = content[line+3].split()
                data = list(map(float, words))
                averages.extend(data[0:3])
                fluctuations.extend(data[3:6])
                words = content[line+4].split()
                data = list(map(float, words))
                averages.extend(data[0:3])
                fluctuations.extend(data[3:6])
                datanames += ['p_xx','p_xy','p_xz','p_yx','p_yy','p_yz','p_zx','p_zy','p_zz']

    # if no overall pressure tensor is included, work out average
    # and fluctuations from conservative, dissipative, random and 
    # kinetic contributions (if available)

    if not totaltensor and numtensor==4:
        avetottensor = [0.0] * 9
        flutottensor = [0.0] * 9
        for i in range(9):
            avetottensor[i] += (averages[-36+i]+averages[-27+i]+averages[-18+i]+averages[-9+i])
            flutottensor[i] += math.sqrt(fluctuations[-36+i]*fluctuations[-36+i] + fluctuations[-27+i]*fluctuations[-27+i] + fluctuations[-18+i]*fluctuations[-18+i] + fluctuations[-9+i]*fluctuations[-9+i])
        averages.extend(avetottensor)
        fluctuations.extend(flutottensor)
        datanames += ['p_xx','p_xy','p_xz','p_yx','p_yy','p_yz','p_zx','p_zy','p_zz']

    return rundata,averages,fluctuations,datanames

if __name__ == '__main__':

    args = docopt(__doc__)
    rho = float(args["--rho"])
    Aii = float(args["--Aii"])
    Aijmin = float(args["--Aijmin"])
    Aijmax = float(args["--Aijmax"])
    dA = float(args["--dA"])
    dz = float(args["--dz"])
    L = float(args["--L"])
    W = float(args["--W"])
    zden = args["--zden"]
    nproc = int(args["--nproc"])
    dlmeso = str(Path(args["--dlmeso"]).resolve())

    if not (os.path.isfile(dlmeso) and os.access(dlmeso, os.X_OK)):
        sys.exit("ERROR: Cannot find DL_MESO_DPD executable (dpd.exe)")

    bo = sys.byteorder
    if(bo == 'big'):
        ri = ">i"
        rd = ">d"
    else:
        ri = "<i"
        rd = "<d"

    if nproc > 1 :
        invoke = "mpirun -np {0:d} {1:s}".format(nproc, dlmeso)
    else :
        invoke = dlmeso

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
        Nztry = Npart//(factors[i]*factors[i])
        Nparttry = Nztry * factors[i] * factors[i]
        if Npart==Nparttry:
            error = (2.*Nztry/L - factors[i]/W) ** 2
            listitem = [Nztry, factors[i]]
            boxlist.append(listitem)
            errors.append(error)

    minerror = errors.index(min(errors))
    Nz = boxlist[minerror][0]
    Nx = boxlist[minerror][1]

    print("Box size: {0:.4f} by {1:.4f} by {2:.4f}, density: {3:.4f},".format(W, W, L, rho))
    print("Number of particles: {0:d} ({1:d} by {2:d} by {3:d})".format(2*Npart, Nx, Nx, 2*Nz))

    # create CONTROL file - same for all simulations

    sc = "DL_MESO Flory-Huggins chi-parameter determination\n\n"
    sc += "volume {0:.4f} {1:.4f} {2:.4f}\n".format(W, W, L)
    sc += "temperature 1.0\n"
    sc += "cutoff 1.0\n\n"
    sc += "timestep 0.01\n"
    sc += "steps 70000\n"
    sc += "equilibration steps 20000\n"
    if zden:
        sc += "zden sample every 100\n"
        sc += "zden binsize {0:f}\n".format(dz)
    else:
        sc += "trajectory 20000 100\n"
    sc += "stack size 100\n"
    sc += "print every 100\n"
    sc += "job time 3600.0\n"
    sc += "close time 200.0\n\n"
    sc += "ensemble nvt mdvv\n\n"
    sc += "l_scr\n\n"
    sc += "finish\n"

    open("CONTROL", "w").write(sc)
    print("CONTROL file saved.")

    # create CONFIG file

    wz = 0.5*L/Nz
    wx = W/Nx
    hz = 0.5*L
    hx = 0.5*W
    cf = "DL_MESO Flory-Huggins chi-parameter determination\n"
    cf += "0\t2\t{0:d}\n".format(2*Npart)
    cf += "{0:16.10f}{1:16.10f}{2:16.10f}\n".format(W, 0.0, 0.0)
    cf += "{0:16.10f}{1:16.10f}{2:16.10f}\n".format(0.0, W, 0.0)
    cf += "{0:16.10f}{1:16.10f}{2:16.10f}\n".format(0.0, 0.0, L)
    for i in range(Npart):
        cf += "{0:s}        {1:d}\n".format("A", i+1)
        ix = i%Nx
        iz = i//(Nx*Nx)
        iy = (i%(Nx*Nx))//Nx
        xx = (ix+0.5)*wx-hx
        yy = (iy+0.5)*wx-hx
        zz = (iz+0.5)*wz-hz
        cf += "{0:16.10f}{1:16.10f}{2:16.10f}\n".format(xx, yy, zz)
    for i in range(Npart):
        cf += "{0:s}        {1:d}\n".format("B", Npart+i+1)
        ix = i%Nx
        iz = i//(Nx*Nx)
        iy = (i%(Nx*Nx))//Nx
        xx = (ix+0.5)*wx-hx
        yy = (iy+0.5)*wx-hx
        zz = (iz+0.5)*wz
        cf += "{0:16.10f}{1:16.10f}{2:16.10f}\n".format(xx, yy, zz)

    open("CONFIG", "w").write(cf)
    print("CONFIG file saved.")

    # open file for recording concentration profiles and chi values

    filename = 'floryhuggins-rho-{0:.3f}.dat'.format(rho)
    fw = open(filename, "a+")

    # loop through unlike interaction parameters

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

    # run DL_MESO_DPD, keeping track of timestep number to display progress bar
    # by reading its OUTPUT file

        description = "Aii = {0:f}, Aij = {1:f}".format(Aii, Aij)
        outfile = "OUTPUT-Aii-{0:f}-Aij-{1:f}-rho-{2:f}".format(Aii, Aij, rho)
        outpipe = open(outfile, 'w')
        dlmesorun = subprocess.Popen(shlex.split(invoke), shell=False, stdout=outpipe)
        stepnum0 = 0
        pbar = tqdm(total=70000, desc='Running '+description)
        terminate = False
        if dlmesorun.poll() is None:
            while not terminate:
                if os.path.getsize(outfile)>0:
                    _, startrun, _, _, numstep, terminate, _ = read_prepare(outfile)
                    if startrun>0:
                        rundata, _, _, _ = read_run(outfile, startrun, terminate)
                        stepnum = int(rundata[-1,0])
                        if stepnum != stepnum0:
                            pbar.update(stepnum-stepnum0)
                            stepnum0 = stepnum
        pbar.close()
        dlmesorun.wait()
        
        print("Completed DL_MESO_DPD run for Aii = {0:f}, Aij = {1:f}".format(Aii, Aij))

    # if using ZDNDAT file, open and get hold of density profiles
    # for species A and all species to calculate concentration profile

        if zden:
            fz = open("ZDNDAT", "r")
            content = fz.readlines()
            numlines = len(content)
            # second line contains number of data sets (should be 3) and
            # number of lines per set
            words = content[1].split()
            if (int(words[0])!=3):
                sys.exit("ERROR: ZDNDAT file does not contain data for two species - cannot continue!")
            nz = int(words[1])
            # setup arrays for sampled density profiles
            rhoall = np.zeros(nz)
            rhospec = np.zeros(nz)
            # now look for first available z-density profile for species named "A"
            for line in range(2, numlines):
                if "A" in content[line]:
                    for data in range(nz):
                        words = content[line+data+1].split()
                        rhoz = float(words[1])
                        rhospec[data] = rhoz
                    break
            # now look for data set starting with "all species":
            # this will be the density profile for all bead species
            for line in range(2, len(content)):
                if "all species" in content[line]:
                    for data in range(nz):
                        words = content[line+data+1].split()
                        rhoz = float(words[1])
                        rhoall[data] = rhoz
                    break
            # finally convert densities to concentrations (volume fractions)
            # and rename ZDNDAT file for run
            volfrac = rhospec/rhoall
            os.rename('ZDNDAT', 'ZDNDAT-Aii-{0:f}-Aij-{1:f}-rho-{2:f}'.format(Aii, Aij, rho))

    # open HISTORY file and check endianness (swap if necessary)

        else:
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
                if endcheck==False:
                    sys.error("ERROR: Cannot read HISTORY file")

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

    # read in trajectory frames and count numbers of first bead type
    # and all beads in slices along z-axis

            for frame in tqdm(range(numframe)):
                time = np.fromfile(fr, dtype = np.dtype(rd), count = 1)
                nbeads = np.fromfile(fr, dtype = np.dtype(ri), count = 1)
                dimx, dimy, dimz, shrdx, shrdy, shrdz = np.fromfile(fr, dtype = np.dtype(rd), count = 6)
    
                if frame == 0:
                    nz = int(dimz/dz)
                    dz = dimz/nz
                    popall = np.zeros(nz)
                    popspec = np.zeros(nz)
    
                gloindex = np.fromfile(fr, dtype = np.dtype(ri), count = nsyst)
                specglobal = []
                for i in range(nsyst):
                    specglobal.append(partproperties[gloindex[i]-1][1]==1)
                for i in range(nsyst):
                    framedata = np.fromfile(fr, dtype = np.dtype(rd), count = (keytrj+1)*3)
                    zpos = int((framedata[2] + 0.5*dimz)/dz)
                    popall[zpos] += 1
                    if(specglobal[i] == True):
                        popspec[zpos] += 1

    # finally divide numbers of first type by numbers of all beads
    # to obtain time-averaged concentrations (volume fractions) of first type

            volfrac = popspec / popall
    
    # work out chi-parameter for current run by sampling concentration
    # in region expected to have more A beads, along with error bars
    # based on concentration variations, and then write interaction
    # parameters, chi-parameter and error, and concentration profile
    # along z-axis to file

        minchi = int(0.15*nz)
        maxchi = int(0.35*nz)
        meanvolfrac = statistics.mean(volfrac[minchi:maxchi])
        stdvolfrac = statistics.stdev(volfrac[minchi:maxchi])
        chi = math.log((1.0-meanvolfrac)/meanvolfrac)/(1.0-2.0*meanvolfrac)
        chimax = math.log((1.0-meanvolfrac-stdvolfrac)/(meanvolfrac+stdvolfrac))/(1.0-2.0*(meanvolfrac+stdvolfrac))
        chimin = math.log((1.0-meanvolfrac+stdvolfrac)/(meanvolfrac-stdvolfrac))/(1.0-2.0*(meanvolfrac-stdvolfrac))
        chierr = max(abs(chimax-chi), abs(chi-chimin))

        fw.write("{0:f},{1:f},{2:f},{3:f},{4:d}\n".format(Aii, Aij, chi, chierr, nz))
        for i in range(nz):
            fw.write("{0:11.7f}      {1:11.7f}\n".format((i+0.5)*dz, volfrac[i]))

        print("Written data for Aii = {0:f}, Aij = {1:f} to {2:s} - chi = {3:f} +/- {4:f}".format(Aii, Aij, filename, chi, chierr))
