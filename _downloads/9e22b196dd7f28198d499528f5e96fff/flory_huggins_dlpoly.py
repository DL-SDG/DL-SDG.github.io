#!/usr/bin/env python3
"""Usage:
    flory_huggins.py [--rho <rho>] [--Aii <Aii>] [--Aijmin <Aijmin>] 
                     [--Aijmax <Aijmax>] [--dA <dA>] [--dz <dz>] 
                     [--L <L>] [--W <W>] [--nproc <nproc>]
                     [--dlpoly <dlpoly>]

Carries out DL_POLY calculations to determine relationship between
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
    --nproc <nproc>     Number of processor cores to run DL_POLY calculations
                        [default: 1]
    --dlpoly <dlpoly>   Location of DL_POLY executable [default: ./DLPOLY.Z]

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
    """Scans DL_POLY OUTPUT file to find essential information for reading further"""
    
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
                if "-------------------------------------" in content[line]:
                    startrun = line
                    break

    # look at next three lines to find column names - all but first column
    # have names of properties being printed - and data lines afterwards
    # to find number of first timestep in run

        if startrun>-1 and startrun+5<numlines:
            names = content[startrun+1].split()
            datanames = names[1:]
            names = content[startrun+2].split()
            datanames.extend(names[1:])
            names = content[startrun+3].split()
            datanames.extend(names[2:])
            words = content[startrun+5].split()
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
                if "-------------------------------------" in content[line] and line+8<numlines:
                    if "-------------------------------------" in content[line+8]:
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
        if "-------------------------------------" in content[line] and line+8<numlines:
            if "-------------------------------------" in content[line+8]:
                words = content[line+1].split()
                timestep = int(words[0])
                instantdata = list(map(float, words[1:]))
                words = content[line+2].split()
                instantdata += list(map(float, words[1:]))
                words = content[line+3].split()
                walltime = float(words[0])
                instantdata += list(map(float, words[1:]))
                words = content[line+5].split()
                runningdata = list(map(float, words[1:]))
                words = content[line+6].split()
                runningdata += list(map(float, words[1:]))
                words = content[line+7].split()
                runningdata += list(map(float, words))
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
    numtensor = 0

    if terminate:
        for line in range(avelines,numlines):
            if "-------------------------------------" in content[line] and line+11<numlines:
                if "-------------------------------------" in content[line+4]:
                    names = content[line+1].split()
                    datanames = names[1:]
                    names = content[line+2].split()
                    datanames += names[1:]
                    names = content[line+3].split()
                    datanames += names[2:]
                    words = content[line+5].split()
                    data = list(map(float, words[1:]))
                    averages.extend(data)
                    words = content[line+6].split()
                    data = list(map(float, words[1:]))
                    averages.extend(data)
                    words = content[line+7].split()
                    data = list(map(float, words[1:]))
                    averages.extend(data)
                    words = content[line+9].split()
                    data = list(map(float, words[1:]))
                    fluctuations.extend(data)
                    words = content[line+10].split()
                    data = list(map(float, words[1:]))
                    fluctuations.extend(data)
                    words = content[line+11].split()
                    data = list(map(float, words[1:]))
                    fluctuations.extend(data)
            elif "conservative contributions" in content[line] or "dissipative contributions" in content[line] or "random contributions" in content[line] or "kinetic contributions" in content[line] and line+4<numlines:
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
            elif "Pressure tensor:" in content[line] and line+4<numlines:
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
    nproc = int(args["--nproc"])
    dlpoly = str(Path(args["--dlpoly"]).resolve())

    if not (os.path.isfile(dlpoly) and os.access(dlpoly, os.X_OK)):
        sys.exit("ERROR: Cannot find DL_POLY executable (DLPOLY.Z)")

    if nproc > 1 :
        invoke = "mpirun -np {0:d} {1:s}".format(nproc, dlpoly)
    else :
        invoke = dlpoly

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

    sc = "title DL_POLY Flory-Huggins chi-parameter determination\n\n"
    sc += "io_units_scheme dpd\n\n"
    sc += "temperature 1.0 dpd_temp\n"
    sc += "cutoff 1.0 dpd_l\n\n"
    sc += "timestep 0.01 dpd_t\n"
    sc += "time_run 70000 steps\n"
    sc += "time_equilibration 20000 steps\n"
    sc += "zden_calculate ON\n"
    sc += "zden_print ON\n"
    sc += "zden_frequency 100 steps\n"
    sc += "zden_binsize {0:f} dpd_l\n".format(dz)
    sc += "stack_size 100 steps\n"
    sc += "print_frequency 100 steps\n"
    sc += "time_job 3600.0 s\n"
    sc += "time_close 200.0 s\n\n"
    sc += "ensemble nvt\n"
    sc += "ensemble_method dpd\n"
    sc += "ensemble_dpd_order mdvv\n"
    sc += "io_file_output SCREEN\n"

    open("CONTROL", "w").write(sc)
    print("CONTROL file saved.")

    # create CONFIG file

    wz = 0.5*L/Nz
    wx = W/Nx
    hz = 0.5*L
    hx = 0.5*W
    cf = "DL_POLY Flory-Huggins chi-parameter determination\n"
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

        sf = "DL_POLY Flory-Huggins chi-parameter determination\n\n"
        sf += "UNITS dpd\n"
        sf += "MOLECULES 2\n"
        sf += "A\n"
        sf += "NUMMOLS {0:d}\n".format(Npart)
        sf += "ATOMS 1\n"
        sf += "A 1.0 0.0 1 0\n"
        sf += "FINISH\n"
        sf += "B\n"
        sf += "NUMMOLS {0:d}\n".format(Npart)
        sf += "ATOMS 1\n"
        sf += "B 1.0 0.0 1 0\n"
        sf += "FINISH\n"
        sf += "VDW 3\n"
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
        dlpolyrun = subprocess.Popen(shlex.split(invoke), shell=False, stderr=outpipe)
        stepnum0 = 0
        pbar = tqdm(total=70000, desc='Running '+description)
        terminate = False
        if dlpolyrun.poll() is None:
            while not terminate:
                if os.path.getsize(outfile)>0:
                    _, startrun, _, _, numstep, terminate, _ = read_prepare(outfile)
                    if startrun>0:
                        rundata, _, _, _ = read_run(outfile, startrun, terminate)
                        if len(rundata)>0:
                            stepnum = int(rundata[-1,0])
                            if stepnum != stepnum0:
                                pbar.update(stepnum-stepnum0)
                                stepnum0 = stepnum
        pbar.close()
        dlpolyrun.wait()
        
        print("Completed DL_POLY run for Aii = {0:f}, Aij = {1:f}".format(Aii, Aij))

    # open ZDNDAT file and get hold of density profiles
    # for species A and B to calculate concentration profile

        fz = open("ZDNDAT", "r")
        content = fz.readlines()
        numlines = len(content)
        # second line contains number of data sets (should be 3) and
        # number of lines per set
        words = content[1].split()
        if (int(words[0])!=2):
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
                    rhoall[data] += rhoz
                break
        # now look for data set starting with "all species":
        # this will be the density profile for all bead species
        for line in range(2, len(content)):
            if "B" in content[line]:
                for data in range(nz):
                    words = content[line+data+1].split()
                    rhoz = float(words[1])
                    rhoall[data] += rhoz
                break
        # finally convert densities to concentrations (volume fractions)
        # and rename ZDNDAT file for run
        volfrac = rhospec/rhoall
        os.rename('ZDNDAT', 'ZDNDAT-Aii-{0:f}-Aij-{1:f}-rho-{2:f}'.format(Aii, Aij, rho))

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
