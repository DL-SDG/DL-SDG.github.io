#!/usr/bin/env python3
"""Usage:
    history_dlm_to_local_vtk.py [--in <histin> --locout <locout> --avout <avout> --nx <nx> --ny <ny> --nz <nz> --lscale <lscale> --tscale <tscale> --mscale <mscale> --first <first> --last <last> --step <step> --averageonly --tabout <tabout> --lineX --lineY --lineZ --ix <ix> --iy <iy> --iz <iz>]

Calculates localised properties (densities, velocities, pressures) by dividing
up volume and assigning particles from DL_MESO_DPD HISTORY file to voxels, 
writing VTK formatted files for each frame and for time-averaged properties
over all given frames, optionally writing files of tabulated time-averaged 
properties along lines in a given direction for plotting 

Options:
    --in <histin>           Name of DL_MESO_DPD-format HISTORY file to read in and
                            convert [default: HISTORY]
    --locout <locout>       Name of VTK files to write instantaneous localised properties
                            for each available trajectory frame [default: local]
    --avout <avout>         Name of VTK file to write time-averaged localised properties
                            based on all given trajectory frames [default: averages] 
    --nx <nx>               Set number of cells in x-direction to <nx> [default: 1]
    --ny <ny>               Set number of cells in y-direction to <ny> [default: 1]
    --nz <nz>               Set number of cells in z-direction to <nz> [default: 1]
    --lscale <lscale>       DPD length scale in Angstroms [default: 1.0]
    --tscale <tscale>       DPD time scale in picoseconds [default: 1.0]
    --mscale <mscale>       DPD mass scale in daltons or unified atomic mass units [default: 1.0]
    --first <first>         Starting DL_MESO_DPD HISTORY file frame number for inclusion
                            in density and order parameter calculations [default: 1]
    --last <last>           Finishing DL_MESO_DPD HISTORY file frame number for inclusion
                            in density and order parameter calculations (value of 0 here 
                            will use last available frame) [default: 0]
    --step <step>           Incrementing number of frames in DL_MESO_DPD HISTORY between
                            frames used for density and order parameter calculations 
                            [default: 1]
    --averageonly           Only write time-averaged properties
    --tabout <tabout>       Name of output file for writing tabulated data (option only
                            activated if name provided)
    --lineX                 Plot tabulated data along line orthogonally to x-axis
    --lineY                 Plot tabulated data along line orthogonally to y-axis
    --lineZ                 Plot tabulated data along line orthogonally to z-axis
    --ix <ix>               Use cell <ix> in x-dimension for plotting data along line (chooses
                            middle cell if not selected) [default: 0]
    --iy <iy>               Use cell <iy> in y-dimension for plotting data along line (chooses
                            middle cell if not selected) [default: 0]
    --iz <iz>               Use cell <iz> in z-dimension for plotting data along line (chooses
                            middle cell if not selected) [default: 0]

michael.seaton@stfc.ac.uk, 25/11/22
"""
from docopt import docopt
from tqdm import tqdm
import dlmhistoryread as dlm
from pyevtk.hl import gridToVTK
import numpy as np
import itertools
import math
import sys

if __name__ == '__main__':
    # first check command-line arguments

    args = docopt(__doc__)
    histin = args["--in"]
    locout = args["--locout"]
    avout = args["--avout"]
    nx = int(args["--nx"])
    ny = int(args["--ny"])
    nz = int(args["--nz"])
    lscale = float(args["--lscale"])
    tscale = float(args["--tscale"])
    mscale = float(args["--mscale"])
    first = int(args["--first"])
    last = int(args["--last"])
    step = int(args["--step"])
    averageonly = args["--averageonly"]
    tabout = args["--tabout"]
    lineX = args["--lineX"]
    lineY = args["--lineY"]
    lineZ = args["--lineZ"]
    ix = int(args["--ix"])
    iy = int(args["--iy"])
    iz = int(args["--iz"])

    # read very beginning of DL_MESO_DPD HISTORY file to determine endianness,
    # sizes of number types, filesize, number of frames and last timestep number
    
    bo, ri, rd, intsize, longintsize, realsize, filesize, numframe, nsteplast = dlm.read_prepare(histin)

    # if not specified last frame to use in command-line argument or value for
    # first and/or last frames are too small/large, reset to defaults:
    # also work out how many frames are going to be used

    if numframe<1:
        sys.exit("No trajectory data available in "+histin+" file to calculate localised properties")

    if first>numframe:
        first = numframe

    first = first - 1

    if last==0 or last<first:
        last = numframe

    numframes = (last - first - 1) // step + 1

    # if total number of cells is 1, ask user to type in numbers of cells
    # to check this is what was requested

    if nx*ny*nz == 1:
        nx = 0
        ny = 0
        nz = 0
        while nx<1:
            nx = int(input("Enter number of cells in x-direction: "))
        while ny<1:
            ny = int(input("Enter number of cells in y-direction: "))
        while nz<1:
            nz = int(input("Enter number of cells in z-direction: "))
 
    # get information from DL_MESO_DPD HISTORY file header, including species,
    # molecule, particle and bond data
    
    nsyst, nusyst, timefirst, timelast, text, keytrj, surfaceprop, speciesprop, moleculeprop, particleprop, bondtable, framesize, headerpos = dlm.read_header(histin, bo, ri, rd, intsize, longintsize, realsize, numframe)

    # get hold of species and molecule types for all particles
    # ready for assignment to species/molecule type grids  

    beadspecies = [x[1] for x in particleprop]
    beadmoltypes = [x[2] for x in particleprop]
    numspecies = len(speciesprop)
    nummoltypes = len(moleculeprop)

    # prepare arrays for time-averaged properties: densities, velocities,
    # total and particle temperatures, velocity and force contributions 
    # to pressure tensors, numbers of frames with at least one particle
    # in each voxel

    density_mean = [np.zeros([nx, ny, nz]) for x in range(numspecies)]
    velocity_mean = [np.zeros([nx, ny, nz]) for x in range(3)]
    temperature_mean = np.zeros([nx, ny, nz])
    partial_temperature_mean = [np.zeros([nx, ny, nz]) for x in range(3)]
    pressure_velocity_mean = [np.zeros([nx, ny, nz]) for x in range(6)]
    pressure_force_mean = [np.zeros([nx, ny, nz]) for x in range(9)]
    number_frame_notempty = np.zeros([nx, ny, nz])

    # major loop through all required frames in DL_MESO_DPD HISTORY file
    
    print('Reading data from trajectories supplied in {0:s} file'.format(histin))

    dimx0 = 0.0
    dimy0 = 0.0
    dimz0 = 0.0

    for frame in tqdm(range(first, last, step)):
        # get all available information from DL_MESO_DPD HISTORY frame
        time, dimx, dimy, dimz, shrdx, shrdy, shrdz, particledata = dlm.read_frame(histin, ri, rd, frame, framesize, headerpos, keytrj)
        # determine grid spacing from box dimensions and specified grid size
        if dimx!=dimx0 or dimy!=dimy0 or dimz!=dimy0:
            dx = dimx / float(nx)
            dy = dimy / float(ny)
            dz = dimz / float(nz)
            deld = np.array([dx, dy, dz])
            vold = dx*dy*dz
            halfd = np.array([0.5*dimx, 0.5*dimy, 0.5*dimz])
            dimx0 = dimx
            dimy0 = dimy
            dimz0 = dimz
            # prepare locations of grid points to write to VTK files
            x = np.zeros((nx+1, ny+1, nz+1))
            y = np.zeros((nx+1, ny+1, nz+1))
            z = np.zeros((nx+1, ny+1, nz+1))
            for k in range(nz+1):
                Z = float(k) * deld[2]
                for j in range(ny+1):
                    Y = float(j) * deld[1]
                    for i in range(nx+1):
                        X = float(i) * deld[0]
                        x[i,j,k] = X
                        y[i,j,k] = Y
                        z[i,j,k] = Z

        # prepare arrays for localised data: numbers of beads for all 
        # and each species and molecule type, densities (for each species),
        # velocities, total and partial temperatures, velocity
        # and force contributions to pressure tensor

        all_beads = np.zeros([nx, ny, nz], dtype=int)
        species_beads = [np.zeros([nx, ny, nz]) for x in range(numspecies)]
        moles_beads = [np.zeros([nx, ny, nz]) for x in range(nummoltypes+1)]
        density = [np.zeros([nx, ny, nz]) for x in range(numspecies)]
        velocity = [np.zeros([nx, ny, nz]) for x in range(3)]
        temperature = np.zeros([nx, ny, nz])
        partial_temperature = [np.zeros([nx, ny, nz]) for x in range(3)]
        pressure_velocity = [np.zeros([nx, ny, nz]) for x in range(6)]
        pressure_force = [np.zeros([nx, ny, nz]) for x in range(9)]

        # go through all particles in trajectory frame, retrieve position,
        # velocity (if available) and force (if available), and use
        # each particle position to find its corresponding voxel and
        # multiplier for moment-of-planes method used for pressure tensor

        for i in range(len(particleprop)):
            xyz = particledata[i][1:4] + halfd
            n0 = (xyz // deld).astype(int)
            a0 = (xyz // deld + np.array([0.5, 0.5, 0.5])) * deld
            sa = np.sign(xyz-a0)
            vxyz = np.zeros(3) if keytrj<1 else particledata[i][4:7]
            fxyz = np.zeros(3) if keytrj<2 else particledata[i][7:10]
            spec = beadspecies[i] - 1
            mole = beadmoltypes[i]
            mass = speciesprop[spec][1]

        # now assign values to each array based on voxel number
        # and species/molecule type (where applicable)

            n0x = n0[0]
            n0y = n0[1]
            n0z = n0[2]
            species_beads[spec][n0x, n0y, n0z] += 1.0
            moles_beads[mole][n0x, n0y, n0z] += 1.0
            density[spec][n0x, n0y, n0z] += mass
            if not speciesprop[spec][4]:
                all_beads[n0x, n0y, n0z] += 1
                velocity[0][n0x, n0y, n0z] += vxyz[0]
                velocity[1][n0x, n0y, n0z] += vxyz[1]
                velocity[2][n0x, n0y, n0z] += vxyz[2]                
                temperature[n0x, n0y, n0z] += mass * (vxyz[0] * vxyz[0] + vxyz[1] * vxyz[1] + vxyz[2] * vxyz[2])
                partial_temperature[0][n0x, n0y, n0z] += mass * vxyz[0] * vxyz[0]
                partial_temperature[1][n0x, n0y, n0z] += mass * vxyz[1] * vxyz[1]
                partial_temperature[2][n0x, n0y, n0z] += mass * vxyz[2] * vxyz[2]
            pressure_velocity[0][n0x, n0y, n0z] += mass * vxyz[0] * vxyz[0]
            pressure_velocity[1][n0x, n0y, n0z] += mass * vxyz[0] * vxyz[1]
            pressure_velocity[2][n0x, n0y, n0z] += mass * vxyz[0] * vxyz[2]
            pressure_velocity[3][n0x, n0y, n0z] += mass * vxyz[1] * vxyz[1]
            pressure_velocity[4][n0x, n0y, n0z] += mass * vxyz[1] * vxyz[2]
            pressure_velocity[5][n0x, n0y, n0z] += mass * vxyz[2] * vxyz[2]
            pressure_force[0][n0x, n0y, n0z] += fxyz[0] * sa[0]
            pressure_force[1][n0x, n0y, n0z] += fxyz[0] * sa[1]
            pressure_force[2][n0x, n0y, n0z] += fxyz[0] * sa[2]
            pressure_force[3][n0x, n0y, n0z] += fxyz[1] * sa[0]
            pressure_force[4][n0x, n0y, n0z] += fxyz[1] * sa[1]
            pressure_force[5][n0x, n0y, n0z] += fxyz[1] * sa[2]
            pressure_force[6][n0x, n0y, n0z] += fxyz[2] * sa[0]
            pressure_force[7][n0x, n0y, n0z] += fxyz[2] * sa[1]
            pressure_force[8][n0x, n0y, n0z] += fxyz[2] * sa[2]

        # after accumulating values over all particles, work out
        # actual properties ready for writing to file

        density = density / vold
        pressure_velocity = pressure_velocity / vold 
        pressure_force[0] = 0.5 * pressure_force[0] / (deld[1] * deld[2])
        pressure_force[1] = 0.5 * pressure_force[1] / (deld[0] * deld[2])
        pressure_force[2] = 0.5 * pressure_force[2] / (deld[0] * deld[1])
        pressure_force[3] = 0.5 * pressure_force[3] / (deld[1] * deld[2])
        pressure_force[4] = 0.5 * pressure_force[4] / (deld[0] * deld[2])
        pressure_force[5] = 0.5 * pressure_force[5] / (deld[0] * deld[1])
        pressure_force[6] = 0.5 * pressure_force[6] / (deld[1] * deld[2])
        pressure_force[7] = 0.5 * pressure_force[7] / (deld[0] * deld[2])
        pressure_force[8] = 0.5 * pressure_force[8] / (deld[0] * deld[1])

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    rbeads = 0.0 if all_beads[i, j, k]==0 else 1.0 / float(all_beads[i, j, k])
                    for spec in range(numspecies):
                        species_beads[spec][i, j, k] = species_beads[spec][i, j, k] * rbeads
                    for mole in range(nummoltypes+1):
                        moles_beads[mole][i, j, k] =  moles_beads[mole][i, j, k] * rbeads
                    velocity[0][i, j, k] = velocity[0][i, j, k] * rbeads
                    velocity[1][i, j, k] = velocity[1][i, j, k] * rbeads
                    velocity[2][i, j, k] = velocity[2][i, j, k] * rbeads
                    temperature[i, j, k] = temperature[i, j, k] * rbeads / 3.0

                    partial_temperature[0][i, j, k] = partial_temperature[0][i, j, k] * rbeads
                    partial_temperature[1][i, j, k] = partial_temperature[1][i, j, k] * rbeads
                    partial_temperature[2][i, j, k] = partial_temperature[2][i, j, k] * rbeads
        
        # add properties to accumulators for time-averaged values

        for i in range(numspecies):
            density_mean[i] += density[i]

        temperature_mean += temperature
        for i in range(3):
            velocity_mean[i] += velocity[i]
            partial_temperature_mean[i] += partial_temperature[i]
        for i in range(6):
            pressure_velocity_mean[i] += pressure_velocity[i]
        for i in range(9):
            pressure_force_mean[i] += pressure_force[i]
        
        # write instantaneous data to VTK structured grid file

        if not averageonly:
            filename = "{0:s}_{1:06d}".format(locout, frame)
            cell = {}
            if keytrj>0:
                cell.update({"velocity": (velocity[0], velocity[1], velocity[2])})
            cell.update({"bead_numbers": all_beads})
            if keytrj>0:
                cell.update({"temperature": temperature})
                cell.update({"temperature_x": partial_temperature[0]})
                cell.update({"temperature_y": partial_temperature[1]})
                cell.update({"temperature_z": partial_temperature[2]})
            for i in range(numspecies):
                specname = speciesprop[i][0].rstrip(' \t\r\n\0')
                cell.update({"density_"+specname: density[i]})
            if keytrj>1:
                pressure = pressure_velocity[0] + pressure_force[0]
                cell.update({"pressure_xx": pressure})
                pressure = pressure_velocity[1] + pressure_force[1]
                cell.update({"pressure_xy": pressure})
                pressure = pressure_velocity[2] + pressure_force[2]
                cell.update({"pressure_xz": pressure})
                pressure = pressure_velocity[1] + pressure_force[3]
                cell.update({"pressure_yx": pressure})
                pressure = pressure_velocity[3] + pressure_force[4]
                cell.update({"pressure_yy": pressure})
                pressure = pressure_velocity[4] + pressure_force[5]
                cell.update({"pressure_yz": pressure})
                pressure = pressure_velocity[2] + pressure_force[6]
                cell.update({"pressure_zx": pressure})
                pressure = pressure_velocity[4] + pressure_force[7]
                cell.update({"pressure_zy": pressure})
                pressure = pressure_velocity[5] + pressure_force[8]
                cell.update({"pressure_zz": pressure})
            for i in range(numspecies):
                specname = speciesprop[i][0].rstrip(' \t\r\n\0')
                cell.update({"species_"+specname: species_beads[i]})
            for i in range(nummoltypes+1):
                specname = moleculeprop[i-1].rstrip(' \t\r\n\0') if i>0 else 'none'
                cell.update({"molecule_"+specname: moles_beads[i]})
            gridToVTK(filename, x, y, z, cellData=cell)

    # find time-averaged values of data and write to VTK structured grid file

    rframes = 0.0 if numframes==0 else 1.0/float(numframes)
    rframesfull = np.reciprocal(number_frame_notempty, where=(number_frame_notempty>0.0))

    for i in range(numspecies):
        density_mean[i] = density_mean[i] * rframes
    temperature_mean = temperature_mean * rframesfull
    for i in range(3):
        velocity_mean[i] = velocity_mean[i] * rframesfull 
        partial_temperature_mean[i] = partial_temperature_mean[i] * rframesfull
    for i in range(6):
        pressure_velocity_mean[i] = pressure_velocity_mean[i] * rframes
    for i in range(9):
        pressure_force_mean[i] = pressure_force_mean[i] * rframes

    cell = {}
    if keytrj>0:
        cell.update({"velocity": (velocity_mean[0], velocity_mean[1], velocity_mean[2])})
    for i in range(numspecies):
        specname = speciesprop[i][0].rstrip(' \t\r\n\0')
        cell.update({"density_"+specname: density_mean[i]})
    if keytrj>0:
        cell.update({"temperature": temperature_mean})
        cell.update({"temperature_x": partial_temperature_mean[0]})
        cell.update({"temperature_y": partial_temperature_mean[1]})
        cell.update({"temperature_z": partial_temperature_mean[2]})
    if keytrj>1:
        pressure = pressure_velocity_mean[0] + pressure_force_mean[0]
        cell.update({"pressure_xx": pressure})
        pressure = pressure_velocity_mean[1] + pressure_force_mean[1]
        cell.update({"pressure_xy": pressure})
        pressure = pressure_velocity_mean[2] + pressure_force_mean[2]
        cell.update({"pressure_xz": pressure})
        pressure = pressure_velocity_mean[1] + pressure_force_mean[3]
        cell.update({"pressure_yx": pressure})
        pressure = pressure_velocity_mean[3] + pressure_force_mean[4]
        cell.update({"pressure_yy": pressure})
        pressure = pressure_velocity_mean[4] + pressure_force_mean[5]
        cell.update({"pressure_yz": pressure})
        pressure = pressure_velocity_mean[2] + pressure_force_mean[6]
        cell.update({"pressure_zx": pressure})
        pressure = pressure_velocity_mean[4] + pressure_force_mean[7]
        cell.update({"pressure_zy": pressure})
        pressure = pressure_velocity_mean[5] + pressure_force_mean[8]
        cell.update({"pressure_zz": pressure})
    gridToVTK(avout, x, y, z, cellData=cell)
    print("Written time-averaged localised properties from {0:d} trajectory frames to {1:s}.vts file".format(numframes, avout))

    # if filename given, write tabulated file with time-averaged data along
    # line in selected direction: if direction not selected, choose direction
    # with largest number of cells; if cell in tangential dimensions not
    # selected or out of range, choose middle cells

    if tabout != None:
        if not lineX and not lineY and not lineZ:
            if nx>=ny and nx>=nz:
                lineX = True
            elif ny>=nx and ny>=nz:
                lineY = True
            else:
                lineZ = True
        datasets = []
        if lineX:
            iy = (iy - 1) if (iy>0 and iy<=ny) else ny//2
            iz = (iz - 1) if (iz>0 and iz<=nz) else nz//2
            cellmid = np.empty(nx)
            for i in range(nx):
                cellmid[i] = (float(i) + 0.5) * dx
            if keytrj>0:
                datasets.append(['velocity_x', velocity_mean[0][:,iy,iz].tolist()])
                datasets.append(['velocity_y', velocity_mean[1][:,iy,iz].tolist()])
                datasets.append(['velocity_z', velocity_mean[2][:,iy,iz].tolist()])
            for i in range(numspecies):
                specname = speciesprop[i][0].rstrip(' \t\r\n\0')
                datasets.append(['density_'+specname, density_mean[i][:,iy,iz]])
            if keytrj>0:
                datasets.append(['temperature', temperature_mean[:,iy,iz]])
                datasets.append(['temperature_x', partial_temperature_mean[0][:,iy,iz]])
                datasets.append(['temperature_y', partial_temperature_mean[1][:,iy,iz]])
                datasets.append(['temperature_z', partial_temperature_mean[2][:,iy,iz]])
            if keytrj>1:
                pressure = pressure_velocity_mean[0] + pressure_force_mean[0]
                datasets.append(['pressure_xx', pressure[:,iy,iz]])
                pressure = pressure_velocity_mean[1] + pressure_force_mean[1]
                datasets.append(['pressure_xy', pressure[:,iy,iz]])
                pressure = pressure_velocity_mean[2] + pressure_force_mean[2]
                datasets.append(['pressure_xz', pressure[:,iy,iz]])
                pressure = pressure_velocity_mean[1] + pressure_force_mean[3]
                datasets.append(['pressure_yx', pressure[:,iy,iz]])
                pressure = pressure_velocity_mean[3] + pressure_force_mean[4]
                datasets.append(['pressure_yy', pressure[:,iy,iz]])
                pressure = pressure_velocity_mean[4] + pressure_force_mean[5]
                datasets.append(['pressure_yz', pressure[:,iy,iz]])
                pressure = pressure_velocity_mean[2] + pressure_force_mean[6]
                datasets.append(['pressure_zx', pressure[:,iy,iz]])
                pressure = pressure_velocity_mean[4] + pressure_force_mean[7]
                datasets.append(['pressure_zy', pressure[:,iy,iz]])
                pressure = pressure_velocity_mean[5] + pressure_force_mean[8]
                datasets.append(['pressure_zz', pressure[:,iy,iz]])
        elif lineY:
            ix = (ix - 1) if (ix>0 and ix<=nx) else nx//2
            iz = (iz - 1) if (iz>0 and iz<=nz) else nz//2
            cellmid = np.empty(ny)
            for i in range(ny):
                cellmid[i] = (float(i) + 0.5) * dy
            if keytrj>0:
                datasets.append(['velocity_x', velocity_mean[0][ix,:,iz].tolist()])
                datasets.append(['velocity_y', velocity_mean[1][ix,:,iz].tolist()])
                datasets.append(['velocity_z', velocity_mean[2][ix,:,iz].tolist()])
            for i in range(numspecies):
                specname = speciesprop[i][0].rstrip(' \t\r\n\0')
                datasets.append(['density_'+specname, density_mean[i][ix,:,iz]])
            if keytrj>0:
                datasets.append(['temperature', temperature_mean[ix,:,iz]])
                datasets.append(['temperature_x', partial_temperature_mean[0][ix,:,iz]])
                datasets.append(['temperature_y', partial_temperature_mean[1][ix,:,iz]])
                datasets.append(['temperature_z', partial_temperature_mean[2][ix,:,iz]])
            if keytrj>1:
                pressure = pressure_velocity_mean[0] + pressure_force_mean[0]
                datasets.append(['pressure_xx', pressure[ix,:,iz]])
                pressure = pressure_velocity_mean[1] + pressure_force_mean[1]
                datasets.append(['pressure_xy', pressure[ix,:,iz]])
                pressure = pressure_velocity_mean[2] + pressure_force_mean[2]
                datasets.append(['pressure_xz', pressure[ix,:,iz]])
                pressure = pressure_velocity_mean[1] + pressure_force_mean[3]
                datasets.append(['pressure_yx', pressure[ix,:,iz]])
                pressure = pressure_velocity_mean[3] + pressure_force_mean[4]
                datasets.append(['pressure_yy', pressure[ix,:,iz]])
                pressure = pressure_velocity_mean[4] + pressure_force_mean[5]
                datasets.append(['pressure_yz', pressure[ix,:,iz]])
                pressure = pressure_velocity_mean[2] + pressure_force_mean[6]
                datasets.append(['pressure_zx', pressure[ix,:,iz]])
                pressure = pressure_velocity_mean[4] + pressure_force_mean[7]
                datasets.append(['pressure_zy', pressure[ix,:,iz]])
                pressure = pressure_velocity_mean[5] + pressure_force_mean[8]
                datasets.append(['pressure_zz', pressure[ix,:,iz]])
        else:
            ix = (ix - 1) if (ix>0 and ix<=nx) else nx//2
            iy = (iy - 1) if (iy>0 and iy<=ny) else ny//2
            cellmid = np.empty(nz)        
            for i in range(nz):
                cellmid[i] = (float(i) + 0.5) * dz
            if keytrj>0:
                datasets.append(['velocity_x', velocity_mean[0][ix,iy,:].tolist()])
                datasets.append(['velocity_y', velocity_mean[1][ix,iy,:].tolist()])
                datasets.append(['velocity_z', velocity_mean[2][ix,iy,:].tolist()])
            for i in range(numspecies):
                specname = speciesprop[i][0].rstrip(' \t\r\n\0')
                datasets.append(['density_'+specname, density_mean[i][ix,iy,:]])
            if keytrj>0:
                datasets.append(['temperature', temperature_mean[ix,iy,:]])
                datasets.append(['temperature_x', partial_temperature_mean[0][ix,iy,:]])
                datasets.append(['temperature_y', partial_temperature_mean[1][ix,iy,:]])
                datasets.append(['temperature_z', partial_temperature_mean[2][ix,iy,:]])
            if keytrj>1:
                pressure = pressure_velocity_mean[0] + pressure_force_mean[0]
                datasets.append(['pressure_xx', pressure[ix,iy,:]])
                pressure = pressure_velocity_mean[1] + pressure_force_mean[1]
                datasets.append(['pressure_xy', pressure[ix,iy,:]])
                pressure = pressure_velocity_mean[2] + pressure_force_mean[2]
                datasets.append(['pressure_xz', pressure[ix,iy,:]])
                pressure = pressure_velocity_mean[1] + pressure_force_mean[3]
                datasets.append(['pressure_yx', pressure[ix,iy,:]])
                pressure = pressure_velocity_mean[3] + pressure_force_mean[4]
                datasets.append(['pressure_yy', pressure[ix,iy,:]])
                pressure = pressure_velocity_mean[4] + pressure_force_mean[5]
                datasets.append(['pressure_yz', pressure[ix,iy,:]])
                pressure = pressure_velocity_mean[2] + pressure_force_mean[6]
                datasets.append(['pressure_zx', pressure[ix,iy,:]])
                pressure = pressure_velocity_mean[4] + pressure_force_mean[7]
                datasets.append(['pressure_zy', pressure[ix,iy,:]])
                pressure = pressure_velocity_mean[5] + pressure_force_mean[8]
                datasets.append(['pressure_zz', pressure[ix,iy,:]])
        so = "#           position"
        for j in range(len(datasets)):
            so += "   {:>17}".format(datasets[j][0])
        so +="\n"
        for i in range(len(cellmid)):
            so += "{0:20.6f}".format(cellmid[i])
            for j in range(len(datasets)):
                so += "{0:20.6f}".format(datasets[j][1][i])
            so += "\n"
        open(tabout,"w").write(so)
        print("Written tabulated time-averaged localised properties from {0:d} trajectory frames to {1:s} file".format(numframes, tabout))

