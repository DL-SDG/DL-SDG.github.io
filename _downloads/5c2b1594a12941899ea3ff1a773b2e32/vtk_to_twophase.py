#!/usr/bin/env python
"""Usage:
    vtk_to_twophase.py (--vtkin <vtkin>) [--lbin <lbin>] 
                       [--threshold <thres>] [--plot]
    
Reads VTK files created by DL_MESO_LBE for simulations of single-component
two phase systems forming a single drop or bubble, works out bulk densities
of vapour and liquid phases, determines centre-of-mass and radius for
drop/bubble, optionally determines size of vapour-liquid interface based
on threshold value given by user

Options:
    -h, --help              Show this screen
    --vtkin <vtkin>         Name of VTK file (either XML or Legacy Structured
                            Grid file) with DL_MESO_LBE simulation snapshot
                            (required input)
    --lbin <lbin>           Name of DL_MESO_LBE lbin.sys input file to read in
                            interaction information for surface tension
                            calculations (not carried out if not specified)
    --threshold <thres>     Threshold value of phase index for vapour phase
                            used to determine interfacial length between vapour
                            and liquid phases (optional calculation activated
                            by setting value above default of zero)
                            [default: 0]
    --plot                  Produce plot of fit to determine drop/bubble radii
                            (and interface length)

michael.seaton@stfc.ac.uk, 15/09/23
"""
from docopt import docopt
import vtk
import sys
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import skimage as ski
import numpy as np
from vtk.util import numpy_support as VN

def fit_ellipse(X):
    """fits points for 2D contour onto function for ellipse using Singular Value Decomposition"""
    """returns radii and transformation matrix for rotations"""

    N = len(X)
    x = X[:, 0]
    y = X[:, 1]
    U, S, V = np.linalg.svd(np.stack((x, y)))
    radii = np.sqrt(2/N)*S
    transform = np.sqrt(2/N) * U.dot(np.diag(S))
    
    return radii, transform

def fit_ellipsoid(X):
    """fits points for 3D isosurface onto function for ellipsoid"""
    """returns radii and transformation matrix for rotations"""

    x = X[:,0]
    y = X[:,1]
    z = X[:,2]
    D = np.array([x*x+y*y-2*z*z, x*x+z*z-2*y*y, 2*x*y, 2*x*z, 2*y*z, 2*x, 2*y, 2*z, 1-0*x])
    d2 = np.array(x*x+y*y+z*z).T
    u = np.linalg.solve(D.dot(D.T), D.dot(d2))
    a = np.array([u[0] + 1 * u[1] - 1])
    b = np.array([u[0] - 2 * u[1] - 1])
    c = np.array([u[1] - 2 * u[0] - 1])
    v = np.concatenate([a, b, c, u[2:]], axis=0).flatten()
    A = np.array([[v[0], v[3], v[4], v[6]], [v[3], v[1], v[5], v[7]], [v[4], v[5], v[2], v[8]], [v[6], v[7], v[8], v[9]]])
    centre = np.linalg.solve(- A[:3,:3], v[6:9])
    translation_matrix = np.eye(4)
    translation_matrix[3, :3] = centre.T
    R = translation_matrix.dot(A).dot(translation_matrix.T)

    evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
    transform = evecs.T
    
    radii = np.sqrt(1. / np.abs(evals))
    radii *= np.sign(evals)

    return radii, transform
    
def read_lbin(lbin):
    """ Read in lbin.sys file to get hold of equation of state information required to find pressures from densities"""

    try:
        with open(lbin) as file:
            content = file.read().splitlines()
        content = list(line for line in content if line)
        ShanChen = False
        Swift = False
        for i in range(len(content)):
            words = content[i].split()
            if words[0] == 'interaction_type':
                ShanChen = words[1].startswith('ShanChen')
                Swift = words[1].startswith('Swift')
        if not ShanChen and not Swift:
            sys.exit("ERROR: Shan-Chen or Swift free-energy interactions not used in DL_MESO_LBE simulation")
    except Exception as e:
        print('Failed to read %s. Reason: %s' % (lbin, e))
        sys.exit("Cannot read data from DL_MESO_LBE input file")

    # search through lbin.sys file to find parameters, equations of state etc.

    potential = 0
    gab = 0.0
    eos_a = 0.0
    eos_b = 0.0
    eos_omega = 0.0
    eos_tc = 0.0
    eos_pc = 0.0
    psi0 = 0.0
    rho0 = 0.0
    gasconst = 1.0
    systemp = 0.0

    for i in range(len(content)):
        words = content[i].split()
        if words[0]=='potential_type' or words[0]=='potential_type_0' or words[0]=='equation_of_state':
            if words[1]=="IdealLattice":
                potential = 0
            elif words[1]=="ShanChen1993":
                potential = 1
            elif words[1]=="ShanChen1994":
                potential = 2
            elif words[1]=="Qian1995":
                potential = 3
            elif words[1]=="Rho":
                potential = 4
            elif words[1]=="Ideal":
                potential = 5
            elif words[1]=="vanderWaals" or words[1]=="vdW":
                potential = 6
            elif words[1]=="CarnahanStarlingvanderWaals" or words[1]=="CSvdW":
                potential = 7
            elif words[1]=="RedlichKwong" or words[1]=="RK":
                potential = 8
            elif words[1]=="SoaveRedlichKwong" or words[1]=="SRK":
                potential = 9
            elif words[1]=="PengRobinson" or words[1]=="PR":
                potential = 10
            elif words[1]=="CarnahanStarlingRedlichKwong" or words[1]=="CSRK":
                potential = 11
        elif words[0] == 'interaction_0' or words[0] == 'interaction_0_0':
            gab = float(words[1])
        elif words[0] == 'density_inc_0':
            rho0 = float(words[1])
        elif words[0] == 'density_ini_0' and rho0==0.0:
            rho0 = float(words[1])
        elif words[0] == 'shanchen_psi0_0':
            psi0 = float(words[1])
        elif words[0] == 'eos_parameter_a_0' or words[0] == 'eos_parameter_a':
            eos_a = float(words[1])
        elif words[0] == 'eos_parameter_b_0' or words[0] == 'eos_parameter_b':
            eos_b = float(words[1])
        elif words[0] == 'critical_temperature_0':
            eos_tc = float(words[1])
        elif words[0] == 'critical_pressure_0':
            eos_pc = float(words[1])
        elif words[0] == 'acentric_factor_0':
            eos_omega = float(words[1])
        elif words[0] == 'gas_constant':
            gasconst = float(words[1])
        elif words[0] == 'temperature_system':
            systemp = float(words[1])

    # go through read-in parameters and get hold of any missing data or rearrange available data
    # (or report if cannot obtain data via other supplied values)
    
    if potential == 1:
        eos_a = rho0
        eos_b = gab * rho0 * rho0 / 6.0
    elif potential == 2:
        eos_a = rho0
        eos_b = gab * psi0 * psi0 / 6.0
    elif potential == 3:
        eos_a = rho0
        eos_b = gab / 6.0
    elif potential == 4:
        eos_a = gab / 6.0
        eos_b = 0.0
    elif potential == 6 and (eos_a==0.0 or eos_b==0):
        if eos_tc==0.0 or eos_pc==0.0:
            sys.exit("ERROR: insufficient parameters or critical properties supplied for van der Waals equation of state")
        else:
            eos_a = 27.0 * gasconst * gasconst * eos_tc * eos_tc / (64.0 * eos_pc)
            eos_b = gasconst * eos_tc / (8.0 * eos_pc)
        if eos_tc==0.0:
            eos_tc = 8.0 * eos_a / (27.0 * b * R)
    elif potential == 7 and (eos_a==0.0 or eos_b==0):
        if eos_tc==0.0 or eos_pc==0.0:
            sys.exit("ERROR: insufficient parameters or critical properties supplied for Carnhan-Starling-van der Waals equation of state")
        else:
            eos_a = 0.49638805772941 * gasconst * gasconst * eos_tc * eos_tc / eos_pc
            eos_b = 0.18729456694673 * gasconst * eos_tc / eos_pc
        if eos_tc==0.0:
            eos_tc =  0.20267685653536 * eos_a / ((eos_b * gasconst)**(2.0/3.0))
    elif potential == 8 and (eos_a==0.0 or eos_b==0):
        if eos_tc==0.0 or eos_pc==0.0:
            sys.exit("ERROR: insufficient parameters or critical properties supplied for Redlich-Kwong equation of state")
        else:
            eos_a = 0.42748023354034 * gasconst * gasconst * eos_tc * eos_tc / eos_pc
            eos_b = 0.08664034996496 * gasconst * eos_tc / eos_pc
        if eos_tc==0.0:
            eos_tc = 0.37731481253489 * eos_a / (eos_b * gasconst)
    elif potential == 9:
        if eos_a==0.0 or eos_b==0:
            if eos_tc==0.0 or eos_pc==0.0:
                sys.exit("ERROR: insufficient parameters or critical properties supplied for Soave-Redlich-Kwong equation of state")
            else:
                eos_a = 0.42748023354034 * gasconst * gasconst * eos_tc * eos_tc / eos_pc
                eos_b = 0.08664034996496 * gasconst * eos_tc / eos_pc
        if eos_tc==0.0:
            eos_tc = 0.20267685653536 * eos_a / (gasconst * eos_b)
    elif potential == 10:
        if eos_a==0.0 or eos_b==0:
            if eos_tc==0.0 or eos_pc==0.0:
                sys.exit("ERROR: insufficient parameters or critical properties supplied for Peng-Robinson equation of state")
            else:
                eos_a = 0.45723552892138 * gasconst * gasconst * eos_tc * eos_tc / eos_pc
                eos_b = 0.07779607390389 * gasconst * eos_tc / eos_pc
        if eos_tc==0.0:
            eos_tc = 0.17014442007035 * eos_a / (gasconst * eos_b)
    elif potential == 11 and (eos_a==0.0 or eos_b==0):
        if eos_tc==0.0 or eos_pc==0.0:
            sys.exit("ERROR: insufficient parameters or critical properties supplied for Carnhan-Starling-Redlich-Kwong equation of state")
        else:
            eos_a = 0.46111979136946 * gasconst * gasconst * eos_tc * eos_tc / eos_pc
            eos_b = 0.10482594711385 * gasconst * eos_tc / eos_pc
        if eos_tc==0:
            eos_tc = 0.2273290998908 * eos_a / ((eos_b * gasconst)**(2.0/3.0))

    return potential, eos_a, eos_b, eos_omega, eos_tc, gasconst, systemp

def pressure_eos(rho, eos, a, b, omega, Tc, R, T):
    """ Calculate pressure given density, equation of state and parameters """

    Tr = T/Tc
    p0 = 0.0
    
    if eos == 0:
    # ideal lattice gas
        p0 = rho / 3.0
    elif eos == 1:
    # Shan-Chen 1993 model
        p0 = rho / 3.0 + b * (1.0 - exp(-rho/a))
    elif eos == 2:
    # Shan-Chen 1994 model
        p0 = rho / 3.0 + b * exp(-(2.0*a/rho))
    elif eos == 3:
    # Qian 1995 model
        p0 = rho / 3.0 + b * a * a * rho * rho / ((a + rho)**2)
    elif eos == 4:
    # density (rho) model
        p0 = rho / 3.0 + a * rho * rho
    elif eos == 5:
    # ideal gas
        p0 = rho * R * T
    elif eos == 6:
    # van der Waals
        p0 = rho * R * T/(1.0 - b * rho) - a * rho * rho
    elif eos == 7:
    # Carnahan-Starling van der Waals
        phi = 0.25 * b * rho
        p0 = rho * R * T * (1.0 + phi + phi * phi - phi * phi * phi) /((1.0-phi)**3) - a * rho * rho
    elif eos == 8:
    # Redlich-Kwong
        p0 = rho * R * T / (1.0 - b * rho) - a * rho * rho / (np.sqrt(T) * (1.0 + b * rho))
    elif eos == 9:
    # Soave-Redlich-Kwong
        alpha = (1.0 + (0.480 + 1.574 * omega - 0.176 * omega * omega) * (1.0 - np.sqrt(Tr)))**2
        p0 = rho * R * T / (1.0 - b * rho) - a * alpha * rho * rho / (1.0 + b * rho)
    elif eos == 10:
    # Peng-Robinson
        alpha = (1.0 + (0.37464 + 1.54226 * omega - 0.26992 * omega * omega) * (1.0 - np.sqrt(Tr)))**2
        p0 = rho * R * T / (1.0 - b * rho) - a * alpha * rho * rho / (1.0 + 2.0 * b * rho - b * b * rho * rho)
    elif eos == 11:
    # Carnahan-Starling Redlich-Kwong
        phi = 0.25 * b * rho
        p0 = rho * R * T * (1.0 + phi + phi * phi - phi * phi * phi) /((1.0-phi)**3) - a * rho * rho

    return p0
    
if __name__=='__main__':
    args = docopt(__doc__)
    vtkin = args["--vtkin"]
    lbin = args["--lbin"]
    thres = float(args["--threshold"])
    plotting = args["--plot"]

    # check if VTK file is Legacy or XML format

    if vtkin[-4:]==".vtk":
        legacyVTK = True
    elif vtkin[-4:]==".vts":
        legacyVTK = False
    else:
        sys.exit("ERROR: selected VTK input file not in recognisable format")

    # determine whether or not looking for interfacial length based
    # on threshold value of phase index given by user

    interface = (thres>0.0)

    # determine whether or not looking for interfacial tension based
    # on whether or not lbin.sys file is specified
    
    tension = (lbin != None)
    
    # read in VTK file and get hold of all relevant information for
    # phase calculations, including size of grid and density data

    if legacyVTK:
        reader = vtk.vtkStructuredPointsReader()
        reader.SetFileName(vtkin)
        reader.ReadAllVectorsOn()
        reader.ReadAllScalarsOn()
        reader.Update()
        data = reader.GetOutput()
    else:
        reader = vtk.vtkXMLStructuredGridReader()
        reader.SetFileName(vtkin)
        reader.Update()
        data = reader.GetOutput()

    extent = data.GetExtent()
    numX = extent[1] - extent[0] + 1
    numY = extent[3] - extent[2] + 1
    numZ = extent[5] - extent[4] + 1

    threeD = (numZ>1)

    numdata = data.GetPointData().GetNumberOfArrays()
    #coords = VN.vtk_to_numpy(reader.GetOutput().GetPoints().GetData()).reshape(numX, numY, numZ, 3)
    #print(coords)

    density = []

    for i in range(numdata):
        namedata = data.GetPointData().GetArrayName(i)
        dataset = VN.vtk_to_numpy(data.GetPointData().GetArray(i))
        if namedata.startswith('density'):
            dataset = dataset.reshape(numZ, numY, numX).transpose(2, 1, 0)
            density.append(dataset)

    # check that only one fluid is used in simulation: if not,
    # exit with error message (need to use different script for
    # multiple fluid systems!)

    if len(density)>1:
        sys.exit("ERROR: More than one fluid in simulation - cannot use this script for analysis!")

    # start by searching for minimum and maximum fluid densities,
    # and assume these are the vapour and liquid densities

    rhoV = np.min(density[0])
    rhoL = np.max(density[0])

    print("Estimated vapour phase density: {0:f}".format(rhoV))
    print("Estimated liquid phase density: {0:f}".format(rhoL))

    # calculate phase indices based on vapour and liquid densities
    # for all grid points, and work out which is dominant phase

    rhoN = np.zeros((numX, numY, numZ))

    for k in range(numZ):
        for j in range(numY):
            for i in range(numX):
                rhoN[i,j,k] = (density[0][i,j,k] - rhoV)/(rhoL-rhoV)

    sumrhoN = np.sum(rhoN)
    meanrhoN = sumrhoN / (numX*numY*numZ)
    liquiddrop = (meanrhoN < 0.5)

    # use phase indices to work out location of drop/bubble centre

    zeta_x = xi_x = 0.0
    zeta_y = xi_y = 0.0
    zeta_z = xi_z = 0.0
    mass = 0.0

    if liquiddrop:
        for k in range(numZ):
            omega_z = 2.0 * math.pi * float(k) / numZ
            for j in range(numY):
                omega_y = 2.0 * math.pi * float(j) / numY
                for i in range(numX):
                    omega_x = 2.0 * math.pi * float(i) / numX
                    zeta_x += rhoN[i,j,k] * math.sin(omega_x)
                    xi_x   += rhoN[i,j,k] * math.cos(omega_x)
                    zeta_y += rhoN[i,j,k] * math.sin(omega_y)
                    xi_y   += rhoN[i,j,k] * math.cos(omega_y)
                    zeta_z += rhoN[i,j,k] * math.sin(omega_z)
                    xi_z   += rhoN[i,j,k] * math.cos(omega_z)
                    mass   += rhoN[i,j,k]
    else:
        for k in range(numZ):
            omega_z = 2.0 * math.pi * float(k) / numZ
            for j in range(numY):
                omega_y = 2.0 * math.pi * float(j) / numY
                for i in range(numX):
                    omega_x = 2.0 * math.pi * float(i) / numX
                    zeta_x += (1.0-rhoN[i,j,k]) * math.sin(omega_x)
                    xi_x   += (1.0-rhoN[i,j,k]) * math.cos(omega_x)
                    zeta_y += (1.0-rhoN[i,j,k]) * math.sin(omega_y)
                    xi_y   += (1.0-rhoN[i,j,k]) * math.cos(omega_y)
                    zeta_z += (1.0-rhoN[i,j,k]) * math.sin(omega_z)
                    xi_z   += (1.0-rhoN[i,j,k]) * math.cos(omega_z)
                    mass   += (1.0-rhoN[i,j,k])
    
    zeta_x /= mass
    xi_x   /= mass
    zeta_y /= mass
    xi_y   /= mass
    zeta_z /= mass
    xi_z   /= mass

    omega_x = math.atan2(-zeta_x, -xi_x) + math.pi
    omega_y = math.atan2(-zeta_y, -xi_y) + math.pi
    if threeD:
        omega_z = math.atan2(-zeta_z, -xi_z) + math.pi
    else:
        omega_z = 0.0
    
    com_x = 0.5 * omega_x * numX / math.pi
    com_y = 0.5 * omega_y * numY / math.pi
    com_z = 0.5 * omega_z * numZ / math.pi

    if liquiddrop:
        print("Centre-of-mass for liquid drop: ({0:f}, {1:f}, {2:f})".format(com_x, com_y, com_z))
    else:
        print("Centre-of-mass for vapour bubble: ({0:f}, {1:f}, {2:f})".format(com_x, com_y, com_z))

    # check centre-of-mass is definitely inside drop or bubble
    # by finding phase index value at nearest grid point: if not,
    # there might be more than one drop/bubble

    cxint = int(com_x+0.5)
    cyint = int(com_y+0.5)
    czint = int(com_z+0.5)

    if (liquiddrop and rhoN[cxint,cyint,czint]<0.5) or (not liquiddrop and rhoN[cxint,cyint,czint]>0.5):
        sys.exit("ERROR: centre-of-mass located outside drop/bubble")
    
    # take phase index values and find contour/surface corresponding to
    # rhoN = 0.5 (boundary between phases), then use to find radii of drop/bubble
    # by fitting contour/surface points to function for ellipse/ellipsoid

    if threeD:
        verts, _, _, _ = ski.measure.marching_cubes(rhoN, 0.5)
        verts_periodic = []
        for contour in verts:
            for i in range(len(contour)):
                xx = contour[i, 0] - com_x
                yy = contour[i, 1] - com_y
                zz = contour[i, 2] - com_z
                xx = xx - round(xx/numX) * numX
                yy = yy - round(yy/numY) * numY
                zz = zz - round(zz/numZ) * numZ
                verts_periodic.append(np.asarray([xx, yy, zz]))
        verts_periodic = np.asarray(verts_periodic)
        radii, transform = fit_ellipsoid(verts_periodic)
        meanradius = (radii[0]*radii[1]*radii[2])**(1.0/3.0)
        if liquiddrop:
            print("Radii (semi-axes) of liquid drop: a = {0:f}, b = {1:f}, c = {2:f}".format(radii[0], radii[1], radii[2]))
            print("Mean radius of liquid drop: {0:f}".format(meanradius))
        else:
            print("Radii (semi-axes) of vapour bubble: a = {0:f}, b = {1:f}, c = {2:f}".format(radii[0], radii[1], radii[2]))
            print("Mean radius of vapour bubble: {0:f}".format((radii[0]*radii[1]*radii[2])**(1.0/3.0)))
            
    else:
        verts = ski.measure.find_contours(rhoN[:,:,0], 0.5)
        verts_periodic = []
        for contour in verts:
            for i in range(len(contour)):
                xx = contour[i, 0] - com_x
                yy = contour[i, 1] - com_y
                xx = xx - round(xx/numX) * numX
                yy = yy - round(yy/numY) * numY
                verts_periodic.append(np.asarray([xx, yy]))
        verts_periodic = np.asarray(verts_periodic)
        radii, transform = fit_ellipse(verts_periodic)
        if liquiddrop:
            print("Radii (semi-axes) of liquid drop: a = {0:f}, b = {1:f}".format(radii[0], radii[1]))
            print("Mean radius of liquid drop: {0:f}".format((radii[0]*radii[1])**0.5))
        else:
            print("Radii (semi-axes) of vapour bubble: a = {0:f}, b = {1:f}".format(radii[0], radii[1]))
            print("Mean radius of vapour bubble: {0:f}".format((radii[0]*radii[1])**0.5))
    
    if tension:
        kurv = 1.0 / meanradius
        eos, a, b, omega, Tc, R, T = read_lbin(lbin)
        p_L = pressure_eos(rhoL, eos, a, b, omega, Tc, R, T)
        p_V = pressure_eos(rhoV, eos, a, b, omega, Tc, R, T)
        print("Pressure of liquid phase: {0:f}".format(p_L))
        print("Pressure of vapour phase: {0:f}".format(p_V))
        print("Mean curvature of drop/bubble: {0:f}".format(kurv))
        print("Estimated surface tension: {0:f}".format(0.5*abs(p_L-p_V)/kurv))
    
    # if requested, obtain contours/isosurfaces for phase indices at
    # specified values to work out thickness of phase interface

    if interface:
        if threeD:
            verts1, _, _, _ = ski.measure.marching_cubes(rhoN, thres)
            verts1_periodic = []
            for contour in verts1:
                for i in range(len(contour)):
                    xx = contour[i, 0] - com_x
                    yy = contour[i, 1] - com_y
                    zz = contour[i, 2] - com_z
                    xx = xx - round(xx/numX) * numX
                    yy = yy - round(yy/numY) * numY
                    zz = zz - round(zz/numZ) * numZ
                    verts1_periodic.append(np.asarray([xx, yy, zz]))
            verts1_periodic = np.asarray(verts1_periodic)
            radii1, transform1 = fit_ellipsoid(verts1_periodic)
            verts2, _, _, _ = ski.measure.marching_cubes(rhoN, 1.0-thres)
            verts2_periodic = []
            for contour in verts2:
                for i in range(len(contour)):
                    xx = contour[i, 0] - com_x
                    yy = contour[i, 1] - com_y
                    zz = contour[i, 2] - com_z
                    xx = xx - round(xx/numX) * numX
                    yy = yy - round(yy/numY) * numY
                    zz = zz - round(zz/numZ) * numZ
                    verts2_periodic.append(np.asarray([xx, yy, zz]))
            verts2_periodic = np.asarray(verts2_periodic)
            radii2, transform2 = fit_ellipsoid(verts2_periodic)
        else:
            verts1 = ski.measure.find_contours(rhoN[:,:,0], thres)
            verts1_periodic = []
            for contour in verts1:
                for i in range(len(contour)):
                    xx = contour[i, 0] - com_x
                    yy = contour[i, 1] - com_y
                    xx = xx - round(xx/numX) * numX
                    yy = yy - round(yy/numY) * numY
                    verts1_periodic.append(np.asarray([xx, yy]))
            verts1_periodic = np.asarray(verts1_periodic)
            radii1, transform1 = fit_ellipse(verts1_periodic)
            verts2 = ski.measure.find_contours(rhoN[:,:,0], 1.0-thres)
            verts2_periodic = []
            for contour in verts2:
                for i in range(len(contour)):
                    xx = contour[i, 0] - com_x
                    yy = contour[i, 1] - com_y
                    xx = xx - round(xx/numX) * numX
                    yy = yy - round(yy/numY) * numY
                    verts2_periodic.append(np.asarray([xx, yy]))
            verts2_periodic = np.asarray(verts2_periodic)
            radii2, transform2 = fit_ellipse(verts2_periodic)
        diffradius = np.abs(radii2-radii1)
        thickness = np.max(diffradius)
        print("Estimated interfacial thickness: {0:f}".format(thickness))
        
    # if selected, plot contour/isosurface of drop/bubble boundary based on
    # fitted function to show fit to data points obtained from phase indices

    if plotting:
        if threeD:
            fig = plt.figure('fit drop/bubble to ellipsoid')
            ax = fig.add_subplot(111, projection='3d')
            u = np.linspace(0.0, 2.0*np.pi, 100)
            v = np.linspace(0.0, np.pi, 100)
            x = radii[0] * np.outer(np.cos(u), np.sin(v))
            y = radii[1] * np.outer(np.sin(u), np.sin(v))
            z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
            for i in range(len(x)):
                for j in range(len(x)):
                    [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], transform)
            
            axes = np.array([[radii[0], 0.0, 0.0], [0.0, radii[1], 0.0], [0.0, 0.0, radii[2]]])
            for i in range(len(axes)):
                axes[i] = np.dot(axes[i], rotation)
            for p in axes:
                X3 = np.linspace(-p[0], p[0], 100)
                Y3 = np.linspace(-p[1], p[1], 100)
                Z3 = np.linspace(-p[2], p[2], 100)
                ax.plot(X3, Y3, Z3, color='b')
            ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color='b', alpha=0.2)
            ax.scatter(verts_periodic[:,0], verts_periodic[:,1], verts_periodic[:,2], marker='o', color='g')
            plt.show()
        else:
            tt = np.linspace(0, 2*np.pi, 1000)
            circle = np.stack((np.cos(tt), np.sin(tt)))
            fit = transform.dot(circle)
            fig = plt.figure('fit drop/bubble to ellipse')
            ax = fig.add_subplot(111)
            ax.set_aspect('equal', adjustable='box')
            ax.plot(verts_periodic[:,0], verts_periodic[:,1], '.')
            ax.plot(fit[0, :], fit[1, :], 'r', label=r'$\rho^N = 0.5$')
            if interface:
                fit1 = transform1.dot(circle)
                ax.plot(fit1[0, :], fit1[1, :], 'g', label=r'$\rho^N = {0:f}$'.format(thres))
                fit2 = transform2.dot(circle)
                ax.plot(fit2[0, :], fit2[1, :], 'b', label=r'$\rho^N = {0:f}$'.format(1.0-thres))
            ax.legend()
            plt.show()

