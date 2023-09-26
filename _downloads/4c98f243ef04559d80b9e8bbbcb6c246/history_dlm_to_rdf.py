#!/usr/bin/env python3
"""Usage:
    history_dlm_to_rdf.py [--in <histin> --out <rdfout> --lscale <lscale> --dr <dr> --rcut <rcut> --first <first> --last <last> --step <step> --fft --fftbin <fftbin> --fftout <fftout>]

Calculates radial distribution functions (RDFs) from DL_MESO_DPD HISTORY file 
for all pairs of particle species, optionally calculates structure factors
from RDFs

Options:
    --in <histin>       Name of DL_MESO_DPD-format HISTORY file to read in and
                        convert [default: HISTORY]
    --out <rdfout>      Name of file to write for RDF data [default: RDFDAT]
    --lscale <lscale>   DPD length scale in Angstroms [default: 1.0]
    --dr <dr>           Histogram bin size for particle pair distances in DPD 
                        length units [default: 0.05]
    --rcut <rcut>       Maximum distance between pairs of particles for RDFs 
                        in DPD length units [default: 2.0]
    --first <first>     Starting DL_MESO_DPD HISTORY file frame number for inclusion
                        in RDF calculations [default: 1]
    --last <last>       Finishing DL_MESO_DPD HISTORY file frame number for inclusion
                        in RDF calculations (value of 0 here will use last available 
                        frame) [default: 0]
    --step <step>       Incrementing number of frames in DL_MESO_DPD HISTORY between
                        frames used for RDF calculations [default: 1]
    --fft               Calculate structure factors from RDFs by carrying out 
                        Fast Fourier Transforms
    --fftbin <fftbin>   Set number of histogram bins for structure factor 
                        calculations (option only used if FFTs to be 
                        calculated, defaults to larger of 500 and double number
                        of bins used for RDF calculations) [default: 500]
    --fftout <fftout>   Name of file to write structure factors 
                        (Fourier-transformed RDFs, only used if FFTs are 
                        calculated) [default: RDFFFT]

michael.seaton@stfc.ac.uk, 14/10/22
based on rdf_lc.py by pv278@cam.ac.uk, 17/01/17
"""
from docopt import docopt
from tqdm import tqdm
import dlmhistoryread as dlm
import numpy as np
from numpy import pi
import itertools
from numba import jit, float64, int64
from numba.typed import Dict
from scipy.fft import dst
import math
import sys

def gridnum2id(n, Ncell):
    """Map 3d grid number to cell ID"""
    return ((n[0] * Ncell[0] + n[1]) * Ncell[1] + n[2])


def id2gridnum(ID, Ncell):
    """Map cell ID to 3d grid number"""
    gn = np.zeros(3).astype(int)
    gn[0] = ID // (Ncell[0]*Ncell[1])
    gn[1] = (ID - gn[0] * Ncell[0] * Ncell[1] ) // Ncell[1]
    gn[2] = ID - (gn[0] * Ncell[0] + gn[1]) * Ncell[1]
    return gn

@jit(float64(float64[:]), nopython=True)
def norm_numba(r):
    """Calculates scalar distance from vector (using Numba JIT compiler to speed this up)"""
    rn = 0.0
    for ri in r:
        rn += ri * ri
    return math.sqrt(rn)

@jit(nopython=True)
def images(rr, box, inv_box, dLx, dLy, dLz, shrx, shry, shrz, wallx, wally, wallz):
    """Calculates minimum distance between pair of particles based on periodic boundary conditions"""
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

    # if reflecting boundaries in use, replace modified components with actual distances

    if wallx:
        rrn[0] = rr[0]
    if wally:
        rrn[1] = rr[1]
    if wallz:
        rrn[2] = rr[2]

    return rrn

@jit(nopython=True)
def distance_vector(xyz, beadspecies, numpairs, surfaceprop, box, dLshear, lc, cp):
    """Generate lists of distances for all pairs of particle species"""
    shrdx = surfaceprop[0]==1
    shrdy = surfaceprop[1]==1
    shrdz = surfaceprop[2]==1
    wallx = surfaceprop[0]>1
    wally = surfaceprop[1]>1
    wallz = surfaceprop[2]>1

    dd = np.zeros(3)

    Nintra = sum([len(v) * (len(v) - 1) // 2 for v in lc.values()])
    Ninter = sum([len(lc[p[0]]) * len(lc[p[1]]) for p in cp])
    dv = np.zeros((numpairs, Nintra + Ninter))
    count = np.zeros(numpairs, dtype=int64)

    inv_box = np.linalg.pinv(box)

    # intracell
    for cell in lc.values():
        Na = len(cell)
        if cell[0]<0:
            continue
        for i in range(Na):
            speci = beadspecies[cell[i]]
            for j in range(i):
                specj = beadspecies[cell[j]]
                specpair = max(speci,specj) * (max(speci,specj) - 1) // 2 + min(speci,specj) - 1
                dd = xyz[cell[i]] - xyz[cell[j]]
                dd = images(dd, box, inv_box, dLshear[0], dLshear[1], dLshear[2], shrdx, shrdy, shrdz, wallx, wally, wallz)
                dv[specpair][count[specpair]] = norm_numba(dd)
                count[specpair] += 1
    # intercell
    for p in cp:
       for i in lc[p[0]]:
            if i<0:
                continue
            speci = beadspecies[i]
            for j in lc[p[1]]:
                if j<0:
                    continue
                specj = beadspecies[j]
                specpair = max(speci,specj) * (max(speci,specj) - 1) // 2 + min(speci,specj) - 1
                dd = xyz[i] - xyz[j]
                dd = images(dd, box, inv_box, dLshear[0], dLshear[1], dLshear[2], shrdx, shrdy, shrdz, wallx, wally, wallz)
                dv[specpair][count[specpair]] = norm_numba(dd)
                count[specpair] += 1
    return dv, count


def cell_pairs(lc, Ncell, surfaces, shearshift):
    """Generate list of unique pairs of neighbouring link cells"""
    cp = []
    for c in lc:
        cp.extend([set([c, i]) for i in neighbour_cells(c, Ncell, surfaces, shearshift) if i != c])
    temp = set([frozenset(i) for i in cp])  # choose unique pairs
    return [tuple(i) for i in temp]

def neighbour_cells(ID, Ncell, surfaces, shearshift):
    """Find all neighbouring cells incl the given one"""
    r = id2gridnum(ID, Ncell)
    neighs = []
    tmp = np.array([-1, 0, 1])
    tmpshr = np.array([-2, -1, 0, 1, 2])
    for p in itertools.product(tmp, repeat=3):
        neigh = r + p
        # check for non-periodic boundaries in each direction and exclude
        # if neighbour crosses periodic boundary
        include = True
        if surfaces[0]>0:
            include = (neigh[0] >= 0 or neigh[0]<Ncell[0])
        if surfaces[1]>0:
            include = (neigh[1] >= 0 or neigh[1]<Ncell[1])
        if surfaces[2]>0:
            include = (neigh[2] >= 0 or neigh[2]<Ncell[2])
        if include:
            neighs.append(neigh%Ncell)
    # deal with any lees-edwards shearing boundaries
    if surfaces[0]==1:
        dy = shearshift[1]
        dz = shearshift[2]
        if r[0]==0:
            for p in itertools.product(tmpshr, repeat=2):
                neigh = (r + [-1, p[0]-dy, p[1]-dz]) % Ncell
                neighs.append(neigh%Ncell)
        elif r[0]==Ncell[0]-1:
            for p in itertools.product(tmpshr, repeat=2):
                neigh = (r + [1, p[0]+dy, p[1]+dz]) % Ncell
                neighs.append(neigh%Ncell)
    if surfaces[1]==1:
        dx = shearshift[0]
        dz = shearshift[2]
        if r[1]==0:
            for p in itertools.product(tmpshr, repeat=2):
                neigh = (r + [p[0]-dx, -1, p[1]-dz]) % Ncell
                neighs.append(neigh%Ncell)
        elif r[1]==Ncell[1]-1:
            for p in itertools.product(tmpshr, repeat=2):
                neigh = (r + [p[0]+dx, 1, p[1]+dz]) % Ncell
                neighs.append(neigh%Ncell)
    if surfaces[2]==1:
        dx = shearshift[0]
        dy = shearshift[1]
        if r[2]==0:
            for p in itertools.product(tmpshr, repeat=2):
                neigh = (r + [p[0]-dx, p[1]-dy, -1]) % Ncell
                neighs.append(neigh%Ncell)
        elif r[2]==Ncell[2]-1:
            for p in itertools.product(tmpshr, repeat=2):
                neigh = (r + [p[0]+dx, p[1]+dy, 1]) % Ncell
                neighs.append(neigh%Ncell)

    return [gridnum2id(neigh, Ncell) for neigh in neighs]

if __name__ == '__main__':
    # first check command-line arguments

    args = docopt(__doc__)
    histin = args["--in"]
    rdfout = args["--out"]
    lscale = float(args["--lscale"])
    dr = float(args["--dr"])
    rcut = float(args["--rcut"])
    first = int(args["--first"])
    last = int(args["--last"])
    step = int(args["--step"])
    dofft = args["--fft"]
    fftbin = int(args["--fftbin"])
    fftout = args["--fftout"]
    
    # read very beginning of DL_MESO_DPD HISTORY file to determine endianness,
    # sizes of number types, filesize, number of frames and last timestep number
    
    bo, ri, rd, intsize, longintsize, realsize, filesize, numframe, nsteplast = dlm.read_prepare(histin)

    # if not specified last frame to use in command-line argument or value for
    # first and/or last frames are too small/large, reset to defaults:
    # also work out how many frames are going to be used

    if numframe<1:
        sys.exit("No trajectory data available in "+histin+" file to calculate RDFs")

    if first>numframe:
        first = numframe

    first = first - 1

    if last==0 or last<first:
        last = numframe

    numframes = (last - first - 1) // step + 1

    # get information from DL_MESO_DPD HISTORY file header, including species,
    # molecule, particle and bond data
    
    nsyst, nusyst, timefirst, timelast, text, keytrj, surfaceprop, speciesprop, moleculeprop, particleprop, bondtable, framesize, headerpos = dlm.read_header(histin, bo, ri, rd, intsize, longintsize, realsize, numframe)

    # set up list of possible species pairs for system

    numspecies = len(speciesprop)
    specpairs = []

    for i in range(numspecies):
        for j in range(i, numspecies):
            specpairs.append([speciesprop[i][0], speciesprop[j][0], i, j])

    numpairs = len(specpairs)

    # work out number of histogram bins from cutoff distance
    # and spacing (and re-adjust spacing if not quite aligned 
    # to number of bins)

    numbins = int(math.ceil(rcut/dr))
    dr = rcut / float(numbins)

    # work out number of histogram bins needed for Fourier
    # transform of RDFs (maximum of value given in command-line,
    # including default of 500, or twice the number of RDF bins)
    # and calculate frequency spacing

    fftbin = max(fftbin, 2*numbins)
    df = pi / (dr * float(fftbin))

    # prepare lists of distances and frequencies

    bins = np.linspace(0.0, rcut, numbins+1)
    r = bins[:-1] + 0.5 * dr

    freq = np.linspace(df, df*fftbin, fftbin)
 
    # set up RDF data arrays to collect numbers of particles in histogram ranges
 
    rdf = np.zeros((numpairs, numbins))
    rdfall = np.zeros(numbins)

    # create lists/arrays of particle species (previously sorted by global indices)
    # and find numbers of each kind

    beadspecies = [x[1] for x in particleprop]
    nspec = np.zeros(numspecies)
    for i in range(numspecies):
        nspec[i] = sum(x==i+1 for x in beadspecies)
    beadspecies = np.asarray(beadspecies)

    # convert surfaceprop to array

    surfaceprop = np.asarray(surfaceprop)

    # major loop through all required frames in DL_MESO_DPD HISTORY file
    
    dimx0 = dimy0 = dimz0 = 0.0
    
    print('Collecting data from trajectories supplied in {0:s} file'.format(histin))

    for frame in tqdm(range(first, last, step)):
        # get all available information from DL_MESO_DPD HISTORY frame
        time, dimx, dimy, dimz, shrdx, shrdy, shrdz, particledata = dlm.read_frame(histin, ri, rd, frame, framesize, headerpos, keytrj)
        # if system volume changes (including first frame) or using lees-edwards
        # shearing, set up or update link-cells and search patterns
        if dimx!=dimx0 or dimy!=dimy0 or dimz!=dimz0 or surfaceprop[0]==1 or surfaceprop[1]==1 or surfaceprop[2]==1:
            L = np.asfarray([dimx, dimy, dimz], np.double)
            box = L * np.eye(3)
            Ncell = (L // rcut).astype(int)
            lc = {}
            lcnb = Dict.empty(key_type=int64, value_type=int64[:])
            for i in range(Ncell[0]*Ncell[1]*Ncell[2]):
                lc[i] = []
            dLshear = np.asfarray([shrdx, shrdy, shrdz], np.double)
            shearshift = np.array([int(shrdx*Ncell[0]/dimx), int(shrdy*Ncell[1]/dimy), int(shrdz*Ncell[2]/dimz)])
            cp = np.array(cell_pairs(lc, Ncell, surfaceprop, shearshift))
            dimx0 = dimx
            dimy0 = dimy
            dimz0 = dimz
            Lx = L / Ncell
        else:
        # if no change in system volume and not using shear,
        # empty link-cell particle lists for reassignment
            for i in range(Ncell[0]*Ncell[1]*Ncell[2]):
                lc[i] = []

        # get particle coordinates as numpy double-precision array
        xyz = [x[:][1:4] for x in particledata]
        xyz = np.asfarray(xyz, np.double)
        # work out which particles belong to which cells and put into dictionary
        N = len(xyz)
        for i in range(N):
            num = (xyz[i] + 0.5*L) // Lx % Ncell
            lc[gridnum2id(num, Ncell)].append(i)
        # put together numba-friendly dictionary of particles in cells
        for i in range(len(lc)):
            if len(lc[i])>0:
                lcnb[i] = np.array(lc[i])
            else:
                lcnb[i] = np.array([-1])
        # collect together distances between particles for each pair of species
        dv, count = distance_vector(xyz, beadspecies, numpairs, surfaceprop, box, dLshear, lcnb, cp)
        for i in range(numpairs):
            countspec = count[i]
            rdf_raw, _ = np.histogram(dv[i][0:countspec], bins)
            rdf[i] += rdf_raw
            rdfall += rdf_raw

    sumrdf = [row[:] for row in rdf]

    # open RDFDAT file and write header at top

    fw = open(rdfout, "w")
    fw.write(text+"\n")
    fw.write('{0:10d}{1:10d}\n\n'.format(numframes, numbins))

    # work out individual and all species pairs radial distribution functions
    # before writing each of them (individual ones first) to RDFDAT file

    for i in range(numpairs):
        if specpairs[i][2]==specpairs[i][3]:
            N = nspec[specpairs[i][2]]
            Np = 0.5 * N * N
            NA = 0.5 * N
        else:
            NA = nspec[specpairs[i][2]]
            NB = nspec[specpairs[i][3]]
            Np = NA * NB
        fw.write('\n{0:8s} {1:8s}\n'.format(specpairs[i][0], specpairs[i][1]))
        sumrdf[i] = sumrdf[i] / (float(numframes) * NA)
        rdf[i] = rdf[i] * L[0] * L[1] * L[2] / (4.0 * pi * dr * (r * r + dr * dr / 12.0) * float(numframes) * Np)
        for j in range(numbins):
            fw.write('{0:14.6e}{1:14.6e}{2:14.6e}\n'.format(r[j]*lscale, rdf[i][j], sum(sumrdf[i][0:j+1])))
        fw.write('\n')
    
    sumrdfall = 2.0 * rdfall / (float(numframes) * nsyst)
    rdfall = rdfall * L[0] * L[1] * L[2] / (2.0 * pi * dr * (r * r + dr * dr / 12.0) * float(numframes) * nsyst * nsyst)
    
    fw.write('\nall species\n')
    for j in range(numbins):
        fw.write('{0:14.6e}{1:14.6e}{2:14.6e}\n'.format(r[j]*lscale, rdfall[j], sum(sumrdfall[0:j+1])))    

    fw.close()
    print('Written RDF data from {0:d} trajectory frames to {1:s} file'.format(numframes, rdfout))

    # if option is selected at command-line, calculate 
    # Fourier (sine) transforms of RDFs to obtain
    # structure factors for each species pair and 
    # for all particles

    if dofft:

    # open RDFFFT file and write header at top

        fw = open(fftout, "w")
        fw.write(text+"\n")
        fw.write('{0:10d}{1:10d}\n\n'.format(numframes, fftbin))

        for i in range(numpairs):
            rdfft = np.zeros(fftbin)
            for j in range(numbins-1):
                rdfft[j] = 0.5 * (rdf[i][j] + rdf[i][j+1]) - 1.0
            rdfft[numbins-1] = 0.0
            rdfft = dst(dr*lscale*(freq/df-0.5)*rdfft)
            fw.write('\n{0:8s} {1:8s}\n'.format(specpairs[i][0], specpairs[i][1]))
            for j in range(fftbin):
                fw.write('{0:14.6e}{1:14.6e}\n'.format(freq[j], 1.0+2.0*pi*dr*lscale*rdfft[j]/freq[j]))
            fw.write('\n')

        rdfft = np.zeros(fftbin)
        for j in range(numbins-1):
            rdfft[j] = 0.5 * (rdfall[j] + rdfall[j+1]) - 1.0
        rdfft[numbins-1] = 0.0
        rdfft = dst(dr*(freq/df-0.5)*rdfft)
        fw.write('\nall species\n')
        for j in range(fftbin):
            fw.write('{0:14.6e}{1:14.6e}\n'.format(freq[j], 1.0+2.0*pi*dr*lscale*rdfft[j]/freq[j]))

        fw.close()
        print('Written structural factor data based on RDFs to {0:s} file'.format(fftout))

