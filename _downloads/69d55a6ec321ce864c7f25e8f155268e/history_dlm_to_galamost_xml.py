#!/usr/bin/env python3
"""Usage:
    history_dlm_to_vtf.py [--con <con> --in <histin> --out <xmlout> --dt <dt> --mscale <mscale> --lscale <lscale> --tscale <tscale> --first <first> --last <last> --step <step>]

Converts DL_MESO_DPD HISTORY file into GALAMOST XML files (to open in e.g. OVITO)

Options:
    --con <con>         Use CONTROL-format file named <con> to find timestep size
                        [default: CONTROL]
    --in <histin>       Name of DL_MESO_DPD-format HISTORY file to read in and
                        convert [default: HISTORY]
    --out <xmlout>      Name of GALAMOST XML files to write without frame numbers or
                        extension [default: traject]
    --dt <dt>           Timestep size used in simulations, needed if no CONTROL
                        file present from which to read value [default: 0.0]
    --mscale <mscale>   DPD mass scale in daltons or unified atomic mass units [default: 1.0]
    --lscale <lscale>   DPD length scale in Angstroms [default: 1.0]
    --tscale <tscale>   DPD time scale in picoseconds [default: 1.0]
    --first <first>     Starting DL_MESO_DPD HISTORY file frame number for inclusion
                        in DL_POLY_4-formatted HISTORY file [default: 1]
    --last <last>       Finishing DL_MESO_DPD HISTORY file frame number for inclusion
                        in DL_POLY_4-formatted HISTORY file (value of 0 here will use last
                        available frame) [default: 0]
    --step <step>       Incrementing number of frames in DL_MESO_DPD HISTORY between
                        frames in DL_POLY_4-formatted HISTORY file [default: 1]

michael.seaton@stfc.ac.uk, 18/08/21
"""
from docopt import docopt
from tqdm import tqdm
import dlmhistoryread as dlm
import numpy as np
import math
import os
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

def galamost_xml_write_frame(filename, nstep, keytrj, dimx, dimy, dimz, numbond, particledata, lscale, vscale, type_txt, mass_txt, charge_txt, diameter_txt, molecule_txt, bond_txt):

    # writes trajectory frame as GALAMOST XML file, assuming information about species,
    # molecules and bonds are pre-prepared for adding as text blocks
    
    nbeads = len(particledata)
    
    # setup main GALAMOST tags
    
    galamost_xml = ET.Element("galamost_xml")
    galamost_xml.set("version","1.3")
    
    # setup configuration tags and include timestep, dimensions and number of particles
    
    configuration = ET.SubElement(galamost_xml, "configuration")
    configuration.set("time_step","{0:d}".format(nstep))
    configuration.set("dimensions","3")
    configuration.set("natoms","{0:d}".format(nbeads))
    
    # setup box tag with system dimensions
    
    box = ET.SubElement(configuration, "box")
    box.set("lx","{0:.6f}".format(lscale*dimx))
    box.set("ly","{0:.6f}".format(lscale*dimy))
    box.set("lz","{0:.6f}".format(lscale*dimz))
    
    # setup position tags and particle positions as text between tags
    
    position = ET.SubElement(configuration, "position")
    position.set("num","{0:d}".format(nbeads))
    position_text = "\n"
    for i in range(nbeads):
        position_text += "{0:12.6f} {1:12.6f} {2:12.6f}\n".format(particledata[i][1]*lscale, particledata[i][2]*lscale, particledata[i][3]*lscale)
    position.text = position_text
    
    # if available, setup velocity tags and particle velocities as text between tags
    
    if keytrj>0:
        velocity = ET.SubElement(configuration, "velocity")
        velocity.set("num","{0:d}".format(nbeads))
        velocity_text = "\n"
        for i in range(nbeads):
            velocity_text += "{0:12.6f} {1:12.6f} {2:12.6f}\n".format(particledata[i][4]*vscale, particledata[i][5]*vscale, particledata[i][6]*vscale)
        velocity.text = velocity_text
            
    # setup type, mass, charge and diameter tags with pre-prepared values for
    # all particles (based on species data)
    
    type = ET.SubElement(configuration, "type")
    type.set("num","{0:d}".format(nbeads))
    type.text = type_txt
    
    mass = ET.SubElement(configuration, "mass")
    mass.set("num","{0:d}".format(nbeads))
    mass.text = mass_txt
    
    charge = ET.SubElement(configuration, "charge")
    charge.set("num","{0:d}".format(nbeads))
    charge.text = charge_txt
    
    diameter = ET.SubElement(configuration, "diameter")
    diameter.set("num","{0:d}".format(nbeads))
    diameter.text = diameter_txt
    
    # if there are any bonds, setup bond and molecule tags with pre-prepared values
    # for bond connectivity and molecule names
    
    if numbond>0:
        bond = ET.SubElement(configuration, "bond")
        bond.set("num","{0:d}".format(numbond))
        bond.text = bond_txt
        molecule = ET.SubElement(configuration, "molecule")
        molecule.set("num","{0:d}".format(nbeads))
        molecule.text = molecule_txt
    
    # put together XML tags and data, and write result to file
    
    tree = ET.ElementTree(galamost_xml)
    ET.indent(tree, space="", level=0)
    tree.write(filename, xml_declaration=True,encoding='UTF-8',method='xml')


if __name__ == '__main__':
    # first check command-line arguments

    args = docopt(__doc__)
    dt = float(args["--dt"])
    con = args["--con"]
    histin = args["--in"]
    xmlout = args["--out"]
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
    # also work out how many frames are going into XML files

    if numframe<1:
        sys.exit("No trajectory data available in "+histin+" file to write to GALAMOST XML files")

    if first>numframe:
        first = numframe

    first = first - 1

    if last==0 or last<first:
        last = numframe

    numframes = (last - first - 1) // step + 1

    # get scaling factors for velocities

    vscale = lscale / tscale
    
    # get information from DL_MESO_DPD HISTORY file header, including species,
    # molecule, particle and bond data
    
    nsyst, nusyst, timefirst, timelast, text, keytrj, surfaceprop, speciesprop, moleculeprop, particleprop, bondtable, framesize, headerpos = dlm.read_header(histin, bo, ri, rd, intsize, longintsize, realsize, numframe)

    # find timestep number for first frame
    
    stepdiff = math.floor((timelast-timefirst)/dt+0.5)
    nstepfirst = nsteplast - stepdiff

    # prepare tags that remain constant for all trajectory frames:
    # particle types, masses, charges, diameters, bonds and molecule types
    
    type_txt = "\n"
    mass_txt = "\n"
    diameter_txt = "\n"
    charge_txt = "\n"
    molecule_txt = "\n"
    for i in range(nsyst):
        spec = particleprop[i][1]-1
        type_txt += "{0:s}\n".format(speciesprop[spec][0])
        mass_txt += "{0:10.6f}\n".format(speciesprop[spec][1]*mscale)
        diameter_txt += "{0:10.6f}\n".format(2.0*lscale*speciesprop[spec][2])
        charge_txt += "{0:10.6f}\n".format(speciesprop[spec][3])
        molecule_txt += "{0:10d}\n".format(particleprop[i][3])

    bond_txt = "\n"
    for i in range(len(bondtable)):
        index = bondtable[i][0] - 1
        moltype = particleprop[index][2]-1
        bond_txt += "{0:8s} {1:11d} {2:11d}\n".format(moleculeprop[moltype], index, bondtable[i][1]-1)
    
    # major loop through all required frames in DL_MESO_DPD HISTORY file
    
    for frame in tqdm(range(first, last, step)):
        framenum = (frame - first) // step
        filename = xmlout+"{0:06d}.xml".format(framenum)
        # get all available information from DL_MESO_DPD HISTORY frame
        time, dimx, dimy, dimz, shrdx, shrdy, shrdz, particledata = dlm.read_frame(histin, ri, rd, frame, framesize, headerpos, keytrj)
        # calculate timestep from specified time
        nstep = nstepfirst + math.floor((time-timefirst)/dt+0.5)
        # put together and write GALMOST XML file
        galamost_xml_write_frame(filename, nstep, keytrj, dimx, dimy, dimz, len(bondtable), particledata, lscale, vscale, type_txt, mass_txt, charge_txt, diameter_txt, molecule_txt, bond_txt)

