#!/usr/bin/env python
"""Usage:
    maxwell.py (--vdW | --CSvdW | --RK | --SRK | --PR | --CSRK) (--Tr <Tr> | --T <T>) [--a <a>] [--b <b>] [--R <R>] [--omega <omega>] [--maxVr <maxVr>] [--subint <subint>]

Calculates saturated pressure and densities of vapour and liquid phases for
given temperature using Maxwell construction, plotting isotherm afterwards.
Can supply just equation of state and reduced temperature (and acentric
factor for relevant equations of state), but parameters a and b can be used
to scale resulting values of pressure and density to 'real' units.

Options:
    -h, --help          Show this screen
    --vdW               Use van der Waals equation of state
    --CSvdW             Use Carnahan-Starling van der Waals equation of state
    --RK                Use Redlich-Kwong equation of state
    --SRK               Use Soave-Redlich-Kwong equation of state
    --PR                Use Peng-Robinson equation of state
    --CSRK              Use Carnahan-Starling Redlich-Kwong equation of state
    --Tr <Tr>           Reduced temperature for Maxwell construction
    --T <T>             Temperature for Maxwell construction
    --a <a>             Parameter a for equation of state (optional if only
                        specifying reduced temperature, used to scale results)
                        [default: 0.0]
    --b <b>             Parameter b for equation of state (optional if only
                        specifying reduced temperature, used to scale results)
                        [default: 0.0]
    --R <R>             Universal gas constant R (optional if only specifying
                        reduced temperature, used to scale results)
                        [default: 1.0]
    --omega <omega>     Acentric factor for equation of state (only required for
                        Soave-Redlich-Kwong and Peng-Robinson equations of
                        state) [default: 0.0]
    --maxVr <maxVr>     Maximum reduced volume required to carry out Maxwell
                        construction (might need to increase this value for
                        lower reduced temperatures) [default: 10.0]
    --subint <subint>   Maximum number of subintervals to use in adaptive
                        integration carried out for Maxwell construction
                        (might need to increase this value for lower
                        reduced temperatures) [default: 50]
    
michael.seaton@stfc.ac.uk, 16/09/23
Based on script at https://scipython.com/blog/the-maxwell-construction/
"""

from docopt import docopt
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import newton
from scipy.signal import argrelextrema

#palette = iter(['#9b59b6', '#4c72b0', '#55a868', '#c44e52', '#dbc256'])
palette = iter(['#4c72b0'])

def eos(type, Tr, Vr, omega):
    """Equation of state: return the reduced pressure from the reduced
    temperature, reduced volume and acentric factor (latter only used
    for Soave-Redlich-Kwong and Peng-Robinson EOS).
    """
    pr = 0.0
    if type==0: # van der Waals
        pr = 8*Tr/(3*Vr-1) - 3/Vr**2
    elif type==1: # Carnahan-Starling van der Waals
        theta = 1.91653293328153028625223271960308792
        rtheta = 1.0/theta
        aa = (1.0+theta*(-16.0+theta*(64.0+theta*(256.0+theta*256.0))))/(1.0+theta*theta*(-64.0+theta*(-256.0+theta*256.0)))
        rzc = 2.0*(4.0*theta-1.0)*(4.0*theta-1.0)*(4.0*theta-1.0)*(4.0*theta-1.0)/(1.0+theta*theta*(-64.0+theta*(-256.0+theta*256.0)))
        pr = rzc*Tr*(64.0*Vr*Vr*Vr+16.0*rtheta*Vr*Vr+4.0*rtheta*rtheta*Vr-rtheta*rtheta*rtheta)/(Vr*(4.0*Vr-rtheta)**3)-aa/(Vr*Vr)
    elif type==2: # Redlich-Kwong
        theta = 1.0 + (2.0**(1.0/3.0)) + (4.0**(1.0/3.0))
        rtheta = 1.0/theta
        pr = 3.0*Tr/(Vr-rtheta) - theta/(Vr*(Vr+rtheta)*Tr**0.5)
    elif type==3: # Soave-Redlich-Kwong
        theta = 1.0 + (2.0**(1.0/3.0)) + (4.0**(1.0/3.0))
        rtheta = 1.0/theta
        sqrtTr = Tr**0.5
        m = 1.0 + (0.480 + omega*(1.574-omega*0.176)) * (1.0-sqrtTr)
        m = m*m
        pr = 3.0*Tr/(Vr-rtheta) - m*theta/(Vr*(Vr+rtheta))
    elif type==4: # Peng-Robinson
        theta = 1.0 + (4.0 - 8.0**0.5)**(1.0/3.0) + (4.0 + 8.0**0.5)**(1.0/3.0)
        rtheta = 1.0/theta
        sqrtTr = Tr**0.5
        m = 1.0 + (0.37464 + omega*(1.54226-omega*0.26992)) * (1.0-sqrtTr)
        m = m*m
        aa = (theta*theta+2.0*theta-1.0)*(theta*theta+2.0*theta-1.0)*rtheta*rtheta/(theta*theta-2.0*theta-1.0)
        rzc = 2.0*(theta+1.0)*(theta-1.0)*(theta-1.0)*rtheta/(theta*theta-2.0*theta-1.0)
        pr = rzc*Tr/(Vr-rtheta) - m*aa/(Vr*Vr + 2.0*Vr*rtheta - rtheta*rtheta)
    elif type==5: # Carnahan-Starling Redlich-Kwong
        theta = 3.00680414886104355791114560215239732
        rtheta = 1.0/theta
        aa = (1.0+theta*(-16.0+theta*(64.0+theta*(256.0+theta*256.0))))*(theta+1.0)*(theta+1.0)*rtheta*rtheta/(1.0+theta*theta*(-64.0+theta*(-256.0+theta*256.0)))
        rzc = (4.0*theta-1.0)*(4.0*theta-1.0)*(4.0*theta-1.0)*(4.0*theta-1.0)*(2.0*theta+1.0)*rtheta/(1.0+theta*theta*(-64.0+theta*(-256.0+theta*256.0)))
        pr = rzc*Tr*(64.0*Vr*Vr*Vr+16.0*rtheta*Vr*Vr+4.0*rtheta*rtheta*Vr-rtheta*rtheta*rtheta)/(Vr*(4.0*Vr-rtheta)**3)-aa/(Vr*(Vr+rtheta)*Tr**0.5)
    
    return pr


def eos_maxwell(type, Tr, Vr, omega):
    """Equation of state with Maxwell construction: return the reduced pressure
    from reduced temperature and volume (and acentric factor if EOS requires it),
    applying the Maxwell construction correction to the unphysical region as
    necessary.
    """

    pr = eos(type, Tr, Vr, omega)

    if Tr >= 1:
        # no unphysical region above the critical temperature:
        # just return reduced pressures
        return pr

    # initial guess for the position of the Maxwell construction line:
    # volume corresponding to the mean pressure between the minimum and
    # maximum in reduced pressure (pr)
    
    iprmin = argrelextrema(pr, np.less)
    iprmax = argrelextrema(pr, np.greater)
    Vr0 = np.mean([Vr[iprmin], Vr[iprmax]])

    # root finding using Newton's (secant) method to determine Vr0,
    # corresponding to equal loop areas for the Maxwell construction
    
    Vr0 = newton(get_area_difference, Vr0, args=(type, Tr, omega))
    pr0 = eos(type, Tr, Vr0, omega)
    Vrmin, Vrmax = get_Vlims(type, Tr, pr0, omega)

    # set the pressure in the Maxwell construction region to constant
    # value (pr0 = saturated/Maxwell pressure)
    
    pr[(Vr >= Vrmin) & (Vr <= Vrmax)] = pr0
    
    return pr

def get_Vlims(type, Tr, pr0, omega):
    """Solve the inverted equation of state for reduced volume:
    return the lowest and highest reduced volumes, such that the
    reduced pressure is pr0. This function only needs to be
    called when Tr<1, i.e. below the critical temperature, when
    there will be at least three roots: for Carnahan-Starling
    quintic equations of state, we choose the third and fifth
    roots (ignoring negative or complex roots and reduced
    volumes below the minimum value for the EOS).
    """

    if type==0: # van der Waals
        eos = np.poly1d( (3*pr0, -(pr0+8*Tr), 9, -3) )
    elif type==1: # Carnahan-Starling van der Waals
        theta = 1.91653293328153028625223271960308792
        rtheta = 1.0/theta
        aa = (1.0+theta*(-16.0+theta*(64.0+theta*(256.0+theta*256.0))))/(1.0+theta*theta*(-64.0+theta*(-256.0+theta*256.0)))
        rZc = 2.0*(4.0*theta-1.0)*(4.0*theta-1.0)*(4.0*theta-1.0)*(4.0*theta-1.0)/(1.0+theta*theta*(-64.0+theta*(-256.0+theta*256.0)))
        eos = np.poly1d( ( 64.0*pr0, -(48.0*pr0*rtheta+64.0*Tr*rZc), (12.0*pr0*rtheta*rtheta-16.0*Tr*rZc*rtheta+64.0*aa), -(pr0*rtheta*rtheta*rtheta+4.0*Tr*rZc*rtheta*rtheta+48.0*aa*rtheta), (Tr*rZc*rtheta*rtheta*rtheta+12.0*aa*rtheta*rtheta), -(aa*rtheta*rtheta*rtheta)) )
    elif type==2: # Redlich-Kwong
        theta = 1.0 + (2.0**(1.0/3.0)) + (4.0**(1.0/3.0))
        rtheta = 1.0/theta
        eos = np.poly1d( ( pr0, -(3.0*Tr), (theta/(Tr**0.5)-3.0*rtheta*Tr-rtheta*rtheta*pr0), -1.0/(Tr**0.5)) )
    elif type==3: # Soave-Redlich-Kwong
        theta = 1.0 + (2.0**(1.0/3.0)) + (4.0**(1.0/3.0))
        rtheta = 1.0/theta
        sqrtTr = Tr**0.5
        m = 1.0 + (0.480 + omega*(1.574-omega*0.176)) * (1.0-sqrtTr)
        m = m*m
        eos = np.poly1d( ( pr0, -(3.0*Tr), (theta*m-3.0*rtheta*Tr-rtheta*rtheta*pr0), -m) )
    elif type==4: # Peng-Robinson
        theta = 1.0 + (4.0 - 8.0**0.5)**(1.0/3.0) + (4.0 + 8.0**0.5)**(1.0/3.0)
        sqrtTr = Tr**0.5
        m = 1.0 + (0.37464 + omega*(1.54226-omega*0.26992)) * (1.0-sqrtTr)
        m = m*m
        aa = theta*theta - 2.0*theta - 1.0
        bb = 2.0*(theta+1.0)*(theta-1.0)*(theta-1.0)
        cc = (theta*theta + 2.0*theta - 1.0)*(theta*theta + 2.0*theta - 1.0)
        eos = np.poly1d( ( (theta*theta*theta*aa*pr0), (theta*theta*aa*pr0-bb*theta*theta*Tr), -(3.0*theta*aa*pr0+2.0*bb*theta*Tr-cc*theta*m), (aa*pr0+bb*Tr-m*cc)))
    elif type==5: # Carnahan-Starling Redlich-Kwong
        theta = 3.00680414886104355791114560215239732
        rtheta = 1.0/theta
        aa = (1.0+theta*(-16.0+theta*(64.0+theta*(256.0+theta*256.0))))*(theta+1.0)*(theta+1.0)*rtheta*rtheta/(1.0+theta*theta*(-64.0+theta*(-256.0+theta*256.0)))/(Tr**0.5)
        rZc = (4.0*theta-1.0)*(4.0*theta-1.0)*(4.0*theta-1.0)*(4.0*theta-1.0)*(2.0*theta+1.0)*rtheta/(1.0+theta*theta*(-64.0+theta*(-256.0+theta*256.0)))
        eos = np.poly1d( ( 64.0*pr0, (16.0*rtheta*pr0 - 64.0*Tr*rZc), (64.0*aa + 36.0*rtheta*rtheta*pr0 - 80.0*rtheta*Tr*rZc), -(48.0*aa*rtheta+11.0*rtheta*rtheta*rtheta*pr0+20.0*rtheta*rtheta*Tr*rZc), (12.0*aa*rtheta*rtheta-rtheta*rtheta*rtheta*rtheta*pr0-3.0*rtheta*rtheta*rtheta*Tr*rZc), (rtheta*rtheta*Tr*rZc-aa*rtheta*rtheta*rtheta)) )

    roots = eos.r
    roots.sort()
    if type==1 or type==5:
        Vrmin, Vrmax = np.real(roots[2]), np.real(roots[4])
    else:
        Vrmin, Vrmax = roots[0], roots[2]
    return Vrmin, Vrmax


def get_area_difference(Vr0, type, Tr, omega):
    """Return the difference in areas of the 'van der Waals' loops: the
    difference between the areas of the loops from Vr0 to Vrmax and from
    Vrmin to Vo, where the reduced pressure from the equation of state
    is the same at Vrmin, Vr0 and Vrmax. This difference will be zero
    when the straight line joining Vrmin and Vrmax at pr0 corresponds
    to the Maxwell construction.
    """

    pr0 = eos(type, Tr, Vr0, omega)
    Vrmin, Vrmax = get_Vlims(type, Tr, pr0, omega)
    return quad(lambda vr: eos(type, Tr, vr, omega) - pr0, Vrmin, Vrmax, limit=subint)[0]


def plot_pV(type,Tr,omega):
    c = next(palette)
    ax.plot(Vr, eos(type, Tr, Vr, omega), lw=2, alpha=0.3, color=c)
    ax.plot(Vr, eos_maxwell(type, Tr, Vr, omega), lw=2, color=c, label='$T_r = {:.4f}$'.format(Tr))

# main program starts here

# read in command-line options and values
args = docopt(__doc__)
vdW = args["--vdW"]
CSvdW = args["--CSvdW"]
RK = args["--RK"]
SRK = args["--SRK"]
PR = args["--PR"]
CSRK = args["--CSRK"]
Tr = eval(args["--Tr"]) if args["--Tr"] != None else 0.0
T = eval(args["--T"])  if args["--T"] != None else 0.0
a = eval(args["--a"])
b = eval(args["--b"])
R = eval(args["--R"])
omega = float(args["--omega"])
maxVr = float(args["--maxVr"])
subint = int(args["--subint"])

if vdW:
    eostype = 0
    print("Using van der Waals cubic equation of state")
elif CSvdW:
    eostype = 1
    print("Using Carnahan-Starling van der Waals hard-sphere (quintic) equation of state")
elif RK:
    eostype = 2
    print("Using Redlich-Kwong cubic equation of state")
elif SRK:
    eostype = 3
    print("Using Soave-Redlich-Kwong cubic equation of state")
elif PR:
    eostype = 4
    print("Using Peng-Robinson cubic equation of state")
elif CSRK:
    eostype = 5
    print("Using Carnahan-Starling Redlich-Kwong hard-sphere (quintic) equation of state")

# check available parameters when specifying temperature
# instead of reduced temperature: if available, find
# critical properties based on EOS and calculate required
# reduced temperature

if T>0.0 and (a==0.0 or b==0.0):
    sys.exit("Insufficient parameters supplied to calculate critical properties")

Tc = pc = Vc = 0.0
if a>0.0 and b>0.0 and R>0.0:
    if vdW:
        Tc = 8.0*a/(27.0*b*R)
        pc = a/(27.0*b*b)
        Vc = 3.0*b
    elif CSvdW:
        theta = 1.91653293328153028625223271960308792
        Tc = 2.0*a*((4.0*theta-1.0)**4)/(b*R*theta*(1.0+theta*(-16.0+theta*(64.0+theta*(256.0+theta*256.0)))))
        pc = a*(1.0+theta*(-64.0+theta*(-256.0+theta*256.0)))/(b*b*theta*theta*(1.0+theta*(-16.0+theta*(64.0+theta*(256.0+theta*256.0)))))
        Vc = theta*b
    elif RK:
        theta = 1.0 + (2.0**(1.0/3.0)) + (4.0**(1.0/3.0))
        Tc = (a*(2.0*theta+1.0)*(theta-1.0)*(theta-1.0)/(b*R*theta*theta*(theta+1.0)*(theta+1.0)))**(2.0/3.0)
        pc = a/(b*b*theta*theta*theta*Tc**0.5)
        Vc = theta*b
    elif SRK:
        theta = 1.0 + (2.0**(1.0/3.0)) + (4.0**(1.0/3.0))
        Tc = a*(2.0*theta+1.0)*(theta-1.0)*(theta-1.0)/(b*R*theta*theta*(theta+1.0)*(theta+1.0))
        pc = a/(b*b*theta*theta*theta)
        Vc = theta*b
    elif PR:
        theta = 1.0 + (4.0 - 8.0**0.5)**(1.0/3.0) + (4.0 + 8.0**0.5)**(1.0/3.0)
        Tc = 2.0*a*(theta+1.0)*(theta-1.0)*(theta-1.0)/(b*R*(theta*theta+2.0*theta-1.0)*(theta*theta+2.0*theta-1.0))
        pc = a*(theta*theta-2.0*theta-1)/(b*b*(theta*theta+2.0*theta-1.0)*(theta*theta+2.0*theta-1.0))
        Vc = theta*b
    elif CSRK:
        theta = 3.00680414886104355791114560215239732
        Tc = (a*(2.0*theta+1.0)*((4.0*theta-1.0)**4)/(b*R*(theta+1.0)*(theta+1.0)*(1.0+theta*(-16.0+theta*(64.0+theta*(256.0+theta*256.0))))))**(2.0/3.0)
        pc = a*(-9.0+theta*(-64.0+theta*(-320.0+theta*(-256.0+theta*256.0))))/(b*b*(theta+1.0)*(theta+1.0)*(1.0+theta*(-16.0+theta*(64.0+theta*(256.0+theta*256.0))))*Tc**0.5)
        Vc = theta*b
    if T>0.0:
        Tr = T/Tc
    else:
        T = Tr*Tc
    print("Derived critical properties from parameters:\nTc = {0:f}\npc = {1:f}\nVc = {2:f}\nrho_c = {3:f}".format(Tc, pc, Vc, 1.0/Vc))

# set range of reduced volumes to search for saturated (Maxwell) pressure
# (minimum based on limit for equation of state and avoiding division-by-zero)

if vdW:
    Vrlow = 1.0/3.0 + 0.001
elif RK or SRK:
    Vrlow = 1.0/(1.0 + (2.0**(1.0/3.0)) + (4.0**(1.0/3.0))) + 0.001
elif PR:
    Vrlow = 1.0/(1.0 + (4.0 - 8.0**0.5)**(1.0/3.0) + (4.0 + 8.0**0.5)**(1.0/3.0)) + 0.001
elif CSvdW:
    Vrlow = 0.25/1.91653293328153028625223271960308792 + 0.001
elif CSRK:
    Vrlow = 0.25/3.00680414886104355791114560215239732 + 0.001

numVr = int(100.0*maxVr+0.5)
Vr = np.linspace(Vrlow, maxVr, numVr)

# first find actual values of pressure and density to print to screen
# (converting to real units if available)

pr = eos(eostype, Tr, Vr, omega)
minpr = min(pr)
minpr = 1.2*minpr if minpr<0.0 else 0.8*minpr
maxpr = max(-minpr, 2.0)

if Tr>=1:
    print("Reduced temperature at or above 1: no coexistence available")
    Vrmax = 15
else:
    iprmin = argrelextrema(pr, np.less)
    iprmax = argrelextrema(pr, np.greater)
    Vr0 = np.mean([Vr[iprmin], Vr[iprmax]])
    pr0 = eos(eostype, Tr, Vr0, omega)
    Vrmin, Vrmax = get_Vlims(eostype, Tr, pr0, omega)
    Vr0 = newton(get_area_difference, Vr0, args=(eostype, Tr, omega))
    pr0 = eos(eostype, Tr, Vr0, omega)
    Vrmin, Vrmax = get_Vlims(eostype, Tr, pr0, omega)
    if pc>0.0:
        print("Temperature T = {0:f} (Tr = {1:f})".format(T, Tr))
        print("Maxwell (saturated) pressure P = {0:f} (Pr = {1:f})".format(pr0*pc, pr0))
        print("Vapour specific volume V_v = {0:f} (V_r,v = {1:f})".format(Vrmax*Vc,Vrmax))
        print("Liquid specific volume V_l = {0:f} (V_r,l = {1:f})".format(Vrmin*Vc,Vrmin))
        print("Vapour density rho_v = {0:f} (rho_r,v = {1:f})".format(1.0/Vrmax/Vc, 1.0/Vrmax))
        print("Liquid density rho_l = {0:f} (rho_r,l = {1:f})".format(1.0/Vrmin/Vc, 1.0/Vrmin))
    else:
        print("Reduced temperature Tr = {0:f}".format(Tr))
        print("Reduced Maxwell (saturated) pressure Pr = {0:f}".format(pr0))
        print("Reduced vapour specific volume Vr_v = {0:f}".format(Vrmax))
        print("Reduced liquid specific volume Vr_l = {0:f}".format(Vrmin))
        print("Reduced vapour density rho_r,v = {0:f}".format(1.0/Vrmax))
        print("Reduced liquid density rho_r,l = {0:f}".format(1.0/Vrmin))

# now plot corresponding isotherm, using lighter line for
# original curve for equation of state and darker line
# for equation of state with Maxwell construction

fig, ax = plt.subplots()

plot_pV(eostype,Tr,omega)

fig.canvas.manager.set_window_title('Maxwell construction')
ax.set_title('$P_r = {0:f}, V_r = {1:f}, {2:f}$'.format(pr0, Vrmin, Vrmax))
ax.set_xlim(0.1, min(maxVr, Vrmax*1.2))
ax.set_xlabel('Reduced volume')
ax.set_ylim(minpr, maxpr)
ax.set_ylabel('Reduced pressure')
ax.legend()

plt.show()

