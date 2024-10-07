#!/usr/bin/env python3

import numpy as np
import read_turbomole as rtm

au2rcm = 219474.63068  # cm-1 / E_h
amu = 1./5.485799090441e-4 # me

mdata = rtm.turbomole_results("testmolecule2")

cvect = mdata.get_couplingvector(True)
Mvect = mdata.get_sqrtMvector(True)
coord,elem,natoms = mdata.get_coords()

grad,coord_grad = mdata.get_gradient(flat=True)

print(cvect)
print(Mvect)
print(grad)

mdata2 = rtm.turbomole_results("testmolecule2a")
coord2,elem2,natoms2 = mdata2.get_coords()

test = (np.array(coord)-np.array(coord2))**2
rmsd = np.sum(test)/len(test)
print(f"RMSD of coordinates read from coord and coord2: {rmsd}")

freqs,Lmat,redmass = mdata2.get_hessian()

# mass weighting of derivative coupling
cvect_w = cvect/Mvect * (1./np.sqrt(amu))

grad_w = grad/Mvect * (1./np.sqrt(amu))

cvect_p = cvect_w.T @ Lmat

grad_p = grad_w.T @ Lmat

# freq. weighting

cvect_pf = []
dsp = []

for cvp,grd,frq in zip(cvect_p,grad_p,freqs):
    if frq < 1.:
        cvpf = 0.
        dspf = 0.
    else:
        frqau = frq/au2rcm
        cvpf = cvp/np.sqrt(frqau)
        dspf = np.sqrt(frqau)*grd / frqau**2
    cvect_pf.append(frq*cvpf)
    dsp.append(dspf)

cvect_pf = np.array(cvect_pf)
dsp = np.array(dsp)

norm = np.linalg.norm(cvect_pf)

osci_list = []
    
for ii in range(3*natoms):
    if freqs[ii] < 1e-3:
        continue
    osci_list.append([freqs[ii],0.5*dsp[ii]*dsp[ii],dsp[ii],redmass[ii]])

for ii in range(3*natoms):
    if freqs[ii] < 1.:
        continue
    
    HR = 0.5*dsp[ii]**2

    D00 = 0.5*(HR**2)*np.exp(-HR)/HR
    D01 = 0.5*(1-HR)**2*np.exp(-HR)

    print(f" {freqs[ii]:8.2f}  {cvect_pf[ii]:12.6f}   {HR:12.6f}  {D00:12.6f}  {D01:12.6f}  ")

print( f"\nNorm of coupling vector: {norm:.2f}\n" )
