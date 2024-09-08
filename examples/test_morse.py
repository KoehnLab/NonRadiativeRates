#!/usr/bin/env python3

import numpy as np

import FcfUtils

Eh2rcm = 219474.63068 # cm-1 / Eh
amu = 1./5.485799090441e-4 # me
a02AA = .529177249 # AA / a0

# use these values:
omega = 3000
xi = 0.002
D = omega/(4*xi)
mred = 1
mredau = mred*amu

Dau = D/Eh2rcm
omau = omega/Eh2rcm

aau = np.sqrt(2.*omau*xi*mredau)

a = aau/a02AA   # a is an inverse distance

dlt = -0.01
dltau = dlt/a02AA

print(f"Settings: omega={omega} cm-1   xi={xi}   delta={dlt} AA")
print(f"  -->     D={D} cm-1   a={a} AA^-1")

re = 2.  # has no effect

m_system = FcfUtils.FcfMorse0(Dau,aau,mredau,dltau,True)

print(f"max. quanta: {m_system.max_n()}")


max_n = 1 #m_system.max_n()

print(max_n)

for ii in range(max_n+1):
    int  = m_system.get_I0n_MI(ii)
    inta = m_system.get_I0n(ii)
    print(f"##### {ii}   {int}   {inta}")
