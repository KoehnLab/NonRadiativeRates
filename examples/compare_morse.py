#!/usr/bin/env python3

import numpy as np
import scipy.special as scsp

from nonradiative_rates import fcf_utils as FcfUtils
from nonradiative_rates.constants import au2rcm, amu, a02AA

# use these values:
omega = 3000
xi = 0.02
D = omega/(4*xi)
mred = 10
mredau = mred*amu

Dau = D/au2rcm
omau = omega/au2rcm

aau = np.sqrt(2.*omau*xi*mredau)

a = aau/a02AA   # a is an inverse distance

dlt = -0.01
dltau = dlt/a02AA

# Huang Rhys:
dQ = np.sqrt(mredau*omau)*dltau
S = 0.5*dQ*dQ

print(f"Settings: omega={omega} cm-1   xi={xi}   delta={dlt} AA")
print(f"  -->     D={D} cm-1   a={a} AA^-1")

print(f"HR factor S={S}")

print(f"args for morse: {Dau} {aau} {mredau} {dltau}")
m_system = FcfUtils.FcfMorse0(Dau,aau,mredau,dltau)

print(f"max. quanta: {m_system.max_n()}")

max_n = m_system.max_n()

print(max_n)

for ii in range(max_n+1):
    eh = omega * (ii+0.5)
    ea = eh - omega*xi*(ii+0.5)**2
    inta = m_system.get_I0n(ii)
    fcfa = inta**2
    fcfh = np.exp(-S)*S**ii/scsp.gamma(ii+1)
    print(f" {ii:3}  {eh:12.2f}  {fcfh:12.4e}    {fcfa:12.4e}    {ea:12.2f}")
