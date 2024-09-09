#!/usr/bin/env python3

import numpy as np
import read_turbomole as rtm

mdata = rtm.turbomole_results("testmolecule2")

cvect = mdata.get_couplingvector(True)
Mvect = mdata.get_sqrtMvector(True)

print(cvect)
print(Mvect)


