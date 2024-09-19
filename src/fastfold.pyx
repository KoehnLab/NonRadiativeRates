import numpy as np

def fold(fcwd,fcf_list,dnu):

    # put integrals on sparse list according to graining
    values = []
    mult = []
    multscr = []

    nbins = len(fcwd)

    for fcf in fcf_list:
        values.append(fcf[0])
        mult.append(int(np.floor(fcf[1]/dnu)))
        multscr.append(0)
        # may add screening here
        # should check that values on mult are not identical (too large grid)

    nmult = len(mult)

    mult = np.array(mult)
    multscr = np.array(multscr)

    scr   = np.zeros((nmult))
    fcwdn = np.zeros((nbins))

    jjmax=0
    # multiply the values on the sparse list onto the current FCWD
    for ii in range(nbins):

        kk = jjmax
        for jj in range(jjmax+1,nmult):
            if mult[jj] <= ii:
                kk+=1
            else:
                break
        jjmax = kk

        multscr = -mult
        multscr += ii
        scr = fcwd[multscr[0:jjmax+1]]

        fcwdn[ii] = np.dot(scr[0:jjmax+1],values[0:jjmax+1])
        

    return fcwdn
