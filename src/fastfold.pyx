import numpy as np

DYTPE=np.float64

def fold(fcwd,fcf_list,float dnu):

    # put integrals on sparse list according to graining
    values = []
    mult = []
    multscr = []

    cdef Py_ssize_t nbins = len(fcwd)

    for fcf in fcf_list:
        values.append(fcf[0])
        mult.append(int(np.floor(fcf[1]/dnu)))
        multscr.append(0)
        # may add screening here
        # should check that values on mult are not identical (too large grid)

    cdef Py_ssize_t nmult = len(mult)

    mult = np.array(mult)
    multscr = np.array(multscr,dtype=np.int64)

    cdef long[:] mult_view = mult
    cdef long[:] multscr_view = multscr

    scr   = np.zeros((nmult))
    fcwdn = np.zeros((nbins))
    values = np.array(values)

    cdef double[:] fcwd_view = fcwd
    cdef double[:] fcwdn_view = fcwdn
    cdef double[:] scr_view = scr
    cdef double[:] values_view = values

    cdef Py_ssize_t ii, jj, kk, jjmax

    jjmax=0
    # multiply the values on the sparse list onto the current FCWD
    for ii in range(nbins):

        kk = jjmax
        for jj in range(jjmax+1,nmult):
            if mult_view[jj] <= ii:
                kk+=1
            else:
                break
        jjmax = kk

        for jj in range(nmult):
            multscr_view[jj] = ii-mult_view[jj]

        for jj in range(jjmax+1):
            scr_view[jj] = fcwd_view[multscr_view[jj]]
        #scr = fcwd[multscr[0:jjmax+1]]

        #fcwdn_view[ii] = np.dot(scr[0:jjmax+1],values[0:jjmax+1])
        fcwdn_view[ii] = 0.
        for jj in range(jjmax+1):
            fcwdn_view[ii] += scr_view[jj]*values_view[jj]
        

    return fcwdn
