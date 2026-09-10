import numpy as np
import scipy.special as scsp

from . import _fastfold as fastfold
from .constants import kBcm

class FcfMorse0:
    """
    this class stores the info needed for computing the Franck-Condon factors of
    two displaced Morse oscillators
    Here, we assume identical parameters for both oscillators and only compute <0|n> integrals

    positive delta means that the oscillator for which we vary the quanta has a longer eq. dist.

    all values should be provided in atomic units
    """
    def __init__(self,D,a,mred,dlt,dbg=False):  # re not needed
        self.D1 = D
        self.alph1 = a
        self.D2 = D
        self.alph2 = a
        self.dlt = dlt
        self.mred = mred
        self.maxll = 10
        self.dbg = dbg
        # etc.

        # compute the basic constants for the Matsumoto/Iwamoto formula
        self.a1 = np.sqrt(2.*self.mred*self.D1)/self.alph1
        self.a2 = np.sqrt(2.*self.mred*self.D2)/self.alph2
        self.alpha = 0.5*(self.alph1+self.alph2)
        
        self.p0 = self.a1 - 0.5
        self.q0 = self.a2 - 0.5
        self.ss = self.alph1/self.alpha
        self.tt = self.alph2/self.alpha
        self.hh = 2.*np.sqrt(self.a1*self.a2)*np.exp((self.alph1-self.alph2)*(self.dlt)*0.25)
        self.h1 = 2*self.a1*np.exp(-self.alph1*self.dlt*0.5)/(self.hh**self.ss)
        self.h2 = 2*self.a2*np.exp(self.alph2*self.dlt*0.5)/(self.hh**self.tt)

        if self.dbg:
            print("Initial settings:")
            print(f"a1: {self.a1} {self.D1} {self.alph1} {self.mred}")
            print(f"a2: {self.a2}")
            print(f"p0: {self.p0}")
            print(f"p0: {self.q0}")


    def F(self,x,U,V,W,s,t):
        """ 
        the approximate expansion of Matsumoto and Iwamoto; a more complete one is given
        by the same authors in
            J. Quant. Spectrosc. Radiat. Transf. 52, 901–913 (1994).
        """
        
        UVW = U + s*V + t*W
        UVW2 = U + s**2*V + t**2*W
        UVW3 = U + s**3*V + t**3*W
        UVW4 = U + s**4*V + t**4*W
        c1 = UVW**2 + UVW2
        c2 = UVW**3 + 3*UVW*UVW2 + UVW3
        c3 = UVW**4 + 6*UVW**2*UVW2 + 7*UVW*UVW3 + UVW4 
        # paper 1
        c3+= -3*s*t*V*W*(s-t)**2 - 3*U* ( (1-s)**2*s*V + (1-t)**2*t*W )
        # paper 2 - identical
        #c3+= -3*s*t*V*W*(s-t)**2 - 3*s*U*V*(1-s)**2 - 3*t*U*W*(1-t)**2
        
        val = 1. + 0.5*c1 * scsp.polygamma(1,x) + c2/6 * scsp.polygamma(2,x) + c3/24 * (3.*scsp.polygamma(1,x)**2 + scsp.polygamma(3,x))

        return val        


    def max_n(self):
        return int(np.floor(self.q0))


    def get_I0n_MI(self,nn):
        """
        get the I(0,n) integrals based on the expansion given by
        
         Matsumoto, A. & Iwamoto, K., J. Quant. Spectrosc. Radiat. Transf. 50, 103–109 (1993)
   
        The expansion is more general (different anharm. and freqs. allowed), but begins to fail for 
        quanta larger than roughly n=8 (see also "F" subroutine)

        """

        if self.dbg:
            print("calculating integral for n = ",nn)

        qn = self.q0 - nn
        # too high quantum number, return none
        if qn <= 0:
            return None

        prefac  = 2/self.alpha * np.sqrt(self.alph1*self.alph2*self.p0*qn)
        if self.dbg:
            print(prefac)
        prefac *= np.sqrt(scsp.gamma(2*qn+nn+1) /(scsp.gamma(2*self.p0+1) * scsp.gamma(nn+1) ))  # gamma for n!
        if self.dbg:
            print(prefac)
        prefac *= scsp.gamma(self.ss*self.p0 + self.tt*qn)/scsp.gamma(2*qn+1)
        if self.dbg:
            print(prefac)
        prefac *= self.h1**self.p0 * self.h2**qn

        if self.dbg:
            print(prefac)

        lfac = 0.
        for ll in range(nn+1):
            l1  = scsp.poch(-nn,ll)*self.h2**ll/(scsp.poch(2*qn+1,ll)*scsp.gamma(ll+1)) # gamma for ll!
            l2  = scsp.gamma(self.ss*self.p0+self.tt*qn+self.tt*ll)/scsp.gamma(self.ss*self.p0+self.tt*qn)
            # get the result of the digamma function 
            arg = self.ss*self.p0 + self.tt*qn + self.tt*ll
            dg = scsp.digamma(arg)
            U = np.exp(dg)
            V = -0.5*self.h1*np.exp(self.ss*dg)
            W = -0.5*self.h2*np.exp(self.tt*dg)

            l3 = np.exp(U+V+W)

            l4 = self.F(arg,U,V,W,self.ss,self.tt)

            lcontr = l1*l2*l3*l4

            lfac += lcontr

            if self.dbg:
                print(f"{ll:3} {l1:16} {l2:16} {l3:16} {l4:16} -->  {lcontr}  sum: {lfac}")

        In0 = prefac*lfac

        return In0
    
    def get_I0n_ori(self,nn):
        """
        Get the I(0,n) Franck-Condon integral of two Morse oscillators
        Restricted to equivalent oscillators (but shifted)
        The expression is as accurate as the underlying implementation of the Gamma and hypergeometric
        functions in scipy are

        Original and debugged expression; if the number of levels becomes to large, it fails as
        it contains several calls to the Gamma function with values on the order of the max. quantum

        The expression is taken from:
          Valiev, R. R. et al., Phys. Chem. Chem. Phys. 25, 6406–6415 (2023). 
        """

        if self.dbg:
            print("calculating integral for n = ",nn)

        if self.p0 != self.q0:
            print("This routine is explicitly only for equiv. Morse osc.")
            raise Exception("not defined")

        # change to notation of that work for convenience
        an = 2.*(self.q0 - nn)
        a0 = 2.*(self.p0)
        alpha = self.alpha
        Delta = np.exp(-alpha*self.dlt)

        # too high quantum number, return none
        if an <= 0:
            return None        
        
        N0 = np.sqrt(alpha*a0/scsp.gamma(a0+1.))
        Nn = np.sqrt(alpha*an*scsp.gamma(nn+1.)/scsp.gamma(an+nn+1))

        A = 0.5*Delta+0.5
        B = 0.5*(a0+an)-1.
        C = an

        #Inf1 = scsp.gamma(1.+B)*scsp.gamma(C+nn+1)/(scsp.gamma(nn+1)*scsp.gamma(C+1.))
        # avoid overflow:
        Inf1a = scsp.gamma(1.+B)/scsp.gamma(nn+1)
        Inf1b = scsp.gamma(C+nn+1)/scsp.gamma(C+1.)

        if self.dbg:
            print(f"Inf1a,Inf1b,B,C: {Inf1a} {Inf1b} {B} {C}")

        Inf1 = Inf1a*Inf1b

        Inf2 = A**(-1-B)
        Inf3 = scsp.hyp2f1(1.+B,-nn,1.+C,1/A)

        if self.dbg:
            print(f"In0 = {N0}*{Nn}/{alpha}*{Delta**(0.5*a0)}*{Inf1}*{Inf2}*{Inf3}")

        In0 = N0*Nn/alpha*Delta**(0.5*a0)*Inf1*Inf2*Inf3

        return In0

    def get_I0n(self,nn):
        """
        Get the I(0,n) Franck-Condon integral of two Morse oscillators
        Restricted to equivalent oscillators (but shifted)
        The expression is as accurate as the underlying implementation of the Gamma and hypergeometric
        functions in scipy are

        Ff the number of levels becomes to large, it fails as it contains several calls 
        to the Gamma function with values on the order of the max. quantum;
        currently no modification to avoid this (use this function to try and get_I0n_ori to verify)

        The expression is taken from:
          Valiev, R. R. et al., Phys. Chem. Chem. Phys. 25, 6406–6415 (2023). 
        """

        if self.dbg:
            print("calculating integral for n = ",nn)

        if self.p0 != self.q0:
            print("This routine is explicitly only for equiv. Morse osc.")
            raise Exception("not defined")

        # change to notation of that work for convenience
        an = 2.*(self.q0 - nn)
        a0 = 2.*(self.p0)
        alpha = self.alpha
        Delta = np.exp(-alpha*self.dlt)

        # too high quantum number, return none
        if an <= 0:
            return None        
        
        N0 = np.sqrt(alpha*a0/scsp.gamma(a0+1.))
        Nn = np.sqrt(alpha*an*scsp.gamma(nn+1.)/scsp.gamma(an+nn+1))

        if self.dbg:
            print(f"N0, Nn, alpha, nn, a0, an: {N0}, {Nn}, {alpha}, {nn}, {a0}, {an}")

        A = 0.5*Delta+0.5
        B = 0.5*(a0+an)-1.
        C = an

        #Inf1 = scsp.gamma(1.+B)*scsp.gamma(C+nn+1)/(scsp.gamma(nn+1)*scsp.gamma(C+1.))
        # avoid overflow:
        Inf1a = scsp.gamma(1.+B)/scsp.gamma(nn+1)
        Inf1b = scsp.gamma(C+nn+1.)/scsp.gamma(C+1.)

        if self.dbg:
            print(f"Inf1a, Inf1b, B, C: {Inf1a} {Inf1b} {B} {C}")

        Inf1 = Inf1a*Inf1b

        Inf2 = A**(-1-B)
        Inf3 = scsp.hyp2f1(1.+B,-nn,1.+C,1/A)

        if self.dbg:
            print(f"In0 = {N0}*{Nn}/{alpha}*{Delta**(0.5*a0)}*{Inf1}*{Inf2}*{Inf3}")

        In0 = N0*Nn/alpha*Delta**(0.5*a0)*Inf1*Inf2*Inf3

        return In0

    def get_b0n(self, nn, h=1e-6):
        """
        Get the b_j Franck–Condon related quantity for two equivalent Morse oscillators.
        Uses numerical derivative dI_n(A,B,C)/dB via four-point central finite differences.

        b_j = (K + ln(2 beta)/alpha) * g_j
              - (1/alpha^2) * N0 * Nn * Delta^(a0/2) * dI_n/dB
        """

        if self.dbg:
            print("calculating b_j for n = ", nn)

        if self.p0 != self.q0:
            print("This routine is explicitly only for equiv. Morse osc.")
            raise Exception("not defined")

        # --- notation consistent with get_I0n ---
        an = 2. * (self.q0 - nn)
        a0 = 2. * self.p0
        alpha = self.alpha
        Delta = np.exp(-alpha * self.dlt)

        if an <= 0:
            return None

        # normalization constants
        N0 = np.sqrt(alpha * a0 / scsp.gamma(a0 + 1.))
        Nn = np.sqrt(alpha * an * scsp.gamma(nn + 1.) / scsp.gamma(an + nn + 1.))

        # hypergeometric parameters
        A = 0.5 * (Delta + 1.)
        B = 0.5 * (a0 + an) - 1.
        C = an

        # --- compute g_j using existing routine ---
        g_j = self.get_I0n(nn)

        # --- numerical derivative dI_n/dB ---
        def In_of_B(Bval):
            Inf1a = scsp.gamma(1. + Bval) / scsp.gamma(nn + 1.)
            Inf1b = scsp.gamma(C + nn + 1.) / scsp.gamma(C + 1.)
            Inf1 = Inf1a * Inf1b

            Inf2 = A ** (-1. - Bval)
            Inf3 = scsp.hyp2f1(1. + Bval, -nn, 1. + C, 1. / A)

            return Inf1 * Inf2 * Inf3

        # --- four-point central difference ---
        dIn_dB = (
            -In_of_B(B + 2.0 * h)
            + 8.0 * In_of_B(B + h)
            - 8.0 * In_of_B(B - h)
            + In_of_B(B - 2.0 * h)
        ) / (12.0 * h)

        # --- physical parameters ---
        K = self.dlt                         # displacement
        beta = (1. / alpha) * np.sqrt(2. * self.D1)

        # --- assemble b_j ---
        bj = ((K + np.log(2. * beta) / alpha) * g_j
              - (1. / alpha**2)
              * N0 * Nn * Delta**(0.5 * a0)
              * dIn_dB)

        if self.dbg:
            print(f"b_j = {bj}")

        return bj

    
    
class FCWD:

    def __init__(self,mode_list,lam_class=0.,T_sim=300.,debug=0,use_cython=True):

        self.mode_list = mode_list
        self.lam_class = lam_class
        self.T_sim = T_sim
        self.dbg = debug
        self.use_cython = use_cython


    def init_Gauss(self,nbins,dnu,nu0,sigma):
        """
        compute a Gaussian on a grid, centered at nu0 and width sigma
        """

        # we assume that bin 1 is zero, so nu0 has to be shifted to match needs
        xvals = np.linspace(0.,dnu*(nbins-1),nbins)

        gauss = 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5*((xvals-nu0)/sigma)**2)

        return gauss


    def get_FCWDori(self,nu_max,dnu):
        """
        compute the FCWD on a grid
        the algorithm goes back to Stein and Rabinowicz (citation to be found)
        and a suggestion by Robert Send (PhD thesis, Karlsruhe 2010)
        """

        # broadening by a classical distr.?
        if self.lam_class > 0.:
            sigma = np.sqrt(2.*self.lam_class*self.T_sim*kBcm)
            if self.dbg > 0:
                print(f"classical distr with sigma = {sigma} cm-1")
        else:
            sigma = 0.

        nu_extra = 6*sigma  # at 6 sigma the Gaussian has decayed to <1e-6

        nbins = int(np.ceil((nu_max+nu_extra)/dnu))

        fcwd = np.zeros((nbins))

        if self.lam_class > 0:
            fcwd = self.init_Gauss(nbins,dnu,nu_extra+self.lam_class,sigma)
        else:
            fcwd[0] = 1.

        # loop over modes
        for fcf_list in self.mode_list:

            # put integrals on sparse list according to graining
            values = []
            mult = []

            for fcf in fcf_list:
                values.append(fcf[0])
                mult.append(int(np.floor(fcf[1]/dnu)))
                # may add screening here
                # should check that values on mult are not identical (too large grid)

            nmult = len(mult)

            scr   = np.zeros((nmult))
            fcwdn = np.zeros((nbins))

            # multiply the values on the sparse list onto the current FCWD
            for ii in range(nbins):

                scr[0:nmult] = 0.
                kk = nmult
                jjmin = 0
                jjmax = nmult-1 #0
                for jj in range(nmult):
                    kk -= 1

                    if mult[kk]>ii:
                        jjmax = kk-1
                        continue
                    if ii-mult[kk] > nbins-1:
                        continue
                    jjmin = kk
                    scr[kk] = fcwd[ii-mult[kk]]

                fcwdn[ii] = np.dot(scr[jjmin:jjmax+1],values[jjmin:jjmax+1])
                #fcwdn[ii] = 0.
                #for jj in range(jjmin,jjmax+1):
                #    fcwdn[ii] += scr[jj]*values[jj]

            fcwd = fcwdn

        return fcwd,-nu_extra
    
    def get_FCWD(self,nu_max,dnu):
        """
        compute the FCWD on a grid
        the algorithm goes back to Stein and Rabinowicz (citation to be found)
        and a suggestion by Robert Send (PhD thesis, Karlsruhe 2010)
        slightly improved version (still slow)
        """

        # broadening by a classical distr.?
        if self.lam_class > 0.:
            sigma = np.sqrt(2.*self.lam_class*self.T_sim*kBcm)
            if self.dbg > 0:
                print(f"classical distr with sigma = {sigma} cm-1")
        else:
            sigma = 0.

        nu_extra = 6*sigma  # at 6 sigma the Gaussian has decayed to <1e-6

        nbins = int(np.ceil((nu_max+nu_extra)/dnu))

        fcwd = np.zeros((nbins))

        if self.lam_class > 0:
            fcwd = self.init_Gauss(nbins,dnu,nu_extra+self.lam_class,sigma)
            if self.dbg > 0:
                test = np.sum(fcwd)*dnu
                print(f"Sum over Gaussian: {test}   first value: {fcwd[0]} ")
        else:
            fcwd[0] = 1.

        # loop over modes
        for fcf_list in self.mode_list:

            if self.use_cython:
                fcwd = fastfold.fold(fcwd,fcf_list,dnu)

            else:
                # put integrals on sparse list according to graining
                values = []
                mult = []
                multscr = []

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

                fcwd = fcwdn

        return fcwd,-nu_extra
