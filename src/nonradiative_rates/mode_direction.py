
# determines phase for normal coordinates:
import numpy as np
from geometric.molecule import Molecule
from geometric.internal import PrimitiveInternalCoordinates, Distance

from .constants import a02AA


# function for constructing B matrix (Jacobian):
def get_Bmatrix(mol, verbose=True):

    # get all internal coordinates (using geometric package)
    ic_all = PrimitiveInternalCoordinates(mol)

    # screen for pure bond stretches:
    bond_pairs = []
    for internal in ic_all.Internals:
        if isinstance(internal,Distance):
            bond_pairs.append((internal.a,internal.b))

    # set up internals with bond stretches only:
    ic_bond = PrimitiveInternalCoordinates(mol)
    # ... overwrite ...
    ic_bond.Internals = [Distance(a,b) for a, b in bond_pairs]


    x = np.array(mol[0].xyzs).flatten()

    if verbose:
        print("\n Internal coordinates:")
        for idx,(a,b) in enumerate(bond_pairs):
            print(f"{idx:6}   {a}   {b}")

    # compute the Bmatrix for molecule 1; has dimension [natoms,ninternals]:
    Bmat = ic_bond.wilsonB(x).T

    return Bmat

# define a norm to distinguish stretching and compression:
def get_norm(vector):
    return np.sum(vector)

# define a function to retrieve the phase factors:
def get_Lmat(moldata, verbose=True):
    """
    returns the transformation matrix L with corrected phase factors
    takes a parsed turbomole_results object with Hessian information
    """

    coord,symbol,_ = moldata.get_coords()
    mass = moldata.get_masses()
    numb = moldata.get_numbers()

    # init geomeTRIC object:
    mol = Molecule()
    mol.elem = [s.capitalize() for s in symbol]
    mol.xyzs = [np.array(coord)*a02AA]

    masses = moldata.get_masses()
    freqs,Lmat,redmass = moldata.get_hessian()

    # construct B matrix:
    Bmat = get_Bmatrix(mol, verbose=verbose)

    # remove mass-weighting:
    masses_ = []
    for ms in masses:
        for idx in range(3):
            masses_.append(ms)
    Lmat_rw = np.zeros(np.shape(Lmat))
    for row_id in range(np.shape(Lmat)[0]):
        Lmat_rw[row_id,:] = (1./np.sqrt(masses_[row_id])) * Lmat[row_id,:]

    # apply B matrix:
    Lmat_rwt = Bmat.T @ Lmat_rw


    # determine whether a mode stretches or compresses bond lengths:
    """
    if a mode is associated with an overall stretching of bond lengths, i.e. the norm is increased,
    it is assumed to have a positive phase factor and a negative phase factor otherwise.
    modes that essentially do not alter bond legnths are assigned zero as prefactor, see also below.

    """
    signs = []
    for column_id in range(np.shape(Lmat)[1]):
        mode_internal = Lmat_rwt[:,column_id]
        mode_norm = get_norm(mode_internal)
        if mode_norm < 0.:
            signs.append(-1.)
        elif mode_norm > 0.:
            signs.append(1.)
        # if a mode does not change bond lengths it does not contribute to the displacement (however, typically non-zero):
        else:
            signs.append(0.)

    if verbose:
        for idx in range(len(signs)):
            print(f"{idx:>5.0f}   {signs[idx]:>5.0f}   {freqs[idx]:>5.0f}")

    Lmat_new = np.zeros(np.shape(Lmat))
    for cidx in range(np.shape(Lmat)[1]):
        Lmat_new[:,cidx] = signs[cidx] * Lmat[:, cidx]

    return Lmat_new
