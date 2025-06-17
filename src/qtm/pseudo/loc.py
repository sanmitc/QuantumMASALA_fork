from __future__ import annotations

__all__ = ["loc_generate_rhoatomic", "loc_generate_pot_rhocore", 'loc_generate_rhoatomic_interpolate', "loc_generate_dpot"]
import numpy as np
from scipy.special import erf

from qtm.crystal.basis_atoms import BasisAtoms
from qtm.gspace import GSpace
from qtm.containers import get_FieldG, FieldGType
from .upf import UPFv2Data

from qtm.constants import PI, RYDBERG, ELECTRON_RYD

from qtm.config import NDArray
from qtm.msg_format import type_mismatch_msg
from qtm.logger import qtmlogger

EPS6 = 1e-6


# def _simpson(f_r: NDArray, r_ab: NDArray):
#     f_times_dr = f_r * r_ab
#     if len(r_ab) % 2 == 0:  # If even # of points <-> odd # of interval
#         f_times_dr = f_times_dr[:-1]  # Discard last point for now
#     f_times_dr[:] *= 1. / 3
#     f_times_dr[..., 1:-1:2] *= 4
#     f_times_dr[..., 2:-1:2] *= 2
#     integral = np.sum(f_times_dr, axis=-1)
#     # Dealing with last interval when odd # of intervals
#     if len(r_ab) % 2 == 0:
#         integral += (-0.5 * f_times_dr[-2] + 4 * f_times_dr[-1])
#         integral += 2.5 * (f_r[-1] * r_ab[-1] / 3.)
#     return integral


def _simpson_old(f, rab):
    r12 = 1 / 3
    f_r = f * rab
    start, stop, step = 0, rab.shape[0] - 2, 2
    return r12 * np.sum(
        f_r[start:stop:step]
        + 4 * f_r[start + 1 : stop + 1 : step]
        + f_r[start + 2 : stop + 2 : step]
    )


def _sph2pw(r: NDArray, r_ab: NDArray, f_times_r2: NDArray, g: NDArray):  # type: ignore
    numg = g.shape[0]
    f_g = np.empty((*f_times_r2.shape[:-1], numg), dtype="c16", like=f_times_r2)
    numr = r.shape[0]

    r_ab = r_ab.copy()
    # Simpson's integration
    r_ab *= 1.0 / 3
    r_ab[1:-1:2] *= 4
    r_ab[2:-1:2] *= 2
    f_g[:] = np.sinc(g * r[0] / PI) * f_times_r2[..., 0] * r_ab[0]

    g = g.reshape(-1, 1)
    f_times_r2 = np.expand_dims(f_times_r2, axis=-2)
    for idxr in range(numr):
        f_g[:] += np.sum(
            np.sinc(g * r[idxr] / PI) * f_times_r2[..., idxr] * r_ab[idxr], axis=-1
        )
    return f_g

def _dsph2pw(r: NDArray, r_ab: NDArray, f_times_r2: NDArray, g: NDArray):
    numg = g.shape[0]
    f_g = np.empty((*f_times_r2.shape[:-1], numg), dtype='c16')
    numr = r.shape[0]

    def dspherical(g,r):
        return np.cos(g*r)/g - np.sin(g*r)/(g**2*r)

    r_ab = r_ab.copy()
    r_ab *= 1. / 3
    r_ab[1:-1:2] *= 4
    r_ab[2:-1:2] *= 2
    f_g[:] = dspherical(g, r[1]) * f_times_r2[..., 1] * r_ab[1]
    g = g.reshape(-1, 1)
    f_times_r2 = np.expand_dims(f_times_r2, axis=-2)
    for idxr in range(2, numr):
        f_g[:] += np.sum(dspherical(g, r[idxr])
                         * f_times_r2[..., idxr] * r_ab[idxr],
                   axis=-1)
    return f_g

def _check_args(sp: BasisAtoms, grho: GSpace):
    if sp.ppdata is None:
        raise ValueError(
            f"{BasisAtoms} instance 'sp' does not have "
            f"pseudopotential data i.e 'sp.ppdata' is None."
        )
    if not isinstance(sp.ppdata, UPFv2Data):
        raise NotImplementedError("only 'UPFv2Data' supported")
    if not isinstance(grho, GSpace):
        raise TypeError(type_mismatch_msg("grho", grho, GSpace))


@qtmlogger.time("rho_generate_atomic")
def loc_generate_rhoatomic(sp: BasisAtoms, grho: GSpace) -> FieldGType:
    """Computes the electron density constructed by superposing atomic charges
    of given atomic species in crystal.

    Parameters
    ----------
    sp : BasisAtoms
        Repesents an atomic species (and corresponding atoms)
        present in the crystal.
    grho : GSpace
        G-Space of the potential/charge density.

    Returns
    -------
    rho_atomic : FieldG
        Atomic Charge density generated from given species.
    """
    _check_args(sp, grho)

    upfdata: UPFv2Data = sp.ppdata
    # Radial Mesh specified in Pseudopotential Data
    g_cryst = grho.g_cryst
    g_norm = grho.g_norm
    FieldG: type[FieldGType] = get_FieldG(grho)

    # Vectorized version
    # struct_fac = FieldG(
    #     np.sum(np.exp((-2 * np.pi * 1j) * (sp.r_cryst.T @ g_cryst)), axis=0)
    # )

    # Looped version
    struct_fac = FieldG(np.zeros(g_cryst.shape[1], dtype=np.complex128))
    for i in range(sp.r_cryst.shape[1]):
        struct_fac += np.exp(-2j * np.pi * (sp.r_cryst[:, i] @ g_cryst))

    r = np.asarray(upfdata.r, like=g_cryst)
    r_ab = np.asarray(upfdata.r_ab, like=g_cryst)
    rhoatom = np.asarray(upfdata.rhoatom, like=g_cryst)

    rho = FieldG.empty(None)

    f_times_r2 = np.empty((1, len(r)), dtype="f8", like=g_cryst)
    f_times_r2[0] = rhoatom

    rho.data[:] = _sph2pw(r, r_ab, f_times_r2, g_norm[:])
    if grho.has_g0:
        rho.data[0] = _simpson_old(rhoatom, r_ab)

    rho *= struct_fac / grho.reallat_dv
    return rho

dg=1E-2
#@qtmlogger.time('loc_generate_atomic_interpolate"')
def loc_generate_rhoatomic_interpolate(sp: BasisAtoms, grho:GSpace) -> FieldGType:
    """It is similar to that of loc_generate_rhoatomic but it creates the Fourier components
    of the atomic charges in a dense G grid and then interpolates it through Lagrange interpolation
    
    Parameters
    ----------
    sp : BasisAtoms
        Repesents an atomic species (and corresponding atoms)
        present in the crystal.
    grho : GSpace
        G-Space of the potential/charge density.

    Returns
    -------
    rho_atomic : FieldG
        Atomic Charge density generated from given species.
    """

    _check_args(sp, grho)

    upfdata: UPFv2Data = sp.ppdata
    # Radial Mesh specified in Pseudopotential Data
    g_cryst = grho.g_cryst
    g_max = np.sqrt(grho.ecut*2)
    num_gfine= int(g_max/dg)+5
    print("the value of g_max is", g_max)
    print("the number of g fine is", num_gfine)
    g_fine = np.linspace(0, g_max, num_gfine)
    FieldG: type[FieldGType] = get_FieldG(grho)

    struct_fac = FieldG(
        np.sum(np.exp((-2 * np.pi * 1j) * (sp.r_cryst.T @ g_cryst)), axis=0)
    )

    r = np.asarray(upfdata.r, like=g_cryst)
    r_ab = np.asarray(upfdata.r_ab, like=g_cryst)
    rhoatom = np.asarray(upfdata.rhoatom, like=g_cryst)

    rho = FieldG.empty(None)
    rho_fine = np.zeros_like(g_fine)

    f_times_r2 = np.empty((1, len(r)), dtype="f8", like=g_cryst)
    f_times_r2[0] = rhoatom

    rho_fine[:] = _sph2pw(r, r_ab, f_times_r2, g_fine[:])
    if grho.has_g0:
        rho_fine[0] = _simpson_old(rhoatom, r_ab)

    ##Now we start the lagrange interpolation
    g_norm=grho.g_norm
    idx_g=np.trunc(g_norm/dg).astype(int)
    np.set_printoptions(threshold=np.inf)
    #print("the value of idx_g is", idx_g)
    print("the maximum value of g_norm is", np.max(g_norm), np.max(idx_g))
    print("the length of g_norm is", len(g_norm))
    with open("idx_g.txt", 'w') as f:
        f.write(f"the maximum value of g_norm is {np.unique(g_norm/dg)}\n")
        f.write(f"the value of idx_g is {np.unique(np.sort(idx_g))}\n")
    np.savetxt("gspc_qm.txt", np.unique(np.trunc(g_norm/dg/1e-8)*1e-8), fmt='%s')

    xmin0=g_norm/dg-idx_g
    xmin1=xmin0-1
    xmin2=xmin0-2
    xmin3=xmin0-3

    rho.data[:]= rho_fine[idx_g+0]*xmin1*xmin2*xmin3/((0-1)*(0-2)*(0-3)) + \
                rho_fine[idx_g+1]*xmin0*xmin2*xmin3/((1-0)*(1-2)*(1-3)) + \
                rho_fine[idx_g+2]*xmin0*xmin1*xmin3/((2-0)*(2-1)*(2-3)) + \
                rho_fine[idx_g+3]*xmin0*xmin1*xmin2/((3-0)*(3-1)*(3-2))
    
    with open('interpolation_data.txt', 'w') as f:
        p=np.unique(np.trunc(g_norm/dg/1e-8)*1e-8)
        idx_g_print=np.trunc(p).astype(int)
        xmin0_print=p-idx_g_print
        xmin1_print=xmin0_print-1
        xmin2_print=xmin0_print-2
        xmin3_print=xmin0_print-3
        f.write(f"value of g_norm is {p}\n")
        f.write(f"value of idx_g is {idx_g_print}\n")
        f.write(f"value of xmin0 is {xmin0_print}\n")
        f.write(f"value of xmin1 is {xmin1_print}\n")
        f.write(f"value of xmin2 is {xmin2_print}\n")
        f.write(f"value of xmin3 is {xmin3_print}\n")
        f.write(f"the value of rho_fine[idx_g+0] is {rho_fine[idx_g_print+0]/grho.reallat_cellvol}\n")
        f.write(f"the value of rho_fine[idx_g+1] is {rho_fine[idx_g_print+1]/grho.reallat_cellvol}\n")
        f.write(f"the value of rho_fine[idx_g+2] is {rho_fine[idx_g_print+2]/grho.reallat_cellvol}\n")
        f.write(f"the value of rho_fine[idx_g+3] is {rho_fine[idx_g_print+3]/grho.reallat_cellvol}\n")
        np.savetxt("i0_qm.txt", idx_g_print, fmt='%s')
        np.savetxt("rho_i0_qm.txt", rho_fine[idx_g_print+0]/grho.reallat_cellvol, fmt='%s')

    rho/=grho.reallat_cellvol
    return rho

@qtmlogger.time("loc_generate")
def loc_generate_pot_rhocore(sp: BasisAtoms, grho: GSpace, mult_struct_fact:bool=False) -> (FieldGType, FieldGType):
    """Computes the local part of pseudopotential and core electron density
    (for NLCC in XC calculation) generated by given atomic species in crystal.

    Parameters
    ----------
    sp : BasisAtoms
        Atomic species (and corresponding atoms) in the crystal.
    grho : GSpace
        G-Space of the potential/charge density

    Returns
    -------
    v_ion : FieldG
        Local part of the pseudopotenital
    rho_core : FieldG
        Charge density of core electrons
    """
    _check_args(sp, grho)

    upfdata: UPFv2Data = sp.ppdata

    # Setting constants and aliases
    cellvol = grho.reallat_cellvol
    _4pibv = 4 * np.pi / cellvol
    _1bv = 1 / cellvol

    g_cryst = grho.g_cryst
    g_norm2 = grho.g_norm2
    g_norm = np.sqrt(g_norm2)
    FieldG: type[FieldGType] = get_FieldG(grho)

    valence = upfdata.z_valence

    sp_r_cryst = np.asarray(sp.r_cryst, like=g_cryst)

    # Vectorized version
    # struct_fac = FieldG(
    #     np.sum(np.exp((-2 * np.pi * 1j) * (sp_r_cryst.T @ g_cryst)), axis=0)
    # )

    # Looped version
    struct_fac = FieldG(np.zeros(g_cryst.shape[1], dtype=np.complex128))
    for i in range(sp_r_cryst.shape[1]):
        struct_fac += np.exp(-2j * np.pi * (sp_r_cryst[:, i] @ g_cryst))

    r = np.asarray(upfdata.r, like=g_cryst)
    r_ab = np.asarray(upfdata.r_ab, like=g_cryst)

    vloc = np.asarray(upfdata.vloc, like=g_cryst)
    if upfdata.core_correction:
        rho_atc = np.asarray(upfdata.rho_atc, like=g_cryst)
    else:
        rho_atc = None

    v_ion = FieldG.empty(None)
    rho_core = FieldG.empty(None)

    f_times_r2 = np.empty(
        (1 + upfdata.core_correction, len(r)), dtype="f8", like=g_cryst
    )
    f_times_r2[0] = (vloc * r + valence * erf(r)) * r
    if upfdata.core_correction:
        f_times_r2[1] = rho_atc * r**2

    f_g = _sph2pw(r, r_ab, f_times_r2, g_norm)
    with np.errstate(divide="ignore"):
        v_ion.data[:] = f_g[0] - valence * np.exp(-g_norm2 / 4) / g_norm2
    if grho.has_g0:
        v_ion.data[0] = _simpson_old(r * (r * vloc + valence), r_ab)

    if upfdata.core_correction:
        rho_core.data[:] = f_g[1]
        if grho.has_g0:
            rho_core.data[0] = _simpson_old(rho_atc * r**2, r_ab)
    else:
        rho_core.data[:] = 0

    N = np.prod(grho.grid_shape)
    v_ion *= _4pibv*N
    rho_core *= _4pibv *N

    if mult_struct_fact:
        v_ion*= struct_fac
        rho_core *= struct_fac

    return v_ion, rho_core


def loc_generate_dpot(sp: BasisAtoms, grho: GSpace) -> FieldGType:
    _check_args(sp, grho)

    upfdata: UPFv2Data = sp.ppdata
    if upfdata.core_correction:
        raise NotImplementedError("core correction is not implemented yet")
    # Radial Mesh specified in Pseudopotential Data
    r = upfdata.r
    r_ab = upfdata.r_ab

    # Setting constants and aliases
    cellvol = grho.reallat_cellvol
    _4pibv = 4 * np.pi / cellvol
    _1bv = 1 / cellvol

    g_norm2 = grho.g_norm2[grho.idxsort]
    g_norm=grho.g_norm[grho.idxsort]
    FieldG: type[FieldGType] = get_FieldG(grho)

    valence = upfdata.z_valence

    vloc = upfdata.vloc
    dv_ion = FieldG.empty(None)

    f_times_r2 = np.empty((1, len(r)), dtype='f8')
    f_times_r2[0] = (vloc * r + valence* erf(r))*r
    f_times_r2[0]/=RYDBERG

    f_g = _dsph2pw(r, r_ab, f_times_r2, g_norm)
    with np.errstate(divide='ignore'):
        dv_ion.data[:] = 0.5*f_g[0]/g_norm + valence * ELECTRON_RYD**2 * np.exp(-g_norm2 / 4) * (1+g_norm2/4)/ g_norm2**2
    if grho.has_g0:
        dv_ion.data[0] = 0

    dv_ion.data[:]*=_4pibv
    return dv_ion
