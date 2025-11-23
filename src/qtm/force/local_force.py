import numpy as np
import gc

from qtm.crystal import Crystal
from qtm.gspace import GSpace
from qtm.containers.field import FieldGType
from qtm.pseudo.loc import loc_generate_pot_rhocore
from qtm.constants import RYDBERG,PI
from qtm.dft import DFTCommMod

from time import perf_counter

##RYDBERG=1/2
###### The energy scale is in RYDbergs all the potential are half than Quantum Espresso.
def force_local(dftcomm: DFTCommMod,
                cryst: Crystal, 
                gspc: GSpace, 
                rho:FieldGType,
                vloc:list,
                gamma_only:bool=False):

    """
    This routine calculates the force caused by the local pseudopotential. 

    Input parameters
    -----------------------------
    dftcomm: The parallelization routine. It has the information about the MPI process. 
    
    crystal: This is a class object having information about the crystal structure of the material.
    
    gspc: The class object having information about the reciprocal lattice vectors or G-vectors.
    
    rho: It contains the fourier components of the electron densities in the reciprocal lattice space basis.
    
    vloc: It contains the Fourier compoenents of the local pseudopotential in the basais of reciprocal lattice vectors. 
    
    gamma_only: A flag that tells us whether to carry out only Gamma point calculation. 


    Output parameters
    ------------------------------
    l_force: a Nx3 array containing the forces. N is the number of atoms in the unit-cell considered. 
  
    ----------------------------------
    The formula for this is given as: math::
    \vec{F}_{loc}= -i\Sigma\sum_{\vec{G}}\vec{G}\exp(i\vec{G}.\vec{R}_\mu)U_{ps}(\vec{G})\rho(\vec{G})
  
    :math: 'U_{ps}(\vec{G})' is denoted by vloc here. It is extracted from the input sorted appropriately w.r.t to the corresponding reciprocal lattice vectors.
  
    :math: '\rho(\vec{G})' is also direcrly taken from the input. Sorting and normalizing is done later. 
  
    :math: '\vec{G}.\vec{R}_\mu' is denoted by gtau, which then is exponentiated.
  
    Later all parts are multiplied together to get the forces as indicated by the equation with proper handling of numpy matrices.
  
    In case of G-space parallelization. The forces from different processors are added up to get the total force. 
    
    """

    #Setting up characteristics of the crystal
    #start_time=perf_counter()
    l_atoms=cryst.l_atoms
    numatoms=[sp.numatoms for sp in l_atoms]
    tot_num = np.sum(numatoms)
    num_typ=len(l_atoms)
    labels=np.repeat([np.arange(num_typ)], [sp.numatoms for sp in l_atoms])
    coords_cart_all = np.concatenate([sp.r_cart for sp in l_atoms], axis=1)
    rho=rho._data[0, rho.gspc.idxsort]/np.prod(rho.gspc.grid_shape)

    #setting up G space characteristics
    idxsort = gspc.idxsort
    numg = gspc.size_g
    cart_g = gspc.g_cart[:,idxsort]
    if cart_g.ndim==3:
        cart_g=cart_g.reshape(cart_g.shape[0],cart_g.shape[-1])
    gtau = coords_cart_all.T @ cart_g
    omega=gspc.reallat_cellvol
    v_loc=np.zeros((tot_num, numg))
    for isp in range(num_typ):
        v_loc[labels==isp]=np.real(vloc[isp].data)
    v_loc=v_loc[:,idxsort]
    fact=2 if gamma_only else 1
    l_force=np.zeros((tot_num, 3))
    vrho=np.multiply(v_loc,(np.imag(np.exp(1j*gtau)*rho)/RYDBERG))
    l_force=np.real(vrho@cart_g.T*omega*fact)
    if l_force.ndim==3:
        l_force=l_force[0]
    if dftcomm.pwgrp_intra!=None: 
        l_force=dftcomm.pwgrp_intra.allreduce(l_force)
    #force_local=cryst.symm.symmetrize_vec(l_force)
    #force_local-=np.mean(force_local, axis=0)

    del v_loc, vrho, rho, labels, coords_cart_all, cryst, gspc, vloc, dftcomm
    gc.collect()
    return l_force
