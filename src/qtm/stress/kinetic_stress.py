import numpy as np
from qtm.dft import kswfn
from qtm.crystal import Crystal
from qtm.constants import ELECTRON_RYD, RY_KBAR
from qtm.dft import DFTCommMod
from qtm.mpi import scatter_slice

__all__=["kinetic_stress"]

def stress_kinetic(dftcomm:DFTCommMod, 
                   numbnd:int, 
                   cryst:Crystal, 
                   wfn_k_group:tuple):
    with dftcomm.kgrp_intra as comm:
        if dftcomm.pwgrp_intra is None:
            band_slice = scatter_slice(numbnd, comm.size, comm.rank)
        else:
            band_slice = np.arange(numbnd)
    omega=cryst.reallat.cellvol
    me_HART=1.
    hcut_HART=1.
    kin_stress=np.zeros((3,3))
    for k_wfn in wfn_k_group:
        for wfn in k_wfn:
            gkcart=wfn.gkspc.gk_cart.T
            occ_num=wfn.occ[band_slice] #Occupation numbers
            k_weight=wfn.k_weight
            evc_gk=wfn.evc_gk[band_slice]
            quant = np.sum(np.abs(evc_gk.data) ** 2*occ_num.reshape(-1,1), axis=0)
            gk_tensor=np.einsum("ij, ik->ijk", gkcart, gkcart)  
            gk_tensor*=quant.reshape(-1,1,1)
            kin_stress_k=np.sum(gk_tensor, axis=0)*k_weight
            kin_stress+=kin_stress_k
    kin_stress*=ELECTRON_RYD**2*hcut_HART**2/me_HART/omega
    '''if dftcomm.kgrp_intra is not None:
        kin_stress = dftcomm.kgrp_intra.allreduce(kin_stress)'''
    #kin_stress=cryst.symm.symmetrize_matrix(kin_stress)
    if dftcomm.image_comm is not None:
        #print("kinetic stress in", dftcomm.image_comm.rank, "=", kin_stress)
        kin_stress = dftcomm.image_comm.allreduce(kin_stress)
    return kin_stress*RY_KBAR