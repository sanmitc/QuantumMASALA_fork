import numpy as np
from qtm.crystal import Crystal
from qtm.pseudo.nloc import NonlocGenerator
from qtm.constants import RYDBERG_HART, RY_KBAR
from qtm.config import NDArray
from qtm.dft import DFTCommMod
from qtm.mpi import scatter_slice
from time import perf_counter


TOL=1e-5

def stress_nonloc(dftcomm:DFTCommMod,
                  numbnd:int,
                  wavefun:tuple,
                   cryst: Crystal,
                   nloc_dij_vkb:list
                   ) -> NDArray:
    assert isinstance(numbnd, int)
    with dftcomm.kgrp_intra as comm:
        if dftcomm.pwgrp_intra is None: band_slice=scatter_slice(numbnd, comm.size, comm.rank)
        else: band_slice=np.arange(numbnd)

    ##Getting the characteristics of the crystal
    omega=cryst.reallat.cellvol
    l_atoms = cryst.l_atoms
    num_typ=len(l_atoms)
    tot_atom = np.sum([sp.numatoms for sp in l_atoms])
    labels=np.repeat(np.arange(len(l_atoms)), [sp.numatoms for sp in l_atoms])
    atom_label= np.concatenate([np.arange(sp.numatoms) for sp in l_atoms])

    ##Initializing the stress tensor
    stress_nl=np.zeros((3,3)).astype(np.complex128)

    ##Looping over the wavefunctions
    for ityp in range(num_typ):
        sp=l_atoms[ityp]
        k_counter=0
        #atom_label_sp=atom_label[inum]
        for wfn in wavefun:
            for k_wfn in wfn:
                ## Getting the evc and gk space characteristics form the wavefunction
                evc=k_wfn.evc_gk[band_slice]
                evc_data=evc.data.T
                gkspace=k_wfn.gkspc
                k_weight=k_wfn.k_weight
                gkcart=gkspace.gk_cart.T
                gknorm=gkspace.gk_norm

                occ_num=k_wfn.occ[band_slice] ##Getting the occupation numbers

                ## Getting the non-local beta projectors and dij matrices from the wavefun
                start_time = perf_counter() 
                k_nonloc= NonlocGenerator(sp=sp,
                                          gwfn=gkspace.gwfn)
                vkb, dij=nloc_dij_vkb[k_counter][ityp]
                dj_vkb, dy_vkb = k_nonloc.gen_vkb_dij_deriv(k_wfn.gkspc)
                vkb=vkb.data
                dj_vkb=dj_vkb.data
                dy_vkbx, dy_vkby, dy_vkbz=dy_vkb
                dy_vkbx=dy_vkbx.data
                dy_vkby=dy_vkby.data
                dy_vkbz=dy_vkbz.data
                dy_vkb=np.array([dy_vkbx, dy_vkby, dy_vkbz])
                
                if dftcomm.image_comm.rank==0: 
                    print("Time taken for non local generator in stress: ", perf_counter() - start_time)
                    print(flush=True)

                start_time = perf_counter()
                #row_vkb=int(vkb_full.data.shape[0]/sp.numatoms)
                #vkb=vkb_full.data[atom_label_sp*row_vkb:(atom_label_sp+1)*row_vkb]
                #dj_vkb=djvkb_full.data[atom_label_sp*row_vkb:(atom_label_sp+1)*row_vkb]
                
                dij_sp=dij/RYDBERG_HART
                '''dy_vkbx=dy_vkbx_full.data[atom_label_sp*row_vkb:(atom_label_sp+1)*row_vkb]
                dy_vkby=dy_vkby_full.data[atom_label_sp*row_vkb:(atom_label_sp+1)*row_vkb]
                dy_vkbz=dy_vkbz_full.data[atom_label_sp*row_vkb:(atom_label_sp+1)*row_vkb]
                dy_vkb=np.array([dy_vkbx, dy_vkby, dy_vkbz])
                dij_sp=dij[atom_label_sp*row_vkb:(atom_label_sp+1)*row_vkb, atom_label_sp*row_vkb:(atom_label_sp+1)*row_vkb]/RYDBERG_HART'''
                if dftcomm.image_comm.rank==0: 
                    print("Time taken for slicing in stress: ", perf_counter() - start_time)
                    print(flush=True)

                ## Calculation of the Diagonal Terms
                start_time = perf_counter()
                betaPsi=np.conj(vkb)@evc_data
                if dftcomm.pwgrp_intra is not None: 
                    betaPsi=dftcomm.pwgrp_intra.allreduce(betaPsi)
                abs2_betaPsi=np.abs(betaPsi)**2*occ_num
                quant=np.sum(dij_sp@abs2_betaPsi)
                quant*=k_weight
                diag_stress=(np.eye(3)*quant).astype(np.complex128)
                stress_nl+=diag_stress
                if dftcomm.image_comm.rank==0: 
                    print("Time taken for diagonal terms in stress: ", perf_counter() - start_time)
                    print(flush=True)
                
                ##The derivative of Spherical Bessel function  
                start_time = perf_counter()
                betaPsi_d=dij_sp@betaPsi           
                beta_dj=dj_vkb.T@betaPsi_d   #(shape is G,numbnd)
                Sigma_j_nl=2*np.real(np.conj(evc_data)*beta_dj)*occ_num
                Sigma_j_nl=np.sum(Sigma_j_nl, axis=1)
                gknorm_nonzero=np.where(gknorm>TOL)
                gknorm_inv=np.zeros_like(gknorm)
                gknorm_inv[gknorm_nonzero]=1/gknorm[gknorm_nonzero]
                gktensor=np.einsum('ij, ik->ijk', gkcart, gkcart)
                gktensor*=((Sigma_j_nl*gknorm_inv).reshape(-1, 1, 1))
                stress_dj=np.sum(gktensor, axis=0)*k_weight
                if dftcomm.pwgrp_intra is not None:
                    stress_dj=dftcomm.pwgrp_intra.allreduce(stress_dj)
                stress_nl+=stress_dj
                if dftcomm.image_comm.rank==0: 
                    print("Time taken for derivative of spherical bessel in stress: ", perf_counter() - start_time)
                    print(flush=True)

                ##The derivative of spherical Harmnomics
                start_time = perf_counter()
                evc_occup=evc_data*occ_num
                if dftcomm.image_comm.rank==0:
                    print("time taken for adding occupation is", perf_counter()-start_time)
                    print(flush=True)
                start_time = perf_counter()
                beta_dy=np.array([betaPsi_d.T@dy_vkb[i] for i in range(3)])
                if dftcomm.image_comm.rank==0:
                    print("shape of dy_vkb is", dy_vkb.shape)
                    print("shape of betaPsi_d is", betaPsi_d.shape)
                    print("shape of beta_dy is", beta_dy.shape)
                    print("time taken for derivative of beta_dy is", perf_counter()-start_time)
                    print(flush=True)
                start_time = perf_counter()
                mult=np.conj(evc_occup.T)
                Sigma_y_nl=beta_dy*mult
                if dftcomm.image_comm.rank==0:
                    print("shape of Sigma_y_nl is", Sigma_y_nl.shape)
                    print("shape of beta_dy is", beta_dy.shape)
                    print("shape of mult is", mult.shape)
                    print("time taken for Sigma_y_nl is", perf_counter()-start_time)
                    print(flush=True)
                start_time = perf_counter()
                Sigma_y_diagonal = np.array([np.sum(Sigma_y_nl[i], axis=0) for i in range(Sigma_y_nl.shape[0])])
                if dftcomm.image_comm.rank==0:
                    print("shape of Sigma_y_diagonal is", Sigma_y_diagonal.shape)
                    print("time taken for Sigma_y_diagonal is", perf_counter()-start_time)
                    print(flush=True)
                start_time = perf_counter()
                stress_dy=2*np.real(Sigma_y_diagonal@gkcart)*k_weight
                if dftcomm.image_comm.rank==0:
                    print("time taken for stress_dy is", perf_counter()-start_time)
                    print(flush=True)
                stress_dy[0,1]=stress_dy[1,0]
                stress_dy[0,2]=stress_dy[2,0]
                stress_dy[1,2]=stress_dy[2,1]
                if dftcomm.pwgrp_intra is not None:
                    stress_dy=dftcomm.pwgrp_intra.allreduce(stress_dy)
                stress_nl+=stress_dy
                if dftcomm.image_comm.rank==0: 
                    print("Time taken for derivative of spherical harmonics in stress: ", perf_counter() - start_time)
                    print(flush=True)

    start_time=perf_counter()
    stress_nl/=omega
    stress_nl=np.real(stress_nl)
    stress_nl=cryst.symm.symmetrize_matrix(stress_nl)
    if dftcomm.image_comm.rank==0: 
        print("Time taken for symmetrization in stress: ", perf_counter() - start_time)
        print(flush=True)

    start_time=perf_counter()
    '''with dftcomm.image_comm as comm:
        comm.Allreduce(comm.IN_PLACE, stress_nl)'''
    '''if dftcomm.image_comm.rank==0: 
        print("Time taken for reduction in stress: ", perf_counter() - start_time)
        print(flush=True)'''
    return stress_nl*RY_KBAR
        

    