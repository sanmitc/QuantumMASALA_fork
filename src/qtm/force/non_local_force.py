import numpy as np
import gc

from qtm.crystal import Crystal
from qtm.constants import RYDBERG_HART
from qtm.config import NDArray
from qtm.dft import DFTCommMod
from qtm.mpi import scatter_slice

def force_nonloc(dftcomm:DFTCommMod,
                 numbnd: int,
                 wavefun: tuple,
                 crystal: Crystal,
                 nloc_dij_vkb:list
                 ) -> NDArray:
    """Calculate the nonlocal force on the atoma from the upf potential
    
    Parameters
    ---------
    numbnd: Number of Bands. It is used for parallelization and also processing the band wise pseudopotential data.
    
    wavefun: A tuple object containing the wavefunction for each k+G vector. If the calculation is not spin-polarized, 
    then the tuple will containe one wavefunction for each k+G vector. Otherwise, there will be two such objects.

    crystal: A crystal object having all the necessary qualities of a crystal like lattice vectors and information of the atomic basis. 

    nloc_dij_vkb: These are the list of some non local projector operators that are directly computed from the scf loop.

    Returns
    ---------
    Nx3 numpy array where N is the number of atoms, representing the forces.


    A brief overview of what is done.
    ---------------------

    Before starting the wavefunction loops, the basic crystal information and machinery of parallelization are being set up.

    In the wavefunction loop we can see one loop of the :math:'\mathbf{k}+\mathbf{G}' vectors, another for spin up and down(for spin polarized calculation)


    The non local force is given by the following formula, math:: 
    F= -i \Omega_{at} \sum_{i, \mathbf{k}, \mathbf{G}, \mathbf{G}'}(\mathbf{G-\mathbf{G'})\psi_i^*(\mathbf{k}+\mathbf{G})\psi_i(\mathbf{k}+\mathbf{G}')(\sum_l U_{ps,l, \mathbf{k}+\mathbf{G}, \mathbf{k}+\mathbf{G}'})


    Now, the psedopotential term can be written as, math::
    \sum_l U_{ps,l, \mathbf{k}+\mathbf{G}, \mathbf{k}+\mathbf{G}'}= \sum_{p,j} \beta_p d_{pj} \beta_j^*

    Now the :math: 'd_{pj}' are the dij_sp and the :math:'\beta' matrices are the vkb which are indexed here.

    After that the above force equation is implemented through the following aalgebraic simplification:
    math::
    A_p(\mathbf{k})=\sum_{\mathbf{G}} (\mathbf{k}+\mathbf{G})\psi_i^*(\mathbf{k}+\mathbf{G})\beta_p
    B_p(\mathbf{k})=\sum_{\mathbf{G}} \psi_i^*(\mathbf{k}+\mathbf{G})\beta_p
    F=2\Omega_{at}\sum_{\mathbf{k},i}(\mathbf{p,j} \text{Imag}(A_p(\mathbf{k})d_{pj}B_j^*(\mathbf{k)))

    The A term is the GBetaPsi and the B term is the BetaPsi indexed in the code. The wavefunction :math:'\psi' is the quantity evc that has been extracted from the wavefunction tuple.
    """
                   

    ##Starting of the parallelization over bands
    assert isinstance(numbnd, int)

    ##No G space parallelization
    if dftcomm.pwgrp_intra==None:
        with dftcomm.kgrp_intra as comm:
            band_slice=scatter_slice(numbnd, comm.size, comm.rank)
    else:
        band_slice=np.arange(numbnd)

    ##Getting the characteristics of the crystal
    l_atoms = crystal.l_atoms
    num_typ=len(l_atoms)
    tot_atom = np.sum([sp.numatoms for sp in l_atoms])

    ##Initializing the force array 
    force_nl=np.zeros((tot_atom, 3))
    k_counter=0
    for wfn in wavefun:
        for k_wfn in wfn:
            ## Getting the evc and gk-space characteristics form the wavefunctio
            evc=k_wfn.evc_gk[band_slice]
            gkspace=k_wfn.gkspc
            k_weight=k_wfn.k_weight
            gkcart=gkspace.gk_cart.T
            dij_vkb=nloc_dij_vkb[k_counter]

            ##Getting the occupation number
            occ_num=k_wfn.occ[band_slice]
            atom_counter=0
            ## Getting the non-local beta projectors and dij matrices from the wavefun
            ##It is being calculated in the root and then broadcasted over to other processors in same k group according to band parallelization
            for ityp in range(num_typ):
                sp=l_atoms[ityp]
                vkb, dij = dij_vkb[ityp]
                row_vkb=int(vkb.data.shape[0]/sp.numatoms)
                dij_sp=np.concatenate([dij[i*row_vkb:(i+1)*row_vkb, i*row_vkb:(i+1)*row_vkb] for i in range(sp.numatoms)], axis=0).T
                dij_sp=dij_sp.reshape(-1, sp.numatoms, row_vkb)
                dij_sp=dij_sp.swapaxes(0,1)
                vkb=vkb.data

                ##Constructing the G\beta\psi
                gkcart_struc=gkcart.reshape(-1,1,3)
                evc_sp=evc.data.T
                Kc=gkcart_struc*evc_sp[:,:,None]
                GbetaPsi=np.einsum("ij, jkl->ikl", vkb, np.conj(Kc))
                GbetaPsi=dftcomm.image_comm.allreduce(GbetaPsi)
                ##Constructing the \beta\psi
                betaPsi=np.conj(vkb)@(evc_sp*occ_num.reshape(1,-1))
                betaPsi=betaPsi.reshape(sp.numatoms, row_vkb, -1)
                betaPsi=dij_sp@betaPsi
                betaPsi=dftcomm.image_comm.allreduce(betaPsi)
                betaPsi=betaPsi.reshape(betaPsi.shape[0]*betaPsi.shape[1], -1)
                ##Multiplying Together
                V_NL_Psi=GbetaPsi*betaPsi.reshape(*betaPsi.shape, 1)
                V_NL_Psi=V_NL_Psi.reshape(sp.numatoms, row_vkb, V_NL_Psi.shape[1], V_NL_Psi.shape[2])
                V_NL_Psi=np.sum(V_NL_Psi, axis=(1,2))
                trace=-2*np.imag(V_NL_Psi)
                ##Multiply by Weight
                trace = trace * k_weight
                force_nl[atom_counter:atom_counter+sp.numatoms]+=trace 
                atom_counter+=sp.numatoms
                del vkb, dij, dij_sp
            del evc, gkspace, k_weight, dij_vkb
        k_counter+=1
    force_nl/=RYDBERG_HART
    force_nl=dftcomm.image_comm.allreduce(force_nl)
    force_nl/=dftcomm.image_comm.size
    del nloc_dij_vkb, wavefun, crystal, dftcomm
    gc.collect()
    return force_nl
