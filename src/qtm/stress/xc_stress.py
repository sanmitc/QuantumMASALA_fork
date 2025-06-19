import numpy as np

from qtm.pot import xc
from qtm.crystal import Crystal
from qtm.gspace import GSpace
from qtm.containers.field import FieldGType, get_FieldG
from qtm.pseudo.loc import loc_generate_pot_rhocore
from qtm.constants import RY_KBAR, RYDBERG
from qtm.config import NDArray

##LIBXC functional is set to None
def stress_xc(cryst:Crystal,
              gspc:GSpace,
              rho:FieldGType,
              xc_compute:tuple
              )->NDArray:
    ## This routine calculates the stress caused by the exchange-correlation potential
    #setting up the characteristics of the crystal
    omega=cryst.reallat.cellvol
    l_atoms = cryst.l_atoms
    libxc_func = xc.get_libxc_func(cryst)
    #print(libxc_func)
    
    v_xc, en_xc, GGA_stress = xc_compute
    rho_r=rho.to_r()
    diag_xc_r=np.real(np.sum(v_xc._data*np.conj(rho_r._data))*gspc.reallat_dv)
    diag_xc=(diag_xc_r-en_xc)
    stress_xc=np.eye(3)*diag_xc
    stress_xc/=omega
    stress_xc+=np.real(GGA_stress)/(RYDBERG*gspc.size_r)
    stress_xc/=RYDBERG
    return stress_xc*RY_KBAR








