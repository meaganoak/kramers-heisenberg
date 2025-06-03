import matplotlib.pyplot as plt
import sys
import numpy as np
import jax.numpy as jnp
import pandas as pd
import h5py
from molcas_suite.extractor import make_extractor as make_molcas_extractor
np.set_printoptions(precision=17)

### Unitary transformation for energies
au2ev = 2.7211386021e1
au2cm = 2.1947463e5
ev2au = (1.0 / au2ev)
c_au = 137.035999084

h5name_file = h5py.File('UO2Cl4_3d4f.rassi.h5', 'r')
soc_energies_au = h5name_file['SOS_ENERGIES'][:]
soc_energies = (soc_energies_au - soc_energies_au[0]) * au2ev

#specific for each system:
N_i = range(0, 1) #initial state 3d^10 4f^14 5f^0 [ground state only, SO State 1]
N_n = range(197, 337) #intermediate states 3d^9 4f^14 5f^1 [SO State 170 - 198]
N_f = range(169, 197) #number of final states 3d^10 4f^13 5f^1 [SO State 198 - 337]
Ei = soc_energies[N_i] # ground state energy
En = soc_energies[N_n] #intermediate state energies
Ef = soc_energies[N_f] # final state energies

#range of E_em and E_em for RIXS map - change based on the experimental data
E_em_grid = np.linspace(3300, 3400, 1000)
E_ex_grid = np.linspace(3700, 3800, 1000)

gamma_n = 5 #broadening of the intermediate state
gamma_f = 1 #broadening of the final state

edipmom_real = h5name_file['SOS_EDIPMOM_REAL']
edipmom_real_x = edipmom_real[0, :, :]
edipmom_real_y = edipmom_real[1, :, :]
edipmom_real_z = edipmom_real[2, :, :]

edipmom_imag = h5name_file['SOS_EDIPMOM_IMAG']
edipmom_imag_x = edipmom_imag[0, :, :]
edipmom_imag_y = edipmom_imag[1, :, :]
edipmom_imag_z = edipmom_imag[2, :, :]

edipmom_complex_x = edipmom_real_x + 1j * edipmom_imag_x
edipmom_complex_y = edipmom_real_y + 1j * edipmom_imag_y
edipmom_complex_z = edipmom_real_z + 1j * edipmom_imag_z

# ⟨n|Dx|i⟩ --> rows = N_n, cols = N_i
edipmom_complex_ni_x = edipmom_complex_x[np.ix_(N_n, N_i)]
edipmom_complex_ni_y = edipmom_complex_y[np.ix_(N_n, N_i)]
edipmom_complex_ni_z = edipmom_complex_z[np.ix_(N_n, N_i)]

# ⟨f|Dx|n⟩ --> rows = N_f, cols = N_n
edipmom_complex_fn_x = edipmom_complex_x[np.ix_(N_f, N_n)]
edipmom_complex_fn_y = edipmom_complex_y[np.ix_(N_f, N_n)]
edipmom_complex_fn_z = edipmom_complex_z[np.ix_(N_f, N_n)]

mu_ni = np.stack([edipmom_complex_ni_x, edipmom_complex_ni_y, edipmom_complex_ni_z], axis=0)
mu_fn = np.stack([edipmom_complex_fn_x, edipmom_complex_fn_y, edipmom_complex_fn_z], axis=0)
print(mu_fn.shape)

#print('mu_ni norm:', np.linalg.norm(mu_ni))
#print('mu_fn norm:', np.linalg.norm(mu_fn))

mu_ni_i = mu_ni[:,:,N_i].squeeze()
#print("⟨n|D|i⟩ max abs:", np.abs(mu_ni).max())
#print("⟨f|D|n⟩ max abs:", np.abs(mu_fn).max())
#print("E_ex_grid range:", E_ex_grid[0], E_ex_grid[-1])
#print("E_em_grid range:", E_em_grid[0], E_em_grid[-1])


denominator = 1.0 / ((En[None, :] - Ei - E_ex_grid[:, None]) - 1j * gamma_n / 2) #reshape En from (N_n,) to (1, N_n), same for E_ex

#print(denominator.shape)#(E_ex, En)


# Expand for broadcasting:
mu_ni_i_exp = mu_ni_i[None, :, :]            # (1, 3, N_n)
denominator_exp = denominator[:, None, :]    # (M, 1, N_n)

# Apply denominator to ⟨n|D|i⟩
mu_ni_weighted = mu_ni_i_exp * denominator_exp  # (M, 3, N_n)
print(mu_ni_weighted.shape)

#mu_fn_T = mu_fn.transpose(0, 2, 1)
#A = np.einsum('anf,man->mf', mu_fn_T, mu_ni_weighted)  # result (M, 3, N_f)
A = np.einsum('afn,mbn->mfab', mu_fn, mu_ni_weighted)  # sum over intermediate states only -> result (E_ex_grid, N_n, cart,cart)

#amplitude of the term over the intermediate states:
I = np.abs(A)**2
print(I.shape)

#second term with broadening of final state:
E_ex_ = E_ex_grid[:, None, None]   # (M,1,1)
E_em_ = E_em_grid[None, None, :]   # (1,1,L)
Ef_   = Ef[None, :, None]           # (1,N_f,1)
Ei_   = Ei[None, None, None]        # scalar broadcasted (1,1,1)

second_term = gamma_f / (((Ef_ - Ei_ - E_ex_ + E_em_)**2) + 0.25*gamma_f**2)
second_term = second_term.squeeze()
both_terms = np.einsum('ifxy,ifz->ixyz', I, second_term)  # final state summation, dims: (E_ex, cart, cart, E_em)

#setting up for total cross section:
### THINK ABOUT CONVERSION LATER#####
au_to_ev = 27.21138602
c_au = 137.035999084
a0_cm = 0.529177e-8
au_area_to_cm2 = a0_cm**2


prefactor_au = (8 * np.pi) / (9 * c_au**4)
sigma_total = prefactor_au * np.einsum('ixyz,i,z->iz', both_terms, E_ex_grid, E_em_grid**3)
sigma_total_normalized = sigma_total / sigma_total.max()

with h5py.File("rixs_map.h5", "w") as f:
    f.create_dataset("E_EX", data=E_ex_grid)
    f.create_dataset("E_EM", data=E_em_grid)
    f.create_dataset("SIGMA_TOTAL", data=sigma_total_normalized)

def plot_rixs_map_from_h5(h5_filename):
    with h5py.File(h5_filename, 'r') as f:
        E_em = f['E_EM'][:]
        E_ex = f['E_EX'][:]
        rixs_map = f['SIGMA_TOTAL'][:]

    print(f"Intensity range: min={rixs_map.min()}, max={rixs_map.max()}")

    plt.figure(figsize=(8, 6))
    vmin, vmax = np.percentile(rixs_map, [1, 99])

    pcm = plt.pcolormesh(E_ex, E_em, rixs_map.T,  # transpose so emission on y-axis
                         shading='auto', cmap='inferno', vmin=vmin, vmax=vmax)

    plt.xlabel('Incident Energy (eV)')
    plt.ylabel('Emission Energy (eV)')
    cbar = plt.colorbar(pcm)
    cbar.set_label('Intensity (a.u.)')

    plt.tight_layout()
    plt.savefig('rixs_map_UO2Cl4_RASSCF_Etrans.png')
    plt.show()

plot_rixs_map_from_h5('rixs_map.h5')
