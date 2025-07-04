import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse

def print_and_plot_rixs_cuts(h5_filename, ex_cuts=None, em_cuts=None):
    with h5py.File(h5_filename, 'r') as f:
        E_ex = f['E_EX'][:]
        E_em = f['E_EM'][:]
        rixs_map = f['SIGMA_TOTAL'][:]

    if ex_cuts is not None:
        for E in ex_cuts:
            idx = np.abs(E_ex - E).argmin()
            profile = rixs_map[idx, :]  # emission profile at fixed incident energy

            plt.figure()
            plt.plot(E_em, profile)
            plt.xlabel('Emission Energy (eV)')
            plt.ylabel('Intensity (arb.)')
            plt.title(f'Emission profile at Incident Energy = {E_ex[idx]:.2f} eV')
            plt.grid(True)
            plt.show()

    if em_cuts is not None:
        for E in em_cuts:
            idx = np.abs(E_em - E).argmin()
            profile = rixs_map[:, idx]  # incident profile at fixed emission energy

            plt.figure()
            plt.plot(E_ex, profile)
            plt.xlabel('Incident Energy (eV)')
            plt.ylabel('Intensity (arb.)')
            plt.title(f'Incident profile at Emission Energy = {E_em[idx]:.2f} eV')
            plt.grid(True)
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot RIXS cuts from .h5 file')
    parser.add_argument('--h5', type=str, required=True, help='Path to RIXS map .h5 file')
    parser.add_argument('--ex-cuts', type=float, nargs='*', help='List of incident energy cuts (eV)')
    parser.add_argument('--em-cuts', type=float, nargs='*', help='List of emission energy cuts (eV)')

    args = parser.parse_args()

    print_and_plot_rixs_cuts(args.h5, ex_cuts=args.ex_cuts, em_cuts=args.em_cuts)
