import numpy as np
from tqdm import tqdm

from src.field import Field
from src.system import System
from src.explicitTerms import Term
from src.fourierfunc import *


def main():

     # Define the grid size.
    grid_size = (100, 100)
    dr=(1.0, 1.0)
    kappa=1.0
    eta=1.0
    k_list, k_grids = momentum_grids(grid_size, dr)
    fourier_operators = k_power_array(k_grids)
    # Initialize the system.
    system = System(grid_size, fourier_operators)

    # Create fields
    system.create_field('phi', k_list, k_grids, dynamic=True)
    system.create_field('mu', k_list, k_grids, dynamic=False)
    system.create_field('iqxphi', k_list, k_grids, dynamic=False)
    system.create_field('iqyphi', k_list, k_grids, dynamic=False)
    system.create_field('vx', k_list, k_grids, dynamic=False)
    system.create_field('vy', k_list, k_grids, dynamic=False)
    system.create_field('sigxx', k_list, k_grids, dynamic=False)
    system.create_field('sigxy', k_list, k_grids, dynamic=False)

    # Initial Conditions
    system.get_field('phi').set_real(0.1*(np.random.rand(*grid_size)-1))
    system.get_field('phi').synchronize_momentum()
    
    
    system.create_term("phi", [("mu", 1)], [-1, 1, 0, 0, 0])
    system.create_term("iqxphi", [("phi", 1)], [1, 0, 1, 0, 0])
    system.create_term("iqyphi", [("phi", 1)], [1, 0, 0, 1, 0])
    system.create_term("phi", [("vx", 1), ("iqxphi", 1)], [-1, 0, 0, 0, 0])
    system.create_term("phi", [("vy", 1), ("iqyphi", 1)], [-1, 0, 0, 0, 0])

    system.create_term("mu", [("phi", 1)], [-1, 0, 0, 0, 0])
    system.create_term("mu", [("phi", 3)], [1, 0, 0, 0, 0])
    system.create_term("mu", [("phi", 1)], [1, 1, 0, 0, 0])

    system.create_term("sigxx", [("iqxphi", 2)], [-kappa/2, 0, 0, 0, 0])
    system.create_term("sigxx", [("iqyphi", 2)], [kappa/2, 0, 0, 0, 0])
    system.create_term("sigxy", [("iqxphi", 1), ("iqyphi", 1)], [-kappa, 0, 0, 0, 0])

    system.create_term("vx", [("sigxx", 1)], [1/eta, 0, 1, 0, 2])
    system.create_term("vx", [("sigxx", 1)], [1/eta, 0, 3, 0, 4])
    system.create_term("vx", [("sigxx", 1)], [-1/eta, 0, 1, 2, 4])
    system.create_term("vx", [("sigxy", 1)], [1/eta, 0, 0, 1, 2])
    system.create_term("vx", [("sigxy", 1)], [2/eta, 0, 2, 1, 4])
 
    system.create_term("vy", [("sigxx", 1)], [1/eta, 0, 2, 1, 4])
    system.create_term("vy", [("sigxx", 1)], [-1/eta, 0, 0, 1, 2])
    system.create_term("vy", [("sigxx", 1)], [-1/eta, 0, 0, 3, 4])
    system.create_term("vy", [("sigxy", 1)], [1/eta, 0, 1, 0, 2])
    system.create_term("vy", [("sigxy", 1)], [2/eta, 0, 1, 2, 4])
    
    phi = system.get_field('phi')

    for t in tqdm(range(40000)):
        # system.update('phi', 0.01)
        # system.update('mu', 0.01)
        # phi.dealias_field()
        # phi.synchronize_real()
        # mu.dealias_field()
        # mu.synchronize_real()

        system.update_system(0.01)


        if t%1000 == 0:
            np.savetxt('data/phi.csv.'+ str(t), phi.get_real(), delimiter=',')


    

def momentum_grids(grid_size, dr):

    k_list = [np.fft.fftfreq(grid_size[i], d=dr[i])*2*np.pi for i in range(len(grid_size))]
    # k is now a list of arrays, each corresponding to k values along one dimension.

    k_grids = np.meshgrid(*k_list, indexing='ij')
    # k_grids is now a list of 2D or 3D (etc.) arrays, each corresponding to k values in one dimension.

    return k_list, k_grids

def k_power_array(k_grids):
    k_squared = sum(ki**2 for ki in k_grids)
    k_abs = np.sqrt(k_squared)
    inv_kAbs = np.divide(1.0, k_abs, where=k_abs!=0)

    k_power_arrays = [k_squared, 1j*k_grids[0], 1j*k_grids[1], inv_kAbs]

    return k_power_arrays


if __name__=="__main__":
    main()
