import numpy as np
from tqdm import tqdm
import os
import shutil
import sys

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.field import Field
from src.system import System
from src.explicitTerms import Term
from src.fourierfunc import *

class SimulationParameters:
    def __init__(self):
        self.outDir = 'data'
        self.grid_size = (128, 128)
        self.dr = (1.0, 1.0)
        self.NSteps = 5000
        self.NSave = 50
        self.dt = 0.01
        self.initPhiNoise = 0.01
    def __str__(self):
        # Generate a string representation of all attributes
        attributes = ", ".join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"SimulationParameters({attributes})"
    
def main():

    params = SimulationParameters()
    print(params)
    manage_outDir(params.outDir)

    k_list, k_grids = momentum_grids(params.grid_size, params.dr)
    fourier_operators = k_power_array(k_grids)
    # Initialize the system.
    system = System(params.grid_size, fourier_operators)

    # Create fields
    system.create_field('phi', k_list, k_grids, dynamic=True) #Time evolution for this field
    system.create_field('mu', k_list, k_grids, dynamic=False) # No time evolution. Statically defined

    # Initial Conditions
    system.get_field('phi').set_real(params.initPhiNoise * np.random.normal(loc=0, scale=1, size=params.grid_size)) # This sets up a simple uncorrelated perturbations of a homgenous state. Change this to any arbtirary initial condition
    system.get_field('phi').synchronize_momentum() ## Make sure to synchorize the initial momentum space values to the initial conditions in real space

    # define the PDEs
    system.create_term("phi", [("mu", 1)], [-1, 1]) #\partial_t \phi = - k^2 (\mu^{1})
    system.create_term("mu", [("phi", 1)], [-1, 0]) # \mu = (-1)*\phi
    system.create_term("mu", [("phi", 3)], [1, 0]) # \mu += (1)*\phi^3
    system.create_term("mu", [("phi", 1)], [1, 1]) # \mu += (1)*(k^2)*\phi

    phi = system.get_field('phi')

    if not os.path.exists(params.outDir):
        os.makedirs(params.outDir)

    for t in tqdm(range(params.NSteps)):
        system.update_system(params.dt)

        if t%params.NSave == 0:
            np.savetxt(params.outDir + '/phi.csv.'+ str(t), phi.get_real(), delimiter=',')



def momentum_grids(grid_size, dr):

    k_list = [np.fft.fftfreq(grid_size[i], d=dr[i])*2*np.pi for i in range(len(grid_size))]
    # k is now a list of arrays, each corresponding to k values along one dimension.
    k_grids = np.meshgrid(*k_list, indexing='ij')
    # k_grids is now a list of 2D or 3D (etc.) arrays, each corresponding to k values in one dimension.

    return k_list, k_grids

def k_power_array(k_grids):
    k_squared = sum(ki**2 for ki in k_grids)
    k_power_arrays = [k_squared] # For model B, all the equations are isotropic so only k^2 is needed

    return k_power_arrays

def manage_outDir(path):
    if os.path.exists(path):
        print('output directory already exists; Overwriting previous output!')
        shutil.rmtree(path)
    os.makedirs(path)



if __name__=="__main__":
    main()
