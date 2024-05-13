import numpy as np
from .explicitTerms import Term
from .fourierfunc import *
class Field:
    def __init__(self, grid_size, k_list, k_grids, is_dynamic, dealias_factor):
        """
        Initialize the field.

        Parameters:
        grid_size -- the size of the grid for this field
        is_dynamic -- a boolean indicating whether this field is dynamic
        """
        self.grid_size = grid_size
        self.is_dynamic = is_dynamic

        self.dealias_factor = dealias_factor

        self.dealias = np.ones_like(k_grids[0], dtype=bool)
        for i, ki in enumerate(k_grids):
            kmax_dealias_i = np.max(np.abs(k_list[i])) * dealias_factor
            self.dealias &= (np.abs(ki) < kmax_dealias_i) 


        self.data_real = np.zeros(grid_size)
        self.data_momentum = np.zeros(grid_size, dtype=np.complex)
        
        self.terms = []

    def add_term(self, term):
        """
        Add a term to the update rule for this field.
        Parameters:
        term -- a Term object
        """
        if self.is_dynamic:
            self.terms.append(term)
        else:
            self.terms.append(term)


    def get_terms(self):
        """
        Get the terms of the update rule for this field.

        Returns:
        terms -- a list of Term objects
        """
        return self.terms

    def get_momentum(self):
        """
        Get the momentum data of this field.

        Returns:
        data_momentum -- a numpy array with the momentum data
        """
        return self.data_momentum
    
    def get_real(self):
        """
        Get the real data of this field.

        Returns:
        data_real -- a numpy array with the real space data
        """
        return self.data_real
    
    def set_momentum(self, value):
        """
        Add a value to the momentum data of this field.

        Parameters:
        value -- a numpy array with the same shape as data_momentum
        """
        self.data_momentum[:] = value

    def set_real(self, value):
        """
        Add a value to the momentum data of this field.

        Parameters:
        value -- a numpy array with the same shape as data_momentum
        """
        self.data_real[:] = value

    def add_to_momentum(self, value):
        """
        Add a value to the momentum data of this field.

        Parameters:
        value -- a numpy array with the same shape as data_momentum
        """
        self.data_momentum += value


    def dealias_field(self):
        self.data_momentum *= self.dealias

    def synchronize_real(self):
        self.data_real = to_real_space(self.data_momentum)

    def synchronize_momentum(self):
        self.data_momentum = to_momentum_space(self.data_real)


        
