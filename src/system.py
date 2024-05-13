from .field import Field
import numpy as np
from .fourierfunc import *
from.explicitTerms import Term

class System:
    def __init__(self, grid_size, fourier_operators):
        """
        Initialize the system without any fields, but with a specified grid size.
        Parameters:
        grid_size -- the size of the grid, given as (num_rows, num_columns)
        """
        self.grid_size = grid_size
        self.fields = {}
        self.fourier_operators = fourier_operators 


    def create_field(self, field_name, k_list, k_grids, dynamic=False, dealias_factor=2/3):
        """
        Create a new field.
        Parameters:
        field_name -- the name of the field
        dynamic -- boolean indicating if the field is dynamic
        """
        if field_name in self.fields:
            raise ValueError(f"A field with name {field_name} already exists.")
        self.fields[field_name] = Field(self.grid_size, k_list, k_grids, dynamic, dealias_factor)

    def get_field(self, field_name):
        """
        Get the field object with the given name.
        Parameters:
        field_name -- the name of the field
        Returns:
        The Field object.
        """
        return self.fields[field_name]
    

    def create_term(self, field_name, fields, exponents):
        """
        Create a term in the update rule for the field with the given name.
        Parameters:
        field_name -- the name of the field
        fields -- a list of field names
        powers -- a list of powers for the spatial derivatives in Fourier space
        """
        assert field_name in self.fields, f"No field named '{field_name}' in the system."
        term = Term(fields, exponents)
        self.fields[field_name].add_term(term)
    

    def update_field(self, field_name, dt):
        """
        Update the field with the given name based on its terms.
        Parameters:
        field_name -- the name of the field
        """
        assert field_name in self.fields, f"No field named '{field_name}' in the system."
        field = self.fields[field_name]

        rhs_hat_total = np.zeros(field.data_momentum.shape, dtype=np.complex)            
        for term in field.get_terms():
            rhs = np.ones(field.data_real.shape)  
            for term_field_name, power in term.fields:
                rhs_field = self.get_field(term_field_name)
                rhs *= np.power(rhs_field.get_real(), power)

            rhs_hat = to_momentum_space(rhs)
            for i, fourier_operator in enumerate(self.fourier_operators):
                rhs_hat *= np.power(fourier_operator, term.exponents[i+1])

            rhs_hat *= term.exponents[0]
            
            rhs_hat_total += rhs_hat            
            
        if field.is_dynamic:
            field.add_to_momentum(dt*rhs_hat_total)
        else:
            field.set_momentum(rhs_hat_total)

    
    def update_system(self, dt):

        for field_name in self.fields:
            self.update_field(field_name, dt)

        for field_name in self.fields:
            self.fields[field_name].dealias_field()
            self.fields[field_name].synchronize_real()

            
            

