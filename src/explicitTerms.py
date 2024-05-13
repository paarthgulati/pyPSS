class Term:
    def __init__(self, fields, exponents):
        """
        Initialize a new term.

        Parameters:
        fields -- a list of tuples where each tuple contains the name of a field and its power
        exponents -- a list of exponents for the spatial derivatives
        """
        self.fields = fields  # List of tuples with field name and power
        self.exponents = exponents  # Exponents for spatial derivatives

