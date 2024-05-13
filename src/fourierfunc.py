from scipy.fft import fftn, ifftn

def to_momentum_space(data_real):
    """
    Convert real-space data to momentum-space data.

    Parameters:
    data_real -- an N-dimensional numpy array representing the field in real space

    Returns:
    An N-dimensional numpy array representing the field in momentum space.
    """
    return fftn(data_real)

def to_real_space(data_momentum):
    """
    Convert momentum-space data to real-space data.

    Parameters:
    data_momentum -- an N-dimensional numpy array representing the field in momentum space

    Returns:
    An N-dimensional numpy array representing the field in real space.
    """
    return ifftn(data_momentum).real
