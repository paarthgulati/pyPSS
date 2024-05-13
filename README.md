# pyPSS
Pseudo spectral solver source code written in python allowing to easily solve arbitrary set of continuum field equations (first-order in time) in $d$ dimension with periodic boundary conditions.
This is very useful to write fast easy solvers with no access to GPUs. For large/long simulations however, or to span phase space of parameters finely, it may be beneficial to switch to cuPSS solvers, which would parallelized. 
