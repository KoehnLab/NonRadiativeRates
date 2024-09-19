# NonRadiativeRates
Collection of python code to compute the bits and pieces needed for non-radiative rates

# Setup
Add `<where_ever_it_is>/NonRadiativeRates/src` to your `$PYTHONPATH`

# Cython

run:
```
cython -3 fastfold.pyx
```
to create the fastfold.c source file. Compile this using
```
gcc -shared -pthread -fPIC -O2 -Wall -fno-strict-aliasing -I/usr/include/python3.9 -o fastfold.so fastfold.c
```
Note: The path to the python include file may be different on other systems.
