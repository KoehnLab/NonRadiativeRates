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

For MAC (in a virtutal environment):
```
brew install pyenv pyenv-virtualenv
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv virtualenv 3.9.5 <name>
pip install -r requirements.txt
pip install --no-build-isolation molmod 
```
then execute:
```
gcc -shared -pthread -fPIC -O2 -Wall -fno-strict-aliasing -I$(python3-config --includes) -undefined dynamic_lookup -o fastfold.so fastfold.c
```
