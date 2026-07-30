# NonRadiativeRates
Collection of python code to compute the bits and pieces needed for non-radiative rates

# Setup
Add `<where_ever_it_is>/NonRadiativeRates/src` to your `$PYTHONPATH`

# Cython

run (inside `src`):
```
python setup.py build_ext --inplace
```
(or `python3` if `python` points to a legacy python2 verion). This compiles the cython file `fastfolg.pyx`.

For MAC (in a virtutal environment):
```
brew install pyenv pyenv-virtualenv
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv virtualenv 3.9.5 <name>
pip install -r requirements.txt
pip install --no-build-isolation molmod 
```
then compile the cython file as described above. 
