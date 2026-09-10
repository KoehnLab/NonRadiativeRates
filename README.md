# NonRadiativeRates
Collection of python code to compute the bits and pieces needed for non-radiative rates

# Setup

This requires a C compiler (to build the `_fastfold` Cython extension). Install the package,
in editable mode, from the repository root:
```
pip install -e .
```

For development (running the test suite):
```
pip install -e ".[dev]"
```

# Tests

Run the test suite with:
```
pytest
```

# Cython

The `_fastfold` extension is built automatically by `pip install -e .`. If you change
`src/nonradiative_rates/_fastfold.pyx` and want to rebuild it in place without reinstalling:
```
python setup.py build_ext --inplace
```
