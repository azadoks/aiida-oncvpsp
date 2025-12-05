# `aiida-oncvpsp`

This is an AiiDA plugin for [ONCVPSP](https://github.com/oncvpsp/oncvpsp).

## Installation

1. Clone `azadoks/oncvpsp:hdf5-output` and install `pyoncvpsp`
    ```bash
    git clone https://github.com/azadoks/oncvpsp
    cd oncvpsp; git checkout -b hdf5-output
    pip install -e .
    # [Optional] Compile and install ONCVPSP v4.0.1 with HDF5 output support
    cmake -S. -Bbuild -DCMAKE_INSTALL_PREFIX=$HOME/.local/share/oncvpsp-4.0.1-hdf5
    cmake --build build --target install --parallel
    ```

2. Clone this repository and install `aiida-oncvpsp`
    ```bash
    git clone https://github.com/azadoks/aiida-oncvpsp
    pip install -e ./aiida-oncvpsp
    ```

3. Use `verdi code create core.code.installed` to set up an ONCVPSP `Code` in AiiDA.
    - You can modify [oncvpsp-sr-4.0.1.yaml](examples/oncvpsp-sr-4.0.1.yaml) and use it to create the `Code` with the `--config` flag.

## Example

```python
# %% Imports
from aiida import load_profile
from aiida import orm
from aiida.engine import run
from aiida.plugins import CalculationFactory
from pyoncvpsp.io import OncvpspInput
from aiida_oncvpsp.calcfunctions import compute_log_der_rmse
OncvpspCalculation = CalculationFactory("oncvpsp.oncvpsp")
load_profile()

# %% Define the calculation input
_input = OncvpspInput.from_dat_string(
"""# ATOM AND REFERENCE CONFIGURATION
#    atsym          z         nc         nv       iexc     psfile
        Li    3.00000          0          2          3       psp8
#        n          l          f
         1          0    2.00000
         2          0    1.00000
# PSEUDOPOTENTIAL AND OPTIMIZATION
#     lmax
         1
#        l         rc         ep       ncon       nbas       qcut
         0    2.00000    0.00000          4          8    6.00000
         1    3.00000    0.10000          4          8    4.00000
# LOCAL POTENTIAL
#     lloc      lpopt      rc(5)     dvloc0
         4          5    1.00000    0.00000
# VANDERBILT-KLEINMAN-BYLANDER PROJECTORS
#        l      nproj       debl
         0          2    0.00000
         1          2    0.75000
# MODEL CORE CHARGE
#    icmod     fcfact
         0    0.00000
# LOG DERIVATIVE ANALYSIS
#    epsh1      epsh2      depsh
  -2.00000    2.00000    0.02000
# OUTPUT GRID
#    rlmax        drl
   6.00000    0.01000
# TEST CONFIGURATIONS
#     ncnf
         2
#    nvcnf
         1
#        n          l          f
         1          0    2.00000
#    nvcnf
         2
#        n          l          f
         1          0    2.00000
         2          1    1.00000
""")
# Modify some parameters
_input.log_derivative_analysis.epsh1 = -12.0
_input.log_derivative_analysis.epsh2 = +12.0
# Build the CalcJob inputs
builder = OncvpspCalculation.get_builder()
# Set the Code
builder.code = orm.load_code("oncvpsp-sr-4.0.1")
# Get the parameters by dumping the OncvpspInput to dict
builder.parameters = orm.Dict(dict=_input.model_dump())
builder.metadata.options.resources = {"num_machines": 1}

#%% Run the calculation (blocking)
result = run(builder)

# %% Compute RMSE errors of the log derivatives
ld_errors = compute_log_der_rmse(
    log_derivatives=result["log_derivatives"],
    e_min=orm.Float(-2.0),
    e_max=orm.Float(6.0)
)
```

## Acknowledgements

[NCCR MARVEL](http://nccr-marvel.ch/) funded by the Swiss National Science Foundation.

<img title="MARVEL Logo" alt="MARVEL Logo" src="docs/MARVEL.png" width=15% align="left">
