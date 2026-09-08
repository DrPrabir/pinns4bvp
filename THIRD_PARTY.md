# Third-Party Software

PINNs4BVP depends on open-source scientific Python software. These packages are
not distributed as source code inside PINNs4BVP; they are installed separately
as Python dependencies.

## Runtime dependencies

### NumPy
- Purpose: numerical arrays and vectorized numerical operations
- License: BSD 3-Clause
- Project: https://numpy.org/
- License information: https://numpy.org/doc/stable/license.html

### SciPy
- Purpose: the v0.1 numerical BVP backend uses `scipy.integrate.solve_bvp`
- License: BSD 3-Clause
- Project: https://scipy.org/
- License information: https://scipy.org/scipylib/license.html

### Matplotlib
- Purpose: plotting and visualization support
- License: Matplotlib license (PSF-based/BSD-compatible)
- Project: https://matplotlib.org/
- License information: https://matplotlib.org/stable/project/license.html

## Development/test dependencies

### pytest
- Purpose: automated testing
- License: MIT
- Project: https://pytest.org/

### setuptools
- Purpose: Python package build/install tooling
- License: MIT
- Project: https://setuptools.pypa.io/

### wheel
- Purpose: Python wheel packaging
- License: MIT
- Project: https://wheel.readthedocs.io/

## Notes

PINNs4BVP is released under the MIT License. The licenses of third-party
packages remain their own and are not replaced by the PINNs4BVP license.
