from aiida.engine import calcfunction
from aiida.plugins import DataFactory
import numpy as np

ArrayData = DataFactory("core.array")
Float = DataFactory("core.float")


@calcfunction
def compute_log_der_rmse(log_derivatives: "ArrayData", e_min: "Float", e_max: "Float") -> "ArrayData":
    ells = log_derivatives.get_array("l")
    energies = log_derivatives.get_array("e")
    ld_ae = log_derivatives.get_array("log_deriv_ae")
    ld_ps = log_derivatives.get_array("log_deriv_ps")
    # Check that all the rows in energies are the same (they always should be)
    # and take the first row
    assert np.allclose(np.diff(energies, axis=0), 0.0)
    energies = energies[0]
    # Sort all data so that energies are in ascending order
    # (usually comes in descending order from ONCVPSP)
    idx = np.argsort(energies)
    energies = energies[idx]
    ld_ae = ld_ae[:, idx]
    ld_ps = ld_ps[:, idx]
    # Select only energies within the specified range
    idx = (energies >= e_min.value) & (energies <= e_max.value)
    energies = energies[idx]
    ld_ae = ld_ae[:, idx]
    ld_ps = ld_ps[:, idx]
    # Compute the mean RMSE for each ell (averaging over kappa)
    errors: np.ndarray = np.zeros(ells.max() + 1)
    counts: np.ndarray = np.zeros(ells.max() + 1, dtype=int)
    for i, ell in enumerate(ells):
        errors[ell] += np.sqrt(np.mean((ld_ae[i] - ld_ps[i]) ** 2))
        counts[ell] += 1
    errors /= counts
    return ArrayData({"l": np.arange(ells.max() + 1), "rmse": errors})
