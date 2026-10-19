import numpy as np
from aiida.engine import calcfunction
from aiida.plugins import DataFactory

ArrayData = DataFactory("core.array")
Float = DataFactory("core.float")


@calcfunction
def compute_log_der_rmse(log_derivatives: "ArrayData", e_min: "Float", e_max: "Float") -> "ArrayData":
    ells = log_derivatives.get_array("l")
    energies = log_derivatives.get_array("e")
    ld_ae = log_derivatives.get_array("log_deriv_ae")
    ld_ps = log_derivatives.get_array("log_deriv_ps")
    # Compute the mean RMSE for each ell (averaging over kappa)
    errors: np.ndarray = np.zeros(ells.max() + 1)
    counts: np.ndarray = np.zeros(ells.max() + 1, dtype=int)
    for i, ell in enumerate(ells):
        # Each row has its own energy grid, NaN-padded by the parser if rows differ in length,
        # so select the energies within the specified range row by row
        mask = np.isfinite(energies[i]) & (energies[i] >= e_min.value) & (energies[i] <= e_max.value)
        if not np.any(mask):
            raise ValueError(f"No log-derivative energies for l={ell} in the range [{e_min.value}, {e_max.value}].")
        errors[ell] += np.sqrt(np.mean((ld_ae[i, mask] - ld_ps[i, mask]) ** 2))
        counts[ell] += 1
    if np.any(counts == 0):
        raise ValueError(f"No log-derivatives found for l={np.flatnonzero(counts == 0).tolist()}.")
    errors /= counts
    return ArrayData({"l": np.arange(ells.max() + 1), "rmse": errors})
