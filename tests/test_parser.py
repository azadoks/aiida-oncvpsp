"""Tests of `OncvpspParser`: exit codes, always-attached `output_parameters`, partial outputs."""

import numpy as np
import pytest
from aiida.plugins import CalculationFactory, ParserFactory
from pyoncvpsp.io.output import Stage

from aiida_oncvpsp.exit_codes import STAGE_EXIT_CODES, exit_status_to_stage

OncvpspCalculation = CalculationFactory("oncvpsp.oncvpsp")

GFORTRAN_STDERR = (
    "At line 436 of file oncvpsp.F90 (unit = 6, file = 'stdout')\n"
    "Fortran runtime error: Expected REAL for item 10 in formatted transfer, got CHARACTER\n"
)


def parse(node):
    parser = ParserFactory("oncvpsp.oncvpsp")(node)
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)
    return results, calcfunction


def test_completed_run(generate_calc_job_node):
    results, calcfunction = parse(generate_calc_job_node("ok"))
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    params = results["output_parameters"].get_dict()
    assert params["completed"] is True
    assert params["failure"] is None
    assert params["last_stage_reached"] == "output"
    assert params["num_errors"] == 0
    assert params["ecut_recommended"] is not None
    assert "ghosts" in params and "num_ghosts" in params
    for label in (
        "local_potential",
        "charge_densities",
        "unscreened_semilocal_potentials",
        "wavefunctions",
        "vkb_projectors",
        "convergence_profiles",
        "log_derivatives",
        "upf",
    ):
        assert label in results, label
    # Records of l-channels have no j; it is stored as NaN, since ArrayData cannot store None
    wavefunctions = results["wavefunctions"]
    assert np.isnan(wavefunctions.get_array("j")).all()
    assert wavefunctions.get_array("l").dtype.kind == "i"
    # Diagnostics warning from run_diag is attributed
    (warning,) = [w for w in params["warnings"] if w["name"] == "lschvkbbe_not_converged"]
    assert warning["stage"] == "diagnostics" and warning["l"] == 0 and warning["attribution"] == "explicit"


def test_completed_run_with_non_fatal_errors(generate_calc_job_node):
    """A skipped test configuration is reported but is not a failure of the pseudization."""
    results, calcfunction = parse(generate_calc_job_node("testcfg_bad"))
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    params = results["output_parameters"].get_dict()
    assert params["failure"] is None
    names = [e["name"] for e in params["errors"]]
    assert "lschvkbb_not_converged" in names
    (skipped,) = [e for e in params["errors"] if e["name"] == "lschvkbb_not_converged"]
    assert skipped["stage"] == "test_config" and skipped["implicated_parameters"] == ["test_configurations__0"]


@pytest.mark.parametrize(
    ("name", "primary", "stage", "ell", "attribution"),
    [
        ("rc_tiny_l1", "vkb_bound_state_not_converged", Stage.VKB_SOLVE, 1, "explicit"),
        ("qcut_high_l1", "first_pswf_has_node", Stage.SEMILOCAL, 1, "inferred"),
        ("rcloc_gt_rc", "input_rc_lt_rcloc", Stage.INPUT, 0, "explicit"),
        ("truncated_semilocal_l1", "incomplete_output", Stage.SEMILOCAL, 1, "inferred"),
    ],
)
def test_failed_run(generate_calc_job_node, name, primary, stage, ell, attribution):
    results, calcfunction = parse(generate_calc_job_node(name))
    status, label = STAGE_EXIT_CODES[stage]
    assert calcfunction.exit_status == status, calcfunction.exit_message
    assert OncvpspCalculation.exit_codes[label].status == status
    assert exit_status_to_stage(calcfunction.exit_status) == stage
    # `output_parameters` is attached on failure, with the attributed failure
    params = results["output_parameters"].get_dict()
    assert params["completed"] is False
    failure = params["failure"]
    assert failure["name"] == primary
    assert failure["stage"] == stage.value
    assert failure["l"] == ell
    assert failure["attribution"] == attribution
    assert failure["implicated_parameters"]
    assert isinstance(failure["primary"]["line_numbers"], list)
    # Partial outputs: nothing physical is available for runs which stopped before the plotting data
    assert "upf" not in results and "log_derivatives" not in results


def test_truncated_semilocal_has_convergence_profiles(generate_calc_job_node):
    results, _ = parse(generate_calc_job_node("truncated_semilocal_l1"))
    assert "convergence_profiles" in results  # the l=0 optimize block was complete
    assert results["convergence_profiles"].get_array("l").tolist() == [0, 0]


def test_stderr_runtime_error_is_merged(generate_calc_job_node):
    results, calcfunction = parse(generate_calc_job_node("truncated_semilocal_l1", stderr=GFORTRAN_STDERR))
    params = results["output_parameters"].get_dict()
    names = [e["name"] for e in params["errors"]]
    assert names == ["fortran_runtime_error"]
    assert params["failure"]["name"] == "fortran_runtime_error"
    assert params["failure"]["stage"] == "semilocal"
    assert calcfunction.exit_status == STAGE_EXIT_CODES[Stage.SEMILOCAL][0]


def test_benign_stderr_is_ignored(generate_calc_job_node):
    results, calcfunction = parse(
        generate_calc_job_node(
            "ok", stderr="Note: The following floating-point exceptions are signalling: IEEE_UNDERFLOW_FLAG\n"
        )
    )
    assert calcfunction.is_finished_ok
    assert results["output_parameters"]["num_errors"] == 0


def test_output_file_missing(generate_calc_job_node):
    _, calcfunction = parse(generate_calc_job_node("ok", stored_as="other.out"))
    assert calcfunction.exit_status == OncvpspCalculation.exit_codes.ERROR_OUTPUT_FILE_MISSING.status
