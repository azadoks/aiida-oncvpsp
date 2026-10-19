"""Consistency of the exit-code tables."""

from aiida.plugins import CalculationFactory
from pyoncvpsp.io.output import ERRORS, Stage

from aiida_oncvpsp.exit_codes import (
    LEGACY_EXIT_CODES,
    LEGACY_GENERIC_EXIT_CODES,
    RETRIEVAL_EXIT_CODES,
    STAGE_EXIT_CODES,
    exit_status_is_failure,
    exit_status_to_stage,
    legacy_exit_codes_for,
)


def test_every_stage_has_a_code():
    assert set(STAGE_EXIT_CODES) == set(Stage)
    statuses = [status for status, _ in STAGE_EXIT_CODES.values()]
    assert len(statuses) == len(set(statuses))
    assert all(600 <= s < 700 for s in statuses)


def test_stage_codes_do_not_collide_with_legacy_codes():
    stage = {status for status, _ in STAGE_EXIT_CODES.values()}
    legacy = set(LEGACY_EXIT_CODES) | set(LEGACY_GENERIC_EXIT_CODES) | set(RETRIEVAL_EXIT_CODES)
    assert not stage & legacy
    # Any ordering of up to 40 pyoncvpsp errors under the legacy scheme stays below 600
    assert max(legacy_exit_codes_for(ERRORS)) < 600


def test_spec_defines_all_codes():
    spec_codes = {code.status: label for label, code in CalculationFactory("oncvpsp.oncvpsp").spec().exit_codes.items()}
    for status, (label, _) in {**RETRIEVAL_EXIT_CODES, **LEGACY_GENERIC_EXIT_CODES}.items():
        assert spec_codes[status] == label
    for status, label in STAGE_EXIT_CODES.values():
        assert spec_codes[status] == label


def test_exit_status_to_stage():
    assert exit_status_to_stage(0) is None
    assert exit_status_to_stage(None) is None
    assert exit_status_to_stage(302) is None
    assert exit_status_to_stage(630) == Stage.SEMILOCAL
    assert exit_status_to_stage(510) == Stage.SEMILOCAL  # legacy first_pswf_has_node
    assert exit_status_to_stage(504) == Stage.TEST_CONFIG  # legacy lschvkbb_not_converged
    assert exit_status_to_stage(508) is None  # legacy no_classical_turning_point: stage unknown
    assert exit_status_to_stage(501) == Stage.UNKNOWN
    assert exit_status_is_failure(630) and exit_status_is_failure(504)
    assert not exit_status_is_failure(0) and not exit_status_is_failure(304)


def test_legacy_names_exist_in_pyoncvpsp():
    names = {e["name"] for e in ERRORS}
    for name, _ in LEGACY_EXIT_CODES.values():
        assert name in names
