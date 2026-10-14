"""Exit codes of `OncvpspCalculation` and their interpretation.

Exit codes are stage-based: a failed run gets the code of the ONCVPSP stage in which its primary
(first) error occurred, as attributed by `pyoncvpsp` (`OncvpspTextParser.failure`). Details
(message name, angular momentum, implicated parameters) are stored in `output_parameters["failure"]`.

Earlier versions of this plugin derived one exit code per error name from the *order* of
`pyoncvpsp.io.ERRORS` (`504 + index`), so those codes changed meaning whenever that list changed.
`LEGACY_EXIT_CODES` records the mapping of the pyoncvpsp revision that was pinned while that scheme was
in use (4f2974c, 9 errors), so that stored calculations can still be interpreted. The stage codes live
in the 600 range so that they can never collide with legacy codes (504-543).
"""

from pyoncvpsp.io.output import Stage

#: Exit codes which do not depend on ONCVPSP's output
RETRIEVAL_EXIT_CODES: dict[int, tuple[str, str]] = {
    301: ("ERROR_NO_RETRIEVED_FOLDER", "The retrieved folder could not be found."),
    302: ("ERROR_OUTPUT_FILE_MISSING", "The output file is missing."),
    303: ("ERROR_OUTPUT_FILE_UNREADABLE", "The output file could not be read."),
    304: ("ERROR_OUTPUT_PARSING_FAILED", "The output file could not be parsed."),
}

#: Legacy generic codes (kept so that stored nodes keep a meaning; no longer returned)
LEGACY_GENERIC_EXIT_CODES: dict[int, tuple[str, str]] = {
    501: (
        "ERROR_RUN_NOT_COMPLETED",
        "(legacy) The ONCVPSP run did not complete successfully, but no other error was found.",
    ),
    502: ("ERROR_UNKNOWN_ERROR", "(legacy) An unknown error occurred."),
    503: ("ERROR_MULTIPLE_ERRORS", "(legacy) Multiple errors were found in the output file."),
}

#: Stage-based exit codes: stage -> (status, label)
STAGE_EXIT_CODES: dict[Stage, tuple[int, str]] = {
    Stage.INPUT: (610, "ERROR_STAGE_INPUT"),
    Stage.AE_REFERENCE: (620, "ERROR_STAGE_AE_REFERENCE"),
    Stage.SEMILOCAL: (630, "ERROR_STAGE_SEMILOCAL"),
    Stage.VKB: (640, "ERROR_STAGE_VKB"),
    Stage.VKB_SOLVE: (650, "ERROR_STAGE_VKB_SOLVE"),
    Stage.MODEL_CORE: (660, "ERROR_STAGE_MODEL_CORE"),
    Stage.DIAGNOSTICS: (670, "ERROR_STAGE_DIAGNOSTICS"),
    Stage.TEST_CONFIG: (680, "ERROR_STAGE_TEST_CONFIG"),
    Stage.LOGDER_PLOT: (690, "ERROR_STAGE_LOGDER_PLOT"),
    Stage.OUTPUT: (695, "ERROR_STAGE_OUTPUT"),
    Stage.UNKNOWN: (699, "ERROR_STAGE_UNKNOWN"),
}

STAGE_EXIT_MESSAGES: dict[Stage, str] = {
    Stage.INPUT: "ONCVPSP rejected the input (check_data) or could not read it.",
    Stage.AE_REFERENCE: "The all-electron reference atom could not be solved.",
    Stage.SEMILOCAL: "Construction of the semilocal pseudopotential (wellstate/optimize/pspot) failed for one channel.",
    Stage.VKB: "Construction of the Vanderbilt-Kleinman-Bylander projectors failed.",
    Stage.VKB_SOLVE: "A valence state could not be solved in the separable (VKB + local) potential.",
    Stage.MODEL_CORE: "Construction of the model core charge failed.",
    Stage.DIAGNOSTICS: "ONCVPSP stopped during the diagnostic tests (run_diag/run_ghosts).",
    Stage.TEST_CONFIG: "ONCVPSP stopped during a test configuration.",
    Stage.LOGDER_PLOT: "ONCVPSP stopped while writing log derivatives or plotting data.",
    Stage.OUTPUT: "ONCVPSP stopped while writing the pseudopotential file.",
    Stage.UNKNOWN: "The ONCVPSP run did not complete and the stage of the failure could not be determined.",
}

#: Exit status -> stage for the stage-based codes
EXIT_STATUS_TO_STAGE: dict[int, Stage] = {status: stage for stage, (status, _) in STAGE_EXIT_CODES.items()}

#: Legacy per-error-name codes (pyoncvpsp 4f2974c ordering): status -> (error name, stage or None)
LEGACY_EXIT_CODES: dict[int, tuple[str, Stage | None]] = {
    504: ("lschvkbb_not_converged", Stage.TEST_CONFIG),
    505: ("wellstate_not_converged", Stage.SEMILOCAL),
    506: ("bad_input", Stage.INPUT),
    507: ("input_file_read_error", Stage.INPUT),
    508: ("no_classical_turning_point", None),
    509: ("no_core_valence_crossing", Stage.MODEL_CORE),
    510: ("first_pswf_has_node", Stage.SEMILOCAL),
    511: ("ps0norm_gt_uunorm", Stage.SEMILOCAL),
    512: ("rxpsh_too_small", Stage.LOGDER_PLOT),
}


def legacy_exit_codes_for(errors: list[dict], offset: int = 504) -> dict[int, str]:
    """Reconstruct the legacy `status -> error name` mapping for an arbitrary `pyoncvpsp.io.ERRORS`
    ordering (useful if calculations were run with a pyoncvpsp revision other than 4f2974c)."""
    return {offset + i: error["name"] for i, error in enumerate(errors)}


def exit_status_to_stage(status: int | None) -> Stage | None:
    """Stage of the failure for a stored exit status (stage-based or legacy), or `None` for success,
    retrieval/parsing failures, and legacy codes without a well-defined stage."""
    if status is None or status == 0:
        return None
    if status in EXIT_STATUS_TO_STAGE:
        return EXIT_STATUS_TO_STAGE[status]
    if status in LEGACY_EXIT_CODES:
        return LEGACY_EXIT_CODES[status][1]
    if status == 501:
        return Stage.UNKNOWN
    return None


def exit_status_is_failure(status: int | None) -> bool:
    """Whether a stored exit status denotes a failed pseudization (as opposed to success or a
    retrieval/parsing problem of the plugin)."""
    return status is not None and status >= 500


__all__ = [
    "EXIT_STATUS_TO_STAGE",
    "LEGACY_EXIT_CODES",
    "LEGACY_GENERIC_EXIT_CODES",
    "RETRIEVAL_EXIT_CODES",
    "STAGE_EXIT_CODES",
    "STAGE_EXIT_MESSAGES",
    "exit_status_is_failure",
    "exit_status_to_stage",
    "legacy_exit_codes_for",
]
