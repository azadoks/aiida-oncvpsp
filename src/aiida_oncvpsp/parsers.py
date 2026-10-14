import io
import logging
from collections.abc import Callable

from aiida import orm, plugins
from aiida.common.exceptions import NotExistent
from aiida.parsers import Parser
import numpy as np
from pyoncvpsp.io import OncvpspTextParser
from pyoncvpsp.io.output import Level, Stage

from .exit_codes import STAGE_EXIT_CODES

ArrayData = plugins.DataFactory("core.array")
XyData = plugins.DataFactory("core.array.xy")
UpfData = plugins.DataFactory("pseudo.upf")
Psp8Data = plugins.DataFactory("pseudo.psp8")

SCHEDULER_STDERR = "_scheduler-stderr.txt"
# Substrings of stderr which indicate that the program crashed (as opposed to benign notes such as
# gfortran's "Note: The following floating-point exceptions are signalling")
RUNTIME_ERROR_MARKERS = ("Fortran runtime error", "forrtl: severe", "Program received signal")


class OncvpspParser(Parser):
    """Parser for ONCVPSP calculations.

    Every output which can be parsed is attached, whether or not the run completed. The exit status
    is 0 if ONCVPSP wrote all of its output and otherwise the stage-based code of the primary failure
    (see `aiida_oncvpsp.exit_codes`), whose details are in `output_parameters["failure"]`.
    """

    def parse(self, **kwargs):
        """Parse the retrieved files."""
        try:
            retrieved = self.retrieved
        except NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER
        try:
            with retrieved.open(self.node.base.attributes.get("output_filename"), "rb") as fp:
                # Decode as latin-1 (as `pyoncvpsp` does), which cannot fail on arbitrary bytes
                text = fp.read().decode("latin-1")
        except FileNotFoundError:
            return self.exit_codes.ERROR_OUTPUT_FILE_MISSING
        except OSError:
            return self.exit_codes.ERROR_OUTPUT_FILE_UNREADABLE
        stderr = self._read_stderr(retrieved)
        if stderr and any(marker in stderr for marker in RUNTIME_ERROR_MARKERS):
            # Let pyoncvpsp pick up the runtime error and attribute it to the last stage reached
            text = text.rstrip("\n") + "\n" + stderr

        parser = OncvpspTextParser(text=text, check=False)
        self._report_messages(parser)

        outputs, build_failed = self._build_outputs(parser)
        for label, node in outputs.items():
            self.out(label, node)

        failure = parser.failure
        if failure is None:
            if build_failed:
                return self.exit_codes.ERROR_OUTPUT_PARSING_FAILED
            return None
        status, label = STAGE_EXIT_CODES.get(failure.stage, STAGE_EXIT_CODES[Stage.UNKNOWN])
        return self.exit_codes[label]

    def _read_stderr(self, retrieved) -> str:
        try:
            with retrieved.open(SCHEDULER_STDERR, "rb") as fp:
                return fp.read().decode("latin-1")
        except (FileNotFoundError, OSError):
            return ""

    def _build_outputs(self, parser: OncvpspTextParser) -> tuple[dict, bool]:
        """Build every output node independently. Returns the nodes and whether any builder raised."""
        builders: dict[str, Callable[[OncvpspTextParser], orm.Data | None]] = {
            "local_potential": _build_local_potential,
            "charge_densities": _build_charge_densities,
            "kinetic_energy_densities": _build_kinetic_energy_densities,
            "kinetic_energy_potentials": _build_kinetic_energy_potentials,
            "unscreened_semilocal_potentials": _build_unscreened_semilocal_potentials,
            "wavefunctions": _build_wavefunctions,
            "vkb_projectors": _build_vkb_projectors,
            "convergence_profiles": _build_convergence_profiles,
            "log_derivatives": _build_log_derivatives,
            "upf": self._build_upf,
            "psp8": self._build_psp8,
        }
        outputs: dict[str, orm.Data] = {}
        failed = False
        for label, builder in builders.items():
            try:
                node = builder(parser)
            except Exception:  # pylint: disable=broad-except
                self.logger.exception("Failed to build output `%s`.", label)
                failed = True
                continue
            if node is not None:
                outputs[label] = node
        try:
            outputs["output_parameters"] = _build_output_parameters(parser)
        except Exception:  # pylint: disable=broad-except
            self.logger.exception("Failed to build `output_parameters`; storing the failure summary only.")
            failed = True
            outputs["output_parameters"] = orm.Dict(_finite_or_none(_failure_summary(parser)))
        return outputs, failed

    def _build_upf(self, parser: OncvpspTextParser):
        if not parser.upf_string:
            return None
        return UpfData(filename=f"{self.node.uuid}.upf", content=io.BytesIO(parser.upf_string.encode("utf-8")))

    def _build_psp8(self, parser: OncvpspTextParser):
        if not parser.psp8_string:
            return None
        return Psp8Data(filename=f"{self.node.uuid}.psp8", content=io.BytesIO(parser.psp8_string.encode("utf-8")))

    def _report_messages(self, parser: OncvpspTextParser) -> None:
        for msg in parser.messages:
            level = logging.ERROR if msg.level == Level.ERROR else logging.WARNING
            where = f"stage={msg.stage.value}" + (f" l={msg.l}" if msg.l is not None else "")
            self.logger.log(level, f"{msg.name}#{msg.line_numbers[0]} [{where}]: {msg.description}")


# --- Output builders ----------------------------------------------------------------------------------


def _failure_summary(parser: OncvpspTextParser) -> dict:
    """Completion state, messages and attributed failure; never depends on the physical outputs."""
    failure = parser.failure
    messages = parser.messages
    return {
        "completed": parser.completed,
        "missing_plot_data": parser.missing_plot_data,
        "last_stage_reached": parser.last_stage_reached.value,
        "warnings": [m.as_dict() for m in messages if m.level == Level.WARNING],
        "errors": [m.as_dict() for m in messages if m.level == Level.ERROR],
        "num_warnings": sum(m.level == Level.WARNING for m in messages),
        "num_errors": sum(m.level == Level.ERROR for m in messages),
        "failure": failure.as_dict() if failure is not None else None,
    }


def _build_output_parameters(parser: OncvpspTextParser) -> orm.Dict:
    summary = _failure_summary(parser)
    optional: dict = {}
    # Each of these may legitimately be unavailable when the run stopped early
    for key, getter in {
        "program_information": lambda: dict(parser.program_information),
        "teter_parameters": lambda: dict(parser.teter_parameters),
        "d2exc_rmse": lambda: {"d2exc_rmse": parser.d2exc_rmse},
        "ecut_recommended": lambda: {"ecut_recommended": parser.ecut_recommended},
        "ghosts": lambda: {
            "ghosts": parser.ghosts,
            "num_ghosts": len(parser.ghosts),
            "num_positive_ghosts": sum(g["eig"] > 0 for g in parser.ghosts),
            "num_negative_ghosts": sum(g["eig"] < 0 for g in parser.ghosts),
        },
    }.items():
        try:
            optional.update(getter())
        except Exception:  # pylint: disable=broad-except
            if parser.completed:
                raise
    return orm.Dict(_finite_or_none({**optional, **summary}))


def _build_local_potential(parser: OncvpspTextParser):
    if not parser.local_potential:
        return None
    xy = XyData()
    xy.set_x(parser.local_potential["r"], "Radius", "Bohr")
    xy.set_y(parser.local_potential["v_loc"], "Local Potential", "Hartree")
    return xy


def _build_charge_densities(parser: OncvpspTextParser):
    if not parser.charge_densities:
        return None
    xy = XyData()
    xy.set_x(parser.charge_densities["r"], "Radius", "Bohr")
    xy.set_y(parser.charge_densities["rho_val"], "Valence Charge Density", "e/Bohr^3")
    xy.set_y(parser.charge_densities["rho_core"], "All-Electron Core Charge Density", "e/Bohr^3")
    xy.set_y(parser.charge_densities["rho_model_core"], "Model Core Charge Density", "e/Bohr^3")
    return xy


def _build_kinetic_energy_densities(parser: OncvpspTextParser):
    if not parser.kinetic_energy_densities:
        return None
    xy = XyData()
    xy.set_x(parser.kinetic_energy_densities["r"], "Radius", "Bohr")
    xy.set_y(parser.kinetic_energy_densities["tau_ps"], "Valence Kinetic Energy Density", "")
    xy.set_y(parser.kinetic_energy_densities["tau_mod"], "Model Core Kinetic Energy Density", "")
    return xy


def _build_kinetic_energy_potentials(parser: OncvpspTextParser):
    if not parser.kinetic_energy_potentials:
        return None
    xy = XyData()
    xy.set_x(parser.kinetic_energy_potentials["r"], "Radius", "Bohr")
    xy.set_y(parser.kinetic_energy_potentials["vtau_ps"], "Valence Kinetic Energy Potential", "")
    xy.set_y(parser.kinetic_energy_potentials["vtau_mod"], "Model Core Kinetic Energy Potential", "")
    return xy


def _ragged(records: list[dict], x_key: str, y_keys: list[str]) -> dict[str, np.ndarray]:
    """Stack per-record 1D arrays into NaN-padded 2D arrays."""
    width = max(len(rec[x_key]) for rec in records)
    arrays = {key: np.full((len(records), width), np.nan) for key in [x_key, *y_keys]}
    for i, rec in enumerate(records):
        for key in [x_key, *y_keys]:
            arrays[key][i, : len(rec[key])] = rec[key]
    return arrays


def _build_records(records: list[dict], x_key: str, y_keys: list[str], label_keys: list[str] | None = None):
    if not records:
        return None
    array = ArrayData()
    labels = label_keys if label_keys is not None else sorted(set(records[0]) - {x_key, *y_keys})
    for key in labels:
        array.set_array(key, np.array([rec[key] for rec in records]))
    for key, values in _ragged(records, x_key, y_keys).items():
        array.set_array(key, values)
    return array


def _build_unscreened_semilocal_potentials(parser: OncvpspTextParser):
    return _build_records(parser.unscreened_semilocal_potentials, "r", ["v_sl"], ["l"])


def _build_wavefunctions(parser: OncvpspTextParser):
    return _build_records(parser.wavefunctions, "r", ["wfn_ae", "wfn_ps"])


def _build_vkb_projectors(parser: OncvpspTextParser):
    return _build_records(parser.vkb_projectors, "r", ["proj"])


def _build_convergence_profiles(parser: OncvpspTextParser):
    return _build_records(parser.convergence_profiles, "ecut", ["eresid"])


def _build_log_derivatives(parser: OncvpspTextParser):
    return _build_records(parser.log_derivatives, "e", ["log_deriv_ae", "log_deriv_ps"])


def _finite_or_none(value):
    """Recursively replace NaN and +/-inf floats with `None`, which the database cannot store."""
    if isinstance(value, dict):
        return {k: _finite_or_none(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_none(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value
