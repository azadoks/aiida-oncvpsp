import io
import logging

from aiida import orm, plugins
from aiida.common.exceptions import NotExistent
from aiida.parsers import Parser
import numpy as np
from pyoncvpsp.io import OncvpspTextParser

ArrayData = plugins.DataFactory("core.array")
XyData = plugins.DataFactory("core.array.xy")
UpfData = plugins.DataFactory("pseudo.upf")
Psp8Data = plugins.DataFactory("pseudo.psp8")


class OncvpspParser(Parser):
    """Parser for ONCVPSP calculations."""

    def parse(self, **kwargs):
        """Parse the retrieved files."""
        # self.retrieved: run folder
        # self.node: OncvpspCalculation
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
        parser = OncvpspTextParser(text=text, check=False)
        self._report_messages(parser.warnings, level="warning")
        self._report_messages(parser.errors, level="error")
        if len(parser.errors) == 1:
            return self.exit_codes.get(parser.errors[0]["name"].upper(), self.exit_codes.ERROR_UNKNOWN_ERROR)
        elif len(parser.errors) > 1:
            return self.exit_codes.ERROR_MULTIPLE_ERRORS
        if not parser.completed:
            return self.exit_codes.ERROR_RUN_NOT_COMPLETED
        # Build all outputs before attaching any, so that a parsing failure doesn't leave partial outputs
        try:
            outputs = self._build_outputs(parser)
        except Exception:
            self.logger.exception("Failed to parse the ONCVPSP output file.")
            return self.exit_codes.ERROR_OUTPUT_PARSING_FAILED
        for label, node in outputs.items():
            self.out(label, node)

    def _build_outputs(self, parser: OncvpspTextParser) -> dict:
        outputs = {}
        if parser.local_potential:
            xy = XyData()
            xy.set_x(parser.local_potential["r"], "Radius", "Bohr")
            xy.set_y(parser.local_potential["v_loc"], "Local Potential", "Hartree")
            outputs["local_potential"] = xy
        if parser.charge_densities:
            xy = XyData()
            xy.set_x(parser.charge_densities["r"], "Radius", "Bohr")
            xy.set_y(parser.charge_densities["rho_val"], "Valence Charge Density", "e/Bohr^3")
            xy.set_y(
                parser.charge_densities["rho_core"],
                "All-Electron Core Charge Density",
                "e/Bohr^3",
            )
            xy.set_y(
                parser.charge_densities["rho_model_core"],
                "Model Core Charge Density",
                "e/Bohr^3",
            )
            outputs["charge_densities"] = xy
        if parser.kinetic_energy_densities:
            xy = XyData()
            xy.set_x(parser.kinetic_energy_densities["r"], "Radius", "Bohr")
            xy.set_y(
                parser.kinetic_energy_densities["tau_ps"],
                "Valence Kinetic Energy Density",
                "",
            )
            xy.set_y(
                parser.kinetic_energy_densities["tau_mod"],
                "Model Core Kinetic Energy Density",
                "",
            )
            outputs["kinetic_energy_densities"] = xy
        if parser.kinetic_energy_potentials:
            xy = XyData()
            xy.set_x(parser.kinetic_energy_potentials["r"], "Radius", "Bohr")
            xy.set_y(
                parser.kinetic_energy_potentials["vtau_ps"],
                "Valence Kinetic Energy Potential",
                "",
            )
            xy.set_y(
                parser.kinetic_energy_potentials["vtau_mod"],
                "Model Core Kinetic Energy Potential",
                "",
            )
            outputs["kinetic_energy_potentials"] = xy
        if parser.unscreened_semilocal_potentials:
            array = ArrayData()
            array.set_array(
                "l",
                np.array(
                    [pot["l"] for pot in parser.unscreened_semilocal_potentials],
                    dtype=int,
                ),
            )
            pot_r = np.full(
                (
                    len(parser.unscreened_semilocal_potentials),
                    max([len(pot["r"]) for pot in parser.unscreened_semilocal_potentials]),
                ),
                np.nan,
            )
            pot_y = np.full_like(pot_r, np.nan)
            for i, pot in enumerate(parser.unscreened_semilocal_potentials):
                pot_r[i, : len(pot["r"])] = pot["r"]
                pot_y[i, : len(pot["v_sl"])] = pot["v_sl"]
            array.set_array("r", pot_r)
            array.set_array("v_sl", pot_y)
            outputs["unscreened_semilocal_potentials"] = array
        if parser.wavefunctions:
            array = ArrayData()
            for key in set(parser.wavefunctions[0].keys()) - set(["r", "wfn_ae", "wfn_ps"]):
                array.set_array(
                    key,
                    np.array([wf[key] for wf in parser.wavefunctions]),
                )
            wfn_r = np.full(
                (
                    len(parser.wavefunctions),
                    max([len(wf["r"]) for wf in parser.wavefunctions]),
                ),
                np.nan,
            )
            wfn_y_ae = np.full_like(wfn_r, np.nan)
            wfn_y_ps = np.full_like(wfn_r, np.nan)
            for i, wf in enumerate(parser.wavefunctions):
                wfn_r[i, : len(wf["r"])] = wf["r"]
                wfn_y_ae[i, : len(wf["wfn_ae"])] = wf["wfn_ae"]
                wfn_y_ps[i, : len(wf["wfn_ps"])] = wf["wfn_ps"]
            array.set_array("r", wfn_r)
            array.set_array("wfn_ae", wfn_y_ae)
            array.set_array("wfn_ps", wfn_y_ps)
            outputs["wavefunctions"] = array
        if parser.vkb_projectors:
            array = ArrayData()
            for key in set(parser.vkb_projectors[0].keys()) - set(["r", "proj"]):
                array.set_array(
                    key,
                    np.array([proj[key] for proj in parser.vkb_projectors]),
                )
            proj_r = np.full(
                (
                    len(parser.vkb_projectors),
                    max([len(proj["r"]) for proj in parser.vkb_projectors]),
                ),
                np.nan,
            )
            proj_y = np.full_like(proj_r, np.nan)
            for i, proj in enumerate(parser.vkb_projectors):
                proj_r[i, : len(proj["r"])] = proj["r"]
                proj_y[i, : len(proj["proj"])] = proj["proj"]
            array.set_array("r", proj_r)
            array.set_array("proj", proj_y)
            outputs["vkb_projectors"] = array
        if parser.convergence_profiles:
            array = ArrayData()
            for key in set(parser.convergence_profiles[0].keys()) - set(["eresid", "ecut"]):
                array.set_array(
                    key,
                    np.array([cp[key] for cp in parser.convergence_profiles]),
                )
            conv_eresid = np.full(
                (
                    len(parser.convergence_profiles),
                    max([len(cp["eresid"]) for cp in parser.convergence_profiles]),
                ),
                np.nan,
            )
            conv_ecut = np.full_like(conv_eresid, np.nan)
            for i, cp in enumerate(parser.convergence_profiles):
                conv_eresid[i, : len(cp["eresid"])] = cp["eresid"]
                conv_ecut[i, : len(cp["ecut"])] = cp["ecut"]
            array.set_array("eresid", conv_eresid)
            array.set_array("ecut", conv_ecut)
            outputs["convergence_profiles"] = array
        if parser.log_derivatives:
            array = ArrayData()
            for key in set(parser.log_derivatives[0].keys()) - set(["e", "log_deriv_ae", "log_deriv_ps"]):
                array.set_array(
                    key,
                    np.array([ld[key] for ld in parser.log_derivatives]),
                )
            ld_e = np.full(
                (
                    len(parser.log_derivatives),
                    max([len(ld["e"]) for ld in parser.log_derivatives]),
                ),
                np.nan,
            )
            ld_y_ae = np.full_like(ld_e, np.nan)
            ld_y_ps = np.full_like(ld_e, np.nan)
            for i, ld in enumerate(parser.log_derivatives):
                ld_e[i, : len(ld["e"])] = ld["e"]
                ld_y_ae[i, : len(ld["log_deriv_ae"])] = ld["log_deriv_ae"]
                ld_y_ps[i, : len(ld["log_deriv_ps"])] = ld["log_deriv_ps"]
            array.set_array("e", ld_e)
            array.set_array("log_deriv_ae", ld_y_ae)
            array.set_array("log_deriv_ps", ld_y_ps)
            outputs["log_derivatives"] = array
        if parser.upf_string:
            upf = UpfData(
                filename=f"{self.node.uuid}.upf",
                content=io.BytesIO(parser.upf_string.encode("utf-8")),
            )
            outputs["upf"] = upf
        if parser.psp8_string:
            psp8 = Psp8Data(
                filename=f"{self.node.uuid}.psp8",
                content=io.BytesIO(parser.psp8_string.encode("utf-8")),
            )
            outputs["psp8"] = psp8
        outputs["output_parameters"] = orm.Dict(
            _finite_or_none(
                {
                    **parser.program_information,
                    **parser.teter_parameters,
                    "d2exc_rmse": parser.d2exc_rmse,
                    "ecut_recommended": parser.ecut_recommended,
                    "ghosts": parser.ghosts,
                    "num_ghosts": len(parser.ghosts),
                    "num_positive_ghosts": sum(g["eig"] > 0 for g in parser.ghosts),
                    "num_negative_ghosts": sum(g["eig"] < 0 for g in parser.ghosts),
                }
            )
        )
        return outputs

    def _report_messages(self, messages: list[dict], level: str | int = "info") -> None:
        if not isinstance(level, int):
            _level = getattr(logging, level.upper(), logging.INFO)
        else:
            _level = level
        for msg in messages:
            self.logger.log(_level, f"{msg['name']}#{msg['line_numbers'][0]}: {msg['description']}")


def _finite_or_none(value):
    """Recursively replace NaN and +/-inf floats with `None`, which the database cannot store."""
    if isinstance(value, dict):
        return {k: _finite_or_none(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_none(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value
