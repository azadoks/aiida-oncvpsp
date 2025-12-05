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
            with retrieved.open(self.node.base.attributes.get("output_filename"), "r") as fp:
                parser = OncvpspTextParser(io=fp, check=False)
        except FileNotFoundError:
            return self.exit_codes.ERROR_OUTPUT_FILE_MISSING
        except OSError:
            return self.exit_codes.ERROR_OUTPUT_FILE_UNREADABLE
        if len(parser.errors) > 0:
            self._report_messages(parser.errors, level="error")
            return self.exit_codes.ERROR_MULTIPLE_ERRORS
        elif len(parser.errors) == 1:
            return self.exit_codes.get(parser.errors[0]["name"], self.exit_codes.ERROR_UNKNOWN_ERROR)
        self._report_messages(parser.warnings, level="warning")
        if not parser.completed:
            return self.exit_codes.ERROR_RUN_NOT_COMPLETED
        if parser.local_potential:
            xy = XyData()
            xy.set_x(parser.local_potential["r"], "Radius", "Bohr")
            xy.set_y(parser.local_potential["v_loc"], "Local Potential", "Hartree")
            self.out("local_potential", xy)
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
            self.out("charge_densities", xy)
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
            self.out("kinetic_energy_densities", xy)
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
            self.out("kinetic_energy_potentials", xy)
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
            self.out("unscreened_semilocal_potentials", array)
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
            self.out("wavefunctions", array)
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
            self.out("vkb_projectors", array)
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
            self.out("convergence_profiles", array)
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
            self.out("log_derivatives", array)
        if parser.upf_string:
            upf = UpfData(
                filename=f"{self.node.uuid}.upf",
                content=io.BytesIO(parser.upf_string.encode("utf-8")),
            )
            self.out("upf", upf)
        if parser.psp8_string:
            psp8 = Psp8Data(
                filename=f"{self.node.uuid}.psp8",
                content=io.BytesIO(parser.psp8_string.encode("utf-8")),
            )
            self.out("psp8", psp8)
        self.out(
            "output_parameters",
            orm.Dict(
                {
                    **parser.program_information,
                    **parser.teter_parameters,
                    "d2exc_rmse": parser.d2exc_rmse,
                    "ecut_recommended": parser.ecut_recommended,
                    "ghosts": parser.ghosts,
                    "num_ghosts": len(parser.ghosts),
                    "num_positive_ghosts": sum(g["sign"] == "+" for g in parser.ghosts),
                    "num_negative_ghosts": sum(g["sign"] == "-" for g in parser.ghosts),
                }
            ),
        )

    def _report_messages(self, messages: list[dict], level: str | int = "info") -> None:
        if not isinstance(level, int):
            _level = getattr(logging, level.upper(), logging.INFO)
        else:
            _level = level
        for msg in messages:
            self.logger.log(_level, f"{msg['name']}#{msg['line_numbers'][0]}: {msg['description']}")
