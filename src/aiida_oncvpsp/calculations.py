from aiida import orm
from aiida.common import datastructures
from aiida.engine import CalcJob
from aiida.plugins import DataFactory
from pyoncvpsp.io import OncvpspInput

from .exit_codes import (
    LEGACY_GENERIC_EXIT_CODES,
    RETRIEVAL_EXIT_CODES,
    STAGE_EXIT_CODES,
    STAGE_EXIT_MESSAGES,
)

ArrayData = DataFactory("core.array")
XyData = DataFactory("core.array.xy")
UpfData = DataFactory("pseudo.upf")
Psp8Data = DataFactory("pseudo.psp8")

INPUT_FORMATS = ("stdin", "dat", "toml")


def validate_input_format(value, _):
    if value not in INPUT_FORMATS:
        return f"Unsupported input format `{value}`; must be one of {INPUT_FORMATS}."


class OncvpspCalculation(CalcJob):
    """Run ONCVPSP on a set of input parameters and parse its stdout.

    Failures are reported through stage-based exit codes (see `aiida_oncvpsp.exit_codes`); the
    attributed failure details (message, angular momentum, implicated parameters) are stored in
    `output_parameters["failure"]`. All outputs which could be parsed are attached even on failure.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)
        # Inputs
        spec.input("metadata.options.parser_name", valid_type=str, default="oncvpsp.oncvpsp")
        spec.input("metadata.options.input_filestem", valid_type=str, default="oncvpsp")
        spec.input("metadata.options.input_format", valid_type=str, default="stdin", validator=validate_input_format)
        spec.input("metadata.options.output_filename", valid_type=str, default="oncvpsp.out")
        spec.input("parameters", valid_type=orm.Dict, help="Input parameters for ONCVPSP")
        # Outputs
        spec.output(
            "output_parameters",
            valid_type=orm.Dict,
            required=True,
            help="Scalar results, warnings, errors and (if the run failed) the attributed failure.",
        )
        spec.output("local_potential", valid_type=XyData, required=False)
        spec.output("charge_densities", valid_type=XyData, required=False)
        spec.output("kinetic_energy_densities", valid_type=XyData, required=False)
        spec.output("kinetic_energy_potentials", valid_type=XyData, required=False)
        spec.output("unscreened_semilocal_potentials", valid_type=ArrayData, required=False)
        spec.output("wavefunctions", valid_type=ArrayData, required=False)
        spec.output("vkb_projectors", valid_type=ArrayData, required=False)
        spec.output("convergence_profiles", valid_type=ArrayData, required=False)
        spec.output("log_derivatives", valid_type=ArrayData, required=False)
        spec.output("upf", valid_type=UpfData, required=False)
        spec.output("psp8", valid_type=Psp8Data, required=False)
        spec.default_output_node = "output_parameters"
        # Exit codes
        for status, (label, message) in RETRIEVAL_EXIT_CODES.items():
            spec.exit_code(status, label, message=message)
        for status, (label, message) in LEGACY_GENERIC_EXIT_CODES.items():
            spec.exit_code(status, label, message=message)
        for stage, (status, label) in STAGE_EXIT_CODES.items():
            spec.exit_code(status, label, message=STAGE_EXIT_MESSAGES[stage])

    def prepare_for_submission(self, folder):
        input_model = OncvpspInput(**self.inputs.parameters.get_dict())
        if self.inputs.metadata.options.input_format == "stdin":
            input_filename = self.inputs.metadata.options.input_filestem + ".in"
            input_text = input_model.as_dat_string()
            stdin_name = input_filename
            input_cmdline_params = []
        elif self.inputs.metadata.options.input_format == "dat":
            input_filename = self.inputs.metadata.options.input_filestem + ".dat"
            input_text = input_model.as_dat_string()
            stdin_name = None
            input_cmdline_params = ["-i", input_filename]
        elif self.inputs.metadata.options.input_format == "toml":
            input_filename = self.inputs.metadata.options.input_filestem + ".toml"
            input_text = input_model.as_toml_string()
            stdin_name = None
            input_cmdline_params = ["-t", input_filename]
        else:
            raise ValueError(f"Unsupported input format: {self.inputs.metadata.options.input_format}")

        with folder.open(input_filename, "w", encoding="utf-8") as handle:
            handle.write(input_text + "\n")

        codeinfo = datastructures.CodeInfo()
        codeinfo.cmdline_params = input_cmdline_params
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdin_name = stdin_name
        codeinfo.stdout_name = self.inputs.metadata.options.output_filename

        calcinfo = datastructures.CalcInfo()
        calcinfo.codes_info = [codeinfo]
        # The scheduler stderr (`_scheduler-stderr.txt`) is always retrieved by AiiDA; it carries
        # Fortran runtime errors, which the parser merges into the message analysis
        calcinfo.retrieve_list = [self.inputs.metadata.options.output_filename]

        return calcinfo
