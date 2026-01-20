from aiida import orm
from aiida.common import datastructures
from aiida.engine import CalcJob
from aiida.plugins import DataFactory
from pyoncvpsp.io import ERRORS as ONCVPSP_ERRORS
from pyoncvpsp.io import OncvpspInput

ArrayData = DataFactory("core.array")
XyData = DataFactory("core.array.xy")
UpfData = DataFactory("pseudo.upf")
Psp8Data = DataFactory("pseudo.psp8")


class OncvpspCalculation(CalcJob):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        # Inputs
        spec.input("metadata.options.parser_name", valid_type=str, default="oncvpsp.oncvpsp")
        spec.input("metadata.options.input_filestem", valid_type=str, default="oncvpsp")
        spec.input("metadata.options.input_format", valid_type=str, default="stdin")
        spec.input("metadata.options.output_filename", valid_type=str, default="oncvpsp.out")
        spec.input("metadata.options.with_hdf5", valid_type=bool, default=False)
        spec.input("parameters", valid_type=orm.Dict, help="Input parameters for ONCVPSP")
        # Outputs
        spec.output("output_parameters", valid_type=orm.Dict, required=True)
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
        spec.exit_code(
            301,
            "ERROR_NO_RETRIEVED_FOLDER",
            message="The retrieved folder could not be found.",
        )
        spec.exit_code(302, "ERROR_OUTPUT_FILE_MISSING", message="The output file is missing.")
        spec.exit_code(
            303,
            "ERROR_OUTPUT_FILE_UNREADABLE",
            message="The output file could not be read.",
        )
        spec.exit_code(
            501,
            "ERROR_RUN_NOT_COMPLETED",
            message="The ONCVPSP run did not complete successfully, but no other error was found.",
        )
        spec.exit_code(502, "ERROR_UNKNOWN_ERROR", message="An error unknown error occurred.")
        for i, error in enumerate(ONCVPSP_ERRORS):
            spec.exit_code(500 + i + 3, error["name"].upper(), message=error["description"])

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
            handle.write(input_text)

        hdf5_cmdline_params = []
        hdf5_retrieve_list = []
        if self.inputs.metadata.options.with_hdf5:
            hdf5_cmdline_params = ["-h5", self.inputs.metadata.options.hdf5_filename]
            hdf5_retrieve_list = [self.inputs.metadata.options.hdf5_filename]

        codeinfo = datastructures.CodeInfo()
        codeinfo.cmdline_params = input_cmdline_params + hdf5_cmdline_params
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdin_name = stdin_name
        codeinfo.stdout_name = self.inputs.metadata.options.output_filename

        calcinfo = datastructures.CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = [self.inputs.metadata.options.output_filename] + hdf5_retrieve_list

        return calcinfo
