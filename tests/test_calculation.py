"""Tests of `OncvpspCalculation.prepare_for_submission`, and of the split run (`oncvpsp*_ae.x`,
`oncvpsp*_ps.x`) through the calculation and the parser."""

import io

import pytest
from aiida import orm
from aiida.plugins import ParserFactory
from conftest import read_fixture
from pyoncvpsp.io import OncvpspInput, OncvpspTextParser

from aiida_oncvpsp.calculations import AE_FILENAME, validate_inputs


@pytest.fixture
def parameters():
    # the input echoed at the top of the Si output
    return orm.Dict(OncvpspInput(**OncvpspTextParser(text=read_fixture("si_ps")).input).model_dump())


def parse(node):
    parser = ParserFactory("oncvpsp.oncvpsp")(node)
    return parser.parse_from_node(node, store_provenance=False)


def test_whole_run(prepare_calculation, parameters):
    calcinfo, folder = prepare_calculation(parameters=parameters)
    (codeinfo,) = calcinfo.codes_info
    assert codeinfo.cmdline_params == []
    assert codeinfo.stdin_name == "oncvpsp.in"
    assert calcinfo.retrieve_list == ["oncvpsp.out"]
    assert calcinfo.local_copy_list == []
    assert "oncvpsp.in" in folder.get_content_list()


def test_ae_stage(prepare_calculation, parameters):
    calcinfo, _ = prepare_calculation(parameters=parameters, metadata={"options": {"write_ae_file": True}})
    (codeinfo,) = calcinfo.codes_info
    assert codeinfo.cmdline_params == ["-a", AE_FILENAME]
    assert calcinfo.retrieve_list == ["oncvpsp.out", AE_FILENAME]


def test_ps_stage(prepare_calculation, parameters):
    ae_file = orm.SinglefileData(io.BytesIO(b"ae data"), filename="si.ae").store()
    calcinfo, _ = prepare_calculation(
        parameters=parameters, ae_file=ae_file, metadata={"options": {"input_format": "dat"}}
    )
    (codeinfo,) = calcinfo.codes_info
    assert codeinfo.cmdline_params == ["-i", "oncvpsp.dat", "-a", AE_FILENAME]
    assert calcinfo.local_copy_list == [(ae_file.uuid, "si.ae", AE_FILENAME)]
    assert calcinfo.retrieve_list == ["oncvpsp.out"]


def test_both_stages_rejected():
    ae_file = orm.SinglefileData(io.BytesIO(b"ae data"), filename="si.ae")
    message = validate_inputs({"ae_file": ae_file, "metadata": {"options": {"write_ae_file": True}}}, None)
    assert message is not None


def test_parse_ae_stage(generate_calc_job_node):
    node = generate_calc_job_node("si_ae", extra_files={AE_FILENAME: b"ae data"})
    results, calcfunction = parse(node)
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    params = results["output_parameters"].get_dict()
    assert params["stage"] == "ae"
    assert params["completed"] is True
    with results["ae_file"].open(mode="rb") as fp:
        assert fp.read() == b"ae data"
    assert "upf" not in results


def test_parse_ae_stage_without_file(generate_calc_job_node):
    results, calcfunction = parse(generate_calc_job_node("si_ae"))
    assert not calcfunction.is_finished_ok
    assert "ae_file" not in results


def test_parse_ps_stage(generate_calc_job_node):
    results, calcfunction = parse(generate_calc_job_node("si_ps"))
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    params = results["output_parameters"].get_dict()
    assert params["stage"] == "ps"
    assert params["ecut_recommended"] is not None
    assert "ae_file" not in results
    # the build of the Si fixture has no ionic tail
    assert params["ionic_tail"] is None and params["coulomb_tail"] is None
