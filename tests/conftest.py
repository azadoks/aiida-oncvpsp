"""Fixtures for the aiida-oncvpsp test suite.

Uses the pytest fixtures shipped with aiida-core (temporary profile with SQLite storage).
"""

import gzip
import io
from pathlib import Path

import pytest
from aiida import orm
from aiida.common.links import LinkType

pytest_plugins = ["aiida.tools.pytest_fixtures"]

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def read_fixture(name: str) -> str:
    """Text of the stdout fixture `name` (without the `.out.gz` suffix)."""
    with gzip.open(FIXTURE_DIR / f"{name}.out.gz", "rt", encoding="latin-1") as f:
        return f.read()


@pytest.fixture
def generate_calc_job_node(aiida_localhost):
    """Create a stored `CalcJobNode` of `OncvpspCalculation` with a retrieved folder containing the
    stdout fixture `name` (and optionally a scheduler stderr)."""

    def _generate(
        name: str,
        stderr: str | None = None,
        output_filename: str = "oncvpsp.out",
        stored_as: str | None = None,
    ) -> orm.CalcJobNode:
        node = orm.CalcJobNode(computer=aiida_localhost, process_type="aiida.calculations:oncvpsp.oncvpsp")
        node.base.attributes.set("output_filename", output_filename)
        node.store()
        retrieved = orm.FolderData()
        retrieved.base.repository.put_object_from_filelike(
            io.BytesIO(read_fixture(name).encode("latin-1")), stored_as or output_filename
        )
        if stderr is not None:
            retrieved.base.repository.put_object_from_filelike(
                io.BytesIO(stderr.encode("latin-1")), "_scheduler-stderr.txt"
            )
        retrieved.base.links.add_incoming(node, link_type=LinkType.CREATE, link_label="retrieved")
        retrieved.store()
        return node

    return _generate
