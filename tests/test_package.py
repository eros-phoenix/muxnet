"""Package metadata that setup.py and operators both rely on."""

import neurons
from neurons.subnet_links import SUBNET_LINKS


def test_spec_version_encodes_major_minor_patch():
    major, minor, patch = (int(part) for part in neurons.__version__.split("."))
    assert neurons.__spec_version__ == (1000 * major) + (10 * minor) + patch


def test_subnet_links_have_unique_names_and_url_fields():
    names = [entry["name"] for entry in SUBNET_LINKS]
    assert names == [entry["name"] for entry in neurons.SUBNET_LINKS]
    assert len(names) == len(set(names))
    for entry in SUBNET_LINKS:
        assert set(entry) == {"name", "url"}
        assert entry["name"].startswith("sn")
