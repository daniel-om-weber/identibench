import pytest


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", default=False, help="run slow tests")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--slow"):
        skip = pytest.mark.skip(reason="need --slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip)
