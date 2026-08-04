from hm_recsys import __version__
from hm_recsys.config import load_config


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_small_config_inherits_base_config() -> None:
    config = load_config("configs/data_small.yaml")
    assert config["sample"]["users"] == 5000
    assert config["resources"]["project_hard_stop_memory_gb"] == 14
    assert config["paths"]["raw_dir"] == "data/raw"
