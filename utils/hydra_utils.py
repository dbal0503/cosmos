from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import sys
import os


def setup_config(config_path):
    """Load config from a relative config path."""
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=os.path.abspath(config_path), version_base=None)
    cfg = compose(config_name="config")
    return cfg


def load_config(project_root, config_dir_path, overrides=None):
    """Load config from an absolute config directory path."""
    sys.path.append(project_root)
    os.environ["PROJECT_ROOT"] = project_root
    GlobalHydra.instance().clear()
    abs_config_dir = os.path.abspath(config_dir_path)
    initialize_config_dir(config_dir=abs_config_dir, version_base=None)
    cfg = compose(config_name="config", overrides=overrides or [])
    # Allow post-hoc attribute mutation
    OmegaConf.set_struct(cfg, False)
    return cfg
