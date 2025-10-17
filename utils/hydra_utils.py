from hydra import compose, core, initialize
from hydra.core.global_hydra import GlobalHydra
import hydra
import sys
import os

def setup_config(config_path):
    # Reset Hydra to avoid conflicts if already initialized
    GlobalHydra.instance().clear()
    # Initialize Hydra and load config manually
    initialize(config_path=config_path, version_base=None)  # Set path to your configs
    # Load the configuration
    cfg = compose(config_name="config")
    return cfg


def load_config(project_root, config_dir_path):
    sys.path.append(project_root)
    os.environ["PROJECT_ROOT"] = project_root
    GlobalHydra.instance().clear()
    hydra.initialize(config_path=config_dir_path, version_base=None)  # Set path to your configs
    cfg = hydra.compose(config_name="config")  # Replace with your main config file
    return cfg