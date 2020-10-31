import yaml
import pathlib

def load_setting():
    root_dir = pathlib.Path(__file__).parents[1]
    setting_file_path = root_dir / "setting.yaml"
    with open(str(setting_file_path), 'r') as f:
        setting = yaml.load(f, Loader=yaml.FullLoader)

    return setting
