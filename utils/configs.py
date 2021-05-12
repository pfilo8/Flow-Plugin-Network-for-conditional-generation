import yaml


def get_config(filename):
    with open(filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            print(exc)


def save_config(config, path):
    with open(path, 'w') as file:
        try:
            config = yaml.safe_dump(config, file)
            return config
        except yaml.YAMLError as exc:
            print(exc)
