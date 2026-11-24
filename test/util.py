from importlib import resources
from pathlib import Path

import pycatkin


def resource_filename(pkg, file):
    with resources.as_file(resources.files(pkg)) as path:
        path = path.parent
    return path / file


if __name__ == "__main__":
    # print(parse("pycatkin"))
    path = resource_filename(pycatkin, 'examples/DMTM/input.json')
    print(path)
    print(Path(path).is_file())