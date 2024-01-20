import os
import sys
from pathlib import Path

from ruamel.yaml import YAML

import larch

config_file = Path(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "_config.yml",
    )
)

if len(sys.argv) >= 2:
    title = sys.argv[1]
else:
    title = larch.__version__
yaml = YAML(typ="rt")  # default, if not specfied, is 'rt' (round-trip)
content = yaml.load(config_file)
content["title"] = title
content["html"]["extra_footer"] = f"<p>Larch {title}</p>"
yaml.dump(content, config_file)
