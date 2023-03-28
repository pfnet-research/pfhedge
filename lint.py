import pathlib
from typing import Optional
from typing import Sequence

import pysen
from pysen import IsortSetting
from pysen.component import ComponentBase
from pysen.manifest import Manifest
from pysen.manifest import ManifestBase


def build(
    components: Sequence[ComponentBase], src_path: Optional[pathlib.Path]
) -> ManifestBase:
    isort_setting: IsortSetting = pysen.IsortSetting.default()
    isort_setting.force_single_line = True
    isort_setting.known_first_party = {"pfhedge"}

    isort = pysen.Isort(setting=isort_setting.to_black_compatible())

    others = [
        component for component in components if not isinstance(component, pysen.Isort)
    ]

    return Manifest([isort, *others])
