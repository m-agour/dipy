import warnings

from dipy.utils.optpkg import optional_package
from dipy.viz.horizon.app import horizon
from dipy.viz.skyline.app import Skyline as skyline

fury_pckg_msg = (
    "You do not have FURY installed. Some visualization functions might not "
    "work for you. Please install or upgrade FURY using pip install -U fury. "
    "For detailed installation instructions visit: https://fury.gl/"
)

fury, has_fury, _ = optional_package(
    "fury", trip_msg=fury_pckg_msg, min_version="0.10.0", max_version="1.0.0"
)


if has_fury:
    from fury import actor, colormap, lib, shaders, ui, utils, window

    try:
        from fury import interactor
    except ImportError:
        interactor = None

    try:
        from fury.data import (
            DATA_DIR as FURY_DATA_DIR,
            fetch_viz_icons,
            read_viz_icons,
        )
    except ImportError:
        FURY_DATA_DIR = None
        fetch_viz_icons = None
        read_viz_icons = None

    from dipy.viz import sh_glyph  # noqa: F401

else:
    warnings.warn(fury_pckg_msg, stacklevel=2)

_, has_mpl, _ = optional_package(
    "matplotlib",
    trip_msg="You do not have Matplotlib installed. Some visualization "
    "functions might not work for you. For installation instructions, "
    "please visit: https://matplotlib.org/",
)

if has_mpl:
    from . import projections
