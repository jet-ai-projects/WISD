"""Keep CPU tests available without installing the Megatron training stack."""

import importlib.util


collect_ignore = []
if importlib.util.find_spec("megatron") is None:
    collect_ignore.append("test_megatron_direct_opd_cp.py")
