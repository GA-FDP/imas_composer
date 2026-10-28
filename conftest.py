"""Applies the FDP environment before pytest imports anything MDSplus-backed.

This is the in-process equivalent of `fdp run pytest`, so the test suite fetches
MDSplus trees and PTDATA from the Pelican origin without a CLI wrapper.

Order here is load-bearing, twice over:

1. BEARER_TOKEN must be set before `import toksearch_d3d`.  That import pulls in
   fdp.environment, which loads libXrdCl, whose Pelican plugin reads the token at
   load time.  ptdata sets BEARER_TOKEN from ~/.fdp/token too, but only when
   ptdata itself is imported -- several lines too late, leaving every pelican://
   read unauthenticated (MDSplus fails with TreeFOPENR, PTDATA silently falls
   back to ptserver and dies on `getservbyname failed for task 'PTSERVER'`).
2. `import toksearch_d3d` must precede any other MDSplus import.  It sets
   XRD_PLUGINCONFDIR before libXrdCl loads; `from omas import ODS` in
   tests/conftest.py would otherwise load it first and never register the plugin.
"""

import os
from pathlib import Path

if "BEARER_TOKEN" not in os.environ:
    _token_file = Path.home() / ".fdp" / "token"
    if _token_file.exists():
        os.environ["BEARER_TOKEN"] = _token_file.read_text().strip()

import toksearch_d3d  # noqa: F401  isort:skip  -- must follow the token, precede omas

from fdp.environment import _resolve_device_handle, resolve_bearer_token
from omas.utilities.omas_mds import set_default_mds_backend
from toksearch_d3d import setup_environment

setup_environment()

if resolve_bearer_token(_resolve_device_handle("d3d")) is None:
    raise RuntimeError(
        "No valid FDP bearer token; Pelican fetches will fail. Run `fdp login`."
    )

# Route the OMAS reference implementation through toksearch as well, so the
# oracle reads from the Pelican origin rather than the atlas thin-client
# server.  The origin is reachable from GitHub CI and tolerates the parallel
# access that -n auto needs; atlas is neither.
set_default_mds_backend("toksearch")
