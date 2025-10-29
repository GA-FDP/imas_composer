"""
OMAS validation tests for Thomson Scattering IDS.

Compare imas_composer Thomson scattering output against OMAS to verify correctness.

Key differences handled:
- OMAS uses nested structure: ods['thomson_scattering']['channel'][i]['n_e']['data']
- imas_composer uses flat arrays: data['thomson_scattering.channel.n_e.data'][i]
- OMAS uses unumpy.uarray for uncertainties
- imas_composer uses separate data and data_error_upper fields
"""

import pytest
import numpy as np
from omas import ODS, mdsvalue
from omas.omas_machine import machine_to_omas

from imas_composer.core import Requirement, RequirementStage
from imas_composer.ids.thomson_scattering import ThomsonScatteringMapper

# Use same reference shot as integration tests
REFERENCE_SHOT = 200000

# Mark all tests as requiring both MDS+ and OMAS
pytestmark = [pytest.mark.integration, pytest.mark.requires_mdsplus, pytest.mark.omas_validation]


def fetch_requirements(requirements: list[Requirement]) -> dict:
    """
    Fetch multiple requirements from MDS+ in a single query.

    Returns dict with requirement keys -> data
    Handles exceptions gracefully (stores Exception objects for failed fetches).
    """
    if not requirements:
        return {}

    # Group by (treename, shot)
    by_tree_shot = {}
    for req in requirements:
        key = (req.treename, req.shot)
        if key not in by_tree_shot:
            by_tree_shot[key] = []
        by_tree_shot[key].append(req)

    # Fetch each tree/shot separately
    raw_data = {}
    for (treename, shot), reqs in by_tree_shot.items():
        tdi_query = {req.mds_path: req.mds_path for req in reqs}

        try:
            result = mdsvalue('d3d', treename=treename, pulse=shot, TDI=tdi_query)
            tree_data = result.raw()

            for req in reqs:
                try:
                    raw_data[req.as_key()] = tree_data[req.mds_path]
                except Exception as e:
                    raw_data[req.as_key()] = e

        except Exception as e:
            for req in reqs:
                raw_data[req.as_key()] = e

    return raw_data


def resolve_requirements_for_ids_path(mapper, shot: int, ids_path: str) -> tuple[dict, dict]:
    """
    Resolve only the requirements needed for a specific IDS path.

    Returns:
        (raw_data, trace) where trace maps ids_path -> list of MDS+ paths used
    """
    raw_data = {}
    trace = {}

    # Find the spec for this IDS path
    if ids_path not in mapper.specs:
        raise ValueError(f"IDS path {ids_path} not found in mapper specs")

    spec = mapper.specs[ids_path]

    # Collect all dependencies recursively
    def collect_dependencies(path: str, visited: set) -> set:
        if path in visited:
            return set()
        visited.add(path)

        if path not in mapper.specs:
            return set()

        spec = mapper.specs[path]
        deps = set(spec.depends_on)

        for dep in spec.depends_on:
            deps.update(collect_dependencies(dep, visited))

        return deps

    all_deps = collect_dependencies(ids_path, set())
    all_deps.add(ids_path)

    # Fetch DIRECT requirements for all dependencies
    direct_reqs = []
    for dep_path in all_deps:
        dep_spec = mapper.specs[dep_path]
        if dep_spec.stage == RequirementStage.DIRECT:
            for req in dep_spec.static_requirements:
                req_with_shot = Requirement(req.mds_path, shot, req.treename)
                direct_reqs.append(req_with_shot)

    if direct_reqs:
        raw_data.update(fetch_requirements(direct_reqs))

    # Fetch DERIVED requirements in dependency order
    # Process DERIVED specs in multiple passes until all are resolved
    all_derived_reqs = []
    processed_derived = set()
    max_passes = 10  # Safety limit

    for _ in range(max_passes):
        pass_reqs = []
        made_progress = False

        for dep_path in all_deps:
            # Skip already processed DERIVED specs
            if dep_path in processed_derived:
                continue

            dep_spec = mapper.specs[dep_path]
            if dep_spec.stage == RequirementStage.DERIVED and dep_spec.derive_requirements:
                try:
                    reqs = dep_spec.derive_requirements(shot, raw_data)
                    pass_reqs.extend(reqs)
                    processed_derived.add(dep_path)
                    made_progress = True
                except (KeyError, Exception):
                    # Dependencies not yet available, will try next pass
                    continue

        if pass_reqs:
            raw_data.update(fetch_requirements(pass_reqs))
            all_derived_reqs.extend(pass_reqs)

        if not made_progress:
            break

    # Build trace: which MDS+ paths were used
    all_reqs = direct_reqs + all_derived_reqs
    mds_paths_used = [req.mds_path for req in all_reqs]
    trace[ids_path] = mds_paths_used

    return raw_data, trace


@pytest.fixture(scope='session')
def omas_thomson_data_cached():
    """Fetch Thomson data using OMAS once per test session (cached)."""
    ods = ODS()
    machine_to_omas(ods, 'd3d', REFERENCE_SHOT, 'thomson_scattering.*')
    return ods


class TestThomsonOMASValidation:
    """Compare Thomson scattering data from imas_composer against OMAS."""

    @pytest.fixture
    def ts_mapper(self):
        """Create Thomson scattering mapper."""
        return ThomsonScatteringMapper()

    @pytest.fixture
    def omas_thomson_data(self, omas_thomson_data_cached):
        """Use cached OMAS Thomson data."""
        return omas_thomson_data_cached

    def _fetch_and_synthesize(self, mapper, ids_path: str):
        """
        Fetch and synthesize a single IDS path independently.

        Returns tuple: (synthesized_value, trace_dict)
        """
        raw_data, trace = resolve_requirements_for_ids_path(mapper, REFERENCE_SHOT, ids_path)

        spec = mapper.specs[ids_path]
        if spec.stage != RequirementStage.COMPUTED or not spec.synthesize:
            raise ValueError(f"{ids_path} is not a COMPUTED stage with synthesize function")

        value = spec.synthesize(REFERENCE_SHOT, raw_data)
        return value, trace

    def test_channel_count_matches(self, omas_thomson_data, ts_mapper):
        """Verify both systems report same number of channels."""
        composer_names, trace = self._fetch_and_synthesize(ts_mapper, 'thomson_scattering.channel.name')

        omas_n_channels = len(omas_thomson_data['thomson_scattering']['channel'])
        composer_n_channels = len(composer_names)

        assert omas_n_channels == composer_n_channels, \
            f"Channel count mismatch: OMAS={omas_n_channels}, composer={composer_n_channels}\n" \
            f"MDS+ paths used: {trace['thomson_scattering.channel.name']}"

    def test_channel_names_match(self, omas_thomson_data, ts_mapper):
        """Verify channel names match."""
        composer_names, trace = self._fetch_and_synthesize(ts_mapper, 'thomson_scattering.channel.name')

        n_channels = len(omas_thomson_data['thomson_scattering']['channel'])

        for i in range(n_channels):
            omas_name = omas_thomson_data['thomson_scattering']['channel'][i]['name']
            composer_name = composer_names[i]

            assert omas_name == composer_name, \
                f"Channel {i} name mismatch: OMAS={omas_name}, composer={composer_name}\n" \
                f"MDS+ paths used: {trace['thomson_scattering.channel.name']}"

    def test_channel_identifiers_match(self, omas_thomson_data, ts_mapper):
        """Verify channel identifiers match."""
        composer_ids, trace = self._fetch_and_synthesize(ts_mapper, 'thomson_scattering.channel.identifier')

        n_channels = len(omas_thomson_data['thomson_scattering']['channel'])

        for i in range(n_channels):
            omas_id = omas_thomson_data['thomson_scattering']['channel'][i]['identifier']
            composer_id = composer_ids[i]

            assert omas_id == composer_id, \
                f"Channel {i} identifier mismatch: OMAS={omas_id}, composer={composer_id}\n" \
                f"MDS+ paths used: {trace['thomson_scattering.channel.identifier']}"

    def test_position_r_matches(self, omas_thomson_data, ts_mapper):
        """Verify R positions match."""
        composer_r, trace = self._fetch_and_synthesize(ts_mapper, 'thomson_scattering.channel.position.r')

        n_channels = len(omas_thomson_data['thomson_scattering']['channel'])

        for i in range(n_channels):
            omas_r = omas_thomson_data['thomson_scattering']['channel'][i]['position']['r']

            try:
                np.testing.assert_allclose(
                    composer_r[i], omas_r,
                    rtol=1e-10, atol=1e-12,
                    err_msg=f"Channel {i} R position mismatch"
                )
            except AssertionError as e:
                print(f"\nMDS+ paths used: {trace['thomson_scattering.channel.position.r']}")
                raise

    def test_position_z_matches(self, omas_thomson_data, ts_mapper):
        """Verify Z positions match."""
        composer_z, trace = self._fetch_and_synthesize(ts_mapper, 'thomson_scattering.channel.position.z')

        n_channels = len(omas_thomson_data['thomson_scattering']['channel'])

        for i in range(n_channels):
            omas_z = omas_thomson_data['thomson_scattering']['channel'][i]['position']['z']

            try:
                np.testing.assert_allclose(
                    composer_z[i], omas_z,
                    rtol=1e-10, atol=1e-12,
                    err_msg=f"Channel {i} Z position mismatch"
                )
            except AssertionError as e:
                print(f"\nMDS+ paths used: {trace['thomson_scattering.channel.position.z']}")
                raise

    def test_position_phi_matches(self, omas_thomson_data, ts_mapper):
        """Verify phi positions match."""
        composer_phi, trace = self._fetch_and_synthesize(ts_mapper, 'thomson_scattering.channel.position.phi')

        n_channels = len(omas_thomson_data['thomson_scattering']['channel'])

        for i in range(n_channels):
            omas_phi = omas_thomson_data['thomson_scattering']['channel'][i]['position']['phi']

            try:
                np.testing.assert_allclose(
                    composer_phi[i], omas_phi,
                    rtol=1e-10, atol=1e-12,
                    err_msg=f"Channel {i} phi position mismatch"
                )
            except AssertionError as e:
                print(f"\nMDS+ paths used: {trace['thomson_scattering.channel.position.phi']}")
                raise

    def test_n_e_time_matches(self, omas_thomson_data, ts_mapper):
        """Verify n_e time arrays match."""
        composer_time, trace = self._fetch_and_synthesize(ts_mapper, 'thomson_scattering.channel.n_e.time')

        n_channels = len(omas_thomson_data['thomson_scattering']['channel'])

        for i in range(n_channels):
            omas_time = omas_thomson_data['thomson_scattering']['channel'][i]['n_e']['time']

            # Convert awkward array element to numpy for comparison
            composer_channel_time = np.asarray(composer_time[i])

            try:
                np.testing.assert_allclose(
                    composer_channel_time, omas_time,
                    rtol=1e-10, atol=1e-12,
                    err_msg=f"Channel {i} n_e time mismatch"
                )
            except AssertionError as e:
                print(f"\nMDS+ paths used: {trace['thomson_scattering.channel.n_e.time']}")
                raise

    def test_t_e_time_matches(self, omas_thomson_data, ts_mapper):
        """Verify t_e time arrays match."""
        composer_time, trace = self._fetch_and_synthesize(ts_mapper, 'thomson_scattering.channel.t_e.time')

        n_channels = len(omas_thomson_data['thomson_scattering']['channel'])

        for i in range(n_channels):
            omas_time = omas_thomson_data['thomson_scattering']['channel'][i]['t_e']['time']

            # Convert awkward array element to numpy for comparison
            composer_channel_time = np.asarray(composer_time[i])

            try:
                np.testing.assert_allclose(
                    composer_channel_time, omas_time,
                    rtol=1e-10, atol=1e-12,
                    err_msg=f"Channel {i} t_e time mismatch"
                )
            except AssertionError as e:
                print(f"\nMDS+ paths used: {trace['thomson_scattering.channel.t_e.time']}")
                raise

    def test_n_e_data_matches(self, omas_thomson_data, ts_mapper):
        """Verify electron density data matches."""
        composer_n_e, trace = self._fetch_and_synthesize(ts_mapper, 'thomson_scattering.channel.n_e.data')

        n_channels = len(omas_thomson_data['thomson_scattering']['channel'])

        for i in range(n_channels):
            omas_n_e = omas_thomson_data['thomson_scattering']['channel'][i]['n_e']['data']
            composer_channel_data = np.asarray(composer_n_e[i])

            try:
                np.testing.assert_allclose(
                    composer_channel_data, omas_n_e,
                    rtol=1e-10, atol=1e-6,
                    err_msg=f"Channel {i} n_e data mismatch"
                )
            except AssertionError as e:
                print(f"\nMDS+ paths used: {trace['thomson_scattering.channel.n_e.data']}")
                raise

    def test_n_e_uncertainty_matches(self, omas_thomson_data, ts_mapper):
        """Verify electron density uncertainties match."""
        composer_n_e_err, trace = self._fetch_and_synthesize(
            ts_mapper, 'thomson_scattering.channel.n_e.data_error_upper'
        )

        n_channels = len(omas_thomson_data['thomson_scattering']['channel'])

        for i in range(n_channels):
            omas_n_e_err = omas_thomson_data['thomson_scattering']['channel'][i]['n_e']['data_error_upper']

            # Convert awkward array element to numpy for comparison
            composer_channel_err = np.asarray(composer_n_e_err[i])

            try:
                np.testing.assert_allclose(
                    composer_channel_err, omas_n_e_err,
                    rtol=1e-10, atol=1e-6,
                    err_msg=f"Channel {i} n_e uncertainty mismatch"
                )
            except AssertionError as e:
                print(f"\nMDS+ paths used: {trace['thomson_scattering.channel.n_e.data_error_upper']}")
                raise

    def test_t_e_data_matches(self, omas_thomson_data, ts_mapper):
        """Verify electron temperature data matches."""
        composer_t_e, trace = self._fetch_and_synthesize(ts_mapper, 'thomson_scattering.channel.t_e.data')

        n_channels = len(omas_thomson_data['thomson_scattering']['channel'])

        for i in range(n_channels):
            omas_t_e = omas_thomson_data['thomson_scattering']['channel'][i]['t_e']['data']

            # Convert awkward array element to numpy for comparison
            composer_channel_data = np.asarray(composer_t_e[i])

            try:
                np.testing.assert_allclose(
                    composer_channel_data, omas_t_e,
                    rtol=1e-10, atol=1e-6,
                    err_msg=f"Channel {i} t_e data mismatch"
                )
            except AssertionError as e:
                print(f"\nMDS+ paths used: {trace['thomson_scattering.channel.t_e.data']}")
                raise

    def test_t_e_uncertainty_matches(self, omas_thomson_data, ts_mapper):
        """Verify electron temperature uncertainties match."""
        composer_t_e_err, trace = self._fetch_and_synthesize(
            ts_mapper, 'thomson_scattering.channel.t_e.data_error_upper'
        )

        n_channels = len(omas_thomson_data['thomson_scattering']['channel'])

        for i in range(n_channels):
            omas_t_e_err = omas_thomson_data['thomson_scattering']['channel'][i]['t_e']['data_error_upper']
            # Convert awkward array element to numpy for comparison
            composer_channel_err = np.asarray(composer_t_e_err[i])

            try:
                np.testing.assert_allclose(
                    composer_channel_err, omas_t_e_err,
                    rtol=1e-10, atol=1e-6,
                    err_msg=f"Channel {i} t_e uncertainty mismatch"
                )
            except AssertionError as e:
                print(f"\nMDS+ paths used: {trace['thomson_scattering.channel.t_e.data_error_upper']}")
                raise
