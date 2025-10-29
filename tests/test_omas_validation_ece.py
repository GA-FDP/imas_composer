"""
OMAS validation tests for ECE IDS.

Compare imas_composer ECE output against OMAS to verify correctness.

Key differences handled:
- OMAS uses nested structure: ods['ece']['channel'][i]['t_e']['data']
- imas_composer uses flat arrays: data['ece.channel.t_e.data'][i]
- OMAS uses unumpy.uarray for uncertainties
- imas_composer uses separate data and data_error_upper fields
"""

import pytest
import numpy as np
from omas import ODS, mdsvalue
from omas.omas_machine import machine_to_omas

from imas_composer.core import Requirement, RequirementStage
from imas_composer.ids.ece import ElectronCyclotronEmissionMapper

# Use same reference shot as integration tests
REFERENCE_SHOT = 200000

# Mark all tests as requiring both MDS+ and OMAS
pytestmark = [pytest.mark.integration, pytest.mark.requires_mdsplus, pytest.mark.omas_validation]


def fetch_requirements(requirements: list[Requirement]) -> dict:
    """
    Fetch multiple requirements from MDS+ in a single query.

    Returns dict with requirement keys -> data
    """
    if not requirements:
        return {}

    # Group by treename
    by_tree = {}
    for req in requirements:
        if req.treename not in by_tree:
            by_tree[req.treename] = []
        by_tree[req.treename].append(req)

    # Fetch each tree separately
    raw_data = {}
    for treename, reqs in by_tree.items():
        tdi_query = {req.mds_path: req.mds_path for req in reqs}
        shot = reqs[0].shot

        result = mdsvalue('d3d', treename=treename, pulse=shot, TDI=tdi_query)
        tree_data = result.raw()

        for req in reqs:
            raw_data[req.as_key()] = tree_data[req.mds_path]

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
def omas_ece_data_cached():
    """Fetch ECE data using OMAS once per test session (cached)."""
    ods = ODS()
    machine_to_omas(ods, 'd3d', REFERENCE_SHOT, 'ece.*')
    return ods


class TestECEOMASValidation:
    """Compare ECE data from imas_composer against OMAS."""

    @pytest.fixture
    def ece_mapper(self):
        """Create ECE mapper."""
        return ElectronCyclotronEmissionMapper(fast_ece=False)

    @pytest.fixture
    def omas_ece_data(self, omas_ece_data_cached):
        """Use cached OMAS ECE data."""
        return omas_ece_data_cached

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

    def test_channel_count_matches(self, omas_ece_data, ece_mapper):
        """Verify both systems report same number of channels."""
        composer_names, trace = self._fetch_and_synthesize(ece_mapper, 'ece.channel.name')

        omas_n_channels = len(omas_ece_data['ece']['channel'])
        composer_n_channels = len(composer_names)

        assert omas_n_channels == composer_n_channels, \
            f"Channel count mismatch: OMAS={omas_n_channels}, composer={composer_n_channels}\n" \
            f"MDS+ paths used: {trace['ece.channel.name']}"

    def test_channel_names_match(self, omas_ece_data, ece_mapper):
        """Verify channel names match."""
        composer_names, trace = self._fetch_and_synthesize(ece_mapper, 'ece.channel.name')

        n_channels = len(omas_ece_data['ece']['channel'])

        for i in range(n_channels):
            omas_name = omas_ece_data['ece']['channel'][i]['name']
            composer_name = composer_names[i]

            assert omas_name == composer_name, \
                f"Channel {i} name mismatch: OMAS={omas_name}, composer={composer_name}\n" \
                f"MDS+ paths used: {trace['ece.channel.name']}"

    def test_channel_identifiers_match(self, omas_ece_data, ece_mapper):
        """Verify channel identifiers match."""
        composer_ids, trace = self._fetch_and_synthesize(ece_mapper, 'ece.channel.identifier')

        n_channels = len(omas_ece_data['ece']['channel'])

        for i in range(n_channels):
            omas_id = omas_ece_data['ece']['channel'][i]['identifier']
            composer_id = composer_ids[i]

            assert omas_id == composer_id, \
                f"Channel {i} identifier mismatch: OMAS={omas_id}, composer={composer_id}\n" \
                f"MDS+ paths used: {trace['ece.channel.identifier']}"

    def test_channel_time_matches(self, omas_ece_data, ece_mapper):
        """Verify time arrays match."""
        composer_time, trace = self._fetch_and_synthesize(ece_mapper, 'ece.channel.time')

        n_channels = len(omas_ece_data['ece']['channel'])

        for i in range(n_channels):
            omas_time = omas_ece_data['ece']['channel'][i]['time']

            try:
                np.testing.assert_allclose(
                    composer_time[i], omas_time,
                    rtol=1e-10, atol=1e-12,
                    err_msg=f"Channel {i} time mismatch"
                )
            except AssertionError as e:
                print(f"\nMDS+ paths used: {trace['ece.channel.time']}")
                raise

    def test_channel_frequency_matches(self, omas_ece_data, ece_mapper):
        """Verify frequency arrays match."""
        composer_freq, trace = self._fetch_and_synthesize(ece_mapper, 'ece.channel.frequency.data')

        n_channels = len(omas_ece_data['ece']['channel'])

        for i in range(n_channels):
            omas_freq = omas_ece_data['ece']['channel'][i]['frequency']['data']

            try:
                np.testing.assert_allclose(
                    composer_freq[i], omas_freq,
                    rtol=1e-10, atol=1e-6,
                    err_msg=f"Channel {i} frequency mismatch"
                )
            except AssertionError as e:
                print(f"\nMDS+ paths used: {trace['ece.channel.frequency.data']}")
                print(f"Composer shape: {composer_freq[i].shape}, OMAS shape: {omas_freq.shape}")
                print(f"Composer[0]={composer_freq[i][0]}, OMAS[0]={omas_freq[0]}")
                raise

    def test_channel_if_bandwidth_matches(self, omas_ece_data, ece_mapper):
        """Verify IF bandwidth matches."""
        composer_bw, trace = self._fetch_and_synthesize(ece_mapper, 'ece.channel.if_bandwidth')

        n_channels = len(omas_ece_data['ece']['channel'])

        for i in range(n_channels):
            omas_bw = omas_ece_data['ece']['channel'][i]['if_bandwidth']

            try:
                np.testing.assert_allclose(
                    composer_bw[i], omas_bw,
                    rtol=1e-10, atol=1e-6,
                    err_msg=f"Channel {i} IF bandwidth mismatch"
                )
            except AssertionError as e:
                print(f"\nMDS+ paths used: {trace['ece.channel.if_bandwidth']}")
                raise

    def test_t_e_data_matches(self, omas_ece_data, ece_mapper):
        """Verify electron temperature data matches."""
        composer_t_e, trace = self._fetch_and_synthesize(ece_mapper, 'ece.channel.t_e.data')

        n_channels = len(omas_ece_data['ece']['channel'])

        for i in range(n_channels):
            # OMAS stores as unumpy.uarray, extract nominal values
            omas_t_e = omas_ece_data['ece']['channel'][i]['t_e']['data']

            try:
                np.testing.assert_allclose(
                    composer_t_e[i], omas_t_e,
                    rtol=1e-10, atol=1e-6,
                    err_msg=f"Channel {i} t_e data mismatch"
                )
            except AssertionError as e:
                print(f"\nMDS+ paths used: {trace['ece.channel.t_e.data']}")
                raise

    def test_t_e_uncertainty_matches(self, omas_ece_data, ece_mapper):
        """Verify electron temperature uncertainties match."""
        composer_t_e_err, trace = self._fetch_and_synthesize(
            ece_mapper, 'ece.channel.t_e.data_error_upper'
        )

        n_channels = len(omas_ece_data['ece']['channel'])

        for i in range(n_channels):
            # OMAS stores as unumpy.uarray, extract std_devs
            omas_t_e_err = omas_ece_data['ece']['channel'][i]['t_e']['data_error_upper']

            try:
                np.testing.assert_allclose(
                    composer_t_e_err[i], omas_t_e_err,
                    rtol=1e-10, atol=1e-6,
                    err_msg=f"Channel {i} t_e uncertainty mismatch"
                )
            except AssertionError as e:
                print(f"\nMDS+ paths used: {trace['ece.channel.t_e.data_error_upper']}")
                raise

    def test_line_of_sight_first_point(self, omas_ece_data, ece_mapper):
        """Verify line of sight first point matches."""
        composer_r, trace_r = self._fetch_and_synthesize(ece_mapper, 'ece.line_of_sight.first_point.r')
        composer_phi, trace_phi = self._fetch_and_synthesize(ece_mapper, 'ece.line_of_sight.first_point.phi')
        composer_z, trace_z = self._fetch_and_synthesize(ece_mapper, 'ece.line_of_sight.first_point.z')

        omas_r = omas_ece_data['ece']['line_of_sight']['first_point']['r']
        omas_phi = omas_ece_data['ece']['line_of_sight']['first_point']['phi']
        omas_z = omas_ece_data['ece']['line_of_sight']['first_point']['z']

        try:
            np.testing.assert_allclose(composer_r, omas_r, rtol=1e-10)
            np.testing.assert_allclose(composer_phi, omas_phi, rtol=1e-10)
            np.testing.assert_allclose(composer_z, omas_z, rtol=1e-10)
        except AssertionError as e:
            print(f"\nR MDS+ paths: {trace_r['ece.line_of_sight.first_point.r']}")
            print(f"Phi MDS+ paths: {trace_phi['ece.line_of_sight.first_point.phi']}")
            print(f"Z MDS+ paths: {trace_z['ece.line_of_sight.first_point.z']}")
            raise

    def test_line_of_sight_second_point(self, omas_ece_data, ece_mapper):
        """Verify line of sight second point matches."""
        composer_r, trace_r = self._fetch_and_synthesize(ece_mapper, 'ece.line_of_sight.second_point.r')
        composer_phi, trace_phi = self._fetch_and_synthesize(ece_mapper, 'ece.line_of_sight.second_point.phi')
        composer_z, trace_z = self._fetch_and_synthesize(ece_mapper, 'ece.line_of_sight.second_point.z')

        omas_r = omas_ece_data['ece']['line_of_sight']['second_point']['r']
        omas_phi = omas_ece_data['ece']['line_of_sight']['second_point']['phi']
        omas_z = omas_ece_data['ece']['line_of_sight']['second_point']['z']

        try:
            np.testing.assert_allclose(composer_r, omas_r, rtol=1e-10)
            np.testing.assert_allclose(composer_phi, omas_phi, rtol=1e-10)
            np.testing.assert_allclose(composer_z, omas_z, rtol=1e-10)
        except AssertionError as e:
            print(f"\nR MDS+ paths: {trace_r['ece.line_of_sight.second_point.r']}")
            print(f"Phi MDS+ paths: {trace_phi['ece.line_of_sight.second_point.phi']}")
            print(f"Z MDS+ paths: {trace_z['ece.line_of_sight.second_point.z']}")
            raise
