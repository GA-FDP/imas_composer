"""
Integration tests for Thomson scattering mapper with real DIII-D MDS+ data.

These tests verify the complete pipeline:
1. Requirement formulation (DIRECT → DERIVED)
2. MDS+ data fetching via omas.mdsvalue
3. Data synthesis (COMPUTED)

Tests use a fixed reference shot to ensure data availability.
"""

import pytest
import numpy as np
from omas import mdsvalue

from imas_composer.core import RequirementStage, Requirement
from imas_composer.ids.thomson_scattering import ThomsonScatteringMapper


# Reference shot with known good Thomson scattering data
REFERENCE_SHOT = 200000

# Mark all tests in this module as integration tests
pytestmark = [pytest.mark.integration, pytest.mark.requires_mdsplus]


@pytest.fixture
def ts_mapper():
    """Create Thomson scattering mapper instance."""
    return ThomsonScatteringMapper()


def fetch_requirement(req: Requirement) -> dict:
    """
    Fetch a single requirement from MDS+.

    Returns dict with requirement key -> data
    """
    result = mdsvalue('d3d', treename=req.treename, pulse=req.shot,
                      TDI={req.mds_path: req.mds_path})
    return result.raw()


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
        # Build TDI query dict
        tdi_query = {req.mds_path: req.mds_path for req in reqs}

        try:
            result = mdsvalue('d3d', treename=treename, pulse=shot, TDI=tdi_query)
            tree_data = result.raw()

            # Store with requirement keys
            for req in reqs:
                try:
                    raw_data[req.as_key()] = tree_data[req.mds_path]
                except Exception as e:
                    # Store exception for this specific path
                    raw_data[req.as_key()] = e

        except Exception as e:
            # If entire query failed, store exception for all reqs
            for req in reqs:
                raw_data[req.as_key()] = e

    return raw_data


class TestThomsonDirectRequirements:
    """Test DIRECT stage requirement fetching."""

    def test_fetch_calib_nums(self, ts_mapper):
        """Test fetching calibration numbers for reference shot."""
        spec = ts_mapper.specs["thomson_scattering._calib_nums"]

        # Update requirement with reference shot
        req = spec.static_requirements[0]
        req_with_shot = Requirement(req.mds_path, REFERENCE_SHOT, req.treename)

        # Fetch
        raw_data = fetch_requirement(req_with_shot)

        # Verify
        assert req.mds_path in raw_data
        calib_nums = raw_data[req.mds_path]

        # Should be an array of calibration set numbers
        assert isinstance(calib_nums, np.ndarray)
        assert len(calib_nums) > 0

        # First element is the calibration set ID
        cal_set = calib_nums[0]
        assert isinstance(cal_set, (int, np.integer))

    def test_only_one_direct_requirement(self, ts_mapper):
        """Verify Thomson has only one DIRECT requirement."""
        direct_specs = [
            path for path, spec in ts_mapper.specs.items()
            if spec.stage == RequirementStage.DIRECT
        ]

        assert len(direct_specs) == 1
        assert direct_specs[0] == "thomson_scattering._calib_nums"


class TestThomsonDerivedRequirements:
    """Test DERIVED stage requirement formulation and fetching."""

    @pytest.fixture
    def calib_data(self, ts_mapper):
        """Fetch calibration numbers as prerequisite."""
        spec = ts_mapper.specs["thomson_scattering._calib_nums"]
        req = spec.static_requirements[0]
        req_with_shot = Requirement(req.mds_path, REFERENCE_SHOT, req.treename)

        raw_data = fetch_requirement(req_with_shot)
        # Store with proper key
        raw_data[req_with_shot.as_key()] = raw_data[req.mds_path]

        return raw_data

    def test_derive_hwmap_requirements(self, ts_mapper, calib_data):
        """Test deriving hardware map requirements from calibration numbers."""
        spec = ts_mapper.specs["thomson_scattering._hwmap"]

        # Derive requirements
        derived_reqs = spec.derive_requirements(REFERENCE_SHOT, calib_data)

        # Should have one requirement per system
        assert len(derived_reqs) == len(ts_mapper.SYSTEMS)

        # Each should be from TSCAL tree with hwmapints
        for req in derived_reqs:
            assert 'hwmapints' in req.mds_path.lower()
            assert req.treename == 'TSCAL'
            # Shot should be calibration set number, not reference shot
            assert req.shot != REFERENCE_SHOT

    def test_fetch_hwmap_data(self, ts_mapper, calib_data):
        """Test fetching hardware map data."""
        spec = ts_mapper.specs["thomson_scattering._hwmap"]

        # Derive requirements
        hwmap_reqs = spec.derive_requirements(REFERENCE_SHOT, calib_data)

        # Fetch
        hwmap_data = fetch_requirements(hwmap_reqs)

        # Verify all systems attempted
        assert len(hwmap_data) == len(ts_mapper.SYSTEMS)

        # At least some systems should have valid data
        valid_systems = [
            key for key, value in hwmap_data.items()
            if not isinstance(value, Exception)
        ]
        assert len(valid_systems) > 0, "At least one system should have hardware map"

        # Verify structure of valid hardware maps
        for key in valid_systems:
            hwmap = hwmap_data[key]
            assert isinstance(hwmap, np.ndarray)

    def test_derive_system_position_requirements(self, ts_mapper, calib_data):
        """Test deriving position requirements for all systems."""
        spec = ts_mapper.specs["thomson_scattering._system_availability"]

        # Derive requirements
        derived_reqs = spec.derive_requirements(REFERENCE_SHOT, calib_data)

        # Should request R, Z, PHI for each system
        expected_count = len(ts_mapper.SYSTEMS) * 3  # 3 coordinates per system
        assert len(derived_reqs) == expected_count

        # Verify structure
        for coord in ['R', 'Z', 'PHI']:
            coord_reqs = [req for req in derived_reqs if ':' + coord in req.mds_path]
            assert len(coord_reqs) == len(ts_mapper.SYSTEMS), \
                f"Should request {coord} for each system"

        # All should be from ELECTRONS tree at reference shot
        for req in derived_reqs:
            assert req.treename == 'ELECTRONS'
            assert req.shot == REFERENCE_SHOT

    def test_fetch_system_position_data(self, ts_mapper, calib_data):
        """Test fetching position data for all systems."""
        spec = ts_mapper.specs["thomson_scattering._system_availability"]

        # Derive and fetch
        pos_reqs = spec.derive_requirements(REFERENCE_SHOT, calib_data)
        pos_data = fetch_requirements(pos_reqs)

        # Check which systems are active
        active_systems = []
        for system in ts_mapper.SYSTEMS:
            r_req = Requirement(f'.TS.BLESSED.{system}:R', REFERENCE_SHOT, 'ELECTRONS')
            r_key = r_req.as_key()

            if r_key in pos_data and not isinstance(pos_data[r_key], Exception):
                r_data = pos_data[r_key]
                if len(r_data) > 0:
                    active_systems.append(system)

        # At least one system should be active for this shot
        assert len(active_systems) > 0, \
            f"At least one Thomson system should be active for shot {REFERENCE_SHOT}"

        # Verify structure for active systems
        for system in active_systems:
            for coord in ['R', 'Z', 'PHI']:
                req = Requirement(f'.TS.BLESSED.{system}:{coord}', REFERENCE_SHOT, 'ELECTRONS')
                key = req.as_key()

                assert key in pos_data
                assert not isinstance(pos_data[key], Exception), \
                    f"{system}:{coord} should be available"

                coord_data = pos_data[key]
                assert isinstance(coord_data, np.ndarray)
                assert len(coord_data) > 0, f"{system}:{coord} should have channels"

    def test_derive_time_requirements(self, ts_mapper, calib_data):
        """Test deriving time base requirements for active systems."""
        # First get system availability
        avail_spec = ts_mapper.specs["thomson_scattering._system_availability"]
        pos_reqs = avail_spec.derive_requirements(REFERENCE_SHOT, calib_data)
        pos_data = fetch_requirements(pos_reqs)

        # Merge into calib_data
        all_data = {**calib_data, **pos_data}

        # Now derive time requirements
        time_spec = ts_mapper.specs["thomson_scattering.channel.n_e.time"]
        time_reqs = time_spec.derive_requirements(REFERENCE_SHOT, all_data)

        # Should only request time for active systems
        assert len(time_reqs) <= len(ts_mapper.SYSTEMS)
        assert len(time_reqs) > 0, "At least one system should be active"

        # All should be TIME requests
        for req in time_reqs:
            assert 'TIME' in req.mds_path
            assert req.treename == 'ELECTRONS'
            assert req.shot == REFERENCE_SHOT

    def test_derive_measurement_requirements(self, ts_mapper, calib_data):
        """Test deriving measurement data requirements (DENSITY and errors)."""
        # First get system availability
        avail_spec = ts_mapper.specs["thomson_scattering._system_availability"]
        pos_reqs = avail_spec.derive_requirements(REFERENCE_SHOT, calib_data)
        pos_data = fetch_requirements(pos_reqs)

        # Merge into calib_data
        all_data = {**calib_data, **pos_data}

        # Derive density measurement requirements
        ne_spec = ts_mapper.specs["thomson_scattering.channel.n_e.data"]
        ne_reqs = ne_spec.derive_requirements(REFERENCE_SHOT, all_data)

        # Should request data + error for each active system
        # So requirements should be even number (pairs)
        assert len(ne_reqs) % 2 == 0, "Should have pairs of (data, error) requests"
        assert len(ne_reqs) > 0, "Should have at least one system"

        # Check for DENSITY and DENSITY_E pairs
        data_reqs = [req for req in ne_reqs if not req.mds_path.endswith('_E')]
        error_reqs = [req for req in ne_reqs if req.mds_path.endswith('_E')]

        assert len(data_reqs) == len(error_reqs), \
            "Should have equal number of data and error requests"

        # Verify structure
        for req in ne_reqs:
            assert 'DENSITY' in req.mds_path
            assert req.treename == 'ELECTRONS'
            assert req.shot == REFERENCE_SHOT


class TestThomsonSynthesis:
    """Test COMPUTED stage synthesis functions."""

    @pytest.fixture
    def fetched_data(self, ts_mapper):
        """Fetch all required data for synthesis tests."""
        # Fetch calibration numbers
        calib_spec = ts_mapper.specs["thomson_scattering._calib_nums"]
        calib_req = Requirement(
            calib_spec.static_requirements[0].mds_path,
            REFERENCE_SHOT,
            calib_spec.static_requirements[0].treename
        )
        raw_data = fetch_requirement(calib_req)
        raw_data[calib_req.as_key()] = raw_data[calib_req.mds_path]

        # Fetch hardware maps
        hwmap_spec = ts_mapper.specs["thomson_scattering._hwmap"]
        hwmap_reqs = hwmap_spec.derive_requirements(REFERENCE_SHOT, raw_data)
        hwmap_data = fetch_requirements(hwmap_reqs)
        raw_data.update(hwmap_data)

        # Fetch system positions
        avail_spec = ts_mapper.specs["thomson_scattering._system_availability"]
        pos_reqs = avail_spec.derive_requirements(REFERENCE_SHOT, raw_data)
        pos_data = fetch_requirements(pos_reqs)
        raw_data.update(pos_data)

        return raw_data

    def test_synthesize_channel_name(self, ts_mapper, fetched_data):
        """Test synthesis of channel names."""
        spec = ts_mapper.specs["thomson_scattering.channel.name"]

        result = spec.synthesize(REFERENCE_SHOT, fetched_data)

        # Verify output
        assert isinstance(result, np.ndarray)
        assert len(result) > 0, "Should have at least one channel"

        # All names should start with TS_
        for name in result:
            assert isinstance(name, str)
            assert name.startswith('TS_'), f"Name should start with 'TS_': {name}"

            # Should contain system name and lens/channel info
            assert any(sys.lower() in name.lower() for sys in ts_mapper.SYSTEMS), \
                f"Name should contain system: {name}"

    def test_synthesize_channel_identifier(self, ts_mapper, fetched_data):
        """Test synthesis of channel identifiers."""
        spec = ts_mapper.specs["thomson_scattering.channel.identifier"]

        result = spec.synthesize(REFERENCE_SHOT, fetched_data)

        # Verify output
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

        # Identifiers should be short codes like T01, D05, C12
        for identifier in result:
            assert isinstance(identifier, str)
            assert len(identifier) == 3, f"Identifier should be 3 chars: {identifier}"
            assert identifier[0] in ['T', 'D', 'C'], \
                f"First char should be system letter: {identifier}"

    def test_synthesize_position_r(self, ts_mapper, fetched_data):
        """Test synthesis of R coordinates."""
        spec = ts_mapper.specs["thomson_scattering.channel.position.r"]

        result = spec.synthesize(REFERENCE_SHOT, fetched_data)

        # Verify output
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

        # R should be in reasonable range for DIII-D
        assert np.all(result > 0.5), "R should be > 0.5m"
        assert np.all(result < 3.0), "R should be < 3.0m"

    def test_synthesize_position_z(self, ts_mapper, fetched_data):
        """Test synthesis of Z coordinates."""
        spec = ts_mapper.specs["thomson_scattering.channel.position.z"]

        result = spec.synthesize(REFERENCE_SHOT, fetched_data)

        # Verify output
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

        # Z should be in reasonable range for DIII-D
        assert np.all(result > -2.0), "Z should be > -2.0m"
        assert np.all(result < 2.0), "Z should be < 2.0m"

    def test_synthesize_position_phi(self, ts_mapper, fetched_data):
        """Test synthesis of phi coordinates."""
        spec = ts_mapper.specs["thomson_scattering.channel.position.phi"]

        result = spec.synthesize(REFERENCE_SHOT, fetched_data)

        # Verify output
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

        # Phi should be in radians
        assert np.all(result >= -np.pi), "Phi should be >= -pi"
        assert np.all(result <= np.pi), "Phi should be <= pi"

        # Should have been converted from degrees with sign flip
        # (original is positive, IMAS convention is negative)
        assert np.all(result <= 0), "Phi should be negative (IMAS convention)"

    def test_position_arrays_same_length(self, ts_mapper, fetched_data):
        """Verify R, Z, Phi arrays have same length."""
        r_spec = ts_mapper.specs["thomson_scattering.channel.position.r"]
        z_spec = ts_mapper.specs["thomson_scattering.channel.position.z"]
        phi_spec = ts_mapper.specs["thomson_scattering.channel.position.phi"]

        r = r_spec.synthesize(REFERENCE_SHOT, fetched_data)
        z = z_spec.synthesize(REFERENCE_SHOT, fetched_data)
        phi = phi_spec.synthesize(REFERENCE_SHOT, fetched_data)

        assert len(r) == len(z) == len(phi), \
            "Position coordinates should have same length"

    def test_channel_metadata_consistent_length(self, ts_mapper, fetched_data):
        """Verify all channel metadata arrays have consistent length."""
        name_spec = ts_mapper.specs["thomson_scattering.channel.name"]
        id_spec = ts_mapper.specs["thomson_scattering.channel.identifier"]
        r_spec = ts_mapper.specs["thomson_scattering.channel.position.r"]

        names = name_spec.synthesize(REFERENCE_SHOT, fetched_data)
        identifiers = id_spec.synthesize(REFERENCE_SHOT, fetched_data)
        r = r_spec.synthesize(REFERENCE_SHOT, fetched_data)

        n_channels = len(names)
        assert len(identifiers) == n_channels, \
            "Identifiers should match channel count"
        assert len(r) == n_channels, \
            "Position should match channel count"


class TestThomsonSystemHandling:
    """Test handling of multiple Thomson systems and inactive systems."""

    @pytest.fixture
    def system_data(self, ts_mapper):
        """Fetch data needed to check system availability."""
        # Fetch calibration numbers
        calib_spec = ts_mapper.specs["thomson_scattering._calib_nums"]
        calib_req = Requirement(
            calib_spec.static_requirements[0].mds_path,
            REFERENCE_SHOT,
            calib_spec.static_requirements[0].treename
        )
        raw_data = fetch_requirement(calib_req)
        raw_data[calib_req.as_key()] = raw_data[calib_req.mds_path]

        # Fetch system positions
        avail_spec = ts_mapper.specs["thomson_scattering._system_availability"]
        pos_reqs = avail_spec.derive_requirements(REFERENCE_SHOT, raw_data)
        pos_data = fetch_requirements(pos_reqs)
        raw_data.update(pos_data)

        return raw_data

    def test_identify_active_systems(self, ts_mapper, system_data):
        """Test identification of which systems are active."""
        active_systems = []
        inactive_systems = []

        for system in ts_mapper.SYSTEMS:
            is_active = ts_mapper._is_system_active(system, REFERENCE_SHOT, system_data)

            if is_active:
                active_systems.append(system)
            else:
                inactive_systems.append(system)

        # Should have at least one active system
        assert len(active_systems) > 0, \
            f"Should have at least one active system for shot {REFERENCE_SHOT}"

        print(f"Active systems: {active_systems}")
        print(f"Inactive systems: {inactive_systems}")

    def test_system_channel_counts(self, ts_mapper, system_data):
        """Test getting channel counts for active systems."""
        for system in ts_mapper.SYSTEMS:
            is_active = ts_mapper._is_system_active(system, REFERENCE_SHOT, system_data)

            if is_active:
                n_channels = ts_mapper._get_system_channel_count(
                    system, REFERENCE_SHOT, system_data
                )
                assert n_channels > 0, f"{system} should have channels"
                assert n_channels < 200, f"{system} channel count sanity check"

    def test_systems_constant(self):
        """Verify SYSTEMS constant is defined correctly."""
        expected_systems = ['TANGENTIAL', 'DIVERTOR', 'CORE']
        mapper = ThomsonScatteringMapper()

        assert mapper.SYSTEMS == expected_systems, \
            "SYSTEMS should contain TANGENTIAL, DIVERTOR, CORE"


class TestThomsonDependencyChain:
    """Test the complete dependency chain through all stages."""

    def test_full_dependency_chain(self, ts_mapper):
        """Test complete dependency resolution from calib_nums to final data."""
        # Stage 1: DIRECT - Fetch calibration numbers
        calib_spec = ts_mapper.specs["thomson_scattering._calib_nums"]
        calib_req = Requirement(
            calib_spec.static_requirements[0].mds_path,
            REFERENCE_SHOT,
            calib_spec.static_requirements[0].treename
        )
        raw_data = fetch_requirement(calib_req)
        raw_data[calib_req.as_key()] = raw_data[calib_req.mds_path]

        # Stage 2: DERIVED - Hardware maps
        hwmap_spec = ts_mapper.specs["thomson_scattering._hwmap"]
        hwmap_reqs = hwmap_spec.derive_requirements(REFERENCE_SHOT, raw_data)
        assert len(hwmap_reqs) > 0, "Should derive hardware map requirements"

        hwmap_data = fetch_requirements(hwmap_reqs)
        raw_data.update(hwmap_data)

        # Stage 2: DERIVED - System positions
        avail_spec = ts_mapper.specs["thomson_scattering._system_availability"]
        pos_reqs = avail_spec.derive_requirements(REFERENCE_SHOT, raw_data)
        assert len(pos_reqs) > 0, "Should derive position requirements"

        pos_data = fetch_requirements(pos_reqs)
        raw_data.update(pos_data)

        # Stage 3: COMPUTED - Channel names
        name_spec = ts_mapper.specs["thomson_scattering.channel.name"]
        names = name_spec.synthesize(REFERENCE_SHOT, raw_data)
        assert len(names) > 0, "Should synthesize channel names"

        # Stage 3: COMPUTED - Positions
        r_spec = ts_mapper.specs["thomson_scattering.channel.position.r"]
        r = r_spec.synthesize(REFERENCE_SHOT, raw_data)
        assert len(r) == len(names), "Position should match channel count"

        print(f"Successfully resolved {len(names)} channels")
