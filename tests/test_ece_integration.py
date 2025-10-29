"""
Integration tests for ECE mapper with real DIII-D MDS+ data.

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
from imas_composer.ids.ece import ElectronCyclotronEmissionMapper


# Reference shot with known good ECE data
REFERENCE_SHOT = 200000

# Mark all tests in this module as integration tests
pytestmark = [pytest.mark.integration, pytest.mark.requires_mdsplus]


@pytest.fixture
def ece_mapper():
    """Create standard ECE mapper instance."""
    return ElectronCyclotronEmissionMapper(fast_ece=False)


@pytest.fixture
def ece_mapper_fast():
    """Create fast ECE mapper instance."""
    return ElectronCyclotronEmissionMapper(fast_ece=True)


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
        # Build TDI query dict
        tdi_query = {req.mds_path: req.mds_path for req in reqs}

        # Use first req's shot (should all be the same)
        shot = reqs[0].shot

        result = mdsvalue('d3d', treename=treename, pulse=shot, TDI=tdi_query)
        tree_data = result.raw()

        # Store with requirement keys
        for req in reqs:
            raw_data[req.as_key()] = tree_data[req.mds_path]

    return raw_data


class TestECEDirectRequirements:
    """Test DIRECT stage requirement fetching."""

    def test_fetch_numch(self, ece_mapper):
        """Test fetching NUMCH for reference shot."""
        spec = ece_mapper.specs["ece._numch"]

        # Update requirement with reference shot
        req = spec.static_requirements[0]
        req_with_shot = Requirement(req.mds_path, REFERENCE_SHOT, req.treename)

        # Fetch
        raw_data = fetch_requirement(req_with_shot)

        # Verify
        assert req.mds_path in raw_data
        numch = raw_data[req.mds_path]
        assert isinstance(numch, (int, np.integer))
        assert numch > 0, "Should have at least one ECE channel"
        assert numch < 100, "Sanity check: shouldn't have more than 100 channels"

    def test_fetch_geometry_setup(self, ece_mapper):
        """Test fetching geometry setup data."""
        spec = ece_mapper.specs["ece._geometry_setup"]

        # Update requirements with reference shot
        reqs_with_shot = [
            Requirement(req.mds_path, REFERENCE_SHOT, req.treename)
            for req in spec.static_requirements
        ]

        # Fetch
        raw_data = fetch_requirements(reqs_with_shot)

        # Verify all three geometry parameters
        for req in reqs_with_shot:
            key = req.as_key()
            assert key in raw_data, f"Missing {req.mds_path}"
            value = raw_data[key]
            assert isinstance(value, (int, float, np.number)), \
                f"{req.mds_path} should be numeric"

    def test_fetch_frequency_setup(self, ece_mapper):
        """Test fetching frequency setup data."""
        spec = ece_mapper.specs["ece._frequency_setup"]

        # Update requirements with reference shot
        reqs_with_shot = [
            Requirement(req.mds_path, REFERENCE_SHOT, req.treename)
            for req in spec.static_requirements
        ]

        # Fetch
        raw_data = fetch_requirements(reqs_with_shot)

        # Verify FREQ and FLTRWID
        for req in reqs_with_shot:
            key = req.as_key()
            assert key in raw_data, f"Missing {req.mds_path}"
            value = raw_data[key]
            assert isinstance(value, np.ndarray), \
                f"{req.mds_path} should be array (one per channel)"
            assert len(value) > 0, f"{req.mds_path} should have data"

    def test_fetch_time_base(self, ece_mapper):
        """Test fetching time base from first channel."""
        spec = ece_mapper.specs["ece._time_base"]

        # Update requirement with reference shot
        req = spec.static_requirements[0]
        req_with_shot = Requirement(req.mds_path, REFERENCE_SHOT, req.treename)

        # Fetch
        raw_data = fetch_requirement(req_with_shot)

        # Verify
        assert req.mds_path in raw_data
        time = raw_data[req.mds_path]
        assert isinstance(time, np.ndarray), "Time should be array"
        assert len(time) > 0, "Time array should not be empty"
        assert time[0] < time[-1], "Time should be monotonically increasing"

    def test_fetch_all_direct_requirements(self, ece_mapper):
        """Test fetching all DIRECT stage requirements at once."""
        # Collect all DIRECT requirements
        direct_reqs = []
        for path, spec in ece_mapper.specs.items():
            if spec.stage == RequirementStage.DIRECT:
                for req in spec.static_requirements:
                    direct_reqs.append(
                        Requirement(req.mds_path, REFERENCE_SHOT, req.treename)
                    )

        # Fetch all at once
        raw_data = fetch_requirements(direct_reqs)

        # Verify all succeeded
        for req in direct_reqs:
            assert req.as_key() in raw_data, f"Failed to fetch {req.mds_path}"


class TestECEDerivedRequirements:
    """Test DERIVED stage requirement formulation and fetching."""

    def test_derive_temperature_requirements(self, ece_mapper):
        """Test deriving temperature channel requirements from NUMCH."""
        # First fetch NUMCH
        numch_spec = ece_mapper.specs["ece._numch"]
        numch_req = Requirement(
            numch_spec.static_requirements[0].mds_path,
            REFERENCE_SHOT,
            numch_spec.static_requirements[0].treename
        )
        raw_data = fetch_requirement(numch_req)

        # Store with proper key
        numch_value = raw_data[numch_req.mds_path]
        raw_data[numch_req.as_key()] = numch_value

        # Now derive temperature requirements
        temp_spec = ece_mapper.specs["ece._temperature_data"]
        derived_reqs = temp_spec.derive_requirements(REFERENCE_SHOT, raw_data)

        # Should have one requirement per channel
        assert len(derived_reqs) == numch_value

        # Each should be a TECE node
        for ich, req in enumerate(derived_reqs, start=1):
            assert f'TECE{ich:02d}' in req.mds_path or f'TECEF{ich:02d}' in req.mds_path
            assert req.shot == REFERENCE_SHOT
            assert req.treename == 'ELECTRONS'

    def test_fetch_derived_temperature_data(self, ece_mapper):
        """Test fetching all temperature channel data."""
        # Fetch NUMCH first
        numch_spec = ece_mapper.specs["ece._numch"]
        numch_req = Requirement(
            numch_spec.static_requirements[0].mds_path,
            REFERENCE_SHOT,
            numch_spec.static_requirements[0].treename
        )
        raw_data = fetch_requirement(numch_req)
        raw_data[numch_req.as_key()] = raw_data[numch_req.mds_path]

        # Derive temperature requirements
        temp_spec = ece_mapper.specs["ece._temperature_data"]
        temp_reqs = temp_spec.derive_requirements(REFERENCE_SHOT, raw_data)

        # Fetch temperature data
        temp_data = fetch_requirements(temp_reqs)

        # Verify all channels fetched successfully
        for req in temp_reqs:
            key = req.as_key()
            assert key in temp_data, f"Failed to fetch {req.mds_path}"

            channel_data = temp_data[key]
            assert isinstance(channel_data, np.ndarray), \
                f"{req.mds_path} should be array"
            assert len(channel_data) > 0, \
                f"{req.mds_path} should have data"


class TestECESynthesis:
    """Test COMPUTED stage synthesis functions."""

    @pytest.fixture
    def fetched_data(self, ece_mapper):
        """Fetch all required data for synthesis tests."""
        # Collect all DIRECT requirements
        direct_reqs = []
        for path, spec in ece_mapper.specs.items():
            if spec.stage == RequirementStage.DIRECT:
                for req in spec.static_requirements:
                    direct_reqs.append(
                        Requirement(req.mds_path, REFERENCE_SHOT, req.treename)
                    )

        # Fetch DIRECT data
        raw_data = fetch_requirements(direct_reqs)

        # Derive and fetch temperature data
        temp_spec = ece_mapper.specs["ece._temperature_data"]

        # Find NUMCH in raw_data
        numch_req = next(req for req in direct_reqs if 'NUMCH' in req.mds_path)
        raw_data[numch_req.as_key()] = raw_data[numch_req.as_key()]

        temp_reqs = temp_spec.derive_requirements(REFERENCE_SHOT, raw_data)
        temp_data = fetch_requirements(temp_reqs)

        # Merge temperature data into raw_data
        raw_data.update(temp_data)

        return raw_data

    def test_synthesize_channel_name(self, ece_mapper, fetched_data):
        """Test synthesis of channel names."""
        spec = ece_mapper.specs["ece.channel.name"]

        result = spec.synthesize(REFERENCE_SHOT, fetched_data)

        # Verify output
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

        # All names should start with 'ECE'
        for name in result:
            assert isinstance(name, str)
            assert name.startswith('ECE')

    def test_synthesize_channel_identifier(self, ece_mapper, fetched_data):
        """Test synthesis of channel identifiers."""
        spec = ece_mapper.specs["ece.channel.identifier"]

        result = spec.synthesize(REFERENCE_SHOT, fetched_data)

        # Verify output
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

        # All identifiers should contain TECE
        for identifier in result:
            assert isinstance(identifier, str)
            assert 'TECE' in identifier

    def test_synthesize_channel_time(self, ece_mapper, fetched_data):
        """Test synthesis of channel time arrays."""
        spec = ece_mapper.specs["ece.channel.time"]

        result = spec.synthesize(REFERENCE_SHOT, fetched_data)

        # Verify output shape
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2, "Time should be 2D (channels x time)"

        n_channels, n_time = result.shape
        assert n_channels > 0
        assert n_time > 0

        # Verify time is in seconds (should be < 10 for typical DIII-D shots)
        assert result[0, 0] < 10, "Time should be converted to seconds"

        # All channels should have same time base
        np.testing.assert_array_equal(result[0], result[1])

    def test_synthesize_channel_frequency(self, ece_mapper, fetched_data):
        """Test synthesis of channel frequency data."""
        spec = ece_mapper.specs["ece.channel.frequency.data"]

        result = spec.synthesize(REFERENCE_SHOT, fetched_data)

        # Verify output shape
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2, "Frequency should be 2D (channels x time)"

        n_channels, n_time = result.shape
        assert n_channels > 0
        assert n_time > 0

        # Verify frequency is in Hz (should be > 1e9 for ECE)
        assert result[0, 0] > 1e9, "Frequency should be in Hz (GHz range)"

        # Frequency should be constant in time for each channel
        for ich in range(n_channels):
            np.testing.assert_allclose(result[ich, 0], result[ich, -1])

    def test_synthesize_channel_if_bandwidth(self, ece_mapper, fetched_data):
        """Test synthesis of IF bandwidth."""
        spec = ece_mapper.specs["ece.channel.if_bandwidth"]

        result = spec.synthesize(REFERENCE_SHOT, fetched_data)

        # Verify output
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

        # Bandwidth should be in Hz (> 1e6)
        assert np.all(result > 1e6), "Bandwidth should be in Hz (MHz range)"

    def test_synthesize_channel_t_e(self, ece_mapper, fetched_data):
        """Test synthesis of electron temperature data."""
        spec = ece_mapper.specs["ece.channel.t_e.data"]

        result = spec.synthesize(REFERENCE_SHOT, fetched_data)

        # Verify output shape
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2, "Temperature should be 2D (channels x time)"

        n_channels, n_time = result.shape
        assert n_channels > 0
        assert n_time > 0

        # Temperature should be in eV (typically 100-10000 for DIII-D)
        assert np.nanmax(result) > 100, "Temperature should be in eV"
        assert np.nanmax(result) < 50000, "Temperature sanity check"

    def test_synthesize_geometry_first_point(self, ece_mapper, fetched_data):
        """Test synthesis of line of sight first point."""
        r_spec = ece_mapper.specs["ece.line_of_sight.first_point.r"]
        phi_spec = ece_mapper.specs["ece.line_of_sight.first_point.phi"]
        z_spec = ece_mapper.specs["ece.line_of_sight.first_point.z"]

        r = r_spec.synthesize(REFERENCE_SHOT, fetched_data)
        phi = phi_spec.synthesize(REFERENCE_SHOT, fetched_data)
        z = z_spec.synthesize(REFERENCE_SHOT, fetched_data)

        # Verify values
        assert isinstance(r, (float, np.floating))
        assert isinstance(phi, (float, np.floating))
        assert isinstance(z, (float, np.floating))

        # R should be at vessel wall (~2.5m for DIII-D)
        assert 2.0 < r < 3.0, "First point R should be near vessel wall"

        # Phi should be in radians
        assert -np.pi <= phi <= np.pi, "Phi should be in radians"

    def test_synthesize_geometry_second_point(self, ece_mapper, fetched_data):
        """Test synthesis of line of sight second point."""
        r_spec = ece_mapper.specs["ece.line_of_sight.second_point.r"]
        phi_spec = ece_mapper.specs["ece.line_of_sight.second_point.phi"]
        z_spec = ece_mapper.specs["ece.line_of_sight.second_point.z"]

        r = r_spec.synthesize(REFERENCE_SHOT, fetched_data)
        phi = phi_spec.synthesize(REFERENCE_SHOT, fetched_data)
        z = z_spec.synthesize(REFERENCE_SHOT, fetched_data)

        # Verify values
        assert isinstance(r, (float, np.floating))
        assert isinstance(phi, (float, np.floating))
        assert isinstance(z, (float, np.floating))

        # R should be in plasma (~0.8m for DIII-D)
        assert 0.5 < r < 1.5, "Second point R should be in plasma region"


class TestECEFastMode:
    """Test fast ECE mode with TECEF nodes."""

    def test_fetch_fast_ece_numch(self, ece_mapper_fast):
        """Test fetching NUMCH for fast ECE."""
        spec = ece_mapper_fast.specs["ece._numch"]

        req = spec.static_requirements[0]
        req_with_shot = Requirement(req.mds_path, REFERENCE_SHOT, req.treename)

        # Verify uses CALF node
        assert 'CALF' in req.mds_path

        # Fetch
        raw_data = fetch_requirement(req_with_shot)

        # Verify
        assert req.mds_path in raw_data
        numch = raw_data[req.mds_path]
        assert isinstance(numch, (int, np.integer))
        assert numch > 0

    def test_fast_ece_uses_tecef_nodes(self, ece_mapper_fast):
        """Verify fast ECE mapper uses TECEF nodes for temperature."""
        # Fetch NUMCH
        numch_spec = ece_mapper_fast.specs["ece._numch"]
        numch_req = Requirement(
            numch_spec.static_requirements[0].mds_path,
            REFERENCE_SHOT,
            numch_spec.static_requirements[0].treename
        )
        raw_data = fetch_requirement(numch_req)
        raw_data[numch_req.as_key()] = raw_data[numch_req.mds_path]

        # Derive temperature requirements
        temp_spec = ece_mapper_fast.specs["ece._temperature_data"]
        temp_reqs = temp_spec.derive_requirements(REFERENCE_SHOT, raw_data)

        # All should use TECEF
        for req in temp_reqs:
            assert 'TECEF' in req.mds_path, "Fast ECE should use TECEF nodes"
