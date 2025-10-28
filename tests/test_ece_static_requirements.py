"""
Tests for ECE mapper static (DIRECT stage) requirements.

These tests verify that we can correctly formulate the MDS+ requirements
for signals that don't depend on other data (DIRECT stage).
"""

import pytest
from imas_composer.core import RequirementStage, Requirement
from imas_composer.ids.ece import ElectronCyclotronEmissionMapper


class TestECEStaticRequirements:
    """Test static requirement formulation for ECE IDS."""

    @pytest.fixture
    def ece_mapper(self):
        """Create standard ECE mapper instance."""
        return ElectronCyclotronEmissionMapper(fast_ece=False)

    @pytest.fixture
    def ece_mapper_fast(self):
        """Create fast ECE mapper instance."""
        return ElectronCyclotronEmissionMapper(fast_ece=True)

    def test_numch_requirements(self, ece_mapper):
        """Test that _numch has correct static requirements."""
        spec = ece_mapper.specs["ece._numch"]

        # Should be DIRECT stage
        assert spec.stage == RequirementStage.DIRECT

        # Should have exactly one requirement
        assert len(spec.static_requirements) == 1

        # Check the requirement details
        req = spec.static_requirements[0]
        assert req.mds_path == '\\ECE::TOP.CAL.NUMCH'
        assert req.shot == 0  # Shot is placeholder in spec
        assert req.treename == 'ELECTRONS'

    def test_numch_requirements_fast(self, ece_mapper_fast):
        """Test that fast ECE _numch uses CALF node."""
        spec = ece_mapper_fast.specs["ece._numch"]

        # Should use CALF for fast ECE
        req = spec.static_requirements[0]
        assert req.mds_path == '\\ECE::TOP.CALF.NUMCH'
        assert req.treename == 'ELECTRONS'

    def test_geometry_setup_requirements(self, ece_mapper):
        """Test that geometry setup has all required MDS+ paths."""
        spec = ece_mapper.specs["ece._geometry_setup"]

        # Should be DIRECT stage
        assert spec.stage == RequirementStage.DIRECT

        # Should have three requirements
        assert len(spec.static_requirements) == 3

        # Extract MDS paths
        paths = [req.mds_path for req in spec.static_requirements]

        # Check expected paths
        assert '\\ECE::TOP.SETUP.ECEPHI' in paths
        assert '\\ECE::TOP.SETUP.ECETHETA' in paths
        assert '\\ECE::TOP.SETUP.ECEZH' in paths

        # All should be from ELECTRONS tree
        for req in spec.static_requirements:
            assert req.treename == 'ELECTRONS'

    def test_frequency_setup_requirements(self, ece_mapper):
        """Test that frequency setup has correct requirements."""
        spec = ece_mapper.specs["ece._frequency_setup"]

        # Should be DIRECT stage
        assert spec.stage == RequirementStage.DIRECT

        # Should have two requirements
        assert len(spec.static_requirements) == 2

        # Extract MDS paths
        paths = [req.mds_path for req in spec.static_requirements]

        # Check expected paths
        assert '\\ECE::TOP.SETUP.FREQ' in paths
        assert '\\ECE::TOP.SETUP.FLTRWID' in paths

    def test_time_base_requirements(self, ece_mapper):
        """Test that time base uses dim_of() for first channel."""
        spec = ece_mapper.specs["ece._time_base"]

        # Should be DIRECT stage
        assert spec.stage == RequirementStage.DIRECT

        # Should have one requirement
        assert len(spec.static_requirements) == 1

        req = spec.static_requirements[0]
        assert req.mds_path == 'dim_of(\\ECE::TOP.TECE.TECE01)'
        assert req.treename == 'ELECTRONS'

    def test_time_base_requirements_fast(self, ece_mapper_fast):
        """Test that fast ECE uses TECEF node for time base."""
        spec = ece_mapper_fast.specs["ece._time_base"]

        req = spec.static_requirements[0]
        assert req.mds_path == 'dim_of(\\ECE::TOP.TECE.TECEF01)'
        assert req.treename == 'ELECTRONS'

    def test_all_direct_specs_are_documented(self, ece_mapper):
        """Verify all DIRECT stage specs have documentation references."""
        for path, spec in ece_mapper.specs.items():
            if spec.stage == RequirementStage.DIRECT:
                assert spec.ids_path is not None, f"{path} missing ids_path"
                assert spec.docs_file is not None, f"{path} missing docs_file"

    def test_direct_specs_have_no_dependencies(self, ece_mapper):
        """DIRECT stage specs should not depend on other specs."""
        for path, spec in ece_mapper.specs.items():
            if spec.stage == RequirementStage.DIRECT:
                assert len(spec.depends_on) == 0, \
                    f"DIRECT stage spec {path} should not have dependencies"

    def test_direct_specs_have_static_requirements(self, ece_mapper):
        """All DIRECT stage specs must have static requirements."""
        for path, spec in ece_mapper.specs.items():
            if spec.stage == RequirementStage.DIRECT:
                assert len(spec.static_requirements) > 0, \
                    f"DIRECT stage spec {path} must have static_requirements"

    def test_requirement_hashing(self):
        """Test that Requirements can be hashed and deduplicated."""
        req1 = Requirement('\\ECE::TOP.CAL.NUMCH', 123456, 'ELECTRONS')
        req2 = Requirement('\\ECE::TOP.CAL.NUMCH', 123456, 'ELECTRONS')
        req3 = Requirement('\\ECE::TOP.CAL.NUMCH', 654321, 'ELECTRONS')

        # Same requirements should hash the same
        assert hash(req1) == hash(req2)

        # Different shot should hash differently
        assert hash(req1) != hash(req3)

        # Should work in sets (deduplication)
        req_set = {req1, req2, req3}
        assert len(req_set) == 2  # req1 and req2 are duplicates

    def test_requirement_as_key(self):
        """Test that Requirements can be used as dict keys."""
        req = Requirement('\\ECE::TOP.CAL.NUMCH', 123456, 'ELECTRONS')

        # Create a key
        key = req.as_key()
        assert isinstance(key, tuple)
        assert key == ('\\ECE::TOP.CAL.NUMCH', 123456, 'ELECTRONS')

        # Should work as dict key
        data = {key: 42}
        assert data[req.as_key()] == 42


class TestECEMapperStructure:
    """Test overall structure and organization of ECE mapper."""

    def test_mapper_initialization(self):
        """Test that mapper initializes correctly."""
        mapper = ElectronCyclotronEmissionMapper(fast_ece=False)

        assert mapper.fast_ece is False
        assert mapper.fast_suffix == ''
        assert mapper.setup_node == '\\ECE::TOP.SETUP.'
        assert mapper.cal_node == '\\ECE::TOP.CAL.'
        assert mapper.tece_node == '\\ECE::TOP.TECE.TECE'

    def test_mapper_initialization_fast(self):
        """Test that fast mapper sets correct paths."""
        mapper = ElectronCyclotronEmissionMapper(fast_ece=True)

        assert mapper.fast_ece is True
        assert mapper.fast_suffix == 'F'
        assert mapper.cal_node == '\\ECE::TOP.CALF.'
        assert mapper.tece_node == '\\ECE::TOP.TECE.TECEF'

    def test_get_specs_returns_dict(self):
        """Test that get_specs() returns the specs dictionary."""
        mapper = ElectronCyclotronEmissionMapper()
        specs = mapper.get_specs()

        assert isinstance(specs, dict)
        assert len(specs) > 0

        # Should contain internal dependencies
        assert "ece._numch" in specs
        assert "ece._geometry_setup" in specs
        assert "ece._frequency_setup" in specs
        assert "ece._time_base" in specs

    def test_all_specs_have_stage(self):
        """Verify every spec has a stage defined."""
        mapper = ElectronCyclotronEmissionMapper()

        for path, spec in mapper.specs.items():
            assert isinstance(spec.stage, RequirementStage), \
                f"Spec {path} has invalid stage"

    def test_internal_specs_use_underscore_prefix(self):
        """Internal dependencies should use underscore prefix."""
        mapper = ElectronCyclotronEmissionMapper()

        internal_specs = [
            "ece._numch",
            "ece._geometry_setup",
            "ece._frequency_setup",
            "ece._time_base",
            "ece._temperature_data"
        ]

        for spec_path in internal_specs:
            assert spec_path in mapper.specs, f"Missing internal spec {spec_path}"
            # Internal specs should not be part of final IDS
            assert spec_path.split('.')[-1].startswith('_')
