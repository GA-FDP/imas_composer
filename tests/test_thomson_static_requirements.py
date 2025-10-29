"""
Tests for Thomson scattering mapper static (DIRECT stage) requirements.

These tests verify that we can correctly formulate the MDS+ requirements
for signals that don't depend on other data (DIRECT stage).
"""

import pytest
from imas_composer.core import RequirementStage, Requirement
from imas_composer.ids.thomson_scattering import ThomsonScatteringMapper


class TestThomsonStaticRequirements:
    """Test static requirement formulation for Thomson scattering IDS."""

    @pytest.fixture
    def ts_mapper(self):
        """Create Thomson scattering mapper instance."""
        return ThomsonScatteringMapper()

    def test_calib_nums_requirements(self, ts_mapper):
        """Test that _calib_nums has correct static requirements."""
        spec = ts_mapper.specs["thomson_scattering._calib_nums"]

        # Should be DIRECT stage
        assert spec.stage == RequirementStage.DIRECT

        # Should have exactly one requirement
        assert len(spec.static_requirements) == 1

        # Check the requirement details
        req = spec.static_requirements[0]
        assert req.mds_path == '.ts.BLESSED.header.calib_nums'
        assert req.shot == 0  # Shot is placeholder in spec
        assert req.treename == 'ELECTRONS'

    def test_calib_nums_uses_blessed_revision(self, ts_mapper):
        """Verify that mapper uses BLESSED revision."""
        assert ts_mapper.REVISION == 'BLESSED'

        spec = ts_mapper.specs["thomson_scattering._calib_nums"]
        req = spec.static_requirements[0]
        assert 'BLESSED' in req.mds_path

    def test_all_direct_specs_are_documented(self, ts_mapper):
        """Verify all DIRECT stage specs have documentation references."""
        for path, spec in ts_mapper.specs.items():
            if spec.stage == RequirementStage.DIRECT:
                assert spec.ids_path is not None, f"{path} missing ids_path"
                assert spec.docs_file is not None, f"{path} missing docs_file"
                assert spec.docs_file == "thomson_scattering.yaml"

    def test_direct_specs_have_no_dependencies(self, ts_mapper):
        """DIRECT stage specs should not depend on other specs."""
        for path, spec in ts_mapper.specs.items():
            if spec.stage == RequirementStage.DIRECT:
                assert len(spec.depends_on) == 0, \
                    f"DIRECT stage spec {path} should not have dependencies"

    def test_direct_specs_have_static_requirements(self, ts_mapper):
        """All DIRECT stage specs must have static requirements."""
        for path, spec in ts_mapper.specs.items():
            if spec.stage == RequirementStage.DIRECT:
                assert len(spec.static_requirements) > 0, \
                    f"DIRECT stage spec {path} must have static_requirements"

    def test_only_calib_nums_is_direct(self, ts_mapper):
        """
        Thomson scattering should only have one DIRECT requirement.
        Everything else depends on calibration numbers.
        """
        direct_specs = [
            path for path, spec in ts_mapper.specs.items()
            if spec.stage == RequirementStage.DIRECT
        ]

        assert len(direct_specs) == 1
        assert direct_specs[0] == "thomson_scattering._calib_nums"


class TestThomsonDerivedDependencies:
    """Test dependency structure for DERIVED stage specs."""

    @pytest.fixture
    def ts_mapper(self):
        """Create Thomson scattering mapper instance."""
        return ThomsonScatteringMapper()

    def test_hwmap_depends_on_calib_nums(self, ts_mapper):
        """Test that hardware map depends on calibration numbers."""
        spec = ts_mapper.specs["thomson_scattering._hwmap"]

        # Should be DERIVED stage
        assert spec.stage == RequirementStage.DERIVED

        # Should depend on calibration numbers
        assert "thomson_scattering._calib_nums" in spec.depends_on

        # Should have derive_requirements function
        assert spec.derive_requirements is not None

    def test_system_availability_depends_on_calib_nums(self, ts_mapper):
        """Test that system availability depends on calibration numbers."""
        spec = ts_mapper.specs["thomson_scattering._system_availability"]

        # Should be DERIVED stage
        assert spec.stage == RequirementStage.DERIVED

        # Should depend on calibration numbers
        assert "thomson_scattering._calib_nums" in spec.depends_on

        # Should have derive_requirements function
        assert spec.derive_requirements is not None

    def test_derived_specs_have_no_static_requirements(self, ts_mapper):
        """DERIVED stage specs should not have static requirements."""
        for path, spec in ts_mapper.specs.items():
            if spec.stage == RequirementStage.DERIVED:
                assert len(spec.static_requirements) == 0, \
                    f"DERIVED stage spec {path} should not have static_requirements"

    def test_derived_specs_have_dependencies(self, ts_mapper):
        """All DERIVED stage specs must have dependencies."""
        for path, spec in ts_mapper.specs.items():
            if spec.stage == RequirementStage.DERIVED:
                assert len(spec.depends_on) > 0, \
                    f"DERIVED stage spec {path} must have dependencies"

    def test_derived_specs_have_derive_function(self, ts_mapper):
        """All DERIVED stage specs must have derive_requirements function."""
        for path, spec in ts_mapper.specs.items():
            if spec.stage == RequirementStage.DERIVED:
                assert spec.derive_requirements is not None, \
                    f"DERIVED stage spec {path} must have derive_requirements"


class TestThomsonComputedDependencies:
    """Test dependency structure for COMPUTED stage specs."""

    @pytest.fixture
    def ts_mapper(self):
        """Create Thomson scattering mapper instance."""
        return ThomsonScatteringMapper()

    def test_channel_name_dependencies(self, ts_mapper):
        """Test channel name depends on hwmap and system availability."""
        spec = ts_mapper.specs["thomson_scattering.channel.name"]

        # Should be COMPUTED stage
        assert spec.stage == RequirementStage.COMPUTED

        # Should depend on both hwmap and system availability
        assert "thomson_scattering._hwmap" in spec.depends_on
        assert "thomson_scattering._system_availability" in spec.depends_on

        # Should have synthesize function
        assert spec.synthesize is not None

    def test_channel_identifier_dependencies(self, ts_mapper):
        """Test channel identifier depends on system availability."""
        spec = ts_mapper.specs["thomson_scattering.channel.identifier"]

        # Should be COMPUTED stage
        assert spec.stage == RequirementStage.COMPUTED

        # Should depend on system availability
        assert "thomson_scattering._system_availability" in spec.depends_on

        # Should have synthesize function
        assert spec.synthesize is not None

    def test_position_dependencies(self, ts_mapper):
        """Test position coordinates depend on system availability."""
        for coord in ['r', 'z', 'phi']:
            spec = ts_mapper.specs[f"thomson_scattering.channel.position.{coord}"]

            # Should be COMPUTED stage
            assert spec.stage == RequirementStage.COMPUTED

            # Should depend on system availability
            assert "thomson_scattering._system_availability" in spec.depends_on

            # Should have synthesize function
            assert spec.synthesize is not None

    def test_measurement_time_dependencies(self, ts_mapper):
        """Test measurement time specs are DERIVED and depend on system availability."""
        for measurement in ['n_e', 't_e']:
            spec = ts_mapper.specs[f"thomson_scattering.channel.{measurement}.time"]

            # Should be DERIVED stage (needs to fetch TIME for active systems)
            assert spec.stage == RequirementStage.DERIVED

            # Should depend on system availability
            assert "thomson_scattering._system_availability" in spec.depends_on

            # Should have derive_requirements function
            assert spec.derive_requirements is not None

    def test_measurement_data_dependencies(self, ts_mapper):
        """Test measurement data specs are DERIVED and depend on system availability."""
        for measurement in ['n_e', 't_e']:
            spec = ts_mapper.specs[f"thomson_scattering.channel.{measurement}.data"]

            # Should be DERIVED stage (needs to fetch DENSITY/TEMP for active systems)
            assert spec.stage == RequirementStage.DERIVED

            # Should depend on system availability
            assert "thomson_scattering._system_availability" in spec.depends_on

            # Should have derive_requirements function
            assert spec.derive_requirements is not None

    def test_computed_specs_have_no_static_requirements(self, ts_mapper):
        """COMPUTED stage specs should not have static requirements."""
        for path, spec in ts_mapper.specs.items():
            if spec.stage == RequirementStage.COMPUTED:
                assert len(spec.static_requirements) == 0, \
                    f"COMPUTED stage spec {path} should not have static_requirements"

    def test_computed_specs_have_synthesize_function(self, ts_mapper):
        """All COMPUTED stage specs must have synthesize function."""
        for path, spec in ts_mapper.specs.items():
            if spec.stage == RequirementStage.COMPUTED:
                assert spec.synthesize is not None, \
                    f"COMPUTED stage spec {path} must have synthesize function"


class TestThomsonMapperStructure:
    """Test overall structure and organization of Thomson scattering mapper."""

    def test_mapper_initialization(self):
        """Test that mapper initializes correctly."""
        mapper = ThomsonScatteringMapper()

        assert mapper.SYSTEMS == ['TANGENTIAL', 'DIVERTOR', 'CORE']
        assert mapper.REVISION == 'BLESSED'
        assert mapper.DOCS_PATH == "thomson_scattering.yaml"

    def test_get_specs_returns_dict(self):
        """Test that get_specs() returns the specs dictionary."""
        mapper = ThomsonScatteringMapper()
        specs = mapper.get_specs()

        assert isinstance(specs, dict)
        assert len(specs) > 0

        # Should contain internal dependencies
        assert "thomson_scattering._calib_nums" in specs
        assert "thomson_scattering._hwmap" in specs
        assert "thomson_scattering._system_availability" in specs

    def test_all_specs_have_stage(self):
        """Verify every spec has a stage defined."""
        mapper = ThomsonScatteringMapper()

        for path, spec in mapper.specs.items():
            assert isinstance(spec.stage, RequirementStage), \
                f"Spec {path} has invalid stage"

    def test_internal_specs_use_underscore_prefix(self):
        """Internal dependencies should use underscore prefix."""
        mapper = ThomsonScatteringMapper()

        internal_specs = [
            "thomson_scattering._calib_nums",
            "thomson_scattering._hwmap",
            "thomson_scattering._system_availability"
        ]

        for spec_path in internal_specs:
            assert spec_path in mapper.specs, f"Missing internal spec {spec_path}"
            # Internal specs should not be part of final IDS
            assert spec_path.split('.')[-1].startswith('_')

    def test_public_specs_no_underscore(self):
        """Public IDS entries should not use underscore prefix."""
        mapper = ThomsonScatteringMapper()

        public_specs = [
            "thomson_scattering.channel.name",
            "thomson_scattering.channel.identifier",
            "thomson_scattering.channel.position.r",
            "thomson_scattering.channel.position.z",
            "thomson_scattering.channel.position.phi",
            "thomson_scattering.channel.n_e.time",
            "thomson_scattering.channel.n_e.data",
            "thomson_scattering.channel.t_e.time",
            "thomson_scattering.channel.t_e.data"
        ]

        for spec_path in public_specs:
            assert spec_path in mapper.specs, f"Missing public spec {spec_path}"
            # Public specs should not have underscore in final component
            final_component = spec_path.split('.')[-1]
            assert not final_component.startswith('_')

    def test_measurement_specs_created_for_ne_and_te(self):
        """Verify that n_e and t_e measurements have time and data specs."""
        mapper = ThomsonScatteringMapper()

        for measurement in ['n_e', 't_e']:
            time_spec = f"thomson_scattering.channel.{measurement}.time"
            data_spec = f"thomson_scattering.channel.{measurement}.data"

            assert time_spec in mapper.specs, f"Missing {time_spec}"
            assert data_spec in mapper.specs, f"Missing {data_spec}"

    def test_position_specs_created_for_all_coords(self):
        """Verify that r, z, phi position specs are created."""
        mapper = ThomsonScatteringMapper()

        for coord in ['r', 'z', 'phi']:
            spec_path = f"thomson_scattering.channel.position.{coord}"
            assert spec_path in mapper.specs, f"Missing {spec_path}"


class TestThomsonDependencyChain:
    """Test the complete dependency chain from calib_nums to final outputs."""

    @pytest.fixture
    def ts_mapper(self):
        """Create Thomson scattering mapper instance."""
        return ThomsonScatteringMapper()

    def test_calib_nums_has_no_dependencies(self, ts_mapper):
        """Root of dependency tree should have no dependencies."""
        spec = ts_mapper.specs["thomson_scattering._calib_nums"]
        assert len(spec.depends_on) == 0

    def test_second_level_depends_on_calib_nums(self, ts_mapper):
        """Second level should only depend on calib_nums."""
        second_level = [
            "thomson_scattering._hwmap",
            "thomson_scattering._system_availability"
        ]

        for spec_path in second_level:
            spec = ts_mapper.specs[spec_path]
            assert "thomson_scattering._calib_nums" in spec.depends_on
            assert spec.stage == RequirementStage.DERIVED

    def test_final_outputs_depend_on_second_level(self, ts_mapper):
        """Final output specs should depend on derived data, not directly on calib_nums."""
        # channel.name needs hwmap (for lens info) and system_availability
        name_spec = ts_mapper.specs["thomson_scattering.channel.name"]
        assert "thomson_scattering._hwmap" in name_spec.depends_on
        assert "thomson_scattering._system_availability" in name_spec.depends_on
        assert "thomson_scattering._calib_nums" not in name_spec.depends_on

        # Other specs just need system_availability
        for spec_path in [
            "thomson_scattering.channel.identifier",
            "thomson_scattering.channel.position.r",
            "thomson_scattering.channel.n_e.time"
        ]:
            spec = ts_mapper.specs[spec_path]
            assert "thomson_scattering._system_availability" in spec.depends_on
            assert "thomson_scattering._calib_nums" not in spec.depends_on
