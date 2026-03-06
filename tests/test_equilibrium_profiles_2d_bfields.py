"""
Test Equilibrium profiles_2d magnetic field components against OMAS physics derivation.

These fields (b_field_tor, b_field_r, b_field_z) use OMAS physics derivation functions
that are not accessible through the standard machine_to_omas pathway. We test them
by explicitly calling the physics derivation method and comparing the results.
"""
import pytest
import numpy as np

from omas import ODS


pytestmark = [pytest.mark.omas_validation]


@pytest.mark.parametrize('field_name', ['b_field_tor', 'b_field_r', 'b_field_z'])
def test_profiles_2d_magnetic_fields(field_name, composer, test_shot):
    """
    Test profiles_2d magnetic field components against OMAS physics derivation.

    This test uses OMAS's physics_derive_equilibrium_profiles_2d_quantity method
    to generate reference data for comparison with imas_composer output.

    Args:
        field_name: One of 'b_field_tor', 'b_field_r', 'b_field_z'
        composer: ImasComposer fixture
        test_shot: Test shot number from parametrized fixture
    """
    # Get composer result
    ids_path = f'equilibrium.time_slice.profiles_2d.{field_name}'

    # Resolve requirements
    raw_data = {}
    while True:
        status, requirements = composer.resolve([ids_path], test_shot, raw_data)
        if status[ids_path]:
            break

        # Fetch requirements from MDSplus
        from omas import mdsvalue
        for req in requirements:
            try:
                mds_data = mdsvalue('d3d', req.treename, req.shot, req.mds_path).raw()
                raw_data[req.as_key()] = mds_data
            except Exception as e:
                pytest.skip(f"MDSplus data unavailable for {req.mds_path}: {e}")

    # Compose the field
    composer_result = composer.compose([ids_path], test_shot, raw_data)
    composer_data = composer_result[ids_path]

    # Get OMAS reference using physics derivation
    ods = ODS()
    with ods.open('d3d', test_shot, options={'EFIT_tree': 'EFIT01'}):
        # Load necessary equilibrium data for physics derivation
        # The physics method needs profiles_2d.psi and grid data to already be loaded
        _ = ods['equilibrium.time_slice.:.profiles_2d.:.psi']
        _ = ods['equilibrium.time_slice.:.profiles_2d.:.grid.dim1']
        _ = ods['equilibrium.time_slice.:.profiles_2d.:.grid.dim2']

        # Also need profiles_1d data for some calculations
        _ = ods['equilibrium.time_slice.:.profiles_1d.psi']
        _ = ods['equilibrium.time_slice.:.profiles_1d.f']

        # Get time points to iterate over
        time_values = ods['equilibrium.time_slice.:.time']
        n_time = len(time_values)

        # Grid index (assume 0 is the only valid index)
        grid_index = 0

        # Derive the field for each time slice
        for time_index in range(n_time):
            # Call OMAS physics derivation method
            # The method modifies the ODS in place
            ods.physics_derive_equilibrium_profiles_2d_quantity(
                time_index=time_index,
                grid_index=grid_index,
                quantity=field_name,
                force_linear_interpolation=True  # Use numpy.gradient like our implementation
            )

        # Extract the derived data
        omas_data = ods[f'equilibrium.time_slice.:.profiles_2d.:.{field_name}']

    # Compare shapes
    assert composer_data.shape == omas_data.shape, (
        f"Shape mismatch for {field_name}: "
        f"composer={composer_data.shape}, omas={omas_data.shape}"
    )

    # Compare values with appropriate tolerances
    # Magnetic field calculations involve gradients and interpolations,
    # so we need somewhat relaxed tolerances
    rtol = 1e-5
    atol = 1e-7

    # Handle the grid dimension in OMAS (axis 1)
    # OMAS has shape (n_time, n_grid, n_r, n_z), composer has (n_time, 1, n_r, n_z)
    # Both should only have 1 grid, so we can squeeze that dimension for comparison
    if omas_data.ndim == 4:
        omas_data = omas_data[:, grid_index, :, :]
        composer_data = composer_data[:, 0, :, :]

    if not np.allclose(composer_data, omas_data, rtol=rtol, atol=atol, equal_nan=True):
        diff = np.abs(composer_data - omas_data)
        max_diff = np.nanmax(diff)
        max_rel_diff = np.nanmax(diff / (np.abs(omas_data) + atol))

        pytest.fail(
            f"Values don't match for {field_name}\n"
            f"Max absolute difference: {max_diff}\n"
            f"Max relative difference: {max_rel_diff}\n"
            f"Tolerance: rtol={rtol}, atol={atol}\n"
            f"Shot: {test_shot}"
        )
