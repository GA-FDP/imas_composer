"""
OMAS Reference Implementation for Thomson Scattering
Source: omas/machine_mappings/d3d.py::thomson_scattering_data (line 789)

This file contains the OMAS implementation for reference when debugging or extending
the imas_composer Thomson scattering mapper.

KEY DIFFERENCES:
- OMAS: Single query dict with all data fetched upfront
- imas_composer: Lazy loading, only fetch what's requested via specs
- OMAS: unumpy.uarray for uncertainties
- imas_composer: Separate .data and .data_error_upper fields
- OMAS: Nested ODS structure
- imas_composer: Flat arrays with awkward for ragged data
"""

def thomson_scattering_data(ods, pulse, revision='BLESSED', _get_measurements=True):
    """
    Loads DIII-D Thomson measurement data

    :param pulse: int

    :param revision: string
        Thomson scattering data revision, like 'BLESSED', 'REVISIONS.REVISION00', etc.
    """
    systems = ['TANGENTIAL', 'DIVERTOR', 'CORE']

    # get the actual data
    query = {'calib_nums': f'.ts.{revision}.header.calib_nums'}
    for system in systems:
        for quantity in ['R', 'Z', 'PHI']:
            query[f'{system}_{quantity}'] = f'.TS.{revision}.{system}:{quantity}'
        if _get_measurements:
            for quantity in ['TEMP', 'TEMP_E', 'DENSITY', 'DENSITY_E', 'TIME']:
                query[f'{system}_{quantity}'] = f'.TS.{revision}.{system}:{quantity}'
    tsdat = mdsvalue('d3d', treename='ELECTRONS', pulse=pulse, TDI=query).raw()

    # Read the Thomson scattering hardware map to figure out which lens each chord looks through
    cal_set = tsdat['calib_nums'][0]
    query = {}
    for system in systems:
        query[f'{system}_hwmapints'] = f'.{system}.hwmapints'
    hw_ints = mdsvalue('d3d', treename='TSCAL', pulse=cal_set, TDI=query).raw()

    # assign data in ODS
    i = 0
    for system in systems:
        if isinstance(tsdat[f'{system}_R'], Exception):
            continue
        nc = len(tsdat[f'{system}_R'])
        if not nc:
            continue

        # determine which lenses were used
        ints = hw_ints[f'{system}_hwmapints']
        if len(np.shape(ints)) < 2:
            # Contingency needed for cases where all view-chords are taken off of divertor laser and reassigned to core
            ints = ints.reshape(1, -1)
        lenses = ints[:, 2]

        # Assign data to ODS
        for j in range(nc):
            ch = ods['thomson_scattering']['channel'][i]
            ch['name'] = 'TS_{system}_r{lens:+0d}_{ch:}'.format(
                system=system.lower(), ch=j, lens=lenses[min(j,len(lenses)-1)]
            )
            ch['identifier'] = f'{system[0]}{j:02d}'
            ch['position']['r'] = tsdat[f'{system}_R'][j]
            ch['position']['z'] = tsdat[f'{system}_Z'][j]
            ch['position']['phi'] = -tsdat[f'{system}_PHI'][j] * np.pi / 180.0
            if _get_measurements:
                ch['n_e.time'] = tsdat[f'{system}_TIME'] / 1e3
                ch['n_e.data'] = unumpy.uarray(tsdat[f'{system}_DENSITY'][j], tsdat[f'{system}_DENSITY_E'][j])
                ch['t_e.time'] = tsdat[f'{system}_TIME'] / 1e3
                ch['t_e.data'] = unumpy.uarray(tsdat[f'{system}_TEMP'][j], tsdat[f'{system}_TEMP_E'][j])
            i += 1
