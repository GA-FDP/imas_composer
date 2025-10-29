"""
OMAS Reference Implementation for ECE (Electron Cyclotron Emission)
Source: omas/machine_mappings/d3d.py::electron_cyclotron_emission_data (line 866)

This file contains the OMAS implementation for reference when debugging or extending
the imas_composer ECE mapper.

KEY DIFFERENCES:
- OMAS: Single query dict with all data fetched upfront
- imas_composer: Lazy loading, only fetch what's requested via specs
- OMAS: unumpy.uarray for uncertainties
- imas_composer: Separate .data and .data_error_upper fields
- OMAS: Nested ODS structure
- imas_composer: Flat arrays (all channels same time base, no ragged arrays needed)
"""

def electron_cyclotron_emission_data(ods, pulse=133221, fast_ece=False, _measurements=True):
    """
    Loads DIII-D Electron cyclotron emission data

    :param pulse: int

    :param fast_ece: bool
            Use data sampled at high frequency
    """
    fast_ece = 'F' if fast_ece else ''
    setup = '\\ECE::TOP.SETUP.'
    cal = '\\ECE::TOP.CAL%s.' % fast_ece
    TECE = '\\ECE::TOP.TECE.TECE' + fast_ece

    query = {}
    for node, quantities in zip([setup, cal], [['ECEPHI', 'ECETHETA', 'ECEZH', 'FREQ', "FLTRWID"], ['NUMCH']]):
        for quantity in quantities:
            query[quantity] = node + quantity
    query['TIME'] = f"dim_of({TECE + '01'})"
    ece_map = mdsvalue('d3d', treename='ELECTRONS', pulse=pulse, TDI=query).raw()
    N_time = len(ece_map['TIME'])
    N_ch = ece_map['NUMCH'].item()

    if _measurements:
        query = {}
        for ich in range(1, N_ch + 1):
            query[f'T{ich}'] = TECE + '{0:02d}'.format(ich)
        ece_data = mdsvalue('d3d', treename='ELECTRONS', pulse=pulse, TDI=query).raw()
        ece_uncertainty = {}
        for key in ece_data:
            # Calculate uncertainties and convert to eV
            # Assumes 7% calibration error (optimisitic) + Poisson uncertainty
            ece_uncertainty[key] = np.sqrt(np.abs(ece_data[key] * 1.e3)) + 70 * np.abs(ece_data[key])

    ods['ece.ids_properties.homogeneous_time'] = 0
    # Not in MDSplus
    if not _measurements:
        points = [{}, {}]
        points[0]['r'] = 2.5
        points[1]['r'] = 0.8
        points[0]['phi'] = np.deg2rad(ece_map['ECEPHI'])
        points[1]['phi'] = np.deg2rad(ece_map['ECEPHI'])
        dz = np.sin(np.deg2rad(ece_map['ECETHETA']))
        points[0]['z'] = ece_map['ECEZH']
        points[1]['z'] = points[0]['z'] + dz
        for entry, point in zip([ods['ece.line_of_sight.first_point'], ods['ece.line_of_sight.second_point']], points):
            for key in point:
                entry[key] = point[key]

    # Assign data to ODS
    f = np.zeros(N_time)
    for ich in range(N_ch):
        ch = ods['ece']['channel'][ich]
        if _measurements:
            ch['t_e']['data'] = unumpy.uarray(ece_data[f'T{ich + 1}'] * 1.0e3,
                                              ece_uncertainty[f'T{ich + 1}'] )# Already converted
        else:
            ch['name'] = 'ECE' + str(ich + 1)
            ch['identifier'] = TECE + '{0:02d}'.format(ich + 1)
            ch['time'] = ece_map['TIME'] * 1.0e-3
            f[:] = ece_map['FREQ'][ich]
            ch['frequency']['data'] = f * 1.0e9
            ch['if_bandwidth'] = ece_map['FLTRWID'][ich] * 1.0e9
