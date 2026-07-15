"""
IRI CAKE Viewer — PyQt GUI for inspecting IRI CAKE equilibrium and profile data.

Replicates the functionality of OMFIT-source/scripts/fetch_IRI_CAKE.py without
any dependency on omas or omfit_classes.  Data is fetched via imas_composer's
simple_load function; IRI run metadata is queried from D3DRDB via d3drdb.py.

Layout (2 × 6 grid of subplots; Eq. CX spans both rows of column 0):
  [        | ne (e)  | Te (e)  | j (current)| Pressure | convergence error ]
  [ Eq. CX | ni (ion)| Ti (ion)| v_tor      | E_r      | Zeff              ]

Usage::

    python -m imas_composer.plots.iri_cake_viewer
    python -m imas_composer.plots.iri_cake_viewer --shot 205055
    python -m imas_composer.plots.iri_cake_viewer --shot 205055 --flavor CAKE_FDP
"""

from __future__ import annotations

import argparse
import sys
import traceback
from typing import Any, Dict, Optional

import numpy as np
import awkward as ak

# ---------------------------------------------------------------------------
# pyqtgraph (and its Qt compatibility layer — uses PySide6 on this system)
# ---------------------------------------------------------------------------
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

import scipy.ndimage
from scipy.interpolate import RegularGridInterpolator
from contourpy import contour_generator

from imas_composer.composer import ImasComposer
from imas_composer.fetchers import simple_load
from .d3drdb import get_iri_upload_ids, list_shots_for_tag, list_all_tags

pg.setConfigOptions(antialias=True, background='w', foreground='k')

# Translation of the matplotlib "tab:*" / named colours used below to hex,
# so pens/brushes render the same as the original matplotlib version.
COLORS = {
    'tab:blue':   '#1f77b4',
    'tab:orange': '#ff7f0e',
    'tab:green':  '#2ca02c',
    'tab:red':    '#d62728',
    'tab:purple': '#9467bd',
    'tab:brown':  '#8c564b',
    'black':      'k',
    'red':        'r',
}


# ---------------------------------------------------------------------------
# IDS fields to load
# ---------------------------------------------------------------------------

EQ_FIELDS = [
    'equilibrium.time',
    'equilibrium.time_slice.profiles_2d.grid.dim1',
    'equilibrium.time_slice.profiles_2d.grid.dim2',
    'equilibrium.time_slice.profiles_2d.psi',
    'equilibrium.time_slice.boundary.outline.r',
    'equilibrium.time_slice.boundary.outline.z',
    'equilibrium.time_slice.boundary.x_point.r',
    'equilibrium.time_slice.boundary.x_point.z',
    'equilibrium.time_slice.profiles_1d.psi',
    'equilibrium.time_slice.profiles_1d.pressure',
    'equilibrium.time_slice.profiles_1d.j_tor',
    'equilibrium.time_slice.profiles_1d.q',
    'equilibrium.time_slice.profiles_1d.rho_tor_norm',
    'equilibrium.time_slice.global_quantities.psi_axis',
    'equilibrium.time_slice.global_quantities.psi_boundary',
    'equilibrium.time_slice.global_quantities.ip',
    'equilibrium.time_slice.global_quantities.q_95',
    'equilibrium.time_slice.global_quantities.beta_normal',
    'equilibrium.time_slice.global_quantities.magnetic_axis.r',
    'equilibrium.time_slice.global_quantities.magnetic_axis.z',
    'equilibrium.time_slice.constraints.pressure.position.psi',
    'equilibrium.time_slice.constraints.pressure.measured',
    'equilibrium.time_slice.constraints.pressure.measured_error_upper',
    'equilibrium.time_slice.constraints.j_tor.position.psi',
    'equilibrium.time_slice.constraints.j_tor.measured',
    'equilibrium.time_slice.convergence.grad_shafranov_deviation_value',
]

WALL_FIELDS = [
    'wall.description_2d.limiter.unit.outline.r',
    'wall.description_2d.limiter.unit.outline.z',
]

SUMMARY_FIELDS = [
    'summary.description',
]

PROF_FIELDS = [
    'core_profiles.time',
    'core_profiles.profiles_1d.grid.rho_pol_norm',
    'core_profiles.profiles_1d.electrons.density',
    'core_profiles.profiles_1d.electrons.density_error_upper',
    'core_profiles.profiles_1d.electrons.temperature',
    'core_profiles.profiles_1d.electrons.temperature_error_upper',
    'core_profiles.profiles_1d.ion.temperature',
    'core_profiles.profiles_1d.ion.temperature_error_upper',
    'core_profiles.profiles_1d.ion.density_thermal',
    'core_profiles.profiles_1d.ion.density_thermal_error_upper',
    'core_profiles.profiles_1d.electrons.density_fit.psi_norm',
    'core_profiles.profiles_1d.electrons.density_fit.measured',
    'core_profiles.profiles_1d.electrons.density_fit.measured_error_upper',
    'core_profiles.profiles_1d.electrons.temperature_fit.psi_norm',
    'core_profiles.profiles_1d.electrons.temperature_fit.measured',
    'core_profiles.profiles_1d.electrons.temperature_fit.measured_error_upper',
    'core_profiles.profiles_1d.ion.label',
    'core_profiles.profiles_1d.ion.density_fit.psi_norm',
    'core_profiles.profiles_1d.ion.density_fit.measured',
    'core_profiles.profiles_1d.ion.density_fit.measured_error_upper',
    'core_profiles.profiles_1d.ion.temperature_fit.psi_norm',
    'core_profiles.profiles_1d.ion.temperature_fit.measured',
    'core_profiles.profiles_1d.ion.temperature_fit.measured_error_upper',
    'core_profiles.profiles_1d.ion.velocity.toroidal',
    'core_profiles.profiles_1d.ion.velocity.toroidal_error_upper',
    'core_profiles.profiles_1d.ion.velocity.toroidal_fit.psi_norm',
    'core_profiles.profiles_1d.ion.velocity.toroidal_fit.measured',
    'core_profiles.profiles_1d.ion.velocity.toroidal_fit.measured_error_upper',
    'core_profiles.profiles_1d.electrons.pressure',
    'core_profiles.profiles_1d.pressure_ion_total',
    'core_profiles.profiles_1d.pressure_ion_non_thermal',
    'core_profiles.profiles_1d.pressure_total',
    'core_profiles.profiles_1d.j_tor',
    'core_profiles.profiles_1d.j_ohmic',
    'core_profiles.profiles_1d.j_bootstrap',
    'core_profiles.profiles_1d.e_field.radial',
    'core_profiles.profiles_1d.zeff',
    'core_profiles.profiles_1d.zeff_error_upper',
]

# CER Zeff overlay — fetched separately and optionally (see DataLoader.run)
CX_ZEFF_FIELDS = [
    'charge_exchange.channel.zeff.data',
    'charge_exchange.channel.zeff.time',
    'charge_exchange.channel.position.r.data',
    'charge_exchange.channel.position.r.time',
    'charge_exchange.channel.position.z.data',
    'charge_exchange.channel.position.z.time',
]


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

class DataLoader(QtCore.QThread):
    """Fetches IDS data in a background thread."""

    loaded = QtCore.Signal(dict)   # emits the data dict on success
    error  = QtCore.Signal(str)    # emits error message on failure
    status = QtCore.Signal(str)    # progress messages

    def __init__(
        self,
        shot: int,
        efit_tree: str,
        efit_run_id: str,
        profiles_tree: str,
        profiles_run_id: str,
        parent=None,
    ):
        super().__init__(parent)
        self.shot = shot
        self.efit_tree = efit_tree
        self.efit_run_id = efit_run_id
        self.profiles_tree = profiles_tree
        self.profiles_run_id = profiles_run_id

    def run(self):
        try:

            composer = ImasComposer(
                efit_tree=self.efit_tree,
                efit_run_id=self.efit_run_id,
                profiles_tree=self.profiles_tree,
                profiles_run_id=self.profiles_run_id,
            )

            self.status.emit("Fetching equilibrium data…")
            eq_data = simple_load(EQ_FIELDS, self.shot, composer=composer)

            self.status.emit("Fetching wall data…")
            wall_data = simple_load(WALL_FIELDS, self.shot, composer=composer)

            self.status.emit("Fetching core profiles…")
            prof_data = simple_load(PROF_FIELDS, self.shot, composer=composer)

            # CER Zeff overlay is optional: a failed charge_exchange fetch must
            # not block the viewer, so its keys are simply absent on failure.
            self.status.emit("Fetching CER Zeff (optional)…")
            try:
                cx_data = simple_load(CX_ZEFF_FIELDS, self.shot, composer=composer)
            except Exception:
                print(traceback.format_exc(), file=sys.stderr)
                cx_data = {}

            # Shot comment is cosmetic (appended to the plot title); a missing
            # \D3D::TOP.COMMENTS:BRIEF node must not block the science panels.
            try:
                summary_data = simple_load(SUMMARY_FIELDS, self.shot, composer=composer)
            except Exception:
                summary_data = {'summary.description': None}

            eq_time = np.asarray(eq_data['equilibrium.time'])
            cp_time = np.asarray(prof_data['core_profiles.time'])
            assert len(eq_time) == len(cp_time), (
                f"equilibrium has {len(eq_time)} time slices but "
                f"core_profiles has {len(cp_time)}"
            )
            assert np.max(np.abs(eq_time - cp_time)) <= 1e-4, (
                "equilibrium and core_profiles time bases differ by more than 0.1 ms"
            )

            self.loaded.emit({**eq_data, **wall_data, **prof_data, **cx_data, **summary_data})

        except Exception:
            self.error.emit(traceback.format_exc())


RDB_TIMEOUT_MS = 10_000


class D3DrdbWorker(QtCore.QThread):
    """Runs a single D3DRDB callable in a background thread."""

    result = QtCore.Signal(object)  # emits the return value on success
    error  = QtCore.Signal(str)     # emits formatted traceback on failure

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            self.result.emit(self._fn(*self._args, **self._kwargs))
        except Exception:
            self.error.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _safe(data: Dict, key: str, t: int = 0):
    """Return data[key][t] if present, else None."""
    arr = data.get(key)
    if arr is None:
        return None
    try:
        if isinstance(arr, ak.Array):
            return np.asarray(arr[t])
        # shape: (n_t, ...)
        if hasattr(arr, '__len__') and len(arr) > t:
            return arr[t]
    except Exception:
        pass
    return None


def _mkcolor(color, alpha: float):
    """A QColor for *color* with the given 0–1 alpha."""
    c = pg.mkColor(color)
    c.setAlphaF(alpha)
    return c


def _cached(p, key: str, factory):
    """Return p's cached item for *key*, creating + adding it to p once.

    Persistent items let a replot update data via ``setData`` instead of the
    matplotlib-style ``clear`` + recreate, which keeps slider scrubbing fast.
    """
    item = p._cache.get(key)
    if item is None:
        item = factory()
        p.addItem(item)
        p._cache[key] = item
    return item


def _band(p, key: str, x, y_lo, y_hi, color, alpha: float = 0.25):
    """Update (creating once) a shaded band between y_lo and y_hi.

    The boundaries are PlotCurveItems (not PlotDataItems): FillBetweenItem reads
    their path directly via getPath(), whereas a PlotDataItem with pen=None never
    populates its internal curve, leaving the fill empty.
    """
    lo = _cached(p, key + ':lo', lambda: pg.PlotCurveItem(pen=None))
    hi = _cached(p, key + ':hi', lambda: pg.PlotCurveItem(pen=None))
    _cached(p, key + ':fill',
            lambda: pg.FillBetweenItem(hi, lo, brush=pg.mkBrush(_mkcolor(color, alpha))))
    lo.setData(x, y_lo)
    hi.setData(x, y_hi)


CONTOUR_LEVELS = np.linspace(0, 1, 12)[1:-1]   # 10 normalised-psi levels


def _contour_xy(R, Z, psi_n, levels=CONTOUR_LEVELS):
    """NaN-separated (x, y) polyline of all contour *levels*, in (R, Z) coords.

    contourpy's C++ engine generates these ~500× faster than pyqtgraph's
    pure-Python marching squares, so the contours can be rebuilt live per frame.
    """
    # contourpy indexes z[row=y, col=x]; psi_n is [r=x, z=y] -> transpose.
    cg = contour_generator(x=R, y=Z, z=psi_n.T)
    xs, ys = [], []
    for lvl in levels:
        for seg in cg.lines(float(lvl)):       # list of (npts, 2) arrays
            xs.append(seg[:, 0]); xs.append([np.nan])
            ys.append(seg[:, 1]); ys.append([np.nan])
    if not xs:
        return EMPTY, EMPTY
    return np.concatenate(xs), np.concatenate(ys)


def plot_equilibrium_cx(p, data: Dict, t: int):
    """Plot equilibrium cross-section: psi contours, LCFS, wall, X-points."""
    blue = COLORS['tab:blue']

    # --- wall ---
    wall_r = data.get('wall.description_2d.limiter.unit.outline.r')
    wall_z = data.get('wall.description_2d.limiter.unit.outline.z')
    wall = _cached(p, 'wall', lambda: pg.PlotDataItem(pen=pg.mkPen('k', width=1.5)))
    if wall_r is not None and wall_z is not None:
        wr = np.asarray(wall_r).ravel()
        wz = np.asarray(wall_z).ravel()
        wall.setData(wr, wz)
        p.setXRange(wr.min(), wr.max(), padding=0)
        p.setYRange(wz.min(), wz.max(), padding=0)

    # --- psi contours (single poly-line, rebuilt live via contourpy) ---
    contours = _cached(p, 'contours', lambda: pg.PlotCurveItem(
        pen=pg.mkPen(_mkcolor(blue, 0.7), width=0.7), connect='finite'))
    dim1 = data.get('equilibrium.time_slice.profiles_2d.grid.dim1')
    dim2 = data.get('equilibrium.time_slice.profiles_2d.grid.dim2')
    psi2d = data.get('equilibrium.time_slice.profiles_2d.psi')
    if dim1 is not None and dim2 is not None and psi2d is not None:
        R = np.asarray(dim1[t, 0, :])   # shape (n_r,)
        Z = np.asarray(dim2[t, 0, :])   # shape (n_z,)
        psi = np.asarray(psi2d[t, 0, :, :])   # shape (n_r, n_z)

        # normalise to [0, 1]
        psi_ax  = float(data['equilibrium.time_slice.global_quantities.psi_axis'][t])
        psi_bdy = float(data['equilibrium.time_slice.global_quantities.psi_boundary'][t])
        psi_n = (psi - psi_ax) / (psi_bdy - psi_ax)

        # sanitise edges
        psi_n_plot = psi_n.copy()
        psi_n_plot[:, -1] = psi_n_plot[:, -2]
        psi_n_plot[-1, :] = psi_n_plot[-2, :]

        # light smoothing for nicer contours (native resolution, ~1 ms)
        psi_smooth = scipy.ndimage.gaussian_filter(psi_n_plot, sigma=1.0)
        contours.setData(*_contour_xy(R, Z, psi_smooth))
    else:
        contours.setData(EMPTY, EMPTY)

    # --- LCFS ---
    bdy_r = data.get('equilibrium.time_slice.boundary.outline.r')
    bdy_z = data.get('equilibrium.time_slice.boundary.outline.z')
    lcfs = _cached(p, 'lcfs', lambda: pg.PlotDataItem(pen=pg.mkPen(blue, width=2)))
    if bdy_r is not None and bdy_z is not None:
        lcfs.setData(np.asarray(bdy_r[t]), np.asarray(bdy_z[t]))

    # --- X-points ---
    xpt_r = data.get('equilibrium.time_slice.boundary.x_point.r')
    xpt_z = data.get('equilibrium.time_slice.boundary.x_point.z')
    xpts = _cached(p, 'xpts', lambda: pg.ScatterPlotItem(
        symbol='x', size=10, pen=pg.mkPen(blue, width=2), brush=blue))
    if xpt_r is not None and xpt_z is not None:
        xr = np.asarray(xpt_r[t]).ravel()
        xz = np.asarray(xpt_z[t]).ravel()
        keep = xr > 0
        xpts.setData(xr[keep], xz[keep])

    # --- magnetic axis ---
    mag_r = data.get('equilibrium.time_slice.global_quantities.magnetic_axis.r')
    mag_z = data.get('equilibrium.time_slice.global_quantities.magnetic_axis.z')
    mag = _cached(p, 'magaxis', lambda: pg.ScatterPlotItem(
        symbol='+', size=10, pen=pg.mkPen(blue, width=2), brush=blue))
    if mag_r is not None and mag_z is not None:
        mag.setData([float(mag_r[t])], [float(mag_z[t])])


def _psi_norm(psi, psi_ax, psi_bdy):
    return (psi - psi_ax) / (psi_bdy - psi_ax)


# Kinetic pressures mapped from core_profiles (OMFIT_PROFS), in (base, colour, label) form
CP_PRESSURES = [
    ('core_profiles.profiles_1d.electrons.pressure',       '#ff7f0e', 'p<sub>e</sub>'),
    ('core_profiles.profiles_1d.pressure_ion_total',       '#2ca02c', 'p<sub>i</sub>'),
    ('core_profiles.profiles_1d.pressure_ion_non_thermal', '#9467bd', 'p<sub>fast</sub>'),
    ('core_profiles.profiles_1d.pressure_total',           'k',       'p<sub>tot</sub>'),
]


PSI_LABEL = 'Ψ<sub>n</sub>'


EMPTY = np.empty(0)


def plot_pressure(p, data: Dict, t: int):
    """Equilibrium fitted pressure + constraints, overlaid with kinetic pressures."""
    psi_ax  = float(data['equilibrium.time_slice.global_quantities.psi_axis'][t])
    psi_bdy = float(data['equilibrium.time_slice.global_quantities.psi_boundary'][t])

    efit = _cached(p, 'efit', lambda: pg.PlotDataItem(
        pen=pg.mkPen(COLORS['tab:blue'], width=1.5), name='EFIT'))
    psi1d = data.get('equilibrium.time_slice.profiles_1d.psi')
    pres  = data.get('equilibrium.time_slice.profiles_1d.pressure')
    if psi1d is not None and pres is not None:
        efit.setData(_psi_norm(psi1d[t], psi_ax, psi_bdy), pres[t] * 1e-3)
    else:
        efit.setData(EMPTY, EMPTY)

    # constraint points (error bars when available, plain markers otherwise)
    errbar = _cached(p, 'cerr', lambda: pg.ErrorBarItem(
        pen=pg.mkPen(_mkcolor('r', 0.3), width=0.8), beam=0.0))
    cpts = _cached(p, 'cpts', lambda: pg.ScatterPlotItem(
        symbol='o', size=4, brush=_mkcolor('r', 0.3), pen=None))
    c_psi = data.get('equilibrium.time_slice.constraints.pressure.position.psi')
    c_meas = data.get('equilibrium.time_slice.constraints.pressure.measured')
    c_err  = data.get('equilibrium.time_slice.constraints.pressure.measured_error_upper')
    if c_psi is not None and c_meas is not None:
        cx = np.asarray(_psi_norm(c_psi[t], psi_ax, psi_bdy))
        cy = np.asarray(c_meas[t]) * 1e-3
        if c_err is not None:
            ce = np.asarray(c_err[t]) * 1e-3
            errbar.setData(x=cx, y=cy, top=ce, bottom=ce)
            cpts.setData(EMPTY, EMPTY)
        else:
            cpts.setData(cx, cy)
            errbar.setData(x=EMPTY, y=EMPTY, top=EMPTY, bottom=EMPTY)
    else:
        cpts.setData(EMPTY, EMPTY)
        errbar.setData(x=EMPTY, y=EMPTY, top=EMPTY, bottom=EMPTY)

    # kinetic pressures from core_profiles (OMFIT_PROFS only)
    xk = _cp_psin(data, t)
    for base, color, label in CP_PRESSURES:
        line = _cached(p, base, lambda c=color, l=label: pg.PlotDataItem(
            pen=pg.mkPen(c, width=1.0), name=l))
        y = _slice(data.get(base), t) if xk is not None else None
        line.setData(xk, y * 1e-3) if y is not None else line.setData(EMPTY, EMPTY)


# Current-density components from core_profiles (OMFIT_PROFS), in (base, colour, label) form
CP_CURRENTS = [
    ('core_profiles.profiles_1d.j_tor',       'k',       'j<sub>tor</sub>'),
    ('core_profiles.profiles_1d.j_ohmic',     '#ff7f0e', 'j<sub>ohm</sub>'),
    ('core_profiles.profiles_1d.j_bootstrap', '#2ca02c', 'j<sub>BS</sub>'),
]


def plot_j_tor(p, data: Dict, t: int):
    """EFIT toroidal current density + constraints, overlaid with the
    core_profiles current-density components (total, ohmic, bootstrap)."""
    psi_ax  = float(data['equilibrium.time_slice.global_quantities.psi_axis'][t])
    psi_bdy = float(data['equilibrium.time_slice.global_quantities.psi_boundary'][t])

    efit = _cached(p, 'efit', lambda: pg.PlotDataItem(
        pen=pg.mkPen(COLORS['tab:blue'], width=1.5), name='EFIT j<sub>tor</sub>'))
    psi1d = data.get('equilibrium.time_slice.profiles_1d.psi')
    jtor  = data.get('equilibrium.time_slice.profiles_1d.j_tor')
    if psi1d is not None and jtor is not None:
        efit.setData(_psi_norm(psi1d[t], psi_ax, psi_bdy), jtor[t] / 1e6)
    else:
        efit.setData(EMPTY, EMPTY)

    # constraint scatter
    cpts = _cached(p, 'cpts', lambda: pg.ScatterPlotItem(
        symbol='o', size=6, brush=_mkcolor('r', 0.4), pen=None))
    c_psi  = data.get('equilibrium.time_slice.constraints.j_tor.position.psi')
    c_meas = data.get('equilibrium.time_slice.constraints.j_tor.measured')
    if c_psi is not None and c_meas is not None:
        cpts.setData(np.asarray(_psi_norm(c_psi[t], psi_ax, psi_bdy)),
                     np.asarray(c_meas[t]) / 1e6)
    else:
        cpts.setData(EMPTY, EMPTY)

    # current-density components from core_profiles (OMFIT_PROFS only)
    # eq and core_profiles share a time base (enforced on load) -> same index t
    xk = _cp_psin(data, t)
    for base, color, label in CP_CURRENTS:
        line = _cached(p, base, lambda c=color, l=label: pg.PlotDataItem(
            pen=pg.mkPen(c, width=1.0), name=l))
        y = _slice(data.get(base), t) if xk is not None else None
        line.setData(xk, y / 1e6) if y is not None else line.setData(EMPTY, EMPTY)


def _cer_zeff_points(data: Dict, t: int):
    """(psi_n, zeff) CER points nearest to eq time slice *t*, or empty arrays.

    Channel (R, Z) positions are mapped to psi_n via the equilibrium 2D psi map.
    The charge_exchange fetch is optional, so missing keys (or empty/ragged
    channels) simply yield no points.
    """
    z_data = data.get('charge_exchange.channel.zeff.data')
    z_time = data.get('charge_exchange.channel.zeff.time')
    pos_r  = data.get('charge_exchange.channel.position.r.data')
    pos_rt = data.get('charge_exchange.channel.position.r.time')
    pos_z  = data.get('charge_exchange.channel.position.z.data')
    pos_zt = data.get('charge_exchange.channel.position.z.time')
    times  = data.get('equilibrium.time')
    dim1   = data.get('equilibrium.time_slice.profiles_2d.grid.dim1')
    dim2   = data.get('equilibrium.time_slice.profiles_2d.grid.dim2')
    psi2d  = data.get('equilibrium.time_slice.profiles_2d.psi')
    if any(v is None for v in (z_data, z_time, pos_r, pos_rt, pos_z, pos_zt,
                               times, dim1, dim2, psi2d)):
        return EMPTY, EMPTY

    psi_ax  = float(data['equilibrium.time_slice.global_quantities.psi_axis'][t])
    psi_bdy = float(data['equilibrium.time_slice.global_quantities.psi_boundary'][t])
    psi_n = (np.asarray(psi2d[t, 0, :, :]) - psi_ax) / (psi_bdy - psi_ax)
    interp = RegularGridInterpolator(
        (np.asarray(dim1[t, 0, :]), np.asarray(dim2[t, 0, :])), psi_n,
        bounds_error=False, fill_value=np.nan)

    t_now = float(times[t])
    # A CER sample "belongs" to this slice if it is closer than half a slice.
    tol = 0.5 * float(np.median(np.diff(np.asarray(times))))

    def nearest(values, time_axis):
        """Sample of *values* nearest to t_now (a lone sample is time-independent)."""
        vals = np.asarray(values)
        if len(vals) == 0:
            return None
        if len(vals) == 1:
            return float(vals[0])
        return float(vals[np.argmin(np.abs(np.asarray(time_axis) - t_now))])

    xs, ys = [], []
    for i in range(len(z_data)):
        zt = np.asarray(z_time[i])
        zv = np.asarray(z_data[i])
        if len(zv) == 0 or len(zt) != len(zv):
            continue
        j = int(np.argmin(np.abs(zt - t_now)))
        if abs(zt[j] - t_now) > tol:
            continue
        r = nearest(pos_r[i], pos_rt[i])
        z = nearest(pos_z[i], pos_zt[i])
        if r is None or z is None:
            continue
        x = float(interp((r, z)))
        if np.isfinite(x) and np.isfinite(zv[j]):
            xs.append(x)
            ys.append(float(zv[j]))
    if not xs:
        return EMPTY, EMPTY
    return np.asarray(xs), np.asarray(ys)


def plot_zeff(p, data: Dict, t: int):
    """OMFIT_PROFS Zeff profile + CER (charge_exchange) point measurements."""
    plot_profile_quantity(p, data, t, 'core_profiles.profiles_1d.zeff', None,
                          COLORS['tab:blue'], label='OMFIT_PROFS')

    pts = _cached(p, 'cer', lambda: pg.ScatterPlotItem(
        symbol='o', size=5, brush=_mkcolor('r', 0.6), pen=None, name='CER'))
    pts.setData(*_cer_zeff_points(data, t))


def plot_convergence_error(p, data: Dict, t: int):
    """Convergence error vs. time (all slices), vertical line at current time."""
    line = _cached(p, 'line', lambda: pg.PlotDataItem(
        pen=pg.mkPen(COLORS['tab:blue'], width=1)))
    tmark = _cached(p, 'tmark', lambda: pg.InfiniteLine(
        angle=90, pen=pg.mkPen('k', width=1, style=QtCore.Qt.PenStyle.DashLine)))

    times = data.get('equilibrium.time')
    cerr  = data.get('equilibrium.time_slice.convergence.grad_shafranov_deviation_value')
    if times is not None and cerr is not None:
        tt = np.asarray(times) * 1e3
        cc = np.asarray(cerr).ravel()
        mask = np.isfinite(cc) & (cc > 0)   # log axis drops non-positive values
        line.setData(tt[mask], cc[mask])
        tmark.setValue(float(times[t]) * 1e3)
    else:
        line.setData(EMPTY, EMPTY)


ION_COLORS = ['#2ca02c', '#9467bd', '#d62728', '#1f77b4', '#ff7f0e', '#8c564b']


def _slice(arr, cp_t: int, ion_index: Optional[int] = None):
    """Return arr[cp_t] (or arr[cp_t][ion_index]) as a numpy array, or None."""
    if arr is None:
        return None
    try:
        sub = arr[cp_t] if ion_index is None else arr[cp_t][ion_index]
    except (TypeError, IndexError):
        return None
    return np.asarray(sub)


def _ion_label(data: Dict, cp_t: int, ion_index: int) -> Optional[str]:
    """Return the species label (e.g. 'D', 'C') for an ion index, or None."""
    labels = data.get('core_profiles.profiles_1d.ion.label')
    if labels is None:
        return None
    try:
        return str(labels[cp_t][ion_index])
    except (TypeError, IndexError):
        return None


def _fit_err(color):
    it = pg.ErrorBarItem(beam=0.0, pen=pg.mkPen(_mkcolor(color, 0.5), width=0.5))
    it.setZValue(-1)
    return it


def _fit_pts(color):
    it = pg.ScatterPlotItem(size=4, brush=_mkcolor(color, 0.5), pen=None)
    it.setZValue(-1)
    return it


def plot_profile_quantity(p, data: Dict, cp_t: int, base: str, fit: Optional[str],
                          color: str, scale: float = 1.0, *,
                          ion_index: Optional[int] = None, label: Optional[str] = None):
    """Plot one quantity of one species: smooth profile + band + raw fit points.

    *p* is a pyqtgraph ``PlotItem`` or a twin ``ViewBox``. Items are cached on
    *p* (keyed by ``base`` + species index) and updated via ``setData`` so a
    replot never rebuilds them. Only the smooth profile line carries *label*, so
    a single legend entry per species covers it and its fit points (shared
    *color*). Returns the profile line item for twin-axis legend registration.
    """
    key = f'{base}:{ion_index}'
    line = _cached(p, key + ':line',
                   lambda: pg.PlotDataItem(pen=pg.mkPen(color, width=1.5), name=label))

    x = _cp_psin(data, cp_t)
    y = _slice(data.get(base), cp_t, ion_index)
    if x is not None and y is not None:
        y_s = y * scale
        yerr = _slice(data.get(base + '_error_upper'), cp_t, ion_index)
        if yerr is not None and len(yerr) > 0:
            ye_s = yerr * scale
            _band(p, key + ':band', x, y_s - ye_s, y_s + ye_s, color)
        else:
            _band(p, key + ':band', EMPTY, EMPTY, EMPTY, color)
        line.setData(x, y_s)
    else:
        line.setData(EMPTY, EMPTY)
        _band(p, key + ':band', EMPTY, EMPTY, EMPTY, color)

    err = _cached(p, key + ':ferr', lambda: _fit_err(color))
    pts = _cached(p, key + ':fpts', lambda: _fit_pts(color))
    fx = fy = fe = None
    if fit is not None:
        fx = _slice(data.get(fit + '.psi_norm'), cp_t, ion_index)
        fy = _slice(data.get(fit + '.measured'), cp_t, ion_index)
    if fx is not None and fy is not None:
        fy = fy * scale
        fe = _slice(data.get(fit + '.measured_error_upper'), cp_t, ion_index)
        mask = np.isfinite(fx) & np.isfinite(fy)
        if fe is not None:
            fe = fe * scale
            # remove NaNs and 100 % uncertainty points
            mask &= np.isfinite(fe)
            mask[mask] &= np.abs(fe[mask]) < np.abs(fy[mask])
        if mask.any():
            pts.setData(fx[mask], fy[mask])
            if fe is not None:
                err.setData(x=fx[mask], y=fy[mask], top=fe[mask], bottom=fe[mask])
            else:
                err.setData(x=EMPTY, y=EMPTY, top=EMPTY, bottom=EMPTY)
        else:
            pts.setData(EMPTY, EMPTY)
            err.setData(x=EMPTY, y=EMPTY, top=EMPTY, bottom=EMPTY)
    else:
        pts.setData(EMPTY, EMPTY)
        err.setData(x=EMPTY, y=EMPTY, top=EMPTY, bottom=EMPTY)

    return line


def _prune_species(container, base: str, keep, legend=None):
    """Drop cached series of *base* whose species index is not in *keep*."""
    for ckey in list(container._cache):
        if not ckey.startswith(base + ':'):
            continue
        idx = ckey[len(base) + 1:].split(':', 1)[0]
        if idx.isdigit() and int(idx) not in keep:
            item = container._cache.pop(ckey)
            container.removeItem(item)
            if legend is not None:
                legend.removeItem(item)


def plot_ion_quantity(p, data: Dict, cp_t: int, base: str, fit: Optional[str],
                      scale: float = 1.0):
    """Plot one ion quantity for every species, one colour per species."""
    y = data.get(base)
    n = len(y[cp_t]) if y is not None else 0
    for i in range(n):
        plot_profile_quantity(p, data, cp_t, base, fit, ION_COLORS[i], scale,
                              ion_index=i, label=_ion_label(data, cp_t, i))
    _prune_species(p, base, set(range(n)))


def plot_ion_density(p, vb2, data: Dict, cp_t: int):
    """Ion density: main ion on the left axis, minorities (×100) on the twin axis."""
    base = 'core_profiles.profiles_1d.ion.density_thermal'
    fit  = 'core_profiles.profiles_1d.ion.density_fit'
    y = data.get(base)
    n = len(y[cp_t]) if y is not None else 0
    for i in range(n):
        target = p if i == 0 else vb2
        scale = 1e-19 if i == 0 else 1e-17
        label = _ion_label(data, cp_t, i)
        line = plot_profile_quantity(target, data, cp_t, base, fit, ION_COLORS[i],
                                     scale, ion_index=i, label=label)
        # twin-axis curves live in vb2, not p, so register them in p's legend by hand
        if i > 0 and p.legend is not None and p.legend.getLabel(line) is None:
            p.legend.addItem(line, label or '')
    # Main axis holds only ion 0; the twin holds ions >= 1.
    _prune_species(p, base, {0} if n else set())
    _prune_species(vb2, base, set(range(1, n)), legend=p.legend)


def _cp_psin(data: Dict, cp_t: int):
    """psi_norm = rho_pol_norm² for the given core_profiles time index."""
    rho = data.get('core_profiles.profiles_1d.grid.rho_pol_norm')
    if rho is None:
        return None
    return np.asarray(rho[cp_t]) ** 2


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class IriCakeViewer(QtWidgets.QMainWindow):

    def __init__(self, shot: int = -1, flavor: str = 'IRI_CAKE01'):
        super().__init__()
        self.setWindowTitle('IRI CAKE Viewer')
        self.resize(1700, 1000)

        self._data: Optional[Dict[str, Any]] = None
        self._loader: Optional[DataLoader] = None
        self._rdb_worker: Optional[D3DrdbWorker] = None
        # Every started QThread lives here until its run() actually returns, so
        # Qt never destroys a still-running thread (which aborts the process).
        self._live_threads: list = []
        self._rdb_timeout: Optional[QtCore.QTimer] = None
        self._pending_load_params: Optional[tuple] = None
        self._shot = shot
        self._flavor = flavor
        # A CLI --shot N is the one auto-load path: fetch it once the shot list
        # has been queried (so the two D3DRDB calls never overlap).
        self._pending_autofetch = shot > 0

        self._build_ui()

        QtCore.QTimer.singleShot(200, self._populate_tags)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ---- control row 1 ----
        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(QtWidgets.QLabel('Tag:'))
        self._flavor_combo = QtWidgets.QComboBox()
        # Items are populated from D3DRDB by _populate_tags() so the list stays up to date.
        self._flavor_combo.setCurrentText(self._flavor)
        self._flavor_combo.setEditable(True)
        self._flavor_combo.setFixedWidth(180)
        row1.addWidget(self._flavor_combo)

        row1.addWidget(QtWidgets.QLabel('Shot:'))
        self._shot_combo = QtWidgets.QComboBox()
        self._shot_combo.setEditable(True)
        self._shot_combo.setFixedWidth(100)
        if self._shot > 0:
            self._shot_combo.setCurrentText(str(self._shot))
        row1.addWidget(self._shot_combo)

        row1.addWidget(QtWidgets.QLabel('EFIT tree:'))
        self._efit_combo = QtWidgets.QComboBox()
        self._efit_combo.addItems(['EFIT'])
        self._efit_combo.setEditable(True)
        self._efit_combo.setFixedWidth(100)
        row1.addWidget(self._efit_combo)

        row1.addWidget(QtWidgets.QLabel('Run ID:'))
        self._efit_id_edit = QtWidgets.QLineEdit()
        self._efit_id_edit.setPlaceholderText('auto')
        self._efit_id_edit.setFixedWidth(50)
        row1.addWidget(self._efit_id_edit)

        row1.addWidget(QtWidgets.QLabel('Profile tree:'))
        self._prof_combo = QtWidgets.QComboBox()
        self._prof_combo.addItems(['OMFIT_PROFS']) #, 'ZIPFIT01', 'ZIPFIT02'
        self._prof_combo.setEditable(True)
        self._prof_combo.setFixedWidth(130)
        row1.addWidget(self._prof_combo)

        row1.addWidget(QtWidgets.QLabel('Run ID:'))
        self._prof_id_edit = QtWidgets.QLineEdit()
        self._prof_id_edit.setPlaceholderText('auto')
        self._prof_id_edit.setFixedWidth(50)
        row1.addWidget(self._prof_id_edit)

        self._fetch_btn = QtWidgets.QPushButton('Fetch Shot')
        self._fetch_btn.setFixedWidth(90)
        self._fetch_btn.clicked.connect(self._trigger_fetch_shot)
        row1.addWidget(self._fetch_btn)

        row1.addStretch()
        root.addLayout(row1)

        # Reset run IDs when the context that determined them changes
        self._shot_combo.currentTextChanged.connect(lambda _: self._reset_run_ids(efit=True, prof=True))
        self._flavor_combo.currentTextChanged.connect(lambda _: self._reset_run_ids(efit=True, prof=True))
        # The available shots depend on the tag, so repopulate the shot list when it changes.
        self._flavor_combo.currentTextChanged.connect(lambda _: self._populate_shots())
        self._efit_combo.currentTextChanged.connect(lambda _: self._reset_run_ids(efit=True, prof=False))
        self._prof_combo.currentTextChanged.connect(lambda _: self._reset_run_ids(efit=False, prof=True))

        # ---- status bar ----
        self._status_label = QtWidgets.QLabel('Ready')
        self._status_label.setStyleSheet('color: grey; font-style: italic;')
        root.addWidget(self._status_label)

        # ---- time slider row ----
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel('Time:'))
        self._time_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._time_slider.setMinimum(0)
        self._time_slider.setMaximum(0)
        self._time_slider.valueChanged.connect(self._on_time_changed)
        row2.addWidget(self._time_slider, stretch=1)
        self._time_label = QtWidgets.QLabel('—')
        self._time_label.setFixedWidth(65)
        row2.addWidget(self._time_label)
        self._scalar_label = QtWidgets.QLabel('')
        row2.addWidget(self._scalar_label)
        root.addLayout(row2)

        # ---- pyqtgraph plotting surface ----
        self._glw = pg.GraphicsLayoutWidget()
        root.addWidget(self._glw, stretch=1)

        self._build_axes()

    def _build_axes(self):
        """Create the 2×6 plot grid (Eq. CX spans both rows of column 0)."""
        self._glw.clear()

        # Row 0: figure-level title spanning all six columns.
        self._title_label = self._glw.addLabel('', row=0, col=0, colspan=6, size='11pt')

        # Eq. CX spans both rows in column 0.
        self._ax_cx     = self._glw.addPlot(row=1, col=0, rowspan=2)
        self._ax_ne     = self._glw.addPlot(row=1, col=1)
        self._ax_te     = self._glw.addPlot(row=1, col=2)
        self._ax_jtor   = self._glw.addPlot(row=1, col=3)
        self._ax_pres   = self._glw.addPlot(row=1, col=4)
        self._ax_conv   = self._glw.addPlot(row=1, col=5)
        self._ax_ni     = self._glw.addPlot(row=2, col=1)
        self._ax_ti     = self._glw.addPlot(row=2, col=2)
        self._ax_vtor   = self._glw.addPlot(row=2, col=3)
        self._ax_efield = self._glw.addPlot(row=2, col=4)
        self._ax_zeff   = self._glw.addPlot(row=2, col=5)

        self._ax_cx.hideButtons()

        # Legends for the panels that overlay several named series.
        for ax in (self._ax_jtor, self._ax_ni, self._ax_ti, self._ax_vtor,
                   self._ax_pres, self._ax_zeff):
            ax.addLegend(offset=(-5, 5), labelTextSize='7pt')

        # Minority-ion density (×100) on a twin y-axis linked to the ni panel.
        self._ax_ni2 = pg.ViewBox()
        self._ax_ni.showAxis('right')
        self._ax_ni.scene().addItem(self._ax_ni2)
        self._ax_ni.getAxis('right').linkToView(self._ax_ni2)
        self._ax_ni2.setXLink(self._ax_ni)
        self._ax_ni.getViewBox().sigResized.connect(
            lambda vb: self._ax_ni2.setGeometry(vb.sceneBoundingRect()))
        self._ax_ni2.setGeometry(self._ax_ni.getViewBox().sceneBoundingRect())

        self._plots = [
            self._ax_cx, self._ax_ne, self._ax_te, self._ax_jtor, self._ax_ni,
            self._ax_ti, self._ax_conv, self._ax_vtor, self._ax_pres, self._ax_efield,
            self._ax_zeff,
        ]

        # Per-panel item cache: plot items are created once and updated via
        # setData, so a replot (e.g. while scrubbing the slider) is cheap.
        for ax in self._plots:
            ax._cache = {}
        self._ax_ni2._cache = {}

        # Static decoration (never changes per time slice) set once here.
        self._ax_cx.setAspectLocked(True)
        self._ax_cx.setLabel('bottom', 'R [m]')
        self._ax_cx.setLabel('left', 'Z [m]')
        self._ax_conv.setLogMode(y=True)
        self._ax_ni.getAxis('right').setLabel(
            'n<sub>C</sub>×100 [10<sup>19</sup> m<sup>-3</sup>]', color=ION_COLORS[1])
        titles = {
            self._ax_ne:     'n<sub>e</sub> [10<sup>19</sup> m<sup>-3</sup>]',
            self._ax_te:     'T<sub>e</sub> [keV]',
            self._ax_jtor:   'j [MA m<sup>-2</sup>]',
            self._ax_ni:     'n<sub>i</sub> [10<sup>19</sup> m<sup>-3</sup>]',
            self._ax_ti:     'T<sub>i</sub> [keV]',
            self._ax_conv:   'Convergence error',
            self._ax_vtor:   'v<sub>tor</sub> [km/s]',
            self._ax_pres:   'Pressure [kPa]',
            self._ax_efield: 'E<sub>r</sub> [kV/m]',
            self._ax_zeff:   'Z<sub>eff</sub>',
        }
        for ax, title in titles.items():
            ax.setTitle(title, size='9pt')
            ax.setLabel('bottom', PSI_LABEL)
        self._ax_conv.setLabel('bottom', 'Time [ms]')

        # Slider throttle: render the first move immediately, then coalesce
        # rapid moves to ~20 fps with a guaranteed trailing render.
        self._replot_pending = False
        self._replot_interval_ms = 50
        self._replot_timer = QtCore.QTimer(self, singleShot=True)
        self._replot_timer.timeout.connect(self._on_replot_timer)

    # ------------------------------------------------------------------
    # Fetch logic
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # D3DRDB helpers (non-blocking)
    # ------------------------------------------------------------------

    def _start_rdb_worker(self, fn, *args, status_msg: str = 'Querying D3DRDB…', **kwargs):
        """Launch *fn* in a D3DrdbWorker with a 10-second timeout watchdog."""
        self._cancel_rdb_worker()
        self._set_buttons_enabled(False)
        self._status_label.setStyleSheet('color: grey; font-style: italic;')
        self._status_label.setText(status_msg)

        self._rdb_worker = D3DrdbWorker(fn, *args, **kwargs)
        self._live_threads.append(self._rdb_worker)
        self._rdb_worker.finished.connect(lambda t=self._rdb_worker: self._reap_thread(t))

        self._rdb_timeout = QtCore.QTimer(singleShot=True)
        self._rdb_timeout.timeout.connect(self._on_rdb_timeout)
        self._rdb_timeout.start(RDB_TIMEOUT_MS)

        return self._rdb_worker

    def _cancel_rdb_worker(self):
        """Stop the timeout and stop listening to the in-flight rdb worker.

        The worker is *not* deleted here: it stays in ``_live_threads`` and
        self-reaps once its (possibly still-running) call returns, so Qt never
        destroys a running QThread.
        """
        if self._rdb_timeout is not None:
            self._rdb_timeout.stop()
            self._rdb_timeout = None
        if self._rdb_worker is not None:
            for sig in (self._rdb_worker.result, self._rdb_worker.error):
                try:
                    sig.disconnect()
                except RuntimeError:
                    pass
            self._rdb_worker = None

    def _reap_thread(self, thread):
        """Drop our reference to a thread once it has truly finished."""
        if thread in self._live_threads:
            self._live_threads.remove(thread)
        if thread is self._loader:
            self._loader = None
        if thread is self._rdb_worker:
            self._rdb_worker = None
        thread.deleteLater()

    def _on_rdb_timeout(self):
        self._cancel_rdb_worker()
        self._set_buttons_enabled(True)
        self._status_label.setStyleSheet('color: orange; font-weight: bold;')
        self._status_label.setText(
            'D3DRDB connection timed out (10 s). '
            'Check network, or enter EFIT / profile run IDs manually.'
        )

    def _on_rdb_error(self, msg: str):
        self._cancel_rdb_worker()
        self._set_buttons_enabled(True)
        last_line = msg.strip().splitlines()[-1]
        self._status_label.setStyleSheet('color: red; font-style: italic;')
        self._status_label.setText(f'D3DRDB error: {last_line}')
        print(msg, file=sys.stderr)

    # ------------------------------------------------------------------
    # Fetch actions
    # ------------------------------------------------------------------

    def _selected_shot(self) -> Optional[int]:
        """Parse the shot combo's current text, or flag a bad entry and return None."""
        text = self._shot_combo.currentText().strip()
        try:
            return int(text)
        except ValueError:
            self._status_label.setStyleSheet('color: red; font-style: italic;')
            self._status_label.setText(f'Invalid shot: {text!r}')
            return None

    def _populate_tags(self):
        """Query D3DRDB for the available tags to populate the tag combo."""
        worker = self._start_rdb_worker(
            list_all_tags,
            status_msg='Querying D3DRDB for tags…',
        )
        worker.result.connect(self._on_tags_found)
        worker.error.connect(self._on_rdb_error)
        worker.start()

    def _on_tags_found(self, tags):
        self._cancel_rdb_worker()
        self._set_buttons_enabled(True)
        # Repopulate without firing the tag-changed handlers for each programmatic change.
        current = self._flavor_combo.currentText()
        self._flavor_combo.blockSignals(True)
        self._flavor_combo.clear()
        self._flavor_combo.addItems(tags)
        self._flavor_combo.setCurrentText(current)
        self._flavor_combo.blockSignals(False)
        # Only one D3DRDB call runs at a time, so fetch shots now that tags are ready.
        self._populate_shots()

    def _populate_shots(self):
        """Query D3DRDB for the shots available under the current tag."""
        worker = self._start_rdb_worker(
            list_shots_for_tag, self._flavor_combo.currentText(),
            status_msg='Querying D3DRDB for shots…',
        )
        worker.result.connect(self._on_shots_found)
        worker.error.connect(self._on_rdb_error)
        worker.start()

    def _on_shots_found(self, shots):
        self._cancel_rdb_worker()
        self._set_buttons_enabled(True)
        # Repopulate without firing the run-id reset for each programmatic change.
        self._shot_combo.blockSignals(True)
        self._shot_combo.clear()
        self._shot_combo.addItems([str(s) for s in shots])   # most-recent first
        self._shot_combo.blockSignals(False)
        self._status_label.setStyleSheet('color: grey; font-style: italic;')
        self._status_label.setText(f'{len(shots)} shots for tag {self._flavor_combo.currentText()}')

        # A CLI --shot N auto-loads once, after the list is available.
        if self._pending_autofetch:
            self._pending_autofetch = False
            self._shot_combo.setCurrentText(str(self._shot))
            self._trigger_fetch_shot()

    def _trigger_fetch_shot(self):
        shot = self._selected_shot()
        if shot is None:
            return
        flavor    = self._flavor_combo.currentText()
        eq_id     = self._efit_id_edit.text().strip()
        prof_id   = self._prof_id_edit.text().strip()
        efit_tree = self._efit_combo.currentText().strip() or 'EFIT'
        prof_tree = self._prof_combo.currentText().strip() or 'OMFIT_PROFS'

        if eq_id and prof_id:
            # Both IDs already provided — skip D3DRDB entirely
            self._start_load(shot, efit_tree, eq_id, prof_tree, prof_id)
            return

        # Save the non-ID params so _on_ids_found can complete the load
        self._pending_load_params = (shot, efit_tree, prof_tree, eq_id, prof_id)
        worker = self._start_rdb_worker(
            get_iri_upload_ids, shot, flavor,
            status_msg=f'Querying D3DRDB for shot {shot}…',
        )
        worker.result.connect(self._on_ids_found)
        worker.error.connect(self._on_rdb_error)
        worker.start()

    def _on_ids_found(self, result):
        self._cancel_rdb_worker()
        auto_prof, auto_eq = result
        shot, efit_tree, prof_tree, eq_id, prof_id = self._pending_load_params
        self._pending_load_params = None

        if not eq_id:
            eq_id = auto_eq
            self._efit_id_edit.setText(eq_id)
        if not prof_id:
            prof_id = auto_prof
            self._prof_id_edit.setText(prof_id)

        self._start_load(shot, efit_tree, eq_id, prof_tree, prof_id)

    def _start_load(self, shot, efit_tree, efit_run_id, profiles_tree, profiles_run_id):
        if self._loader is not None and self._loader.isRunning():
            # Detach the in-flight loader instead of terminate()-ing it:
            # killing a thread mid IMAS/MDSplus call corrupts state and crashes.
            # It stays in _live_threads and self-reaps when run() returns.
            for sig in (self._loader.loaded, self._loader.error, self._loader.status):
                try:
                    sig.disconnect()
                except RuntimeError:
                    pass
            self._loader = None

        self._set_buttons_enabled(False)
        self._status_label.setStyleSheet('color: grey; font-style: italic;')
        self._status_label.setText(
            f'Loading shot {shot}  EFIT={efit_tree}{efit_run_id}  '
            f'PROFS={profiles_tree}{profiles_run_id}…'
        )
        # Restore panel visibility in case a previous fetch showed an error
        for ax in self._plots:
            ax.setVisible(True)

        self._loader = DataLoader(shot, efit_tree, efit_run_id, profiles_tree, profiles_run_id)
        self._live_threads.append(self._loader)
        self._loader.finished.connect(lambda t=self._loader: self._reap_thread(t))
        self._loader.status.connect(self._status_label.setText)
        self._loader.loaded.connect(self._on_load_finished)
        self._loader.error.connect(self._on_load_error)
        self._loader.start()

    def _on_load_finished(self, data: Dict):
        self._data = data
        self._set_buttons_enabled(True)

        times = data.get('equilibrium.time')
        if times is not None:
            n = len(times)
            self._time_slider.setMaximum(max(n - 1, 0))
            self._time_slider.setValue(n // 2)
        else:
            self._time_slider.setMaximum(0)
            self._time_slider.setValue(0)

        self._status_label.setStyleSheet('color: grey; font-style: italic;')
        self._status_label.setText(
            f'Loaded shot {self._shot_combo.currentText()}  —  '
            f'{len(np.asarray(times)) if times is not None else 0} time slices'
        )
        self._replot()

    def _on_load_error(self, msg: str):
        self._set_buttons_enabled(True)
        print(msg, file=sys.stderr)
        self._show_fetch_error(msg)

    def _show_fetch_error(self, msg: str):
        """Clear all plots and show a concise error in the status bar."""
        self._status_label.setStyleSheet('color: red; font-style: italic;')
        self._status_label.setText('Load error - see console')

        self._ax_ni2.clear()
        self._ax_ni2._cache.clear()
        for ax in self._plots:
            ax.clear()
            ax._cache.clear()   # force a clean item rebuild on the next load
            ax.setVisible(False)

        self._title_label.setText('Load error - see console')

    def _reset_run_ids(self, *, efit: bool, prof: bool):
        if efit:
            self._efit_id_edit.clear()
        if prof:
            self._prof_id_edit.clear()

    def _set_buttons_enabled(self, enabled: bool):
        self._fetch_btn.setEnabled(enabled)

    def closeEvent(self, event):
        """Wait for in-flight threads so none is destroyed while still running."""
        for thread in list(self._live_threads):
            try:
                thread.wait(5000)
            except RuntimeError:
                pass
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _on_time_changed(self, value: int):
        if self._data is None:
            return
        times = self._data.get('equilibrium.time')
        if times is not None and value < len(times):
            t_ms = float(times[value]) * 1e3
            self._time_label.setText(f'{t_ms:.1f} ms')
            self._update_scalars(value)
        # Throttle: render the first move now, coalesce the rest (see _build_axes).
        self._replot_pending = True
        if not self._replot_timer.isActive():
            self._replot_pending = False
            self._replot()
            self._replot_timer.start(self._replot_interval_ms)

    def _on_replot_timer(self):
        if self._replot_pending:
            self._replot_pending = False
            self._replot()
            self._replot_timer.start(self._replot_interval_ms)

    def _update_scalars(self, t: int):
        d = self._data
        if d is None:
            return
        parts = []
        ip = d.get('equilibrium.time_slice.global_quantities.ip')
        if ip is not None:
            parts.append(f'Ip={float(ip[t])*1e-6:.2f} MA')
        q95 = d.get('equilibrium.time_slice.global_quantities.q_95')
        if q95 is not None:
            parts.append(f'q₉₅={float(q95[t]):.2f}')
        bn = d.get('equilibrium.time_slice.global_quantities.beta_normal')
        if bn is not None:
            parts.append(f'βN={float(bn[t]):.2f}')
        self._scalar_label.setText('   '.join(parts))

    def _replot(self):
        if self._data is None:
            return

        t = self._time_slider.value()
        d = self._data

        # equilibrium and core_profiles share a time base (enforced on load)
        times_eq = d.get('equilibrium.time')

        orange, blue = COLORS['tab:orange'], COLORS['tab:blue']
        cp = 'core_profiles.profiles_1d'
        for fn, ax in [
            (lambda ax: plot_equilibrium_cx(ax, d, t),        self._ax_cx),
            (lambda ax: plot_profile_quantity(
                ax, d, t, f'{cp}.electrons.density', f'{cp}.electrons.density_fit',
                orange, 1e-19), self._ax_ne),
            (lambda ax: plot_ion_density(self._ax_ni, self._ax_ni2, d, t), self._ax_ni),
            (lambda ax: plot_profile_quantity(
                ax, d, t, f'{cp}.electrons.temperature', f'{cp}.electrons.temperature_fit',
                orange, 1e-3), self._ax_te),
            (lambda ax: plot_ion_quantity(
                ax, d, t, f'{cp}.ion.temperature', f'{cp}.ion.temperature_fit', 1e-3),
                self._ax_ti),
            (lambda ax: plot_ion_quantity(
                ax, d, t, f'{cp}.ion.velocity.toroidal', f'{cp}.ion.velocity.toroidal_fit',
                1e-3), self._ax_vtor),
            (lambda ax: plot_j_tor(ax, d, t),                 self._ax_jtor),
            (lambda ax: plot_convergence_error(ax, d, t),     self._ax_conv),
            (lambda ax: plot_pressure(ax, d, t),              self._ax_pres),
            (lambda ax: plot_profile_quantity(
                ax, d, t, f'{cp}.e_field.radial', None, blue, 1e-3), self._ax_efield),
            (lambda ax: plot_zeff(ax, d, t),                  self._ax_zeff),
        ]:
            try:
                fn(ax)
            except Exception as e:
                # Keep persistent items intact (do not clear the cache); just
                # surface the error in the panel title.
                print(traceback.format_exc(), file=sys.stderr)
                ax.setTitle(str(e), color='r', size='7pt')

        # Title
        if times_eq is not None and t < len(times_eq):
            shot = self._shot_combo.currentText()
            t_s = float(times_eq[t])
            title = f'DIII-D #{shot}  @  {t_s*1e3:.1f} ms'
            desc = str(d.get('summary.description') or '').strip()
            if desc:
                title += f'  —  {desc}'
            self._title_label.setText(title)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='IRI CAKE Viewer')
    parser.add_argument('--shot', type=int, default=-1,
                        help='Shot number (-1 = latest)')
    parser.add_argument('--flavor', type=str, default='IRI_CAKE01',
                        help='IRI CAKE tag/flavor')
    args = parser.parse_args()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = IriCakeViewer(shot=args.shot, flavor=args.flavor)
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
