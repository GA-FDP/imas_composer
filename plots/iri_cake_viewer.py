"""
IRI CAKE Viewer — PyQt GUI for inspecting IRI CAKE equilibrium and profile data.

Replicates the functionality of OMFIT-source/scripts/fetch_IRI_CAKE.py without
any dependency on omas or omfit_classes.  Data is fetched via imas_composer's
simple_load function; IRI run metadata is queried from D3DRDB via d3drdb.py.

Layout (2 × 4 grid of subplots):
  [Eq. CX   | ne (e)  | Te (e)  | j_tor OR convergence error ]
  [  (tall) | ni (ion)| Ti (ion)| Pressure + constraints      ]

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
# Qt (via pyqtgraph's compatibility layer — uses PySide6 on this system)
# ---------------------------------------------------------------------------
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

# ---------------------------------------------------------------------------
# Matplotlib embedded in Qt
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import scipy.ndimage

from imas_composer.composer import ImasComposer
from imas_composer.fetchers import simple_load
from d3drdb import get_iri_upload_ids, get_max_iri_shot_and_ids


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
]


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

class DataLoader(QtCore.QThread):
    """Fetches IDS data in a background thread."""

    finished = QtCore.Signal(dict)   # emits the data dict on success
    error    = QtCore.Signal(str)    # emits error message on failure
    status   = QtCore.Signal(str)    # progress messages

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

            self.finished.emit({**eq_data, **wall_data, **prof_data})

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


def _nearest_time_index(times: np.ndarray, target: float) -> int:
    """Return index of the time in *times* closest to *target*."""
    return int(np.argmin(np.abs(np.asarray(times) - target)))


def plot_equilibrium_cx(ax, data: Dict, t: int):
    """Plot equilibrium cross-section: psi contours, LCFS, wall, X-points."""
    ax.clear()
    ax.set_aspect('equal')
    ax.set_frame_on(False)

    # --- wall ---
    wall_r = data.get('wall.description_2d.limiter.unit.outline.r')
    wall_z = data.get('wall.description_2d.limiter.unit.outline.z')
    wall_patch = None
    if wall_r is not None and wall_z is not None:
        wr = np.asarray(wall_r).ravel()
        wz = np.asarray(wall_z).ravel()
        ax.plot(wr, wz, 'k', linewidth=1.5)
        ax.set_xlim(wr.min(), wr.max())
        ax.set_ylim(wz.min(), wz.max())
        # clip mask for contours
        verts = np.column_stack([wr, wz])
        wall_path = mpath.Path(verts)
        wall_patch = mpatches.PathPatch(wall_path, facecolor='none', edgecolor='none')
        ax.add_patch(wall_patch)

    # --- psi contours ---
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

        # smooth for nicer contours
        psi_smooth = scipy.ndimage.zoom(psi_n_plot, 3, order=3)
        R_sm = np.linspace(R[0], R[-1], psi_smooth.shape[0])
        Z_sm = np.linspace(Z[0], Z[-1], psi_smooth.shape[1])

        levels = np.linspace(0, 1, 12)[1:-1]
        Rg, Zg = np.meshgrid(R_sm, Z_sm, indexing='ij')
        cs = ax.contour(Rg, Zg, psi_smooth, levels=levels, linewidths=0.7, colors='tab:blue', alpha=0.7)
        if wall_patch is not None:
            cs.set_clip_path(wall_patch)

    # --- LCFS ---
    bdy_r = data.get('equilibrium.time_slice.boundary.outline.r')
    bdy_z = data.get('equilibrium.time_slice.boundary.outline.z')
    if bdy_r is not None and bdy_z is not None:
        br = np.asarray(bdy_r[t])
        bz = np.asarray(bdy_z[t])
        ax.plot(br, bz, 'tab:blue', linewidth=2)

    # --- X-points ---
    xpt_r = data.get('equilibrium.time_slice.boundary.x_point.r')
    xpt_z = data.get('equilibrium.time_slice.boundary.x_point.z')
    if xpt_r is not None and xpt_z is not None:
        xr = np.asarray(xpt_r[t])
        xz = np.asarray(xpt_z[t])
        for r, z in zip(xr.ravel(), xz.ravel()):
            if r > 0:
                ax.plot(r, z, 'x', color='tab:blue', markersize=8, markeredgewidth=2)

    # --- magnetic axis ---
    mag_r = data.get('equilibrium.time_slice.global_quantities.magnetic_axis.r')
    mag_z = data.get('equilibrium.time_slice.global_quantities.magnetic_axis.z')
    if mag_r is not None and mag_z is not None:
        ax.plot(float(mag_r[t]), float(mag_z[t]), '+', color='tab:blue', markersize=8, markeredgewidth=2)

    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


def _psi_norm(psi, psi_ax, psi_bdy):
    return (psi - psi_ax) / (psi_bdy - psi_ax)


def plot_pressure(ax, data: Dict, t: int):
    """Fitted pressure profile + constraint points."""
    ax.clear()

    psi_ax  = float(data['equilibrium.time_slice.global_quantities.psi_axis'][t])
    psi_bdy = float(data['equilibrium.time_slice.global_quantities.psi_boundary'][t])

    psi1d = data.get('equilibrium.time_slice.profiles_1d.psi')
    pres  = data.get('equilibrium.time_slice.profiles_1d.pressure')
    if psi1d is not None and pres is not None:
        x = _psi_norm(psi1d[t], psi_ax, psi_bdy)
        ax.plot(x, pres[t] * 1e-3, color='tab:blue', linewidth=1.5)

    # constraint points
    c_psi = data.get('equilibrium.time_slice.constraints.pressure.position.psi')
    c_meas = data.get('equilibrium.time_slice.constraints.pressure.measured')
    c_err  = data.get('equilibrium.time_slice.constraints.pressure.measured_error_upper')
    if c_psi is not None and c_meas is not None:
        cx = _psi_norm(c_psi[t], psi_ax, psi_bdy)
        cy = np.asarray(c_meas[t]) * 1e-3
        if c_err is not None:
            ce = np.asarray(c_err[t]) * 1e-3
            ax.errorbar(cx, cy, yerr=ce, fmt='', color='red', alpha=0.3, linewidth=0.8)
        else:
            ax.plot(cx, cy, '.', color='red', alpha=0.3)

    ax.set_title(r'Pressure [kPa]', y=0.9, va='top', fontsize=9)
    ax.set_xlabel(r'$\Psi_\mathrm{n}$', fontsize=8)
    ax.tick_params(labelsize=7)


def plot_j_tor(ax, data: Dict, t: int):
    """Toroidal current density profile + constraint scatter."""
    ax.clear()

    psi_ax  = float(data['equilibrium.time_slice.global_quantities.psi_axis'][t])
    psi_bdy = float(data['equilibrium.time_slice.global_quantities.psi_boundary'][t])

    psi1d = data.get('equilibrium.time_slice.profiles_1d.psi')
    jtor  = data.get('equilibrium.time_slice.profiles_1d.j_tor')
    if psi1d is not None and jtor is not None:
        x = _psi_norm(psi1d[t], psi_ax, psi_bdy)
        ax.plot(x, jtor[t] / 1e6, color='tab:blue', linewidth=1.5)

    # constraint scatter
    c_psi  = data.get('equilibrium.time_slice.constraints.j_tor.position.psi')
    c_meas = data.get('equilibrium.time_slice.constraints.j_tor.measured')
    if c_psi is not None and c_meas is not None:
        cx = _psi_norm(c_psi[t], psi_ax, psi_bdy)
        cy = np.asarray(c_meas[t]) / 1e6
        ax.plot(cx, cy, 'o', color='red', alpha=0.4, markersize=3)

    ax.set_title(r'$j_\mathrm{tor}$ [MA m$^{-2}$]', y=0.9, va='top', fontsize=9)
    ax.set_xlabel(r'$\Psi_\mathrm{n}$', fontsize=8)
    ax.tick_params(labelsize=7)


def plot_convergence_error(ax, data: Dict, t: int):
    """Convergence error vs. time (all slices), vertical line at current time."""
    ax.clear()

    times = data.get('equilibrium.time')
    cerr  = data.get('equilibrium.time_slice.convergence.grad_shafranov_deviation_value')
    if times is not None and cerr is not None:
        ax.plot(np.asarray(times) * 1e3, np.asarray(cerr).ravel(), color='tab:blue', linewidth=1)
        ax.axvline(float(times[t]) * 1e3, color='k', linewidth=1, linestyle='--')
        ax.set_yscale('log')

    ax.set_title('Convergence error', y=0.9, va='top', fontsize=9)
    ax.set_xlabel('Time [ms]', fontsize=8)
    ax.tick_params(labelsize=7)


def _plot_profile(ax, x, y, yerr, fit_x, fit_y, fit_err, color_band, color_pts, title, xlabel, scale=1.0, clear_ax=True):
    """Generic 1D profile plot with error band and raw-data error bars."""
    if clear_ax:
        ax.clear()
    if x is not None and y is not None:
        y_s = np.asarray(y) * scale
        if yerr is not None:
            ye_s = np.asarray(yerr) * scale
            ax.fill_between(x, y_s - ye_s, y_s + ye_s, alpha=0.25, color=color_band)
        ax.plot(x, y_s, color=color_band, linewidth=1.5)

    if fit_x is not None and fit_y is not None and fit_err is not None:
        fx = np.asarray(fit_x)
        fy = np.asarray(fit_y) * scale
        fe = np.asarray(fit_err) * scale
        # remove NaNs and 100 % uncertainty points
        mask = np.isfinite(fx) & np.isfinite(fy) & np.isfinite(fe)
        mask[mask] &= np.abs(fe[mask]) < np.abs(fy[mask])
        if mask.any():
            ax.errorbar(fx[mask], fy[mask], fe[mask],
                        fmt='.', color=color_pts, alpha=0.5, markersize=3,
                        linewidth=0.5, zorder=-1)

    ax.set_title(title, y=0.9, va='top', fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(labelsize=7)


def plot_electron_density(ax, data: Dict, cp_t: int):
    x    = _cp_psin(data, cp_t)
    y    = data.get('core_profiles.profiles_1d.electrons.density')
    yerr = data.get('core_profiles.profiles_1d.electrons.density_error_upper')
    fx   = data.get('core_profiles.profiles_1d.electrons.density_fit.psi_norm')
    fy   = data.get('core_profiles.profiles_1d.electrons.density_fit.measured')
    fe   = data.get('core_profiles.profiles_1d.electrons.density_fit.measured_error_upper')
    _plot_profile(
        ax,
        x,
        None if y is None else y[cp_t],
        None if yerr is None else yerr[cp_t],
        None if fx is None else fx[cp_t],
        None if fy is None else fy[cp_t],
        None if fe is None else fe[cp_t],
        'tab:orange', 'tab:orange',
        r'$n_e$ [$10^{19}$ m$^{-3}$]',
        r'$\Psi_\mathrm{n}$',
        scale=1e-19,
    )


def plot_electron_temperature(ax, data: Dict, cp_t: int):
    x    = _cp_psin(data, cp_t)
    y    = data.get('core_profiles.profiles_1d.electrons.temperature')
    yerr = data.get('core_profiles.profiles_1d.electrons.temperature_error_upper')
    fx   = data.get('core_profiles.profiles_1d.electrons.temperature_fit.psi_norm')
    fy   = data.get('core_profiles.profiles_1d.electrons.temperature_fit.measured')
    fe   = data.get('core_profiles.profiles_1d.electrons.temperature_fit.measured_error_upper')
    _plot_profile(
        ax,
        x,
        None if y is None else y[cp_t],
        None if yerr is None else yerr[cp_t],
        None if fx is None else fx[cp_t],
        None if fy is None else fy[cp_t],
        None if fe is None else fe[cp_t],
        'tab:orange', 'tab:orange',
        r'$T_e$ [keV]',
        r'$\Psi_\mathrm{n}$',
        scale=1e-3,
    )


def plot_ion_density(ax, data: Dict, cp_t: int):
    x    = _cp_psin(data, cp_t)
    y    = data.get('core_profiles.profiles_1d.ion.density_thermal')
    yerr = data.get('core_profiles.profiles_1d.ion.density_thermal_error_upper')
    colors = ['tab:green','tab:purple','tab:red','tab:blue','tab:orange','tab:black']
    if y is not None:
        for i in range(len(y[cp_t])):
            _plot_profile(
                ax, x,
                y[cp_t][i],
                None if yerr is None else yerr[cp_t][i],
                None, None, None,
                colors[i], colors[i],
                r'$n_i$ [$10^{19}$ m$^{-3}$]',
                r'$\Psi_\mathrm{n}$',
                scale=1e-19,
                clear_ax=not i
            )


def plot_ion_temperature(ax, data: Dict, cp_t: int):
    x    = _cp_psin(data, cp_t)
    y    = data.get('core_profiles.profiles_1d.ion.temperature')
    yerr = data.get('core_profiles.profiles_1d.ion.temperature_error_upper')
    colors = ['tab:green','tab:purple','tab:red','tab:blue','tab:orange','tab:black']
    if y is not None:
        for i in range(len(y[cp_t])):
            _plot_profile(
                ax, x,
                y[cp_t][i],
                None if yerr is None else yerr[cp_t][i],
                None, None, None,
                colors[i], colors[i],
                r'$T_i$ [keV]',
                r'$\Psi_\mathrm{n}$',
                scale=1e-3,
                clear_ax=not i
            )


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

    def __init__(self, shot: int = -1, flavor: str = 'CAKE_FDP'):
        super().__init__()
        self.setWindowTitle('IRI CAKE Viewer')
        self.resize(1400, 700)

        self._data: Optional[Dict[str, Any]] = None
        self._loader: Optional[DataLoader] = None
        self._shot = shot
        self._flavor = flavor

        self._build_ui()

        if shot > 0:
            QtCore.QTimer.singleShot(200, self._trigger_fetch_shot)
        elif shot == -1:
            QtCore.QTimer.singleShot(200, self._fetch_latest)

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
        row1.addWidget(QtWidgets.QLabel('Shot:'))
        self._shot_spin = QtWidgets.QSpinBox()
        self._shot_spin.setRange(-1, 999999)
        self._shot_spin.setValue(self._shot if self._shot > 0 else 205055)
        self._shot_spin.setFixedWidth(80)
        row1.addWidget(self._shot_spin)

        row1.addWidget(QtWidgets.QLabel('Tag:'))
        self._flavor_combo = QtWidgets.QComboBox()
        self._flavor_combo.addItems(['CAKE_FDP', 'IRI_CAKE01', 'IRI_CAKE02', 'cake_nersc_testing', 'cake_nersc_testing_2'])
        self._flavor_combo.setCurrentText(self._flavor)
        self._flavor_combo.setEditable(True)
        self._flavor_combo.setFixedWidth(180)
        row1.addWidget(self._flavor_combo)

        row1.addWidget(QtWidgets.QLabel('EFIT tree:'))
        self._efit_combo = QtWidgets.QComboBox()
        self._efit_combo.addItems(['EFIT']) #'EFIT01', 'EFIT02', 'EFIT02er', 'EFIT03', 'EFIT_CAKE01', 'EFIT_CAKE02'
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

        self._latest_btn = QtWidgets.QPushButton('Latest Shot')
        self._latest_btn.setFixedWidth(90)
        self._latest_btn.clicked.connect(self._fetch_latest)
        row1.addWidget(self._latest_btn)

        self._fetch_btn = QtWidgets.QPushButton('Fetch Shot')
        self._fetch_btn.setFixedWidth(90)
        self._fetch_btn.clicked.connect(self._trigger_fetch_shot)
        row1.addWidget(self._fetch_btn)

        self._error_mode_cb = QtWidgets.QCheckBox('Plot convergence error')
        row1.addWidget(self._error_mode_cb)
        self._error_mode_cb.stateChanged.connect(self._replot)

        row1.addStretch()
        root.addLayout(row1)

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

        # ---- matplotlib canvas ----
        self._fig = Figure(figsize=(14, 5), tight_layout=True)
        self._canvas = FigureCanvas(self._fig)
        root.addWidget(self._canvas, stretch=1)

        self._build_axes()

    def _build_axes(self):
        """Create the 2×4 subplot grid."""
        self._fig.clf()
        gs = gridspec.GridSpec(
            2, 4,
            figure=self._fig,
            left=0.05, right=0.98,
            top=0.93, bottom=0.10,
            wspace=0.35, hspace=0.40,
        )
        # Eq. CX spans both rows in column 0
        self._ax_cx   = self._fig.add_subplot(gs[:, 0])
        self._ax_ne   = self._fig.add_subplot(gs[0, 1])
        self._ax_ni   = self._fig.add_subplot(gs[1, 1])
        self._ax_te   = self._fig.add_subplot(gs[0, 2])
        self._ax_ti   = self._fig.add_subplot(gs[1, 2])
        self._ax_jtor = self._fig.add_subplot(gs[0, 3])  # or convergence error
        self._ax_pres = self._fig.add_subplot(gs[1, 3])

        self._title_text = self._fig.text(0.5, 0.97, '', ha='center', va='top', fontsize=11)
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Fetch logic
    # ------------------------------------------------------------------

    def _fetch_latest(self):
        self._status_label.setText('Querying D3DRDB for latest shot…')
        QtWidgets.QApplication.processEvents()
        try:
            shot, prof_id, eq_id, comment = get_max_iri_shot_and_ids()
        except Exception as e:
            self._status_label.setText(f'D3DRDB error: {e}')
            return
        self._shot_spin.setValue(shot)
        self._efit_id_edit.setText(eq_id)
        self._prof_id_edit.setText(prof_id)
        self._efit_combo.setCurrentText('EFIT')
        self._prof_combo.setCurrentText('OMFIT_PROFS')
        self._start_load(shot, 'EFIT', eq_id, 'OMFIT_PROFS', prof_id)

    def _trigger_fetch_shot(self):
        shot = self._shot_spin.value()
        flavor = self._flavor_combo.currentText()

        eq_id   = self._efit_id_edit.text().strip()
        prof_id = self._prof_id_edit.text().strip()
        efit_tree  = self._efit_combo.currentText().strip() or 'EFIT'
        prof_tree  = self._prof_combo.currentText().strip() or 'OMFIT_PROFS'

        # Auto-lookup IDs from D3DRDB if not manually set
        if not eq_id or not prof_id:
            self._status_label.setText(f'Querying D3DRDB for shot {shot}…')
            QtWidgets.QApplication.processEvents()
            try:
                auto_prof, auto_eq = get_iri_upload_ids(shot, flavor)
                if not prof_id:
                    prof_id = auto_prof
                    self._prof_id_edit.setText(prof_id)
                if not eq_id:
                    eq_id = auto_eq
                    self._efit_id_edit.setText(eq_id)
            except Exception as e:
                self._status_label.setText(f'D3DRDB lookup failed: {e}')
                if not eq_id or not prof_id:
                    return

        self._start_load(shot, efit_tree, eq_id, prof_tree, prof_id)

    def _start_load(self, shot, efit_tree, efit_run_id, profiles_tree, profiles_run_id):
        if self._loader is not None and self._loader.isRunning():
            self._loader.terminate()
            self._loader.wait()

        self._set_buttons_enabled(False)
        self._status_label.setText(
            f'Loading shot {shot}  EFIT={efit_tree}{efit_run_id}  '
            f'PROFS={profiles_tree}{profiles_run_id}…'
        )

        self._loader = DataLoader(shot, efit_tree, efit_run_id, profiles_tree, profiles_run_id)
        self._loader.status.connect(self._status_label.setText)
        self._loader.finished.connect(self._on_load_finished)
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

        self._status_label.setText(
            f'Loaded shot {self._shot_spin.value()}  —  '
            f'{len(np.asarray(times)) if times is not None else 0} time slices'
        )
        self._replot()

    def _on_load_error(self, msg: str):
        self._set_buttons_enabled(True)
        self._status_label.setText('Load error — see console')
        print(msg, file=sys.stderr)

    def _set_buttons_enabled(self, enabled: bool):
        self._fetch_btn.setEnabled(enabled)
        self._latest_btn.setEnabled(enabled)

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
        self._replot()

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
        error_mode = self._error_mode_cb.isChecked()

        # Equilibrium time for core-profiles time-matching
        times_eq = d.get('equilibrium.time')
        times_cp = d.get('core_profiles.time')
        cp_t = 0
        if times_eq is not None and times_cp is not None and t < len(times_eq):
            cp_t = _nearest_time_index(np.asarray(times_cp), float(times_eq[t]))

        try:
            plot_equilibrium_cx(self._ax_cx, d, t)
        except Exception as e:
            self._ax_cx.clear()
            self._ax_cx.text(0.5, 0.5, str(e), transform=self._ax_cx.transAxes,
                             ha='center', va='center', fontsize=7, wrap=True)

        for fn, ax in [
            (lambda ax: plot_electron_density(ax, d, cp_t),   self._ax_ne),
            (lambda ax: plot_ion_density(ax, d, cp_t),        self._ax_ni),
            (lambda ax: plot_electron_temperature(ax, d, cp_t), self._ax_te),
            (lambda ax: plot_ion_temperature(ax, d, cp_t),    self._ax_ti),
            (lambda ax: plot_pressure(ax, d, t),              self._ax_pres),
        ]:
            try:
                fn(ax)
            except Exception as e:
                ax.clear()
                ax.text(0.5, 0.5, str(e), transform=ax.transAxes,
                        ha='center', va='center', fontsize=7, wrap=True)

        try:
            if error_mode:
                plot_convergence_error(self._ax_jtor, d, t)
            else:
                plot_j_tor(self._ax_jtor, d, t)
        except Exception as e:
            self._ax_jtor.clear()
            self._ax_jtor.text(0.5, 0.5, str(e), transform=self._ax_jtor.transAxes,
                               ha='center', va='center', fontsize=7, wrap=True)

        # Title
        if times_eq is not None and t < len(times_eq):
            shot = self._shot_spin.value()
            t_s = float(times_eq[t])
            self._title_text.set_text(f'DIII-D #{shot}  @  {t_s*1e3:.1f} ms')

        self._canvas.draw_idle()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='IRI CAKE Viewer')
    parser.add_argument('--shot', type=int, default=-1,
                        help='Shot number (-1 = latest)')
    parser.add_argument('--flavor', type=str, default='CAKE_FDP',
                        help='IRI CAKE tag/flavor')
    args = parser.parse_args()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = IriCakeViewer(shot=args.shot, flavor=args.flavor)
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
