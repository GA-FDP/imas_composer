"""
Reusable Qt control row that resolves an IRI CAKE run against D3DRDB.

A host window adds a :class:`CakeSelector`, connects to its ``selected`` signal and receives the
shot together with the EFIT and profile trees and run IDs of the chosen CAKE run.  The D3DRDB
queries (tag list, shot list, upload IDs) run in background threads with a timeout watchdog, so the
host never blocks on the database.

The widget also owns the shared status line: the host reports its own progress through
:meth:`CakeSelector.set_status` so that everything the user sees appears in one place.
"""

from __future__ import annotations

import sys
import time
import traceback
from typing import List, Optional

from pyqtgraph.Qt import QtCore, QtWidgets

from imas_composer.rdb.d3drdb import get_iri_upload_ids, list_shots_for_tag, list_all_tags

RDB_TIMEOUT_MS = 20_000

# Status line styles, keyed by the level passed to CakeSelector.set_status().
STATUS_STYLES = {
    'info':    'color: grey; font-style: italic;',
    'warning': 'color: orange; font-weight: bold;',
    'error':   'color: red; font-style: italic;',
}


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


class CakeSelector(QtWidgets.QWidget):
    """Tag / shot / tree / run-ID row that resolves an IRI CAKE run against D3DRDB.

    Emits ``selected(shot, efit_tree, efit_run_id, profiles_tree, profiles_run_id)`` once every
    field of a run is known — either because the user filled both run IDs in by hand, or because
    D3DRDB resolved them from the shot and the tag.
    """

    selected = QtCore.Signal(int, str, str, str, str)

    def __init__(self, shot: int = -1, flavor: str = 'IRI_CAKE01',
                 efit_tree: str = 'EFIT', efit_run_id: str = '',
                 profiles_tree: str = 'OMFIT_PROFS', profiles_run_id: str = '',
                 parent=None):
        super().__init__(parent)

        self._rdb_worker: Optional[D3DrdbWorker] = None
        # Every started QThread lives here until its run() actually returns, so
        # Qt never destroys a still-running thread (which aborts the process).
        self._live_threads: list = []
        self._rdb_timeout: Optional[QtCore.QTimer] = None
        self._pending_load_params: Optional[tuple] = None
        self._shot = shot
        self._flavor = flavor
        # A preselected shot is the one auto-load path: fetch it once the shot list
        # has been queried (so the two D3DRDB calls never overlap).
        self._pending_autofetch = shot > 0
        self.rdb_fetch_start = None

        self._build_ui(efit_tree, efit_run_id, profiles_tree, profiles_run_id)

        QtCore.QTimer.singleShot(200, self._populate_tags)
        # A run whose IDs are already known needs no D3DRDB, so it must not wait for the shot list either.
        if self._pending_autofetch and efit_run_id and profiles_run_id:
            self._pending_autofetch = False
            QtCore.QTimer.singleShot(0, self.fetch_shot)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self, efit_tree: str, efit_run_id: str, profiles_tree: str, profiles_run_id: str):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

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
        self._efit_combo.addItems([efit_tree])
        self._efit_combo.setEditable(True)
        self._efit_combo.setFixedWidth(100)
        row1.addWidget(self._efit_combo)

        row1.addWidget(QtWidgets.QLabel('Run ID:'))
        self._efit_id_edit = QtWidgets.QLineEdit(efit_run_id)
        self._efit_id_edit.setPlaceholderText('auto')
        self._efit_id_edit.setFixedWidth(50)
        row1.addWidget(self._efit_id_edit)

        row1.addWidget(QtWidgets.QLabel('Profile tree:'))
        self._prof_combo = QtWidgets.QComboBox()
        self._prof_combo.addItems([profiles_tree])
        self._prof_combo.setEditable(True)
        self._prof_combo.setFixedWidth(130)
        row1.addWidget(self._prof_combo)

        row1.addWidget(QtWidgets.QLabel('Run ID:'))
        self._prof_id_edit = QtWidgets.QLineEdit(profiles_run_id)
        self._prof_id_edit.setPlaceholderText('auto')
        self._prof_id_edit.setFixedWidth(50)
        row1.addWidget(self._prof_id_edit)

        self._fetch_btn = QtWidgets.QPushButton('Fetch Shot')
        self._fetch_btn.setFixedWidth(90)
        self._fetch_btn.clicked.connect(self.fetch_shot)
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
        # Committing a shot fetches it right away. currentTextChanged fires per keystroke of the
        # editable combo and must not reach D3DRDB, so only these two commit.
        self._shot_combo.activated.connect(lambda _: self.fetch_shot())
        self._shot_combo.lineEdit().returnPressed.connect(self.fetch_shot)

        self._status_label = QtWidgets.QLabel('Ready')
        self._status_label.setStyleSheet(STATUS_STYLES['info'])
        root.addWidget(self._status_label)

    # ------------------------------------------------------------------
    # Host interface
    # ------------------------------------------------------------------

    def set_status(self, message: str, level: str = 'info'):
        """Write *message* to the shared status line, styled by *level*."""
        if level not in STATUS_STYLES:
            raise ValueError(f"Unknown status level {level!r}, use one of {sorted(STATUS_STYLES)}")
        self._status_label.setStyleSheet(STATUS_STYLES[level])
        self._status_label.setText(message)

    def set_busy(self, busy: bool):
        """Disable the fetch button while the host is loading."""
        self._fetch_btn.setEnabled(not busy)

    def wait_for_threads(self, msecs: int = 5000):
        """Wait for the in-flight D3DRDB threads, for the host's closeEvent."""
        for thread in list(self._live_threads):
            try:
                thread.wait(msecs)
            except RuntimeError:
                pass

    # ------------------------------------------------------------------
    # D3DRDB helpers (non-blocking)
    # ------------------------------------------------------------------

    def _start_rdb_worker(self, fn, *args, status_msg: str = 'Querying D3DRDB…', **kwargs):
        """Launch *fn* in a D3DrdbWorker with a timeout watchdog."""
        self._cancel_rdb_worker()
        self.set_busy(True)
        self.set_status(status_msg)

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
        if thread is self._rdb_worker:
            self._rdb_worker = None
        thread.deleteLater()

    def _on_rdb_timeout(self):
        self._cancel_rdb_worker()
        self.set_busy(False)
        self.set_status(
            f'D3DRDB connection timed out ({RDB_TIMEOUT_MS // 1000} s). '
            'Check network, or enter EFIT / profile run IDs manually.',
            'warning',
        )

    def _on_rdb_error(self, msg: str):
        self._cancel_rdb_worker()
        self.set_busy(False)
        last_line = msg.strip().splitlines()[-1]
        self.set_status(f'D3DRDB error: {last_line}', 'error')
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
            self.set_status(f'Invalid shot: {text!r}', 'error')
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

    def _on_tags_found(self, tags: List[str]):
        self._cancel_rdb_worker()
        self.set_busy(False)
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
        self.rdb_fetch_start = time.time()
        worker = self._start_rdb_worker(
            list_shots_for_tag, self._flavor_combo.currentText(),
            status_msg='Querying D3DRDB for shots…',
        )
        worker.result.connect(self._on_shots_found)
        worker.error.connect(self._on_rdb_error)
        worker.start()

    def _on_shots_found(self, shots: List[int]):
        self._cancel_rdb_worker()
        self.set_busy(False)
        # Repopulate without firing the run-id reset for each programmatic change.
        self._shot_combo.blockSignals(True)
        self._shot_combo.clear()
        self._shot_combo.addItems([str(s) for s in shots])   # most-recent first
        self._shot_combo.blockSignals(False)
        time_elapsed = -1.0
        if self.rdb_fetch_start is not None:
            time_elapsed = time.time() - self.rdb_fetch_start
        self.set_status(
            f'{len(shots)} shots for tag {self._flavor_combo.currentText()} in {time_elapsed:1.2f} s'
        )

        # A preselected shot auto-loads once, after the list is available. Its run IDs were passed
        # in together with it, so selecting it must not run them through _reset_run_ids.
        if self._pending_autofetch:
            self._pending_autofetch = False
            self._shot_combo.blockSignals(True)
            self._shot_combo.setCurrentText(str(self._shot))
            self._shot_combo.blockSignals(False)
            self.fetch_shot()

    def fetch_shot(self):
        """Resolve the current selection and emit ``selected`` once it is complete."""
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
            self.selected.emit(shot, efit_tree, eq_id, prof_tree, prof_id)
            return

        # Save the non-ID params so _on_ids_found can complete the selection
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
        self.selected.emit(shot, efit_tree, eq_id, prof_tree, prof_id)

    def _reset_run_ids(self, *, efit: bool, prof: bool):
        if efit:
            self._efit_id_edit.clear()
        if prof:
            self._prof_id_edit.clear()
