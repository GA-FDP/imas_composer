"""
D3DRDB connection utilities for querying IRI CAKE run metadata.

Replaces omfit_classes.omfit_rdb.OMFITrdb and omfit_classes.omfit_iri.available_iri_results
without any OMFIT dependency.

Credentials are read from ~/D3DRDB.sybase_login or ~/d3drdb.sybase_login.
The file format is either:
  - Single line: hostname:username:password  (last two colon-split parts used)
  - Two-line:    username\\npassword          (last two lines used)
"""

import ctypes.util
import os
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

try:
    import pyodbc
except ImportError:
    pyodbc = None


SERVER = 'd3drdb.gat.com'
PORT = 8001
DATABASE = 'code_rundb'


def _read_sybase_credentials() -> Tuple[str, str]:
    """Read username and password from ~/D3DRDB.sybase_login or ~/d3drdb.sybase_login."""
    for fname in ['D3DRDB.sybase_login', 'd3drdb.sybase_login']:
        fpath = os.path.join(os.path.expanduser('~'), fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, 'r') as f:
            content = f.read().strip()
        # Format 1: "hostname:username:password" — take the last two colon-split parts
        parts = content.split(':')
        if len(parts) >= 2:
            return parts[-2].strip(), parts[-1].strip()
        # Format 2: last two non-empty lines are username, password
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        if len(lines) >= 2:
            return lines[-2], lines[-1]
    return '', ''


def _find_tds_driver() -> str:
    """Return the pyodbc driver string for FreeTDS/TDS ODBC."""
    if pyodbc is not None:
        for d in pyodbc.drivers():
            if 'tds' in d.lower() or 'freetds' in d.lower():
                return d
    # Try well-known shared library paths
    lib = ctypes.util.find_library('tdsodbc')
    if lib:
        return lib
    for path in ['/usr/lib/x86_64-linux-gnu/odbc/libtdsodbc.so',
                 '/usr/local/lib/libtdsodbc.so',
                 '/usr/lib/libtdsodbc.so.0']:
        if os.path.exists(path):
            return path
    raise RuntimeError(
        "No TDS ODBC driver found. Install FreeTDS: sudo apt install tdsodbc"
    )


class D3DRDB:
    """
    Minimal pyodbc wrapper for the D3D relational database.

    Usage::

        db = D3DRDB()
        rows = db.query("SELECT * FROM iri_run_log WHERE shot=205055")
        # rows is a list of dicts keyed by UPPERCASE column name
        db.close()
    """

    def __init__(self, username: str = '', password: str = ''):
        if not username or not password:
            username, password = _read_sybase_credentials()
        if not username or not password:
            username, password = 'guest', 'guest_pwd'
            print('No username or password; Attempting to use guest credentials')
        self._username = username
        self._password = password
        self._cnxn = None
        self._cursor = None

    def _connect(self):
        if pyodbc is None:
            raise RuntimeError("pyodbc is required: pip install pyodbc")

        driver = _find_tds_driver()
        # Use TDS 7.0 for compatibility with d3drdb.gat.com
        # Note: TDS 8.0 worked on RHEL but not WSL Ubuntu with FreeTDS 1.5.18+
        # Newer FreeTDS versions require explicit Encrypt=no for unencrypted connections
        conn_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={SERVER};"
            f"PORT={PORT};"
            f"DATABASE={DATABASE};"
            f"UID={self._username};"
            f"PWD={self._password};"
            "TDS_Version=7.0;"
            "Login Timeout=30;"
        )
        self._cnxn = pyodbc.connect(conn_str)
        self._cursor = self._cnxn.cursor()

    def query(self, sql: str) -> List[Dict[str, Any]]:
        """Execute *sql* and return rows as a list of dicts with UPPERCASE keys."""
        if self._cnxn is None:
            self._connect()
        self._cursor.execute(sql)
        columns = [d[0].upper() for d in self._cursor.description]
        return [dict(zip(columns, row)) for row in self._cursor.fetchall()]

    def close(self):
        if self._cnxn is not None:
            try:
                self._cnxn.close()
            except Exception:
                pass
            self._cnxn = None
            self._cursor = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Public helpers (mirrors of the OMFIT fetch_IRI_CAKE.py functions)
# ---------------------------------------------------------------------------

def available_iri_results(
    shot: int,
    tag: Optional[str] = 'CAKE_FDP',
    ignore_ignore: bool = False,
) -> Dict[int, Dict]:
    """
    Return a dict keyed by IRI_ID of available IRI CAKE runs for *shot*.

    Each value is a dict with all columns from iri_run_log plus a
    ``results_uploaded`` sub-dict keyed by CODE_NAME.

    Returns empty dict if nothing is found.
    """
    with D3DRDB() as db:
        sql = f"SELECT * FROM iri_run_log WHERE experiment='DIII-D' AND shot={shot}"
        if tag is not None:
            sql += f" AND tag='{tag}'"
        if not ignore_ignore:
            sql += " AND ignore='False'"

        runs = db.query(sql)
        if not runs:
            return {}

        result: Dict[int, Dict] = {}
        for run in runs:
            iri_id = run['IRI_ID']
            run_dict = {k: v for k, v in run.items()}

            # Normalise RUN_DATE to datetime
            rd = run_dict.get('RUN_DATE')
            if rd is None:
                run_dict['RUN_DATE'] = datetime(2000, 1, 1)
            elif not isinstance(rd, datetime):
                try:
                    run_dict['RUN_DATE'] = datetime.fromisoformat(str(rd))
                except Exception:
                    run_dict['RUN_DATE'] = datetime(2000, 1, 1)

            uploads = db.query(
                f"SELECT * FROM iri_upload_log WHERE iri_id={iri_id}"
            )
            run_dict['results_uploaded'] = {}
            for upload in uploads:
                code = upload.get('CODE_NAME', 'unknown')
                run_dict['results_uploaded'][code] = {
                    k: v for k, v in upload.items()
                }

            result[iri_id] = run_dict

    return result


def get_iri_upload_ids(shot: int, tag: str = 'CAKE_FDP') -> Tuple[str, str]:
    """
    Return (prof_id, eq_id) 3- and 2-digit suffixes for the most recent
    valid IRI CAKE run for *shot*.

    Raises ValueError with a helpful message if nothing is found.
    """
    iri_runs = available_iri_results(shot, tag)

    if not iri_runs:
        all_runs = available_iri_results(shot, tag=None)
        available_tags = sorted({r['TAG'] for r in all_runs.values()})
        raise ValueError(
            f"No IRI CAKE data for shot {shot} with tag '{tag}'.\n"
            f"Available tags: {available_tags or ['(none found)']}"
        )

    iri_num = max(iri_runs)
    if iri_runs[iri_num]['RUN_DATE'] < datetime(2024, 6, 25):
        raise ValueError(f"No valid IRI CAKE data for shot {shot} (run too old)")

    uploads = iri_runs[iri_num]['results_uploaded']
    prof_id = str(uploads['OMFIT_CAKE_PROF']['UPLOAD_ID'])[-3:]
    eq_id   = str(uploads['OMFIT_CAKE_EFIT']['UPLOAD_ID'])[-2:]
    return prof_id, eq_id


def get_max_iri_shot_and_ids() -> Tuple[int, str, str, str]:
    """
    Return (shot, prof_id, eq_id, comment) for the most recent IRI run
    that has an OMFIT_PROFS upload.
    """
    with D3DRDB() as db:
        rows = db.query(
            "SELECT max(iri_id) as iri_id FROM iri_upload_log "
            "WHERE upload_tree='OMFIT_PROFS'"
        )
        max_iri_id = rows[0]['IRI_ID']

        shot = db.query(
            f"SELECT shot FROM iri_run_log WHERE iri_id={max_iri_id}"
        )[0]['SHOT']

        prof_id = str(db.query(
            f"SELECT upload_id FROM iri_upload_log "
            f"WHERE upload_tree='OMFIT_PROFS' AND iri_id={max_iri_id}"
        )[0]['UPLOAD_ID'])[-3:]

        eq_row = db.query(
            f"SELECT upload_id, comments FROM iri_upload_log "
            f"WHERE upload_tree='EFIT' AND iri_id={max_iri_id}"
        )[0]
        eq_id   = str(eq_row['UPLOAD_ID'])[-2:]
        comment = eq_row.get('COMMENTS', '') or ''

    return shot, prof_id, eq_id, comment


def list_available_tags(shot: int) -> List[str]:
    """Return all TAG values found for *shot* (ignoring the ignore flag)."""
    runs = available_iri_results(shot, tag=None, ignore_ignore=True)
    return sorted({r['TAG'] for r in runs.values()})
