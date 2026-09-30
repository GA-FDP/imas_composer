"""
Tests of the D3DRDB code run database queries.

They need the D3DRDB server (d3drdb.gat.com), so they are integration tests.
"""

import pytest

from imas_composer.rdb.d3drdb import list_standard_efit_trees

# A shot with the standard EFIT runs and an IRI CAKE run in the EFIT tree (pulse 17358701)
SHOT_WITH_CAKE_RUN = 173587


@pytest.mark.integration
def test_list_standard_efit_trees():
    """The standard per-shot EFIT trees, without the EFIT tree the IRI CAKE runs are stored in."""
    trees = list_standard_efit_trees(SHOT_WITH_CAKE_RUN)

    assert {'EFIT01', 'EFIT02', 'EFIT03', 'EFIT04', 'EFIT05', 'EFIT06'} <= set(trees)
    assert 'EFIT' not in trees
    assert trees == sorted(trees)


@pytest.mark.integration
def test_list_standard_efit_trees_of_a_shot_without_runs():
    # shot 1 is a test shot with EFIT runs, a negative shot has none
    with pytest.raises(ValueError, match='No standard EFIT run for shot -1'):
        list_standard_efit_trees(-1)
