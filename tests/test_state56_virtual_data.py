import hashlib
import numpy as np
import pytest

from scripts.state56_virtual_data import State56SidecarStore


class _FakeTable:
    def __init__(self,row):self.row=row
    def to_pylist(self):return [self.row]


class _FakeDataset:
    def __init__(self,row):self.row=row
    def take(self,positions,columns):
        assert positions==[0]
        return _FakeTable({key:self.row[key] for key in columns})


def _store_and_row():
    state=np.arange(3*56,dtype=np.float32).reshape(3,56)
    state_sha=hashlib.sha256(np.ascontiguousarray(state,dtype='<f4').tobytes()).hexdigest()
    row={'index':{'release_row_index':7,'uuid':'u'},'window':{'source_total_frames':3},'state':state.tolist(),'state_sha256':state_sha,'provenance':{'source_row_payload_sha256':'a'*64}}
    store=State56SidecarStore.__new__(State56SidecarStore);store._positions={7:0};store.dataset=_FakeDataset(row)
    return store,state


def test_state56_sidecar_load_verifies_join_and_state_sha():
    store,state=_store_and_row();row=store.load(7,expected_uuid='u',expected_source_payload_sha256='a'*64)
    assert np.array_equal(row['state'],state)


def test_state56_sidecar_load_rejects_wrong_source_payload():
    store,_=_store_and_row()
    with pytest.raises(ValueError,match='source payload mismatch'):
        store.load(7,expected_uuid='u',expected_source_payload_sha256='b'*64)


def test_state56_sidecar_load_rejects_rows_outside_grade_a():
    store,_=_store_and_row()
    with pytest.raises(ValueError,match='outside the Grade-A'):
        store.load(8,expected_uuid='u',expected_source_payload_sha256='a'*64)
