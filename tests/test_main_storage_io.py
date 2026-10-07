import errno
import json
from pathlib import Path
from uuid import uuid4

import pytest

from rulervla.training import storage_io


def test_local_progress_survives_nas_outage_and_mirror_recovers(monkeypatch):
    root=Path(__file__).parents[1]/'outputs'/('storage_test_'+uuid4().hex)
    records=storage_io.RunRecords(root/'local',root/'nas')
    original=storage_io.atomic_json
    def unavailable(path,payload):
        if Path(path).parent==records.archive:
            raise OSError(errno.EHOSTDOWN,'Host is down')
        original(path,payload)
    monkeypatch.setattr(storage_io,'atomic_json',unavailable)
    records.write(records.archive/'status.json',{'optimizer_updates':222})
    assert json.loads((records.local/'status.json').read_text())['optimizer_updates']==222
    monkeypatch.setattr(storage_io,'atomic_json',original)
    records.write(records.archive/'status.json',{'optimizer_updates':223})
    assert json.loads((records.archive/'status.json').read_text())['optimizer_updates']==223


def test_retry_preserves_operation_and_refuses_non_transient_errors(monkeypatch):
    monkeypatch.setattr(storage_io.time,'sleep',lambda seconds:None)
    calls=[]
    def operation():
        calls.append(1)
        if len(calls)==1:raise OSError(errno.EHOSTDOWN,'Host is down')
        return 'same sample'
    assert storage_io.retry_io(operation)=='same sample' and len(calls)==2
    with pytest.raises(OSError) as failure:
        storage_io.retry_io(lambda: (_ for _ in ()).throw(OSError(errno.EACCES,'Forbidden')))
    assert failure.value.errno==errno.EACCES


def test_publish_retry_preserves_interrupted_copy_and_never_overwrites_complete(monkeypatch):
    root=Path(__file__).parents[1]/'outputs'/('publish_test_'+uuid4().hex)
    root.mkdir(parents=True)
    destination=root/'4999';calls=[]
    monkeypatch.setattr(storage_io.time,'sleep',lambda seconds:None)
    def publish(source,target,completed_updates):
        calls.append(completed_updates)
        partial=target.with_name(target.name+'.partial')
        partial.mkdir();(partial/'bytes').write_text('evidence')
        if len(calls)==1:raise OSError(errno.EHOSTDOWN,'Host is down')
        (partial/'V03_COPY_COMPLETE.json').write_text(str(completed_updates))
        partial.rename(target)
        return completed_updates
    def verify(target,expected_updates):
        assert (target/'V03_COPY_COMPLETE.json').read_text()==str(expected_updates)
        return expected_updates
    assert storage_io.publish_with_retry(root/'source',destination,5000,publish,verify)==5000
    assert len(list(root.glob('4999.partial.failed_*')))==1
    assert storage_io.publish_with_retry(root/'source',destination,5000,publish,verify)==5000
    assert len(calls)==2
