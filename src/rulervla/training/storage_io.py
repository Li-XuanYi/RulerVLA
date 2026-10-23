"""Local progress records and bounded retry for transient shared-storage errors."""
import errno
import json
import os
from pathlib import Path
import time
from uuid import uuid4
from zipfile import BadZipFile


TRANSIENT = {errno.EIO, errno.EHOSTDOWN, errno.EHOSTUNREACH, errno.ENOTCONN,
             errno.ETIMEDOUT, errno.ECONNRESET, errno.ESTALE}


def _transient_storage_cause(error):
    """Recognize only actual underlying transient OS errors, even if wrapped.

    zipfile may translate a CIFS EHOSTDOWN raised during ZIP-end-directory
    reads into BadZipFile, preserving the OSError as __context__. A genuinely
    corrupted NPZ must still fail fast rather than silently skip samples.
    """
    pending = [error]
    visited = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, OSError) and current.errno in TRANSIENT:
            return current
        for nested in (current.__cause__, current.__context__):
            if nested is not None:
                pending.append(nested)
    return None


def retry_io(operation, *, timeout=300, interval=10):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return operation()
        except (OSError, BadZipFile) as error:
            underlying = _transient_storage_cause(error)
            if underlying is None or time.monotonic() >= deadline:
                raise
            print('Transient Dataset2 I/O; preserving state and retrying: ' +
                  repr(underlying) + ' (raised as ' + type(error).__name__ + ')', flush=True)
            time.sleep(interval)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def publish_with_retry(source, destination, updates, publisher, verifier):
    """Retry copy without overwriting complete checkpoints or failed partial evidence."""
    destination=Path(destination)
    partial=destination.with_name(destination.name+'.partial')
    def operation():
        if destination.exists():
            return verifier(destination,expected_updates=updates)
        if partial.exists():
            if (partial/'V03_COPY_COMPLETE.json').is_file():
                result=verifier(partial,expected_updates=updates)
                os.rename(partial,destination)
                return result
            preserved=partial.with_name(partial.name+'.failed_'+uuid4().hex)
            os.rename(partial,preserved)
            print('Interrupted checkpoint copy retained: '+str(preserved),flush=True)
        return publisher(source,destination,completed_updates=updates)
    return retry_io(operation)


class RunRecords:
    """Local copy is authoritative; a dropped NAS status mirror cannot kill training."""
    def __init__(self, local_root, archive_root):
        self.local = Path(local_root)
        self.archive = Path(archive_root)
        self.local.mkdir(parents=True, exist_ok=True)

    def write(self, path, payload):
        relative = Path(path).relative_to(self.archive)
        atomic_json(self.local/relative, payload)
        try:
            atomic_json(path, payload)
        except OSError as error:
            if error.errno not in TRANSIENT:
                raise
            print('Dataset2 progress mirror unavailable; local record retained: '+repr(error), flush=True)
