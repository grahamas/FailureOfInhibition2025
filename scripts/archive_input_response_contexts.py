"""Losslessly archive completed geometry searches while preserving resume checks.

Only the geometry context directory is packed. Summary tables, independent
confirmations, trajectories and original checkpoint bytes remain available.
Run after a study's checksum manifest is written only if it will be regenerated.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
import shutil
import tarfile
import tomllib


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_relative(name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'invalid checkpoint path: {name}')
    return path


def verify_checkpoint(folder):
    data = tomllib.loads((folder / 'done.toml').read_text())
    for name, expected in data['files'].items():
        if digest(folder / safe_relative(name)) != expected:
            raise ValueError(f'checkpoint mismatch: {folder / name}')
    return data


def verify_archive(path):
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(set(names)) != len(names) or any(not m.isfile() for m in members):
            raise ValueError('archive must contain unique regular files')
        original = archive.extractfile('_checkpoint/done.toml').read()
        files = tomllib.loads(original.decode())['files']
        expected = {k: v for k, v in files.items()
                    if safe_relative(k).parts[0] == 'contexts'}
        if set(names) != set(expected) | {'_checkpoint/done.toml'}:
            raise ValueError('archive contents do not match original checkpoint')
        for name, value in expected.items():
            if hashlib.sha256(archive.extractfile(name).read()).hexdigest() != value:
                raise ValueError(f'archive mismatch: {name}')
    return original, expected


def archive_geometry(folder):
    folder = Path(folder)
    summary = tomllib.loads((folder / 'summary.toml').read_text())
    confirmation = folder.parent / 'geometry_confirmation'
    if not summary.get('screen', False):
        # Detailed confirmation reads raw contexts; finish it before packing.
        if not (confirmation / 'done.toml').is_file():
            return False
        verify_checkpoint(confirmation)
    marker = folder / 'done.toml'
    original = marker.read_bytes()
    data = verify_checkpoint(folder)
    contexts = folder / 'contexts'
    archive_path = folder / 'contexts.tar.gz'
    record_path = folder / 'context_archive.json'
    if archive_path.name in data['files']:
        archived_marker, expected = verify_archive(archive_path)
        if record_path.name not in data['files']:
            raise ValueError('archived checkpoint has no provenance record')
        # Recover a crash after publishing the marker but before removing raw
        # copies. Verify them before deletion so unexpected edits are retained.
        if contexts.exists():
            verify_raw(folder, expected)
            shutil.rmtree(contexts)
        return False
    if not contexts.is_dir():
        raise ValueError('raw contexts are missing from an unpacked checkpoint')
    expected = {k: v for k, v in data['files'].items()
                if safe_relative(k).parts[0] == 'contexts'}
    verify_raw(folder, expected)
    if archive_path.exists():
        archived_marker, archived_files = verify_archive(archive_path)
        if archived_marker != original or archived_files != expected:
            raise ValueError('existing archive belongs to a different checkpoint')
    else:
        temporary = folder / 'contexts.tar.gz.tmp'
        try:
            with tarfile.open(temporary, 'w:gz', compresslevel=6) as archive:
                info = tarfile.TarInfo('_checkpoint/done.toml')
                info.size = len(original)
                archive.addfile(info, io.BytesIO(original))
                for name in sorted(expected):
                    archive.add(folder / name, arcname=name, recursive=False)
            archived_marker, archived_files = verify_archive(temporary)
            if archived_marker != original or archived_files != expected:
                raise ValueError('new archive differs from original checkpoint')
            os.replace(temporary, archive_path)
        finally:
            temporary.unlink(missing_ok=True)
    raw_bytes = sum((folder / name).stat().st_size for name in expected)
    record = {
        'format': 1, 'scope': 'geometry/contexts only; numerical data unchanged',
        'original_checkpoint_sha256': hashlib.sha256(original).hexdigest(),
        'archive_sha256': digest(archive_path), 'archived_files': len(expected),
        'unpacked_bytes': raw_bytes, 'archive_bytes': archive_path.stat().st_size,
        'original_checkpoint_member': '_checkpoint/done.toml',
        'restore': 'Extract contexts/ members into this geometry directory.',
        'script_sha256': digest(Path(__file__)),
    }
    temporary_record = record_path.with_suffix('.json.tmp')
    temporary_record.write_text(json.dumps(record, indent=2) + '\n')
    os.replace(temporary_record, record_path)
    remaining = {k: v for k, v in data['files'].items() if k not in expected}
    remaining.update({archive_path.name: digest(archive_path),
                      record_path.name: digest(record_path)})
    # Existing Julia checked_unit reads this same file/hash contract. Keep the
    # original marker inside the archive for per-file verification and replay.
    text = '[files]\n' + ''.join(
        f'{json.dumps(k)} = {json.dumps(v)}\n' for k, v in sorted(remaining.items()))
    temporary_marker = marker.with_suffix('.toml.tmp')
    temporary_marker.write_text(text)
    os.replace(temporary_marker, marker)
    verify_checkpoint(folder)
    verify_raw(folder, expected)
    shutil.rmtree(contexts)
    return True


def verify_raw(folder, expected):
    files = list((folder / 'contexts').rglob('*'))
    if any(p.is_symlink() for p in files):
        raise ValueError('context symlinks are not supported')
    actual = {str(p.relative_to(folder)) for p in files if p.is_file()}
    if actual != set(expected):
        raise ValueError('context files differ from original checkpoint')
    for name, value in expected.items():
        if digest(folder / name) != value:
            raise ValueError(f'raw context mismatch: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    args = parser.parse_args()
    if (args.study / 'checksums.toml').exists():
        parser.error('study already has a final manifest; archive before finalization')
    count = 0
    for category in ('anchors', 'expansion', 'representatives'):
        for folder in sorted((args.study / category).glob('*/geometry')):
            if not (folder / 'done.toml').is_file():
                continue
            # Completed archives are checked by ordinary resume. Avoid
            # repeatedly decompressing them during incremental batch execution.
            data = tomllib.loads((folder / 'done.toml').read_text())
            if 'contexts.tar.gz' in data['files'] and not (folder / 'contexts').exists():
                continue
            count += archive_geometry(folder)
    print(json.dumps({'newly_archived_geometries': count}), flush=True)


if __name__ == '__main__':
    main()
