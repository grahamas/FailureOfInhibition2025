"""Lossless geometry packing and checkpoint failure tests (standard library)."""
from pathlib import Path
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    'archives', Path(__file__).parents[1] / 'scripts/archive_input_response_contexts.py')
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.geometry = self.root / 'geometry'
        self.contexts = self.geometry / 'contexts/0.0_0.0'
        self.contexts.mkdir(parents=True)
        (self.geometry / 'summary.toml').write_text('screen = true\n')
        (self.contexts / 'input.toml').write_text('B_E = 0.0\nB_I = 0.0\n')
        (self.contexts / 'context.toml').write_text('raw = "preserve every byte"\n')
        self.marker()
        self.original = (self.geometry / 'done.toml').read_bytes()

    def marker(self):
        files = {str(p.relative_to(self.geometry)): A.digest(p)
                 for p in self.geometry.rglob('*') if p.is_file() and p.name != 'done.toml'}
        (self.geometry / 'done.toml').write_text('[files]\n' + ''.join(
            f'{json.dumps(k)} = {json.dumps(v)}\n' for k, v in files.items()))

    def test_roundtrip_and_resume(self):
        self.assertTrue(A.archive_geometry(self.geometry))
        self.assertFalse((self.geometry / 'contexts').exists())
        original, files = A.verify_archive(self.geometry / 'contexts.tar.gz')
        self.assertEqual(original, self.original)
        self.assertEqual(len(files), 2)
        A.verify_checkpoint(self.geometry)
        self.assertFalse(A.archive_geometry(self.geometry))

    def test_changed_raw_file_is_retained(self):
        (self.contexts / 'context.toml').write_text('changed = true\n')
        with self.assertRaises(ValueError):
            A.archive_geometry(self.geometry)
        self.assertTrue(self.contexts.is_dir())
        self.assertFalse((self.geometry / 'contexts.tar.gz').exists())

    def test_untracked_raw_file_is_retained(self):
        (self.contexts / 'extra.txt').write_text('untracked data')
        with self.assertRaises(ValueError):
            A.archive_geometry(self.geometry)
        self.assertTrue((self.contexts / 'extra.txt').is_file())

    def test_symlink_is_rejected(self):
        path = self.contexts / 'context.toml'
        contents = path.read_bytes()
        target = self.root / 'outside.toml'
        target.write_bytes(contents)
        path.unlink()
        path.symlink_to(target)
        with self.assertRaises(ValueError):
            A.archive_geometry(self.geometry)
        self.assertTrue(target.is_file())

    def test_detailed_search_waits_for_confirmation(self):
        (self.geometry / 'summary.toml').write_text('screen = false\n')
        self.marker()
        self.assertFalse(A.archive_geometry(self.geometry))
        self.assertTrue(self.contexts.is_dir())
        confirmation = self.root / 'geometry_confirmation'
        confirmation.mkdir()
        (confirmation / 'done.toml').write_text('[files]\n')
        self.assertTrue(A.archive_geometry(self.geometry))

    def test_archive_damage_breaks_resume(self):
        A.archive_geometry(self.geometry)
        with (self.geometry / 'contexts.tar.gz').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaises(ValueError):
            A.archive_geometry(self.geometry)

    def test_resume_after_archive_written_before_marker(self):
        backup = self.root / 'backup'
        shutil.copytree(self.geometry / 'contexts', backup)
        A.archive_geometry(self.geometry)
        shutil.copytree(backup, self.geometry / 'contexts')
        (self.geometry / 'done.toml').write_bytes(self.original)
        self.assertTrue(A.archive_geometry(self.geometry))
        A.verify_checkpoint(self.geometry)

    def test_resume_after_marker_written_before_raw_deletion(self):
        backup = self.root / 'backup'
        shutil.copytree(self.geometry / 'contexts', backup)
        A.archive_geometry(self.geometry)
        shutil.copytree(backup, self.geometry / 'contexts')
        self.assertFalse(A.archive_geometry(self.geometry))
        self.assertFalse((self.geometry / 'contexts').exists())
        A.verify_checkpoint(self.geometry)

    def test_cli_refuses_a_finalized_study(self):
        (self.root / 'checksums.toml').write_text('[files]\n')
        result = subprocess.run([sys.executable, str(SPEC.origin), str(self.root)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('already has a final manifest', result.stderr)
        self.assertEqual((self.geometry / 'done.toml').read_bytes(), self.original)
        self.assertTrue(self.contexts.is_dir())


if __name__ == '__main__':
    unittest.main()
