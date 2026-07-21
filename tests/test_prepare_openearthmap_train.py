import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / 'tools' / 'dataset_converters' / 'prepare_openearthmap_train.py'


class PrepareOpenEarthMapTrainTest(unittest.TestCase):

    def _sample(self, subset, region, name):
        image = subset / region / 'images' / name
        label = subset / region / 'labels' / name
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b'image')
        label.write_bytes(b'label')
        return image, label

    def test_official_basename_manifest_is_materialized(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            subset = root / 'OpenEarthMap_wo_xBD'
            first_image, first_label = self._sample(
                subset, 'aachen', 'aachen_1.tif')
            self._sample(subset, 'tokyo', 'tokyo_1.tif')
            (subset / 'train.txt').write_text('aachen_1.tif\n', encoding='utf-8')

            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(root)],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            image_link = root / 'img_dir' / 'train' / 'aachen_1.tif'
            label_link = root / 'ann_dir' / 'train' / 'aachen_1.tif'
            self.assertTrue(image_link.is_symlink())
            self.assertTrue(label_link.is_symlink())
            self.assertEqual(image_link.resolve(), first_image.resolve())
            self.assertEqual(label_link.resolve(), first_label.resolve())
            report = json.loads(
                (root / 'train_prepare_report.json').read_text(
                    encoding='utf-8'))
            self.assertEqual(report['selected_pairs'], 1)
            self.assertEqual(report['source_layout'], 'nested_subset_root')
            self.assertEqual(report['output_layout'],
                             'flat_img_dir_and_ann_dir')
            self.assertIsNone(report['runtime_manifest'])
            self.assertEqual(report['label_policy'],
                             'preserve original ids 0..8; reduce_zero_label=False')

    def test_flattened_server_layout_is_auto_detected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image, label = self._sample(root, 'accra', 'accra_1.tif')
            # Existing converted validation folders may coexist with raw cities.
            (root / 'img_dir' / 'val').mkdir(parents=True)
            (root / 'ann_dir' / 'val').mkdir(parents=True)
            (root / 'train.txt').write_text('accra_1.tif\n', encoding='utf-8')

            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(root)],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            image_link = root / 'img_dir' / 'train' / 'accra_1.tif'
            label_link = root / 'ann_dir' / 'train' / 'accra_1.tif'
            self.assertTrue(image_link.is_symlink())
            self.assertTrue(label_link.is_symlink())
            self.assertEqual(image_link.resolve(), image.resolve())
            self.assertEqual(label_link.resolve(), label.resolve())
            report = json.loads(
                (root / 'train_prepare_report.json').read_text(
                    encoding='utf-8'))
            self.assertEqual(report['source_layout'], 'flattened_data_root')
            self.assertEqual(report['subset_root'], str(root.resolve()))

    def test_duplicate_basename_fails_instead_of_silent_selection(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            subset = root / 'OpenEarthMap_wo_xBD'
            self._sample(subset, 'region_a', 'shared.tif')
            self._sample(subset, 'region_b', 'shared.tif')
            (subset / 'train.txt').write_text('shared.tif\n', encoding='utf-8')

            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(root), '--dry-run'],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn('ambiguous: shared.tif', result.stderr)

    def test_region_qualified_duplicate_basename_cannot_be_flattened(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            subset = root / 'OpenEarthMap_wo_xBD'
            self._sample(subset, 'region_a', 'shared.tif')
            self._sample(subset, 'region_b', 'shared.tif')
            (subset / 'train.txt').write_text(
                'region_a/images/shared.tif\nregion_b/images/shared.tif\n',
                encoding='utf-8')

            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(root), '--dry-run'],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn('duplicate basenames', result.stderr)

    def test_allow_missing_materializes_available_manifest_entries(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image, label = self._sample(root, 'accra', 'accra_1.tif')
            (root / 'train.txt').write_text(
                'accra_1.tif\nadelaide_10.tif\n', encoding='utf-8')

            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(root), '--allow-missing'],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                (root / 'img_dir' / 'train' / 'accra_1.tif').resolve(),
                image.resolve())
            self.assertEqual(
                (root / 'ann_dir' / 'train' / 'accra_1.tif').resolve(),
                label.resolve())
            report = json.loads(
                (root / 'train_prepare_report.json').read_text(
                    encoding='utf-8'))
            self.assertEqual(report['manifest']['manifest_entries'], 2)
            self.assertEqual(report['manifest']['resolved_entries'], 1)
            self.assertEqual(report['manifest']['resolved_ratio'], 0.5)
            self.assertEqual(report['manifest']['skipped_missing_count'], 1)
            self.assertEqual(
                report['manifest']['skipped_missing_entries'],
                [{'line': 2, 'token': 'adelaide_10.tif'}])
            self.assertIn('WARNING: skipped 1/2', result.stderr)


if __name__ == '__main__':
    unittest.main()
