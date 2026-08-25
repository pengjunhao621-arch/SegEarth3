import json
import os
import unittest

from role_functional_text_definitions import (
    JOINT_ROLE_VIEW_OPERATOR_NAMES,
    load_joint_role_view_registry,
    load_role_functional_text_bank,
    load_role_text_selection_registry,
)
from tools.summarize_joint_role_view import _expected_variants


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = ('uavid', 'isaid', 'voc20', 'cityscapes')
CLASS_FILES = {
    'uavid': 'cls_uavid.txt',
    'voc20': 'cls_voc20.txt',
    'cityscapes': 'cls_city_scapes.txt',
    'isaid': 'cls_iSAID.txt',
}


def class_contract(dataset):
    path = os.path.join(ROOT, 'configs', CLASS_FILES[dataset])
    with open(path, encoding='utf-8') as handle:
        rows = [
            [value.strip() for value in line.strip().split(',')]
            for line in handle if line.strip()]
    return [row[0] for row in rows], rows


class DomainExtensionJointProfileTest(unittest.TestCase):

    def test_prompt_banks_match_official_class_files(self):
        for dataset in DATASETS:
            names, official = class_contract(dataset)
            path = os.path.join(
                ROOT, 'configs', 'prompt_banks',
                'role_functional_text_v3', f'{dataset}.json')
            bank = load_role_functional_text_bank(path, names, official)
            self.assertTrue(bank['frozen_before_evaluation'])
            self.assertEqual(len(bank['classes']), len(names))
            self.assertGreaterEqual(bank['_prompt_count'], 8)

    def test_discovery_registry_covers_all_role_compositions(self):
        path = os.path.join(
            ROOT, 'configs', 'experiments',
            'joint_role_view_domain_extension_v1.json')
        for dataset in DATASETS:
            record = load_joint_role_view_registry(path, dataset)
            self.assertEqual(len(record['role_candidates']), 36)
            self.assertEqual(record['anchor_candidate'], 'anchor')
            self.assertIsNone(record['prior_view_miou'])
            self.assertTrue(all(
                candidate['source_miou'] is None
                for candidate in record['role_candidates']))
            payload = {
                'role_candidates': record['role_candidates'],
                'anchor_candidate': record['anchor_candidate'],
            }
            self.assertEqual(
                len(_expected_variants(payload)),
                1 + 36 * len(JOINT_ROLE_VIEW_OPERATOR_NAMES))

    def test_selection_and_view_contracts_are_dataset_specific(self):
        selection_path = os.path.join(
            ROOT, 'configs', 'experiments',
            'role_text_domain_extension_v1.json')
        view_path = os.path.join(
            ROOT, 'configs', 'experiments',
            'role_visual_field_domain_extension_v1.json')
        with open(view_path, encoding='utf-8') as handle:
            views = json.load(handle)['datasets']
        for dataset in DATASETS:
            selection = load_role_text_selection_registry(
                selection_path, dataset)
            self.assertEqual(selection['_best_overall_slots'], (0, 0, 0))
            self.assertEqual(
                views[dataset]['reference_endpoint'], 'global')
            self.assertGreater(
                views[dataset]['context_size'], views[dataset]['fine_size'])


if __name__ == '__main__':
    unittest.main()
