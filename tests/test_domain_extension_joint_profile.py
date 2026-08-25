import json
import os
import tempfile
import unittest

from role_functional_text_definitions import (
    JOINT_ROLE_VIEW_OPERATOR_NAMES,
    load_joint_role_view_registry,
    load_role_functional_text_bank,
    load_role_text_selection_registry,
)
from tools.summarize_joint_role_view import _expected_variants
from tools.build_sequential_joint_registry import build_registry


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
                'operator_specs': [
                    {'name': name} for name in JOINT_ROLE_VIEW_OPERATOR_NAMES],
            }
            self.assertEqual(
                len(_expected_variants(payload)),
                1 + 36 * len(JOINT_ROLE_VIEW_OPERATOR_NAMES))

    def test_role_only_contract_and_selected_joint_registry(self):
        source_path = os.path.join(
            ROOT, 'configs', 'experiments',
            'joint_role_view_domain_extension_v1.json')
        source = load_joint_role_view_registry(source_path, 'isaid')
        payload = {
            'role_candidates': source['role_candidates'],
            'anchor_candidate': source['anchor_candidate'],
            'operator_specs': [{'name': 'global'}],
        }
        self.assertEqual(len(_expected_variants(payload)), 37)

        summary = {'profiles': [
            dict(dataset='isaid', profile='official', candidate_id='anchor',
                 slots='0+0+0', admission='native', miou=27.6),
            dict(dataset='isaid', profile='role_only_best',
                 candidate_id='p1_s2_i1', slots='1+2+1',
                 admission='native', miou=29.1),
        ]}
        registry = build_registry(summary, source_path, 'isaid')
        selection = registry['selection']
        self.assertEqual(selection['selected_candidate'], 'p1_s2_i1')
        self.assertAlmostEqual(selection['selected_gain'], 1.5)
        candidates = registry['datasets']['isaid']['role_candidates']
        self.assertEqual([value['id'] for value in candidates], [
            'anchor', 'p1_s2_i1'])

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'selected.json')
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(registry, handle)
            loaded = load_joint_role_view_registry(path, 'isaid')
        self.assertEqual(len(loaded['role_candidates']), 2)
        self.assertEqual(loaded['current_role_candidate'], 'p1_s2_i1')

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
