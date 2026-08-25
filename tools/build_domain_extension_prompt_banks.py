#!/usr/bin/env python3
"""Build frozen Role prompt banks for UAVid and the domain extensions.

The generated prompts follow the same functional contract as the retained
remote-sensing banks: Presence receives existence-oriented noun phrases,
Semantic receives dense coverage/appearance phrases, and Instance receives
object- or region-organization phrases.  The generated JSON files are checked
in so server evaluation never invokes an LLM or changes prompts online.
"""

import json
import os


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(
    ROOT, 'configs', 'prompt_banks', 'role_functional_text_v3')
EXPERIMENT_DIR = os.path.join(ROOT, 'configs', 'experiments')


DISPLAY_NAMES = {
    'trafficlight': 'traffic light',
    'firehydrant': 'fire hydrant',
    'stopsign': 'stop sign',
    'parkingmeter': 'parking meter',
    'sportsball': 'sports ball',
    'baseballbat': 'baseball bat',
    'baseballglove': 'baseball glove',
    'tennisracket': 'tennis racket',
    'wineglass': 'wine glass',
    'hotdog': 'hot dog',
    'pottedplant': 'potted plant',
    'diningtable': 'dining table',
    'cellphone': 'cell phone',
    'teddybear': 'teddy bear',
    'hairdrier': 'hair drier',
    'playingfield': 'playing field',
    'waterdrops': 'water drops',
}


APPEARANCE = {
    'person': 'complete human figure with head, torso and limbs',
    'bicycle': 'two-wheeled bicycle frame and wheels',
    'car': 'complete car body with a compact rectangular footprint',
    'motorcycle': 'two-wheeled motor vehicle silhouette',
    'airplane': 'aircraft body with wings and tail',
    'aeroplane': 'aircraft body with wings and tail',
    'bus': 'large elongated passenger vehicle body',
    'train': 'connected rail vehicle body',
    'truck': 'large road vehicle with cab and cargo body',
    'boat': 'complete floating vessel hull',
    'bird': 'complete bird body and visible wings',
    'cat': 'complete cat body and silhouette',
    'dog': 'complete dog body and silhouette',
    'horse': 'complete horse body and legs',
    'cow': 'complete cow body and legs',
    'sheep': 'complete sheep body and woolly silhouette',
    'chair': 'chair seat, back and supporting legs',
    'table': 'complete tabletop and supporting structure',
    'dining table': 'complete dining tabletop and supporting structure',
    'sofa': 'complete upholstered sofa body',
    'television monitor': 'rectangular display screen and frame',
    'road': 'continuous drivable road surface',
    'sidewalk': 'continuous pedestrian pavement beside a road',
    'building': 'complete building facade and visible structure',
    'wall': 'continuous vertical wall surface',
    'fence': 'thin connected fence structure',
    'pole': 'narrow vertical pole structure',
    'tree': 'complete tree canopy and trunk region',
    'grass': 'continuous low green vegetation cover',
    'sky': 'continuous open sky region',
    'river': 'elongated continuous water channel',
    'sea': 'large continuous water surface',
    'water other': 'continuous visible water surface',
    'railroad': 'parallel connected railway track region',
    'playing field': 'bounded open sports field surface',
    'floor wood': 'continuous wooden floor surface',
    'floor tile': 'continuous tiled floor surface',
    'wall brick': 'continuous brick wall surface',
    'window other': 'complete window opening and frame',
}


ISAID_PROMPTS = {
    'background': {
        'presence': ['visible image background', 'non-target background area'],
        'semantic': [
            'remaining background pixels',
            'continuous non-target image region',
            'unlabeled overhead background surface'],
        'instance': [
            'connected background component',
            'bounded non-target background region'],
    },
    'ship': {
        'presence': ['water vessel', 'visible ships'],
        'semantic': [
            'complete elongated ship footprint',
            'ship-shaped region on the water surface',
            'overhead vessel body with bow and stern'],
        'instance': ['individual ship instance', 'separate complete vessel'],
    },
    'store tank': {
        'presence': ['storage tank', 'visible storage tanks'],
        'semantic': [
            'complete circular storage-tank footprint',
            'round industrial tank roof region',
            'overhead cylindrical storage tank'],
        'instance': [
            'individual storage tank', 'separate complete circular tank'],
    },
    'baseball diamond': {
        'presence': ['baseball field diamond', 'visible baseball diamonds'],
        'semantic': [
            'complete baseball infield diamond',
            'diamond-shaped baseball field surface',
            'overhead baseball diamond with infield boundary'],
        'instance': [
            'individual baseball diamond', 'separate complete baseball field'],
    },
    'tennis court': {
        'presence': ['tennis playing court', 'visible tennis courts'],
        'semantic': [
            'complete rectangular tennis-court surface',
            'tennis court with line-marked playing area',
            'overhead tennis court rectangle'],
        'instance': ['individual tennis court', 'separate complete tennis court'],
    },
    'basketball court': {
        'presence': ['basketball playing court', 'visible basketball courts'],
        'semantic': [
            'complete rectangular basketball-court surface',
            'basketball court with line-marked playing area',
            'overhead basketball court rectangle'],
        'instance': [
            'individual basketball court', 'separate complete basketball court'],
    },
    'ground track field': {
        'presence': ['athletic track field', 'visible running-track fields'],
        'semantic': [
            'complete oval athletic-track region',
            'running track surrounding a central field',
            'overhead track-and-field facility'],
        'instance': [
            'individual ground track field', 'separate complete athletic track'],
    },
    'bridge': {
        'presence': ['transport bridge', 'visible bridges'],
        'semantic': [
            'complete elongated bridge deck',
            'bridge span crossing water or land',
            'overhead connected bridge structure'],
        'instance': ['individual bridge', 'separate complete bridge span'],
    },
    'large vehicle': {
        'presence': ['large road vehicle', 'visible large vehicles'],
        'semantic': [
            'complete large-vehicle footprint',
            'elongated large vehicle body from overhead',
            'large truck or bus roof region'],
        'instance': [
            'individual large vehicle', 'separate complete large vehicle'],
    },
    'small vehicle': {
        'presence': ['small road vehicle', 'visible small vehicles'],
        'semantic': [
            'complete compact vehicle footprint',
            'small rectangular vehicle body from overhead',
            'compact car-like roof region'],
        'instance': [
            'individual small vehicle', 'separate complete small vehicle'],
    },
    'helicopter': {
        'presence': ['rotary-wing aircraft', 'visible helicopters'],
        'semantic': [
            'complete helicopter footprint',
            'helicopter body with rotor extent',
            'overhead rotary-wing aircraft silhouette'],
        'instance': ['individual helicopter', 'separate complete helicopter'],
    },
    'swimming pool': {
        'presence': ['outdoor swimming pool', 'visible swimming pools'],
        'semantic': [
            'complete bounded swimming-pool surface',
            'rectangular blue pool-water region',
            'overhead outdoor pool footprint'],
        'instance': ['individual swimming pool', 'separate complete pool basin'],
    },
    'roundabout': {
        'presence': ['road roundabout', 'visible roundabouts'],
        'semantic': [
            'complete circular road junction',
            'ring-shaped traffic roundabout surface',
            'overhead roundabout with central island'],
        'instance': ['individual roundabout', 'separate complete road roundabout'],
    },
    'soccer ball field': {
        'presence': ['soccer playing field', 'visible soccer fields'],
        'semantic': [
            'complete rectangular soccer-field surface',
            'soccer pitch with line-marked boundary',
            'overhead grass football field'],
        'instance': ['individual soccer field', 'separate complete soccer pitch'],
    },
    'plane': {
        'presence': ['fixed-wing aircraft', 'visible airplanes'],
        'semantic': [
            'complete airplane footprint',
            'aircraft body with wings and tail',
            'overhead fixed-wing aircraft silhouette'],
        'instance': ['individual airplane', 'separate complete aircraft'],
    },
    'harbor': {
        'presence': ['port harbor', 'visible harbor area'],
        'semantic': [
            'complete connected harbor region',
            'port water and dock interface',
            'overhead sheltered harbor basin'],
        'instance': ['connected harbor component', 'bounded complete port region'],
    },
}


UAVID_PROMPTS = {
    'background': {
        'presence': ['visible image background', 'non-target urban background'],
        'semantic': [
            'remaining background pixels',
            'continuous non-target aerial background',
            'unclassified overhead background surface'],
        'instance': [
            'connected background component',
            'bounded non-target background region'],
    },
    'building': {
        'presence': ['visible buildings', 'one or more building structures'],
        'semantic': [
            'complete building footprint',
            'rectilinear rooftop surface',
            'overhead building roofs with sharp boundaries'],
        'instance': [
            'individual building region',
            'separate complete building footprint'],
    },
    'road': {
        'presence': ['visible roads', 'road network'],
        'semantic': [
            'continuous drivable road surface',
            'elongated paved roadway',
            'overhead connected road network'],
        'instance': [
            'connected road component',
            'bounded complete road region'],
    },
    'car': {
        'presence': ['visible cars', 'one or more road vehicles'],
        'semantic': [
            'complete compact car footprint',
            'rectangular vehicle roof surface',
            'small overhead vehicle body'],
        'instance': [
            'individual car',
            'separate complete vehicle footprint'],
    },
    'tree': {
        'presence': ['visible trees', 'tree canopy cover'],
        'semantic': [
            'complete tree canopy extent',
            'coarse-textured tree crowns',
            'overhead individual and grouped tree canopies'],
        'instance': [
            'individual tree crown',
            'separate complete canopy region'],
    },
    'vegetation': {
        'presence': ['visible low vegetation', 'vegetation cover'],
        'semantic': [
            'continuous low-vegetation surface',
            'fine-textured vegetated ground',
            'overhead coherent vegetation cover'],
        'instance': [
            'connected vegetation component',
            'bounded vegetation patch'],
    },
    'human': {
        'presence': ['visible people', 'one or more humans'],
        'semantic': [
            'complete human figure extent',
            'small upright human silhouette',
            'visible pedestrian body'],
        'instance': [
            'individual person',
            'separate complete human figure'],
    },
}


def parse_classes(path):
    rows = []
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append([value.strip() for value in line.split(',')])
    return rows


def display_name(name):
    value = DISPLAY_NAMES.get(name, name.replace('-', ' '))
    return ' '.join(value.split())


def plural_phrase(noun):
    irregular = {
        'person': 'people', 'sheep': 'sheep', 'fish': 'fish',
        'bus': 'buses', 'sports ball': 'sports balls',
    }
    if noun in irregular:
        return irregular[noun]
    if noun.endswith(('s', 'x', 'ch', 'sh')):
        return noun + 'es'
    if noun.endswith('y') and len(noun) > 1 and noun[-2] not in 'aeiou':
        return noun[:-1] + 'ies'
    return noun + 's'


def generic_prompts(name, is_thing):
    noun = display_name(name)
    visual = APPEARANCE.get(noun)
    if is_thing:
        return {
            'presence': [f'visible {noun}', f'one or more {plural_phrase(noun)}'],
            'semantic': [
                f'complete visible {noun} extent',
                f'{noun} silhouette and surface',
                visual or f'whole {noun} object including its boundary'],
            'instance': [
                f'individual {noun} instance',
                f'separate complete {noun} object'],
        }
    return {
        'presence': [f'visible {noun}', f'{noun} area'],
        'semantic': [
            f'continuous {noun} region',
            f'{noun} pixels with coherent appearance',
            visual or f'spatially connected {noun} surface'],
        'instance': [
            f'connected {noun} component',
            f'bounded {noun} region'],
    }


def dataset_specs():
    return {
        'uavid': dict(
            label='UAVid', class_file='configs/cls_uavid.txt',
            thing=lambda index, name: name in {
                'building', 'car', 'tree', 'human'}),
        'voc20': dict(
            label='Pascal VOC20', class_file='configs/cls_voc20.txt',
            thing=lambda index, name: True),
        'cityscapes': dict(
            label='Cityscapes', class_file='configs/cls_city_scapes.txt',
            thing=lambda index, name: index >= 11),
        'isaid': dict(
            label='iSAID', class_file='configs/cls_iSAID.txt',
            thing=lambda index, name: name not in {
                'background', 'ground track field', 'harbor'}),
    }


def build_bank(key, spec):
    classes = []
    rows = parse_classes(os.path.join(ROOT, spec['class_file']))
    for index, official in enumerate(rows):
        name = official[0]
        special = {
            'isaid': ISAID_PROMPTS,
            'uavid': UAVID_PROMPTS,
        }.get(key)
        prompts = (
            special[name] if special is not None
            else generic_prompts(name, spec['thing'](index, name)))
        classes.append({
            'id': index,
            'name': name,
            'official_prompts': official,
            'presence_candidates': prompts['presence'],
            'semantic_candidates': prompts['semantic'],
            'instance_candidates': prompts['instance'],
        })
    return {
        'schema_version': 1,
        'protocol': 'role_functional_text_screen_v1',
        'dataset': spec['label'],
        'generator': (
            'Codex; frozen role-functional candidates generated before '
            'evaluation from dataset ontology and visual geometry'),
        'frozen_before_evaluation': True,
        'candidate_semantics': {
            'presence': [
                'P1: canonical visual existence or equivalent noun phrase',
                'P2: multiplicity/area-oriented concept existence'],
            'semantic': [
                'S1: complete dense extent or spatial continuity',
                'S2: silhouette/surface or coherent region appearance',
                'S3: dataset-view-specific geometry and visual structure'],
            'instance': [
                'I1: individual object or connected component',
                'I2: separated/bounded complete object or region'],
        },
        'classes': classes,
    }


def write_json(path, payload):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(os.path.relpath(path, ROOT))


def build_selection_registry(specs):
    datasets = {}
    for key in specs:
        datasets[key] = {
            'prompt_bank': (
                f'configs/prompt_banks/role_functional_text_v3/{key}.json'),
            'best_overall': {
                'slots': [0, 0, 0], 'admission': 'native',
                'status': 'discovery_anchor'},
            'best_all_nonzero': {
                'slots': [1, 1, 1], 'admission': 'native',
                'status': 'discovery_control'},
            'instance_diagnostic_slot': 1,
        }
    return {'schema_version': 1, 'datasets': datasets}


def build_visual_registry():
    return {
        'schema_version': 1,
        'protocol': 'role_visual_field_v1',
        'note': (
            'Dataset-aligned Local/Global discovery fields. Global is the '
            'official full-image observation; Local preserves finer object '
            'detail through non-overlapping crops.'),
        'datasets': {
            'voc20': {
                'source_mode': 'image', 'fine_size': 384,
                'context_size': 768, 'reference_endpoint': 'global'},
            'cityscapes': {
                'source_mode': 'image', 'fine_size': 512,
                'context_size': 1024, 'reference_endpoint': 'global'},
            'isaid': {
                'source_mode': 'image', 'fine_size': 512,
                'context_size': 896, 'reference_endpoint': 'global'},
            'uavid': {
                'source_mode': 'image', 'fine_size': 640,
                'context_size': 1280, 'reference_endpoint': 'global'},
        },
    }


def all_role_candidates():
    candidates = []
    for presence in range(3):
        for semantic in range(4):
            for instance in range(3):
                slots = [presence, semantic, instance]
                identifier = (
                    'anchor' if slots == [0, 0, 0]
                    else f'p{presence}_s{semantic}_i{instance}')
                candidates.append({
                    'id': identifier,
                    'slots': slots,
                    'admission': 'native',
                })
    return candidates


def build_joint_registry(specs):
    candidates = all_role_candidates()
    return {
        'schema_version': 1,
        'protocol': 'joint_role_view_profile_v1',
        'stage': 'dataset_profile_discovery',
        'note': (
            'Exhaustive 3x4x3 Role compositions share one cached prompt bank; '
            'each is evaluated with the common 15-operator Local/Global '
            'family. source_miou/prior_view_miou are intentionally absent '
            'because these datasets have no prior Role-View screen.'),
        'datasets': {
            key: {
                'anchor_candidate': 'anchor',
                'current_role_candidate': 'anchor',
                'prior_view_operator': 'global',
                'role_candidates': candidates,
            }
            for key in specs
        },
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    specs = dataset_specs()
    for key, spec in specs.items():
        output = os.path.join(OUTPUT_DIR, f'{key}.json')
        write_json(output, build_bank(key, spec))
    write_json(
        os.path.join(EXPERIMENT_DIR, 'role_text_domain_extension_v1.json'),
        build_selection_registry(specs))
    write_json(
        os.path.join(EXPERIMENT_DIR, 'role_visual_field_domain_extension_v1.json'),
        build_visual_registry())
    write_json(
        os.path.join(EXPERIMENT_DIR, 'joint_role_view_domain_extension_v1.json'),
        build_joint_registry(specs))


if __name__ == '__main__':
    main()
