import json
import math
import os
from collections import defaultdict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from PIL import Image, ImageFilter

from region_hypothesis_diagnostic import RegionHypothesisDiagnosticMixin
from region_contrastive_readout import RegionContrastiveReadoutMixin
from candidate_region_quality_diagnostic import (
    CandidateRegionQualityDiagnosticMixin,
)
from query_topology_diagnostic import QueryTopologyDiagnosticMixin
from rethinking_reviewer import RethinkingReviewerMixin
from concept_specificity_diagnostic import ConceptSpecificityDiagnosticMixin
from cross_image_bank_diagnostic import CrossImageBankDiagnosticMixin
from ontology_self_verification import OntologySelfVerificationMixin
from ontology_readout_oracle import OntologyReadoutOracleMixin
from sam3_geometry_requery_diagnostic import Sam3GeometryRequeryDiagnosticMixin
from self_prompted_concept_verification import (
    SelfPromptedConceptVerificationMixin,
)
from evidence_enhancement import EvidenceEnhancementMixin
from structure_aware_recalibration import StructureAwareRecalibrationMixin
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


@MODELS.register_module()
class SegEarthOV3Segmentation(
        CrossImageBankDiagnosticMixin,
        OntologySelfVerificationMixin,
        OntologyReadoutOracleMixin,
        EvidenceEnhancementMixin,
        StructureAwareRecalibrationMixin,
        SelfPromptedConceptVerificationMixin,
        Sam3GeometryRequeryDiagnosticMixin,
        ConceptSpecificityDiagnosticMixin,
        RethinkingReviewerMixin,
        QueryTopologyDiagnosticMixin,
        CandidateRegionQualityDiagnosticMixin,
        RegionHypothesisDiagnosticMixin,
        RegionContrastiveReadoutMixin,
        BaseSegmentor):
    def __init__(self, classname_path,
                 device=torch.device('cuda'),
                 prob_thd=0.0,
                 bg_idx=0,
                 slide_stride=0,
                 slide_crop=0,
                 confidence_threshold=0.5,
                 use_sem_seg=True,
                 use_presence_score=True,
                 use_transformer_decoder=True,
                 instance_score_type='presence',
                 dump_evidence_stats=False,
                 evidence_stats_path=None,
                 dump_competition_stats=False,
                 competition_stats_path=None,
                 use_reject_aware_calibration=False,
                 use_reject_recovery=True,
                 use_competition_suppression=False,
                 reject_recovery_bg_role='auto',
                 reject_recovery_factor=0.25,
                 reject_recovery_semantic_thd=0.02,
                 reject_recovery_instance_thd=0.01,
                 reject_recovery_margin_thd=0.02,
                 use_residual_background_modeling=False,
                 dump_residual_background_stats=False,
                 residual_background_stats_path=None,
                 residual_background_route='threshold_only',
                 residual_background_score_factor=0.25,
                 residual_background_min_score=0.02,
                 residual_background_min_semantic=0.03,
                 residual_background_min_instance=0.01,
                 residual_background_min_presence=0.0,
                 residual_background_min_local=0.55,
                 residual_background_min_fg_bg_margin=0.0,
                 residual_background_min_reliability_margin=0.02,
                 residual_background_semantic_weight=0.25,
                 residual_background_instance_weight=0.25,
                 residual_background_presence_weight=0.10,
                 residual_background_local_weight=0.10,
                 residual_background_bg_weight=0.50,
                 residual_background_local_kernel=7,
                 residual_background_require_head_support=True,
                 dump_background_separability_stats=False,
                 background_separability_stats_path=None,
                 background_separability_score_thresholds='0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.30,0.50',
                 background_separability_margin_thresholds='-0.10,-0.05,0.00,0.02,0.05,0.10,0.20,0.40',
                 background_separability_reliability_thresholds='-0.20,-0.10,-0.05,0.00,0.02,0.05,0.10,0.20,0.40',
                 background_separability_local_thresholds='0.40,0.50,0.60,0.70,0.80,0.90',
                 competition_top2_ratio=0.30,
                 competition_sem_inst_gap=0.15,
                 competition_penalty=1.0,
                 use_evidence_competition_graph=False,
                 ecg_top2_ratio=0.75,
                 ecg_margin_thd=0.05,
                 ecg_semantic_thd=0.03,
                 ecg_instance_thd=0.01,
                 ecg_agreement_gap=0.01,
                 ecg_sem_over_inst_gap=0.10,
                 ecg_sem_only_transfer=0.25,
                 ecg_boost_scale=0.50,
                 ecg_suppress_scale=0.10,
                 dump_oracle_stats=False,
                 oracle_stats_path=None,
                 oracle_topk=3,
                 use_topk_candidate_verifier=False,
                 dump_topk_verifier_stats=False,
                 topk_verifier_stats_path=None,
                 topk_verifier_k=3,
                 topk_verifier_apply_mode='all',
                 topk_verifier_require_final_candidate=True,
                 topk_verifier_min_aux_rank=0.34,
                 topk_verifier_low_margin=0.08,
                 topk_verifier_confident_margin=0.25,
                 topk_verifier_aux_rank_weight=0.35,
                 topk_verifier_vote_weight=0.10,
                 topk_verifier_local_weight=0.15,
                 topk_verifier_final_weight=0.05,
                 topk_verifier_local_kernel=7,
                 dump_error_rank_stats=False,
                 error_rank_stats_path=None,
                 error_rank_topk=3,
                 use_top2_risk_arbitration=False,
                 dump_top2_risk_stats=False,
                 top2_risk_stats_path=None,
                 top2_risk_topk=3,
                 top2_risk_min_margin=0.10,
                 top2_risk_max_margin=0.80,
                 top2_risk_min_sem_adv=0.10,
                 top2_risk_min_inst_adv=0.10,
                 top2_risk_min_agreement_adv=0.04,
                 top2_risk_min_top1_sem_only=0.20,
                 top2_risk_require_head_disagree=False,
                 top2_risk_bg_policy='avoid_top2_bg',
                 top2_risk_mode='source_advantage',
                 dump_multiview_oracle_stats=False,
                 multiview_oracle_stats_path=None,
                 multiview_oracle_topk=3,
                 multiview_oracle_views='hflip,rot90',
                 multiview_oracle_slide_views='',
                 multiview_oracle_include_base=True,
                 multiview_oracle_empty_cache=True,
                 dump_seed_separability_stats=False,
                 seed_separability_stats_path=None,
                 seed_dataset_name=None,
                 seed_final_score_thd=0.5,
                 seed_margin_thd=0.10,
                 seed_local_kernel=7,
                 seed_local_consistency_thd=0.80,
                 seed_core_kernel=5,
                 seed_core_consistency_thd=0.999,
                 seed_region_min_pixels=1,
                 seed_region_purity_thd=0.95,
                 seed_similarity_eps=1e-6,
                 seed_similarity_spaces='evidence',
                 dump_raw_mask_oracle_stats=False,
                 raw_mask_oracle_stats_path=None,
                 raw_mask_oracle_topk=10,
                 raw_mask_oracle_bin_thd=0.5,
                 raw_mask_oracle_min_pixels=16,
                 dump_candidate_quality_stats=False,
                 candidate_quality_stats_path=None,
                 candidate_quality_clean_purity_thd=0.8,
                 dump_prompt_competition_stats=False,
                 prompt_competition_stats_path=None,
                 prompt_competition_class_names=None,
                 prompt_competition_templates='{class}|remote sensing {class}|aerial image {class}|satellite image {class}|{class} land cover|{class} region',
                 prompt_competition_max_variants=6,
                 prompt_competition_use_builtin_variants=True,
                 dump_pair_prompt_competition_stats=False,
                 pair_prompt_competition_stats_path=None,
                 pair_prompt_competition_pairs=None,
                 pair_prompt_competition_templates='{class}, not {competitor}|{class} and not {competitor}|{class} region, excluding {competitor}|remote sensing {class}, not {competitor}|aerial image {class}, not {competitor}',
                 pair_prompt_competition_max_variants=6,
                 pair_prompt_competition_use_builtin_variants=True,
                 dump_geometry_context_stats=False,
                 geometry_context_stats_path=None,
                 geometry_context_score_thd=0.5,
                 geometry_context_min_pixels=64,
                 geometry_context_core_kernel=5,
                 dump_weak_support_stats=False,
                 weak_support_stats_path=None,
                 weak_support_thresholds='0.10,0.20,0.30,0.40,0.50',
                 weak_support_sources='final,semantic,instance,any_head',
                 weak_support_min_pair_pixels=1,
                 dump_reject_recovery_precision_stats=False,
                 reject_recovery_precision_stats_path=None,
                 reject_recovery_precision_score_bins='0.00,0.03,0.05,0.07,0.10,0.15,0.20,0.30',
                 reject_recovery_precision_semantic_thds='0.03,0.05,0.10',
                 reject_recovery_precision_instance_thds='0.01,0.03,0.05',
                 reject_recovery_precision_margin_thds='0.00,0.02,0.05',
                 reject_recovery_precision_local_thds='0.50,0.70,0.85',
                 reject_recovery_precision_use_pe=False,
                 reject_recovery_precision_pe_spaces='pe_layer_1,pe_layer_2',
                 dump_context_requery_stats=False,
                 context_requery_stats_path=None,
                 context_requery_pairs=None,
                 context_requery_modes='full,bbox1.5,bbox2.0,bbox3.0',
                 context_requery_thresholds='0.05,0.10,0.20,0.50',
                 context_requery_min_pair_pixels=256,
                 context_requery_min_crop_size=512,
                 context_requery_max_crop_size=1536,
                 dump_scale_requery_stats=False,
                 scale_requery_stats_path=None,
                 scale_requery_pairs=None,
                 scale_requery_scales='0.50,0.75,1.00,1.25,1.50',
                 scale_requery_thresholds='0.05,0.10,0.20,0.50',
                 scale_requery_min_pair_pixels=256,
                 scale_requery_max_side=1536,
                 dump_scale_stability_stats=False,
                 scale_stability_stats_path=None,
                 scale_stability_pairs=None,
                 scale_stability_scale=0.50,
                 scale_stability_topk=3,
                 scale_stability_drop_thresholds='0.02,0.05,0.10',
                 scale_stability_min_pair_pixels=256,
                 scale_stability_max_side=1536,
                 use_scale_stability_rerank=False,
                 dump_scale_stability_rerank_stats=False,
                 scale_stability_rerank_stats_path=None,
                 scale_stability_rerank_pairs=None,
                 scale_stability_rerank_pair_file=None,
                 scale_stability_rerank_scale=0.50,
                 scale_stability_rerank_topk=3,
                 scale_stability_rerank_gate='relative',
                 scale_stability_rerank_drop_threshold=0.05,
                 scale_stability_rerank_min_scaled_margin=0.0,
                 scale_stability_rerank_min_base_margin=-1.0,
                 scale_stability_rerank_max_base_margin=0.0,
                 scale_stability_rerank_max_target_rank=3,
                 scale_stability_rerank_margin_boost=1e-4,
                 scale_stability_rerank_max_side=1536,
                 scale_stability_rerank_pair_mode='manual',
                 scale_stability_rerank_query_mode='all_classes',
                 scale_stability_rerank_auto_min_pixels=256,
                 scale_stability_rerank_auto_exclude_bg=True,
                 scale_stability_rerank_bg_names='background,clutter,other',
                 scale_stability_rerank_max_risk_classes=12,
                 dump_expert_reliability_stats=False,
                 expert_reliability_stats_path=None,
                 expert_reliability_min_gate_pixels=1,
                 expert_reliability_local_kernel=7,
                 dump_evidence_bias_stats=False,
                 evidence_bias_stats_path=None,
                 evidence_bias_pairs=None,
                 evidence_bias_pair_mode='manual',
                 evidence_bias_probes='scale:0.5,blur:1.5,hflip,rot90,pad:128',
                 evidence_bias_heads='final,semantic,instance',
                 evidence_bias_support_thresholds='0.05,0.10,0.20',
                 evidence_bias_min_pair_pixels=256,
                 evidence_bias_max_gt_pairs=12,
                 evidence_bias_max_side=1536,
                 evidence_bias_empty_cache=True,
                 dump_topk_conflict_bias_stats=False,
                 topk_conflict_bias_stats_path=None,
                 topk_conflict_topk=3,
                 topk_conflict_probes='scale:0.5,scale:0.75,blur:1.5,sharpen,hflip,rot90,pad:128',
                 topk_conflict_heads='final,semantic,instance',
                 topk_conflict_support_thresholds='0.05,0.10,0.20',
                 topk_conflict_min_pair_pixels=256,
                 topk_conflict_max_pairs=20,
                 topk_conflict_max_side=1536,
                 topk_conflict_local_kernel=7,
                 topk_conflict_empty_cache=True,
                 dump_internal_selection_gap_stats=False,
                 internal_selection_gap_stats_path=None,
                 internal_selection_sources='final,semantic,instance,fusion_no_presence,semantic_presence,instance_presence,raw_object,raw_presence,encoder_all,vision,pe_all',
                 internal_selection_topk=3,
                 internal_selection_max_side=512,
                 internal_selection_raw_topk=10,
                 internal_selection_min_pair_pixels=16,
                 internal_selection_seed_rule='final_score_margin_sem_inst_final_agree_local_core',
                 internal_selection_conservative_min_consensus=2,
                 internal_selection_conservative_min_reliability=0.0,
                 dump_candidate_internal_verifier_stats=False,
                 candidate_internal_verifier_stats_path=None,
                 candidate_internal_topk=3,
                 candidate_internal_max_side=512,
                 candidate_internal_min_pair_pixels=16,
                 candidate_internal_delta_thresholds='-0.50,-0.25,0.00,0.10,0.25,0.50,1.00',
                 candidate_internal_vote_thresholds='1,2,3',
                 candidate_internal_null_trials=3,
                 candidate_internal_local_kernel=7,
                 candidate_internal_dump_matched_stats=False,
                 candidate_internal_matched_families='semantic,instance,no_presence,presence_readout,raw_mask,encoder,visual,support_topology',
                 candidate_internal_matched_ranks='2,3',
                 candidate_internal_matched_regimes='argmax_competition',
                 candidate_internal_matched_margin_bins='0.00,0.05,0.10,0.20,0.50,inf',
                 candidate_internal_matched_score_thresholds='-1.00,-0.50,-0.25,0.00,0.10,0.25,0.50,1.00,2.00',
                 candidate_internal_matched_null_trials=2,
                 candidate_internal_matched_auc_bins=64,
                 candidate_internal_matched_auc_min=-4.0,
                 candidate_internal_matched_auc_max=4.0,
                 dump_position_bias_stats=False,
                 position_bias_stats_path=None,
                 position_bias_layers='0,1,2',
                 position_bias_ranks='2,4,8',
                 position_bias_controls='permuted,random',
                 position_bias_random_trials=1,
                 position_bias_max_side=256,
                 position_bias_seed_rule='final_score_margin_sem_inst_final_agree_local_core',
                 dump_scene_common_bias_stats=False,
                 scene_common_bias_stats_path=None,
                 scene_common_layers='0,2',
                 scene_common_variants='global,robust,class_balanced,local,shared,controls',
                 scene_common_strengths='0.25,0.50,0.75,1.00',
                 scene_common_local_kernels='7,15',
                 scene_common_trim_quantile=0.10,
                 scene_common_random_trials=1,
                 scene_common_compute_spectrum=True,
                 scene_common_max_side=256,
                 scene_common_seed_rule='final_score_margin_sem_inst_final_agree_local_core',
                 dump_candidate_residual_trajectory_stats=False,
                 candidate_residual_trajectory_stats_path=None,
                 candidate_residual_layers='0,2',
                 candidate_residual_variants='global,class_balanced,controls',
                 candidate_residual_strengths='0.00,0.25,0.50,0.75,1.00',
                 candidate_residual_scores='margin_slope,endpoint_margin_gain,candidate_gain,competitor_suppression,independent_support,candidate_gain_fraction,crossing_score',
                 candidate_residual_ranks='2,3',
                 candidate_residual_regimes='argmax_competition',
                 candidate_residual_margin_bins='0.00,0.05,0.10,0.20,0.50,inf',
                 candidate_residual_min_pair_pixels=16,
                 candidate_residual_null_trials=2,
                 candidate_residual_auc_bins=64,
                 candidate_residual_auc_min=-4.0,
                 candidate_residual_auc_max=4.0,
                 candidate_residual_max_side=256,
                 candidate_residual_seed_rule='final_score_margin_sem_inst_final_agree_local_core',
                 dump_candidate_residual_miou_stats=False,
                 candidate_residual_miou_stats_path=None,
                 candidate_residual_miou_scores='margin_slope,endpoint_margin_gain',
                 candidate_residual_miou_ranks='2',
                 candidate_residual_miou_units='pixel,component,raw_mask',
                 candidate_residual_miou_absolute_thresholds='0.00,0.02,0.05,0.10,0.20',
                 candidate_residual_miou_coverages='0.001,0.0025,0.005,0.01,0.02,0.05',
                 candidate_residual_miou_region_reducer='mean,p25',
                 candidate_residual_miou_region_min_pixels=8,
                 candidate_residual_miou_raw_bin_thd=0.5,
                 candidate_residual_miou_max_side=256,
                 dump_ontology_readout_oracle_stats=False,
                 ontology_readout_oracle_stats_path=None,
                 ontology_readout_oracle_sources=(
                     'semantic,instance,raw_mask,presence_gated,pe_layer0'),
                 ontology_readout_oracle_max_side=256,
                 ontology_readout_oracle_topk=3,
                 ontology_readout_oracle_min_pair_pixels=16,
                 dump_state_action_atlas_stats=False,
                 state_action_atlas_stats_path=None,
                 state_action_atlas_actions='baseline,presence_sqrt,presence_bounded_sqrt,presence_guarded,semantic_presence,instance_presence,fusion_no_presence,semantic_no_presence,instance_no_presence,prompt_mean',
                 state_action_apply='baseline',
                 state_action_presence_gamma=0.50,
                 state_action_presence_blend=1.00,
                 state_action_presence_max_boost=0.20,
                 state_action_presence_floor=0.25,
                 state_action_support_threshold=0.10,
                 state_action_low_presence_threshold=0.25,
                 state_action_margin_threshold=0.05,
                 state_action_local_kernel=7,
                 state_action_feature_max_side=512,
                 use_region_contrastive_readout=False,
                 dump_region_contrastive_readout_stats=False,
                 region_contrastive_readout_stats_path=None,
                 region_readout_max_side=256,
                 region_readout_max_regions=192,
                 region_readout_masks_per_prompt=8,
                 region_readout_mask_threshold=0.50,
                 region_readout_min_pixels=16,
                 region_readout_dedup_iou=0.90,
                 region_readout_ring_kernel=7,
                 region_readout_top_fraction=0.25,
                 region_readout_presence_weight=0.10,
                 region_readout_topk=3,
                 region_readout_blends='0.25,0.50,1.00',
                 region_readout_mix_sources='true_zcontrast_top,true_zcontrast_top_presence,rect_zcontrast_top,shift_zcontrast_top,classperm_zcontrast_top,pixel_semantic',
                 region_readout_apply_source='true_zcontrast_top_presence',
                 region_readout_apply_blend=0.50,
                 region_readout_apply_topk=3,
                 region_readout_min_margin=0.10,
                 region_readout_min_pair_pixels=16,
                 region_readout_high_purity_threshold=0.80,
                 region_readout_projection='class_max',
                 region_readout_route_variant='legacy',
                 region_readout_foreground_blend=0.25,
                 region_readout_reject_blend=0.25,
                 region_readout_background_blend=0.25,
                 region_readout_formula_agreement=1,
                 region_readout_agreement_sources=(
                     'true_inside_mean,true_inside_top,'
                     'true_contrast_top,true_zcontrast_top'),
                 dump_region_hypothesis_v2_stats=False,
                 region_hypothesis_v2_stats_path=None,
                 region_hypothesis_v2_formulas=(
                     'inside_mean,inside_top,contrast_mean,contrast_top,'
                     'zcontrast_mean,zcontrast_top,origin_prompt,'
                     'contrast_presence'),
                 region_hypothesis_v2_formula_selectors=(
                     'max_margin,majority_margin,origin_guarded,'
                     'role_routed'),
                 region_hypothesis_v2_projections=(
                     'class_wise_max,region_winner,soft_mask_mixture,'
                     'baseline_compete,overlap_abstain'),
                 region_hypothesis_v2_selector_thresholds=(
                     '0.45,0.55,0.65,0.75'),
                 region_hypothesis_v2_quality_bins=(
                     '0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.01'),
                 region_hypothesis_v2_selector_projection=(
                     'overlap_abstain'),
                 region_hypothesis_v2_presence_weight=0.10,
                 region_hypothesis_v2_blend=0.25,
                 region_hypothesis_v2_min_margin=0.10,
                 region_hypothesis_v2_topk=3,
                 region_hypothesis_v2_overlap_max=1,
                 region_hypothesis_v2_baseline_advantage=0.0,
                 region_hypothesis_v2_core_kernel=5,
                 region_hypothesis_v2_high_purity=0.80,
                 dump_candidate_region_quality_stats=False,
                 candidate_region_quality_stats_path=None,
                 candidate_region_quality_sources=(
                     'raw_instance,grouped_instance,semantic_cc,hybrid'),
                 candidate_region_quality_max_side=256,
                 candidate_region_quality_max_regions=384,
                 candidate_region_quality_min_pixels=16,
                 candidate_region_quality_gt_min_pixels=16,
                 candidate_region_quality_high_purity=0.80,
                 candidate_region_quality_group_iou=0.35,
                 candidate_region_quality_group_containment=0.75,
                 candidate_region_quality_semantic_threshold=0.35,
                 candidate_region_quality_semantic_max_per_class=32,
                 candidate_region_quality_hybrid_dedup_iou=0.90,
                 candidate_region_quality_fragment_coverage=0.10,
                 dump_query_topology_stats=False,
                 query_topology_stats_path=None,
                 query_topology_max_side=256,
                 query_topology_max_regions=256,
                 query_topology_min_pixels=16,
                 query_topology_support_threshold=0.05,
                 query_topology_mask_threshold=0.50,
                 query_topology_semantic_threshold=0.35,
                 query_topology_action_min_overlap=0.05,
                 query_topology_action_padding=4,
                 query_topology_include_background=True,
                 query_topology_save_npz=False,
                 query_topology_artifact_dir=(
                     'logs/query_topology/artifacts'),
                 query_topology_max_saved_images=16,
                 dump_reviewer_cache=False,
                 reviewer_cache_dir=None,
                 reviewer_dataset_name=None,
                 reviewer_max_side=256,
                 reviewer_cache_samples_per_image=4096,
                 reviewer_cache_hard_keep_margin=0.15,
                 reviewer_topk=3,
                 reviewer_local_kernel=7,
                 use_learned_reviewer=False,
                 reviewer_checkpoint=None,
                 reviewer_variant='internal_trajectory',
                 reviewer_inference_batch_size=32768,
                 reviewer_gate_threshold=None,
                 reviewer_action_margin=None,
                 reviewer_min_boost=1e-4,
                 dump_learned_reviewer_stats=False,
                 learned_reviewer_stats_path=None,
                 use_geoer_router=False,
                 dump_geoer_router_stats=False,
                 geoer_router_stats_path=None,
                 geoer_use_scale_stability=True,
                 geoer_use_reject_recovery=False,
                 geoer_reject_recovery_datasets='vaihingen,potsdam',
                 geoer_reject_recovery_bg_names='clutter',
                 geoer_reject_recovery_score_thd=0.05,
                 geoer_reject_recovery_semantic_thd=0.03,
                 geoer_reject_recovery_instance_thd=0.01,
                 geoer_reject_recovery_margin_thd=0.0,
                 geoer_reject_recovery_local_thd=0.70,
                 geoer_reject_recovery_require_bg_margin=True,
                 use_coco_sec_fusion=False,
                 dump_coco_sec_stats=False,
                 coco_sec_stats_path=None,
                 coco_sec_lambda=0.7,
                 coco_sec_temperature=1.0,
                 coco_sec_eps=1e-4,
                 coco_sec_center_prior=True,
                 coco_sec_synonym_reduce='logsumexp',
                 dump_active_concept_stats=False,
                 active_concept_stats_path=None,
                 use_active_concept_pruning=False,
                 active_concept_apply_variant='conflict_preserving',
                 active_concept_variants='gt_oracle,presence_only,multi_evidence,conflict_preserving',
                 active_concept_presence_thd=0.20,
                 active_concept_score_thd=None,
                 active_concept_area_thd=0.001,
                 active_concept_pred_area_thd=0.0005,
                 active_concept_topk=3,
                 active_concept_topk_area_thd=0.0005,
                 active_concept_conflict_mode='role_topk',
                 active_concept_v2_context_area_thd=0.0005,
                 active_concept_v2_context_topk_area_thd=0.0002,
                 active_concept_v2_context_presence_thd=0.10,
                 active_concept_v2_min_strong_signals=2,
                 active_concept_keep_bg=True,
                 active_concept_suppress_value=-10000.0,
                 dump_local_active_set_stats=False,
                 local_active_set_stats_path=None,
                 local_active_max_side=256,
                 local_active_windows='17,33,65',
                 local_active_topk=3,
                 local_active_score_thd=None,
                 local_active_area_thresholds='0.001,0.005,0.01,0.02,0.05',
                 local_active_sources='topk,final,semantic,instance,agreement',
                 local_active_min_pair_pixels=32,
                 dump_prompt_winner_attribution_stats=False,
                 prompt_winner_attribution_stats_path=None,
                 prompt_winner_max_side=256,
                 prompt_winner_sources='final,semantic,instance',
                 prompt_winner_min_pair_pixels=32,
                 dump_region_prompt_identity_stats=False,
                 region_prompt_identity_stats_path=None,
                 region_prompt_identity_max_side=256,
                 region_prompt_identity_bin_thd=0.5,
                 region_prompt_identity_min_pixels=16,
                 region_prompt_identity_group_iou=0.5,
                 region_prompt_identity_group_containment=0.75,
                 region_prompt_identity_max_regions=256,
                 region_prompt_identity_ring_kernel=15,
                 region_prompt_identity_min_pair_pixels=16,
                 use_ontology_self_verification=False,
                 dump_ontology_self_verification_stats=False,
                 ontology_self_verification_stats_path=None,
                 ontology_self_verification_spaces='evidence,vision',
                 ontology_self_verification_seed_rule='final_score_margin_any_head_agree_local_core',
                 ontology_self_verification_topk=3,
                 ontology_self_verification_min_similarity_margin=0.35,
                 ontology_self_verification_max_base_gap=0.35,
                 ontology_self_verification_min_candidate_score=-1.0,
                 ontology_self_verification_logit_boost=1e-4,
                 ontology_self_verification_apply_scope='non_bg_to_non_bg',
                 ontology_self_verification_require_top1_seed=True,
                 ontology_self_verification_include_roles='',
                 ontology_self_verification_exclude_roles='catch_all',
                 ontology_self_verification_seed_include_roles='',
                 ontology_self_verification_seed_exclude_roles='catch_all',
                 dump_sam3_geometry_requery_stats=False,
                 sam3_geometry_requery_stats_path=None,
                 sam3_geometry_requery_topk=3,
                 sam3_geometry_requery_max_pairs=4,
                 sam3_geometry_requery_min_pair_pixels=64,
                 sam3_geometry_requery_min_region_pixels=32,
                 sam3_geometry_requery_score_thd=None,
                 sam3_geometry_requery_box_modes='target_pos,pred_neg,target_pos_pred_neg',
                 sam3_geometry_requery_prompt_mode='best_query',
                 sam3_geometry_requery_max_side=1024,
                 sam3_geometry_requery_min_box_size=4,
                 sam3_geometry_requery_context_scale=1.0,
                 sam3_geometry_requery_include_bg=True,
                 sam3_geometry_requery_empty_cache=True,
                 dump_self_prompted_concept_verification_stats=False,
                 self_prompted_concept_verification_stats_path=None,
                 self_prompted_concept_verification_topk=3,
                 self_prompted_concept_verification_max_pairs=3,
                 self_prompted_concept_verification_min_pair_pixels=96,
                 self_prompted_concept_verification_min_region_pixels=32,
                 self_prompted_concept_verification_score_thd=None,
                 self_prompted_concept_verification_region_fraction=0.35,
                 self_prompted_concept_verification_max_base_gap=1.0,
                 self_prompted_concept_verification_min_candidate_score=-1.0,
                 self_prompted_concept_verification_prompt_mode='best_query',
                 self_prompted_concept_verification_max_side=1024,
                 self_prompted_concept_verification_min_box_size=4,
                 self_prompted_concept_verification_context_scale=1.05,
                 self_prompted_concept_verification_include_bg=True,
                 self_prompted_concept_verification_empty_cache=True,
                 use_evidence_enhancement=False,
                 dump_evidence_enhancement_stats=False,
                 evidence_enhancement_stats_path=None,
                 evidence_enhancement_classes='all',
                 evidence_enhancement_source='semantic',
                 evidence_enhancement_presence_mode='prompt',
                 evidence_enhancement_reduce='max',
                 evidence_enhancement_combine='max',
                 evidence_enhancement_alpha=1.0,
                 evidence_enhancement_max_prompts=4,
                 evidence_enhancement_use_ontology_prompts=True,
                 evidence_enhancement_templates='{class}|remote sensing {class}|aerial image {class}|satellite image {class}|{class} land cover|{class} region',
                 use_structure_aware_recalibration=False,
                 dump_structure_aware_recalibration_stats=False,
                 structure_aware_recalibration_stats_path=None,
                 structure_recalibration_variant='final_context',
                 structure_recalibration_kernel=9,
                 structure_recalibration_alpha=0.25,
                 structure_recalibration_min_context_gain=0.01,
                 structure_recalibration_min_context_score=0.0,
                 structure_recalibration_max_base_margin=1.0,
                 structure_recalibration_protect_bg=True,
                 structure_recalibration_include_roles='all',
                 structure_recalibration_exclude_roles='catch_all',
                 structure_validity_topk=3,
                 structure_validity_high_conf_margin=0.20,
                 structure_validity_min_pair_pixels=32,
                 dump_concept_specificity_stats=False,
                 concept_specificity_stats_path=None,
                 use_concept_specificity_pruning=False,
                 concept_specificity_apply_ranker='specificity_combo',
                 concept_specificity_max_side=512,
                 concept_specificity_score_thd=None,
                 concept_specificity_prune_active_count=None,
                 concept_specificity_keep_bg=True,
                 concept_specificity_suppress_value=-10000.0,
                 dump_cross_image_bank_stats=False,
                 cross_image_bank_stats_path=None,
                 cross_image_bank_spaces='vision,pe_layer_0,pe_layer_1,pe_layer_2',
                 cross_image_bank_seed_rule='final_score_margin_sem_inst_final_agree_local_core',
                 cross_image_bank_min_seed_pixels=4,
                 cross_image_bank_min_pair_pixels=4,
                 cross_image_bank_vector_decimals=6,
                 cross_image_bank_feature_max_side=1024,
                 cross_image_bank_reencode_missing_features=True,
                 use_cross_image_bank_recomposition=False,
                 dump_cross_image_bank_recomposition_stats=False,
                 cross_image_bank_recomposition_stats_path=None,
                 cross_image_bank_file=None,
                 cross_image_bank_dataset_name=None,
                 cross_image_bank_apply_space='pe_layer_0',
                 cross_image_bank_apply_variant='image_centered',
                 cross_image_bank_apply_seed_rule='final_score_margin_sem_inst_final_agree_local_core',
                 cross_image_bank_apply_topk=3,
                 cross_image_bank_apply_min_bank_margin=0.10,
                 cross_image_bank_apply_max_base_gap=0.60,
                 cross_image_bank_apply_min_candidate_score=-1.0,
                 cross_image_bank_apply_min_candidate_affinity=-1.0,
                 cross_image_bank_apply_logit_boost=1e-4,
                 cross_image_bank_apply_min_class_weight=32,
                 cross_image_bank_apply_min_seed_pixels=4,
                 cross_image_bank_apply_exclude_current_image=True,
                 cross_image_bank_apply_protect_bg=True,
                 cross_image_bank_apply_exclude_bg_candidate=True,
                 cross_image_bank_apply_require_base_non_bg=True,
                 **kwargs):
        super().__init__()

        self.device = _resolve_inference_device(device)
        # Initialize SAM3 model
        # Keep SAM3 out of nn.Module registration. Some SAM3 buffers are complex
        # tensors, and DDP/NCCL cannot broadcast ComplexFloat during pure eval.
        # A tiny float parameter below lets MMEngine wrap this segmentor for
        # distributed test while each rank owns its local SAM3 processor.
        self._ddp_dummy_param = nn.Parameter(
            torch.zeros(1, device=self.device),
            requires_grad=True,
        )
        sam3_model = build_sam3_image_model(
            bpe_path="./sam3/assets/bpe_simple_vocab_16e6.txt.gz", 
            checkpoint_path='weights/sam3/sam3.pt', 
            device='cuda' if self.device.type == 'cuda' else str(self.device)
        )
        sam3_model = sam3_model.to(self.device).eval()
        self.processor = Sam3Processor(sam3_model, confidence_threshold=confidence_threshold, device=self.device)
        self.query_words, self.query_idx = get_cls_idx(classname_path)
        self.num_cls = max(self.query_idx) + 1
        self.num_queries = len(self.query_idx)
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(self.device)

        self.prob_thd = prob_thd
        self.bg_idx = bg_idx
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.confidence_threshold = confidence_threshold
        self.use_sem_seg = use_sem_seg
        self.use_presence_score = use_presence_score
        self.use_transformer_decoder = use_transformer_decoder
        self.instance_score_type = instance_score_type
        self.dump_active_concept_stats = dump_active_concept_stats
        self.active_concept_stats_path = active_concept_stats_path
        self.use_active_concept_pruning = use_active_concept_pruning
        self.active_concept_apply_variant = active_concept_apply_variant
        self.active_concept_variants = active_concept_variants
        self.active_concept_presence_thd = active_concept_presence_thd
        self.active_concept_score_thd = active_concept_score_thd
        self.active_concept_area_thd = active_concept_area_thd
        self.active_concept_pred_area_thd = active_concept_pred_area_thd
        self.active_concept_topk = active_concept_topk
        self.active_concept_topk_area_thd = active_concept_topk_area_thd
        self.active_concept_conflict_mode = active_concept_conflict_mode
        self.active_concept_v2_context_area_thd = (
            active_concept_v2_context_area_thd)
        self.active_concept_v2_context_topk_area_thd = (
            active_concept_v2_context_topk_area_thd)
        self.active_concept_v2_context_presence_thd = (
            active_concept_v2_context_presence_thd)
        self.active_concept_v2_min_strong_signals = (
            active_concept_v2_min_strong_signals)
        self.active_concept_keep_bg = active_concept_keep_bg
        self.active_concept_suppress_value = active_concept_suppress_value
        self._active_concept_stats_file = None
        self.dump_local_active_set_stats = dump_local_active_set_stats
        self.local_active_set_stats_path = local_active_set_stats_path
        self.local_active_max_side = local_active_max_side
        self.local_active_windows = local_active_windows
        self.local_active_topk = local_active_topk
        self.local_active_score_thd = local_active_score_thd
        self.local_active_area_thresholds = local_active_area_thresholds
        self.local_active_sources = local_active_sources
        self.local_active_min_pair_pixels = local_active_min_pair_pixels
        self._local_active_set_stats_file = None
        self.dump_prompt_winner_attribution_stats = (
            dump_prompt_winner_attribution_stats)
        self.prompt_winner_attribution_stats_path = (
            prompt_winner_attribution_stats_path)
        self.prompt_winner_max_side = prompt_winner_max_side
        self.prompt_winner_sources = prompt_winner_sources
        self.prompt_winner_min_pair_pixels = prompt_winner_min_pair_pixels
        self._prompt_winner_attribution_stats_file = None
        self.dump_region_prompt_identity_stats = (
            dump_region_prompt_identity_stats)
        self.region_prompt_identity_stats_path = (
            region_prompt_identity_stats_path)
        self.region_prompt_identity_max_side = region_prompt_identity_max_side
        self.region_prompt_identity_bin_thd = region_prompt_identity_bin_thd
        self.region_prompt_identity_min_pixels = (
            region_prompt_identity_min_pixels)
        self.region_prompt_identity_group_iou = (
            region_prompt_identity_group_iou)
        self.region_prompt_identity_group_containment = (
            region_prompt_identity_group_containment)
        self.region_prompt_identity_max_regions = (
            region_prompt_identity_max_regions)
        self.region_prompt_identity_ring_kernel = (
            region_prompt_identity_ring_kernel)
        self.region_prompt_identity_min_pair_pixels = (
            region_prompt_identity_min_pair_pixels)
        self._region_prompt_identity_stats_file = None
        self.use_ontology_self_verification = bool(
            use_ontology_self_verification)
        self.dump_ontology_self_verification_stats = bool(
            dump_ontology_self_verification_stats)
        self.ontology_self_verification_stats_path = (
            ontology_self_verification_stats_path)
        self.ontology_self_verification_spaces = (
            ontology_self_verification_spaces)
        self.ontology_self_verification_seed_rule = (
            ontology_self_verification_seed_rule)
        self.ontology_self_verification_topk = (
            ontology_self_verification_topk)
        self.ontology_self_verification_min_similarity_margin = float(
            ontology_self_verification_min_similarity_margin)
        self.ontology_self_verification_max_base_gap = float(
            ontology_self_verification_max_base_gap)
        self.ontology_self_verification_min_candidate_score = float(
            ontology_self_verification_min_candidate_score)
        self.ontology_self_verification_logit_boost = float(
            ontology_self_verification_logit_boost)
        self.ontology_self_verification_apply_scope = (
            ontology_self_verification_apply_scope)
        self.ontology_self_verification_require_top1_seed = bool(
            ontology_self_verification_require_top1_seed)
        self.ontology_self_verification_include_roles = (
            ontology_self_verification_include_roles)
        self.ontology_self_verification_exclude_roles = (
            ontology_self_verification_exclude_roles)
        self.ontology_self_verification_seed_include_roles = (
            ontology_self_verification_seed_include_roles)
        self.ontology_self_verification_seed_exclude_roles = (
            ontology_self_verification_seed_exclude_roles)
        self._ontology_self_verification_stats_file = None
        self.dump_sam3_geometry_requery_stats = bool(
            dump_sam3_geometry_requery_stats)
        self.sam3_geometry_requery_stats_path = (
            sam3_geometry_requery_stats_path)
        self.sam3_geometry_requery_topk = int(sam3_geometry_requery_topk)
        self.sam3_geometry_requery_max_pairs = int(
            sam3_geometry_requery_max_pairs)
        self.sam3_geometry_requery_min_pair_pixels = int(
            sam3_geometry_requery_min_pair_pixels)
        self.sam3_geometry_requery_min_region_pixels = int(
            sam3_geometry_requery_min_region_pixels)
        self.sam3_geometry_requery_score_thd = (
            sam3_geometry_requery_score_thd)
        self.sam3_geometry_requery_box_modes = (
            sam3_geometry_requery_box_modes)
        self.sam3_geometry_requery_prompt_mode = (
            sam3_geometry_requery_prompt_mode)
        self.sam3_geometry_requery_max_side = int(
            sam3_geometry_requery_max_side)
        self.sam3_geometry_requery_min_box_size = int(
            sam3_geometry_requery_min_box_size)
        self.sam3_geometry_requery_context_scale = float(
            sam3_geometry_requery_context_scale)
        self.sam3_geometry_requery_include_bg = (
            sam3_geometry_requery_include_bg)
        self.sam3_geometry_requery_empty_cache = (
            sam3_geometry_requery_empty_cache)
        self._sam3_geometry_requery_stats_file = None
        self.dump_self_prompted_concept_verification_stats = bool(
            dump_self_prompted_concept_verification_stats)
        self.self_prompted_concept_verification_stats_path = (
            self_prompted_concept_verification_stats_path)
        self.self_prompted_concept_verification_topk = int(
            self_prompted_concept_verification_topk)
        self.self_prompted_concept_verification_max_pairs = int(
            self_prompted_concept_verification_max_pairs)
        self.self_prompted_concept_verification_min_pair_pixels = int(
            self_prompted_concept_verification_min_pair_pixels)
        self.self_prompted_concept_verification_min_region_pixels = int(
            self_prompted_concept_verification_min_region_pixels)
        self.self_prompted_concept_verification_score_thd = (
            self_prompted_concept_verification_score_thd)
        self.self_prompted_concept_verification_region_fraction = float(
            self_prompted_concept_verification_region_fraction)
        self.self_prompted_concept_verification_max_base_gap = float(
            self_prompted_concept_verification_max_base_gap)
        self.self_prompted_concept_verification_min_candidate_score = float(
            self_prompted_concept_verification_min_candidate_score)
        self.self_prompted_concept_verification_prompt_mode = (
            self_prompted_concept_verification_prompt_mode)
        self.self_prompted_concept_verification_max_side = int(
            self_prompted_concept_verification_max_side)
        self.self_prompted_concept_verification_min_box_size = int(
            self_prompted_concept_verification_min_box_size)
        self.self_prompted_concept_verification_context_scale = float(
            self_prompted_concept_verification_context_scale)
        self.self_prompted_concept_verification_include_bg = (
            self_prompted_concept_verification_include_bg)
        self.self_prompted_concept_verification_empty_cache = (
            self_prompted_concept_verification_empty_cache)
        self._self_prompted_concept_verification_stats_file = None
        self.use_evidence_enhancement = bool(use_evidence_enhancement)
        self.dump_evidence_enhancement_stats = bool(
            dump_evidence_enhancement_stats)
        self.evidence_enhancement_stats_path = (
            evidence_enhancement_stats_path)
        self.evidence_enhancement_classes = evidence_enhancement_classes
        self.evidence_enhancement_source = evidence_enhancement_source
        self.evidence_enhancement_presence_mode = (
            evidence_enhancement_presence_mode)
        self.evidence_enhancement_reduce = evidence_enhancement_reduce
        self.evidence_enhancement_combine = evidence_enhancement_combine
        self.evidence_enhancement_alpha = float(evidence_enhancement_alpha)
        self.evidence_enhancement_max_prompts = int(
            evidence_enhancement_max_prompts)
        self.evidence_enhancement_use_ontology_prompts = (
            evidence_enhancement_use_ontology_prompts)
        self.evidence_enhancement_templates = evidence_enhancement_templates
        self._evidence_enhancement_stats_file = None
        self.use_structure_aware_recalibration = bool(
            use_structure_aware_recalibration)
        self.dump_structure_aware_recalibration_stats = bool(
            dump_structure_aware_recalibration_stats)
        self.structure_aware_recalibration_stats_path = (
            structure_aware_recalibration_stats_path)
        self.structure_recalibration_variant = (
            structure_recalibration_variant)
        self.structure_recalibration_kernel = int(
            structure_recalibration_kernel)
        self.structure_recalibration_alpha = float(
            structure_recalibration_alpha)
        self.structure_recalibration_min_context_gain = float(
            structure_recalibration_min_context_gain)
        self.structure_recalibration_min_context_score = float(
            structure_recalibration_min_context_score)
        self.structure_recalibration_max_base_margin = float(
            structure_recalibration_max_base_margin)
        self.structure_recalibration_protect_bg = bool(
            structure_recalibration_protect_bg)
        self.structure_recalibration_include_roles = (
            structure_recalibration_include_roles)
        self.structure_recalibration_exclude_roles = (
            structure_recalibration_exclude_roles)
        self.structure_validity_topk = int(structure_validity_topk)
        self.structure_validity_high_conf_margin = float(
            structure_validity_high_conf_margin)
        self.structure_validity_min_pair_pixels = int(
            structure_validity_min_pair_pixels)
        self._structure_aware_recalibration_stats_file = None
        self.dump_concept_specificity_stats = dump_concept_specificity_stats
        self.concept_specificity_stats_path = concept_specificity_stats_path
        self.use_concept_specificity_pruning = use_concept_specificity_pruning
        self.concept_specificity_apply_ranker = concept_specificity_apply_ranker
        self.concept_specificity_max_side = concept_specificity_max_side
        self.concept_specificity_score_thd = concept_specificity_score_thd
        self.concept_specificity_prune_active_count = (
            concept_specificity_prune_active_count)
        self.concept_specificity_keep_bg = concept_specificity_keep_bg
        self.concept_specificity_suppress_value = (
            concept_specificity_suppress_value)
        self._concept_specificity_stats_file = None
        self.dump_cross_image_bank_stats = dump_cross_image_bank_stats
        self.cross_image_bank_stats_path = cross_image_bank_stats_path
        self.cross_image_bank_spaces = cross_image_bank_spaces
        self.cross_image_bank_seed_rule = cross_image_bank_seed_rule
        self.cross_image_bank_min_seed_pixels = cross_image_bank_min_seed_pixels
        self.cross_image_bank_min_pair_pixels = cross_image_bank_min_pair_pixels
        self.cross_image_bank_vector_decimals = cross_image_bank_vector_decimals
        self.cross_image_bank_feature_max_side = cross_image_bank_feature_max_side
        self.cross_image_bank_reencode_missing_features = (
            cross_image_bank_reencode_missing_features)
        self._cross_image_bank_stats_file = None
        self.use_cross_image_bank_recomposition = (
            use_cross_image_bank_recomposition)
        self.dump_cross_image_bank_recomposition_stats = (
            dump_cross_image_bank_recomposition_stats)
        self.cross_image_bank_recomposition_stats_path = (
            cross_image_bank_recomposition_stats_path)
        self.cross_image_bank_file = cross_image_bank_file
        self.cross_image_bank_dataset_name = cross_image_bank_dataset_name
        self.cross_image_bank_apply_space = cross_image_bank_apply_space
        self.cross_image_bank_apply_variant = cross_image_bank_apply_variant
        self.cross_image_bank_apply_seed_rule = (
            cross_image_bank_apply_seed_rule)
        self.cross_image_bank_apply_topk = cross_image_bank_apply_topk
        self.cross_image_bank_apply_min_bank_margin = (
            cross_image_bank_apply_min_bank_margin)
        self.cross_image_bank_apply_max_base_gap = (
            cross_image_bank_apply_max_base_gap)
        self.cross_image_bank_apply_min_candidate_score = (
            cross_image_bank_apply_min_candidate_score)
        self.cross_image_bank_apply_min_candidate_affinity = (
            cross_image_bank_apply_min_candidate_affinity)
        self.cross_image_bank_apply_logit_boost = (
            cross_image_bank_apply_logit_boost)
        self.cross_image_bank_apply_min_class_weight = (
            cross_image_bank_apply_min_class_weight)
        self.cross_image_bank_apply_min_seed_pixels = (
            cross_image_bank_apply_min_seed_pixels)
        self.cross_image_bank_apply_exclude_current_image = (
            cross_image_bank_apply_exclude_current_image)
        self.cross_image_bank_apply_protect_bg = (
            cross_image_bank_apply_protect_bg)
        self.cross_image_bank_apply_exclude_bg_candidate = (
            cross_image_bank_apply_exclude_bg_candidate)
        self.cross_image_bank_apply_require_base_non_bg = (
            cross_image_bank_apply_require_base_non_bg)
        self._cross_image_bank_recomposition_stats_file = None
        self.dump_evidence_stats = dump_evidence_stats
        self.evidence_stats_path = evidence_stats_path
        self._evidence_stats_file = None
        self.dump_competition_stats = dump_competition_stats
        self.competition_stats_path = competition_stats_path
        self._competition_stats_file = None
        self.use_reject_aware_calibration = use_reject_aware_calibration
        self.use_reject_recovery = use_reject_recovery
        self.use_competition_suppression = use_competition_suppression
        self.reject_recovery_bg_role = reject_recovery_bg_role
        self.reject_recovery_factor = reject_recovery_factor
        self.reject_recovery_semantic_thd = reject_recovery_semantic_thd
        self.reject_recovery_instance_thd = reject_recovery_instance_thd
        self.reject_recovery_margin_thd = reject_recovery_margin_thd
        self.use_residual_background_modeling = bool(
            use_residual_background_modeling)
        self.dump_residual_background_stats = bool(
            dump_residual_background_stats)
        self.residual_background_stats_path = (
            residual_background_stats_path)
        self.residual_background_route = str(residual_background_route)
        self.residual_background_score_factor = float(
            residual_background_score_factor)
        self.residual_background_min_score = float(
            residual_background_min_score)
        self.residual_background_min_semantic = float(
            residual_background_min_semantic)
        self.residual_background_min_instance = float(
            residual_background_min_instance)
        self.residual_background_min_presence = float(
            residual_background_min_presence)
        self.residual_background_min_local = float(
            residual_background_min_local)
        self.residual_background_min_fg_bg_margin = float(
            residual_background_min_fg_bg_margin)
        self.residual_background_min_reliability_margin = float(
            residual_background_min_reliability_margin)
        self.residual_background_semantic_weight = float(
            residual_background_semantic_weight)
        self.residual_background_instance_weight = float(
            residual_background_instance_weight)
        self.residual_background_presence_weight = float(
            residual_background_presence_weight)
        self.residual_background_local_weight = float(
            residual_background_local_weight)
        self.residual_background_bg_weight = float(
            residual_background_bg_weight)
        self.residual_background_local_kernel = int(
            residual_background_local_kernel)
        self.residual_background_require_head_support = bool(
            residual_background_require_head_support)
        self._residual_background_stats_file = None
        self.dump_background_separability_stats = bool(
            dump_background_separability_stats)
        self.background_separability_stats_path = (
            background_separability_stats_path)
        self.background_separability_score_thresholds = (
            background_separability_score_thresholds)
        self.background_separability_margin_thresholds = (
            background_separability_margin_thresholds)
        self.background_separability_reliability_thresholds = (
            background_separability_reliability_thresholds)
        self.background_separability_local_thresholds = (
            background_separability_local_thresholds)
        self._background_separability_stats_file = None
        self.competition_top2_ratio = competition_top2_ratio
        self.competition_sem_inst_gap = competition_sem_inst_gap
        self.competition_penalty = competition_penalty
        self.use_evidence_competition_graph = use_evidence_competition_graph
        self.ecg_top2_ratio = ecg_top2_ratio
        self.ecg_margin_thd = ecg_margin_thd
        self.ecg_semantic_thd = ecg_semantic_thd
        self.ecg_instance_thd = ecg_instance_thd
        self.ecg_agreement_gap = ecg_agreement_gap
        self.ecg_sem_over_inst_gap = ecg_sem_over_inst_gap
        self.ecg_sem_only_transfer = ecg_sem_only_transfer
        self.ecg_boost_scale = ecg_boost_scale
        self.ecg_suppress_scale = ecg_suppress_scale
        self.dump_oracle_stats = dump_oracle_stats
        self.oracle_stats_path = oracle_stats_path
        self.oracle_topk = oracle_topk
        self._oracle_stats_file = None
        self.use_topk_candidate_verifier = use_topk_candidate_verifier
        self.dump_topk_verifier_stats = dump_topk_verifier_stats
        self.topk_verifier_stats_path = topk_verifier_stats_path
        self.topk_verifier_k = topk_verifier_k
        self.topk_verifier_apply_mode = topk_verifier_apply_mode
        self.topk_verifier_require_final_candidate = topk_verifier_require_final_candidate
        self.topk_verifier_min_aux_rank = topk_verifier_min_aux_rank
        self.topk_verifier_low_margin = topk_verifier_low_margin
        self.topk_verifier_confident_margin = topk_verifier_confident_margin
        self.topk_verifier_aux_rank_weight = topk_verifier_aux_rank_weight
        self.topk_verifier_vote_weight = topk_verifier_vote_weight
        self.topk_verifier_local_weight = topk_verifier_local_weight
        self.topk_verifier_final_weight = topk_verifier_final_weight
        self.topk_verifier_local_kernel = topk_verifier_local_kernel
        self._topk_verifier_stats_file = None
        self.dump_error_rank_stats = dump_error_rank_stats
        self.error_rank_stats_path = error_rank_stats_path
        self.error_rank_topk = error_rank_topk
        self._error_rank_stats_file = None
        self.use_top2_risk_arbitration = use_top2_risk_arbitration
        self.dump_top2_risk_stats = dump_top2_risk_stats
        self.top2_risk_stats_path = top2_risk_stats_path
        self.top2_risk_topk = top2_risk_topk
        self.top2_risk_min_margin = top2_risk_min_margin
        self.top2_risk_max_margin = top2_risk_max_margin
        self.top2_risk_min_sem_adv = top2_risk_min_sem_adv
        self.top2_risk_min_inst_adv = top2_risk_min_inst_adv
        self.top2_risk_min_agreement_adv = top2_risk_min_agreement_adv
        self.top2_risk_min_top1_sem_only = top2_risk_min_top1_sem_only
        self.top2_risk_require_head_disagree = top2_risk_require_head_disagree
        self.top2_risk_bg_policy = top2_risk_bg_policy
        self.top2_risk_mode = top2_risk_mode
        self._top2_risk_stats_file = None
        self.dump_multiview_oracle_stats = dump_multiview_oracle_stats
        self.multiview_oracle_stats_path = multiview_oracle_stats_path
        self.multiview_oracle_topk = multiview_oracle_topk
        self.multiview_oracle_views = multiview_oracle_views
        self.multiview_oracle_slide_views = multiview_oracle_slide_views
        self.multiview_oracle_include_base = multiview_oracle_include_base
        self.multiview_oracle_empty_cache = multiview_oracle_empty_cache
        self._multiview_oracle_stats_file = None
        self.dump_seed_separability_stats = dump_seed_separability_stats
        self.seed_separability_stats_path = seed_separability_stats_path
        self.seed_dataset_name = seed_dataset_name or _infer_dataset_name(classname_path)
        self.seed_final_score_thd = seed_final_score_thd
        self.seed_margin_thd = seed_margin_thd
        self.seed_local_kernel = seed_local_kernel
        self.seed_local_consistency_thd = seed_local_consistency_thd
        self.seed_core_kernel = seed_core_kernel
        self.seed_core_consistency_thd = seed_core_consistency_thd
        self.seed_region_min_pixels = seed_region_min_pixels
        self.seed_region_purity_thd = seed_region_purity_thd
        self.seed_similarity_eps = seed_similarity_eps
        self.seed_similarity_spaces = seed_similarity_spaces
        self._seed_separability_stats_file = None
        self.dump_raw_mask_oracle_stats = dump_raw_mask_oracle_stats
        self.raw_mask_oracle_stats_path = raw_mask_oracle_stats_path
        self.raw_mask_oracle_topk = raw_mask_oracle_topk
        self.raw_mask_oracle_bin_thd = raw_mask_oracle_bin_thd
        self.raw_mask_oracle_min_pixels = raw_mask_oracle_min_pixels
        self._raw_mask_oracle_stats_file = None
        self.dump_candidate_quality_stats = dump_candidate_quality_stats
        self.candidate_quality_stats_path = candidate_quality_stats_path
        self.candidate_quality_clean_purity_thd = candidate_quality_clean_purity_thd
        self._candidate_quality_stats_file = None
        self.dump_prompt_competition_stats = dump_prompt_competition_stats
        self.prompt_competition_stats_path = prompt_competition_stats_path
        self.prompt_competition_class_names = prompt_competition_class_names
        self.prompt_competition_templates = prompt_competition_templates
        self.prompt_competition_max_variants = prompt_competition_max_variants
        self.prompt_competition_use_builtin_variants = prompt_competition_use_builtin_variants
        self._prompt_competition_stats_file = None
        self.dump_pair_prompt_competition_stats = dump_pair_prompt_competition_stats
        self.pair_prompt_competition_stats_path = pair_prompt_competition_stats_path
        self.pair_prompt_competition_pairs = pair_prompt_competition_pairs
        self.pair_prompt_competition_templates = pair_prompt_competition_templates
        self.pair_prompt_competition_max_variants = pair_prompt_competition_max_variants
        self.pair_prompt_competition_use_builtin_variants = pair_prompt_competition_use_builtin_variants
        self._pair_prompt_competition_stats_file = None
        self.dump_geometry_context_stats = dump_geometry_context_stats
        self.geometry_context_stats_path = geometry_context_stats_path
        self.geometry_context_score_thd = geometry_context_score_thd
        self.geometry_context_min_pixels = geometry_context_min_pixels
        self.geometry_context_core_kernel = geometry_context_core_kernel
        self._geometry_context_stats_file = None
        self.dump_weak_support_stats = dump_weak_support_stats
        self.weak_support_stats_path = weak_support_stats_path
        self.weak_support_thresholds = weak_support_thresholds
        self.weak_support_sources = weak_support_sources
        self.weak_support_min_pair_pixels = weak_support_min_pair_pixels
        self._weak_support_stats_file = None
        self.dump_reject_recovery_precision_stats = dump_reject_recovery_precision_stats
        self.reject_recovery_precision_stats_path = reject_recovery_precision_stats_path
        self.reject_recovery_precision_score_bins = reject_recovery_precision_score_bins
        self.reject_recovery_precision_semantic_thds = reject_recovery_precision_semantic_thds
        self.reject_recovery_precision_instance_thds = reject_recovery_precision_instance_thds
        self.reject_recovery_precision_margin_thds = reject_recovery_precision_margin_thds
        self.reject_recovery_precision_local_thds = reject_recovery_precision_local_thds
        self.reject_recovery_precision_use_pe = reject_recovery_precision_use_pe
        self.reject_recovery_precision_pe_spaces = reject_recovery_precision_pe_spaces
        self._reject_recovery_precision_stats_file = None
        self.dump_context_requery_stats = dump_context_requery_stats
        self.context_requery_stats_path = context_requery_stats_path
        self.context_requery_pairs = context_requery_pairs
        self.context_requery_modes = context_requery_modes
        self.context_requery_thresholds = context_requery_thresholds
        self.context_requery_min_pair_pixels = context_requery_min_pair_pixels
        self.context_requery_min_crop_size = context_requery_min_crop_size
        self.context_requery_max_crop_size = context_requery_max_crop_size
        self._context_requery_stats_file = None
        self.dump_scale_requery_stats = dump_scale_requery_stats
        self.scale_requery_stats_path = scale_requery_stats_path
        self.scale_requery_pairs = scale_requery_pairs
        self.scale_requery_scales = scale_requery_scales
        self.scale_requery_thresholds = scale_requery_thresholds
        self.scale_requery_min_pair_pixels = scale_requery_min_pair_pixels
        self.scale_requery_max_side = scale_requery_max_side
        self._scale_requery_stats_file = None
        self.dump_scale_stability_stats = dump_scale_stability_stats
        self.scale_stability_stats_path = scale_stability_stats_path
        self.scale_stability_pairs = scale_stability_pairs
        self.scale_stability_scale = scale_stability_scale
        self.scale_stability_topk = scale_stability_topk
        self.scale_stability_drop_thresholds = scale_stability_drop_thresholds
        self.scale_stability_min_pair_pixels = scale_stability_min_pair_pixels
        self.scale_stability_max_side = scale_stability_max_side
        self._scale_stability_stats_file = None
        self.use_scale_stability_rerank = use_scale_stability_rerank
        self.dump_scale_stability_rerank_stats = dump_scale_stability_rerank_stats
        self.scale_stability_rerank_stats_path = scale_stability_rerank_stats_path
        self.scale_stability_rerank_pairs = scale_stability_rerank_pairs
        self.scale_stability_rerank_pair_file = scale_stability_rerank_pair_file
        self.scale_stability_rerank_scale = scale_stability_rerank_scale
        self.scale_stability_rerank_topk = scale_stability_rerank_topk
        self.scale_stability_rerank_gate = scale_stability_rerank_gate
        self.scale_stability_rerank_drop_threshold = scale_stability_rerank_drop_threshold
        self.scale_stability_rerank_min_scaled_margin = scale_stability_rerank_min_scaled_margin
        self.scale_stability_rerank_min_base_margin = scale_stability_rerank_min_base_margin
        self.scale_stability_rerank_max_base_margin = scale_stability_rerank_max_base_margin
        self.scale_stability_rerank_max_target_rank = scale_stability_rerank_max_target_rank
        self.scale_stability_rerank_margin_boost = scale_stability_rerank_margin_boost
        self.scale_stability_rerank_max_side = scale_stability_rerank_max_side
        self.scale_stability_rerank_pair_mode = scale_stability_rerank_pair_mode
        self.scale_stability_rerank_query_mode = scale_stability_rerank_query_mode
        self.scale_stability_rerank_auto_min_pixels = scale_stability_rerank_auto_min_pixels
        self.scale_stability_rerank_auto_exclude_bg = scale_stability_rerank_auto_exclude_bg
        self.scale_stability_rerank_bg_names = scale_stability_rerank_bg_names
        self.scale_stability_rerank_max_risk_classes = scale_stability_rerank_max_risk_classes
        self._scale_stability_rerank_stats_file = None
        self.dump_expert_reliability_stats = dump_expert_reliability_stats
        self.expert_reliability_stats_path = expert_reliability_stats_path
        self.expert_reliability_min_gate_pixels = expert_reliability_min_gate_pixels
        self.expert_reliability_local_kernel = expert_reliability_local_kernel
        self._expert_reliability_stats_file = None
        self.dump_evidence_bias_stats = dump_evidence_bias_stats
        self.evidence_bias_stats_path = evidence_bias_stats_path
        self.evidence_bias_pairs = evidence_bias_pairs
        self.evidence_bias_pair_mode = evidence_bias_pair_mode
        self.evidence_bias_probes = evidence_bias_probes
        self.evidence_bias_heads = evidence_bias_heads
        self.evidence_bias_support_thresholds = evidence_bias_support_thresholds
        self.evidence_bias_min_pair_pixels = evidence_bias_min_pair_pixels
        self.evidence_bias_max_gt_pairs = evidence_bias_max_gt_pairs
        self.evidence_bias_max_side = evidence_bias_max_side
        self.evidence_bias_empty_cache = evidence_bias_empty_cache
        self._evidence_bias_stats_file = None
        self.dump_topk_conflict_bias_stats = dump_topk_conflict_bias_stats
        self.topk_conflict_bias_stats_path = topk_conflict_bias_stats_path
        self.topk_conflict_topk = topk_conflict_topk
        self.topk_conflict_probes = topk_conflict_probes
        self.topk_conflict_heads = topk_conflict_heads
        self.topk_conflict_support_thresholds = topk_conflict_support_thresholds
        self.topk_conflict_min_pair_pixels = topk_conflict_min_pair_pixels
        self.topk_conflict_max_pairs = topk_conflict_max_pairs
        self.topk_conflict_max_side = topk_conflict_max_side
        self.topk_conflict_local_kernel = topk_conflict_local_kernel
        self.topk_conflict_empty_cache = topk_conflict_empty_cache
        self._topk_conflict_bias_stats_file = None
        self.dump_internal_selection_gap_stats = dump_internal_selection_gap_stats
        self.internal_selection_gap_stats_path = internal_selection_gap_stats_path
        self.internal_selection_sources = internal_selection_sources
        self.internal_selection_topk = internal_selection_topk
        self.internal_selection_max_side = internal_selection_max_side
        self.internal_selection_raw_topk = internal_selection_raw_topk
        self.internal_selection_min_pair_pixels = internal_selection_min_pair_pixels
        self.internal_selection_seed_rule = internal_selection_seed_rule
        self.internal_selection_conservative_min_consensus = (
            internal_selection_conservative_min_consensus)
        self.internal_selection_conservative_min_reliability = (
            internal_selection_conservative_min_reliability)
        self._internal_selection_gap_stats_file = None
        self.dump_candidate_internal_verifier_stats = (
            dump_candidate_internal_verifier_stats)
        self.candidate_internal_verifier_stats_path = (
            candidate_internal_verifier_stats_path)
        self.candidate_internal_topk = candidate_internal_topk
        self.candidate_internal_max_side = candidate_internal_max_side
        self.candidate_internal_min_pair_pixels = (
            candidate_internal_min_pair_pixels)
        self.candidate_internal_delta_thresholds = (
            candidate_internal_delta_thresholds)
        self.candidate_internal_vote_thresholds = (
            candidate_internal_vote_thresholds)
        self.candidate_internal_null_trials = candidate_internal_null_trials
        self.candidate_internal_local_kernel = candidate_internal_local_kernel
        self.candidate_internal_dump_matched_stats = (
            candidate_internal_dump_matched_stats)
        self.candidate_internal_matched_families = (
            candidate_internal_matched_families)
        self.candidate_internal_matched_ranks = (
            candidate_internal_matched_ranks)
        self.candidate_internal_matched_regimes = (
            candidate_internal_matched_regimes)
        self.candidate_internal_matched_margin_bins = (
            candidate_internal_matched_margin_bins)
        self.candidate_internal_matched_score_thresholds = (
            candidate_internal_matched_score_thresholds)
        self.candidate_internal_matched_null_trials = (
            candidate_internal_matched_null_trials)
        self.candidate_internal_matched_auc_bins = (
            candidate_internal_matched_auc_bins)
        self.candidate_internal_matched_auc_min = (
            candidate_internal_matched_auc_min)
        self.candidate_internal_matched_auc_max = (
            candidate_internal_matched_auc_max)
        self._candidate_internal_verifier_stats_file = None
        self.dump_position_bias_stats = dump_position_bias_stats
        self.position_bias_stats_path = position_bias_stats_path
        self.position_bias_layers = position_bias_layers
        self.position_bias_ranks = position_bias_ranks
        self.position_bias_controls = position_bias_controls
        self.position_bias_random_trials = position_bias_random_trials
        self.position_bias_max_side = position_bias_max_side
        self.position_bias_seed_rule = position_bias_seed_rule
        self._position_bias_stats_file = None
        self._position_bias_basis_cache = {}
        self._position_bias_control_basis_cache = {}
        self.dump_scene_common_bias_stats = dump_scene_common_bias_stats
        self.scene_common_bias_stats_path = scene_common_bias_stats_path
        self.scene_common_layers = scene_common_layers
        self.scene_common_variants = scene_common_variants
        self.scene_common_strengths = scene_common_strengths
        self.scene_common_local_kernels = scene_common_local_kernels
        self.scene_common_trim_quantile = scene_common_trim_quantile
        self.scene_common_random_trials = scene_common_random_trials
        self.scene_common_compute_spectrum = scene_common_compute_spectrum
        self.scene_common_max_side = scene_common_max_side
        self.scene_common_seed_rule = scene_common_seed_rule
        self._scene_common_bias_stats_file = None
        self.dump_candidate_residual_trajectory_stats = (
            dump_candidate_residual_trajectory_stats)
        self.candidate_residual_trajectory_stats_path = (
            candidate_residual_trajectory_stats_path)
        self.candidate_residual_layers = candidate_residual_layers
        self.candidate_residual_variants = candidate_residual_variants
        self.candidate_residual_strengths = candidate_residual_strengths
        self.candidate_residual_scores = candidate_residual_scores
        self.candidate_residual_ranks = candidate_residual_ranks
        self.candidate_residual_regimes = candidate_residual_regimes
        self.candidate_residual_margin_bins = (
            candidate_residual_margin_bins)
        self.candidate_residual_min_pair_pixels = (
            candidate_residual_min_pair_pixels)
        self.candidate_residual_null_trials = (
            candidate_residual_null_trials)
        self.candidate_residual_auc_bins = candidate_residual_auc_bins
        self.candidate_residual_auc_min = candidate_residual_auc_min
        self.candidate_residual_auc_max = candidate_residual_auc_max
        self.candidate_residual_max_side = candidate_residual_max_side
        self.candidate_residual_seed_rule = candidate_residual_seed_rule
        self._candidate_residual_trajectory_stats_file = None
        self.dump_candidate_residual_miou_stats = (
            dump_candidate_residual_miou_stats)
        self.candidate_residual_miou_stats_path = (
            candidate_residual_miou_stats_path)
        self.candidate_residual_miou_scores = (
            candidate_residual_miou_scores)
        self.candidate_residual_miou_ranks = (
            candidate_residual_miou_ranks)
        self.candidate_residual_miou_units = (
            candidate_residual_miou_units)
        self.candidate_residual_miou_absolute_thresholds = (
            candidate_residual_miou_absolute_thresholds)
        self.candidate_residual_miou_coverages = (
            candidate_residual_miou_coverages)
        self.candidate_residual_miou_region_reducer = (
            candidate_residual_miou_region_reducer)
        self.candidate_residual_miou_region_min_pixels = (
            candidate_residual_miou_region_min_pixels)
        self.candidate_residual_miou_raw_bin_thd = (
            candidate_residual_miou_raw_bin_thd)
        self.candidate_residual_miou_max_side = (
            candidate_residual_miou_max_side)
        self._candidate_residual_miou_stats_file = None
        self.dump_ontology_readout_oracle_stats = bool(
            dump_ontology_readout_oracle_stats)
        self.ontology_readout_oracle_stats_path = (
            ontology_readout_oracle_stats_path)
        self.ontology_readout_oracle_sources = (
            ontology_readout_oracle_sources)
        self.ontology_readout_oracle_max_side = int(
            ontology_readout_oracle_max_side)
        self.ontology_readout_oracle_topk = int(
            ontology_readout_oracle_topk)
        self.ontology_readout_oracle_min_pair_pixels = int(
            ontology_readout_oracle_min_pair_pixels)
        self._ontology_readout_oracle_stats_file = None
        self.dump_state_action_atlas_stats = (
            dump_state_action_atlas_stats)
        self.state_action_atlas_stats_path = (
            state_action_atlas_stats_path)
        self.state_action_atlas_actions = state_action_atlas_actions
        self.state_action_apply = state_action_apply
        self.state_action_presence_gamma = state_action_presence_gamma
        self.state_action_presence_blend = state_action_presence_blend
        self.state_action_presence_max_boost = (
            state_action_presence_max_boost)
        self.state_action_presence_floor = state_action_presence_floor
        self.state_action_support_threshold = (
            state_action_support_threshold)
        self.state_action_low_presence_threshold = (
            state_action_low_presence_threshold)
        self.state_action_margin_threshold = state_action_margin_threshold
        self.state_action_local_kernel = state_action_local_kernel
        self.state_action_feature_max_side = (
            state_action_feature_max_side)
        self._state_action_atlas_stats_file = None
        self.use_region_contrastive_readout = (
            use_region_contrastive_readout)
        self.dump_region_contrastive_readout_stats = (
            dump_region_contrastive_readout_stats)
        self.region_contrastive_readout_stats_path = (
            region_contrastive_readout_stats_path)
        self.region_readout_max_side = region_readout_max_side
        self.region_readout_max_regions = region_readout_max_regions
        self.region_readout_masks_per_prompt = (
            region_readout_masks_per_prompt)
        self.region_readout_mask_threshold = (
            region_readout_mask_threshold)
        self.region_readout_min_pixels = region_readout_min_pixels
        self.region_readout_dedup_iou = region_readout_dedup_iou
        self.region_readout_ring_kernel = region_readout_ring_kernel
        self.region_readout_top_fraction = region_readout_top_fraction
        self.region_readout_presence_weight = (
            region_readout_presence_weight)
        self.region_readout_topk = region_readout_topk
        self.region_readout_blends = region_readout_blends
        self.region_readout_mix_sources = region_readout_mix_sources
        self.region_readout_apply_source = region_readout_apply_source
        self.region_readout_apply_blend = region_readout_apply_blend
        self.region_readout_apply_topk = region_readout_apply_topk
        self.region_readout_min_margin = region_readout_min_margin
        self.region_readout_min_pair_pixels = (
            region_readout_min_pair_pixels)
        self.region_readout_high_purity_threshold = (
            region_readout_high_purity_threshold)
        self.region_readout_projection = region_readout_projection
        self.region_readout_route_variant = (
            region_readout_route_variant)
        self.region_readout_foreground_blend = (
            region_readout_foreground_blend)
        self.region_readout_reject_blend = (
            region_readout_reject_blend)
        self.region_readout_background_blend = (
            region_readout_background_blend)
        self.region_readout_formula_agreement = (
            region_readout_formula_agreement)
        self.region_readout_agreement_sources = (
            region_readout_agreement_sources)
        self._region_contrastive_readout_stats_file = None
        self.dump_region_hypothesis_v2_stats = (
            dump_region_hypothesis_v2_stats)
        self.region_hypothesis_v2_stats_path = (
            region_hypothesis_v2_stats_path)
        self.region_hypothesis_v2_formulas = (
            region_hypothesis_v2_formulas)
        self.region_hypothesis_v2_formula_selectors = (
            region_hypothesis_v2_formula_selectors)
        self.region_hypothesis_v2_projections = (
            region_hypothesis_v2_projections)
        self.region_hypothesis_v2_selector_thresholds = (
            region_hypothesis_v2_selector_thresholds)
        self.region_hypothesis_v2_quality_bins = (
            region_hypothesis_v2_quality_bins)
        self.region_hypothesis_v2_selector_projection = (
            region_hypothesis_v2_selector_projection)
        self.region_hypothesis_v2_presence_weight = (
            region_hypothesis_v2_presence_weight)
        self.region_hypothesis_v2_blend = (
            region_hypothesis_v2_blend)
        self.region_hypothesis_v2_min_margin = (
            region_hypothesis_v2_min_margin)
        self.region_hypothesis_v2_topk = (
            region_hypothesis_v2_topk)
        self.region_hypothesis_v2_overlap_max = (
            region_hypothesis_v2_overlap_max)
        self.region_hypothesis_v2_baseline_advantage = (
            region_hypothesis_v2_baseline_advantage)
        self.region_hypothesis_v2_core_kernel = (
            region_hypothesis_v2_core_kernel)
        self.region_hypothesis_v2_high_purity = (
            region_hypothesis_v2_high_purity)
        self._region_hypothesis_v2_stats_file = None
        self.dump_candidate_region_quality_stats = bool(
            dump_candidate_region_quality_stats)
        self.candidate_region_quality_stats_path = (
            candidate_region_quality_stats_path)
        self.candidate_region_quality_sources = (
            candidate_region_quality_sources)
        self.candidate_region_quality_max_side = int(
            candidate_region_quality_max_side)
        self.candidate_region_quality_max_regions = int(
            candidate_region_quality_max_regions)
        self.candidate_region_quality_min_pixels = int(
            candidate_region_quality_min_pixels)
        self.candidate_region_quality_gt_min_pixels = int(
            candidate_region_quality_gt_min_pixels)
        self.candidate_region_quality_high_purity = float(
            candidate_region_quality_high_purity)
        self.candidate_region_quality_group_iou = float(
            candidate_region_quality_group_iou)
        self.candidate_region_quality_group_containment = float(
            candidate_region_quality_group_containment)
        self.candidate_region_quality_semantic_threshold = float(
            candidate_region_quality_semantic_threshold)
        self.candidate_region_quality_semantic_max_per_class = int(
            candidate_region_quality_semantic_max_per_class)
        self.candidate_region_quality_hybrid_dedup_iou = float(
            candidate_region_quality_hybrid_dedup_iou)
        self.candidate_region_quality_fragment_coverage = float(
            candidate_region_quality_fragment_coverage)
        self._candidate_region_quality_stats_file = None
        self.dump_query_topology_stats = bool(
            dump_query_topology_stats)
        self.query_topology_stats_path = query_topology_stats_path
        self.query_topology_max_side = int(query_topology_max_side)
        self.query_topology_max_regions = int(
            query_topology_max_regions)
        self.query_topology_min_pixels = int(
            query_topology_min_pixels)
        self.query_topology_support_threshold = float(
            query_topology_support_threshold)
        self.query_topology_mask_threshold = float(
            query_topology_mask_threshold)
        self.query_topology_semantic_threshold = float(
            query_topology_semantic_threshold)
        self.query_topology_action_min_overlap = float(
            query_topology_action_min_overlap)
        self.query_topology_action_padding = int(
            query_topology_action_padding)
        self.query_topology_include_background = bool(
            query_topology_include_background)
        self.query_topology_save_npz = bool(
            query_topology_save_npz)
        self.query_topology_artifact_dir = (
            query_topology_artifact_dir)
        self.query_topology_max_saved_images = int(
            query_topology_max_saved_images)
        self._query_topology_saved_images = 0
        self._query_topology_stats_file = None
        self.dump_reviewer_cache = bool(dump_reviewer_cache)
        self.reviewer_cache_dir = reviewer_cache_dir
        self.reviewer_dataset_name = reviewer_dataset_name
        self.reviewer_max_side = int(reviewer_max_side)
        self.reviewer_cache_samples_per_image = int(
            reviewer_cache_samples_per_image)
        self.reviewer_cache_hard_keep_margin = float(
            reviewer_cache_hard_keep_margin)
        self.reviewer_topk = int(reviewer_topk)
        self.reviewer_local_kernel = int(reviewer_local_kernel)
        self.use_learned_reviewer = bool(use_learned_reviewer)
        self.reviewer_checkpoint = reviewer_checkpoint
        self.reviewer_variant = str(reviewer_variant)
        self.reviewer_inference_batch_size = int(
            reviewer_inference_batch_size)
        self.reviewer_gate_threshold = reviewer_gate_threshold
        self.reviewer_action_margin = reviewer_action_margin
        self.reviewer_min_boost = float(reviewer_min_boost)
        self.dump_learned_reviewer_stats = bool(
            dump_learned_reviewer_stats)
        self.learned_reviewer_stats_path = (
            learned_reviewer_stats_path)
        self._learned_reviewer_model = None
        self._reviewer_checkpoint_meta = None
        self._learned_reviewer_stats_file = None
        if self._uses_rethinking_reviewer() and self.reviewer_topk < 2:
            raise ValueError('reviewer_topk must be at least 2.')
        if (
                self.dump_reviewer_cache
                and not self.reviewer_cache_dir):
            raise ValueError(
                'dump_reviewer_cache=True requires reviewer_cache_dir.')
        self.use_geoer_router = use_geoer_router
        self.dump_geoer_router_stats = dump_geoer_router_stats
        self.geoer_router_stats_path = geoer_router_stats_path
        self.geoer_use_scale_stability = geoer_use_scale_stability
        self.geoer_use_reject_recovery = geoer_use_reject_recovery
        self.geoer_reject_recovery_datasets = geoer_reject_recovery_datasets
        self.geoer_reject_recovery_bg_names = geoer_reject_recovery_bg_names
        self.geoer_reject_recovery_score_thd = geoer_reject_recovery_score_thd
        self.geoer_reject_recovery_semantic_thd = geoer_reject_recovery_semantic_thd
        self.geoer_reject_recovery_instance_thd = geoer_reject_recovery_instance_thd
        self.geoer_reject_recovery_margin_thd = geoer_reject_recovery_margin_thd
        self.geoer_reject_recovery_local_thd = geoer_reject_recovery_local_thd
        self.geoer_reject_recovery_require_bg_margin = geoer_reject_recovery_require_bg_margin
        self._geoer_router_stats_file = None
        self.use_coco_sec_fusion = use_coco_sec_fusion
        self.dump_coco_sec_stats = dump_coco_sec_stats
        self.coco_sec_stats_path = coco_sec_stats_path
        self.coco_sec_lambda = coco_sec_lambda
        self.coco_sec_temperature = coco_sec_temperature
        self.coco_sec_eps = coco_sec_eps
        self.coco_sec_center_prior = coco_sec_center_prior
        self.coco_sec_synonym_reduce = coco_sec_synonym_reduce
        self._coco_sec_stats_file = None
        self._coco_sec_query_text_features = None
        self.class_names = _build_class_names(self.query_words, self.query_idx, self.num_cls)
        if self.instance_score_type not in ('presence', 'raw'):
            raise ValueError(
                "instance_score_type must be 'presence' or 'raw', "
                f"but got {self.instance_score_type!r}")

    def _get_instance_score(self, inference_state, inst_id):
        if self.instance_score_type == 'raw':
            return inference_state['object_score_raw'][inst_id]
        return inference_state['object_score_presence'][inst_id]

    def _get_internal_selection_sources(self):
        sources = self.internal_selection_sources
        if sources is None:
            sources = []
        elif isinstance(sources, str):
            sources = sources.split(',')
        elif not isinstance(sources, (list, tuple)):
            sources = [sources]
        normalized = []
        for source in sources:
            source = str(source).strip().lower()
            if source and source not in normalized:
                normalized.append(source)
        return normalized

    def _uses_internal_selection_analysis(self):
        return (
            self.dump_internal_selection_gap_stats
            or self.dump_candidate_internal_verifier_stats
            or self.dump_position_bias_stats
            or self.dump_scene_common_bias_stats
            or self.dump_candidate_residual_trajectory_stats
            or self.dump_candidate_residual_miou_stats
            or self._uses_ontology_readout_oracle()
        )

    def _get_active_internal_sources(self):
        sources = (
            self._get_internal_selection_sources()
            if self._uses_internal_selection_analysis()
            else [])
        if self._uses_rethinking_reviewer():
            for source in self._reviewer_internal_sources():
                if source not in sources:
                    sources.append(source)
        if self._uses_ontology_readout_oracle():
            for source in self._ontology_readout_internal_sources():
                if source not in sources:
                    sources.append(source)
        return sources

    def _uses_internal_evidence_diagnostics(self):
        return (
            self._uses_internal_selection_analysis()
            or self._uses_rethinking_reviewer())

    def _candidate_residual_miou_uses_raw_masks(self):
        return (
            self.dump_candidate_residual_miou_stats
            and 'raw_mask' in self._parse_name_list(
                self.candidate_residual_miou_units)
        )

    def _internal_selection_wants(self, source):
        sources = self._get_active_internal_sources()
        if source in sources:
            return True
        if source.startswith('encoder_level_') and 'encoder_all' in sources:
            return True
        if source.startswith('pe_layer_') and 'pe_all' in sources:
            return True
        return False

    def _internal_selection_diag_shape(self, height, width):
        max_sides = []
        if self.dump_internal_selection_gap_stats:
            max_sides.append(int(self.internal_selection_max_side))
        if self.dump_candidate_internal_verifier_stats:
            max_sides.append(int(self.candidate_internal_max_side))
        if self.dump_position_bias_stats:
            max_sides.append(int(self.position_bias_max_side))
        if self.dump_scene_common_bias_stats:
            max_sides.append(int(self.scene_common_max_side))
        if self.dump_candidate_residual_trajectory_stats:
            max_sides.append(int(self.candidate_residual_max_side))
        if self.dump_candidate_residual_miou_stats:
            max_sides.append(int(self.candidate_residual_miou_max_side))
        if self._uses_ontology_readout_oracle():
            max_sides.append(int(self.ontology_readout_oracle_max_side))
        if self._uses_rethinking_reviewer():
            max_sides.append(int(self.reviewer_max_side))
        max_side = max(1, min(max_sides or [self.internal_selection_max_side]))
        scale = min(1.0, float(max_side) / float(max(height, width)))
        diag_h = max(1, int(round(height * scale)))
        diag_w = max(1, int(round(width * scale)))
        return diag_h, diag_w

    @staticmethod
    def _interpolate_float32(tensor, size):
        """Resize diagnostic maps in FP32 for older CUDA/PyTorch builds."""
        tensor = tensor.float()
        if tensor.device.type == 'cuda':
            with torch.autocast(device_type='cuda', enabled=False):
                return F.interpolate(
                    tensor,
                    size=size,
                    mode='bilinear',
                    align_corners=False,
                )
        return F.interpolate(
            tensor,
            size=size,
            mode='bilinear',
            align_corners=False,
        )

    def _build_internal_raw_query_map(self, inference_state, output_shape,
                                      score_type):
        raw_masks = inference_state.get('raw_masks_logits_lowres')
        if score_type == 'raw_object':
            scores = inference_state.get('raw_object_score')
        else:
            scores = inference_state.get('raw_object_score_presence')
        if raw_masks is None or scores is None:
            return None
        if raw_masks.numel() == 0 or scores.numel() == 0:
            return None
        count = min(int(raw_masks.shape[0]), int(scores.numel()))
        topk = max(1, min(int(self.internal_selection_raw_topk), count))
        selected = torch.topk(scores[:count].detach().float(), k=topk).indices
        masks = torch.sigmoid(self._interpolate_float32(
            raw_masks[selected].detach().unsqueeze(1),
            output_shape,
        ).squeeze(1))
        weighted = masks * scores[selected].detach().float()[:, None, None]
        return weighted.max(dim=0)[0]

    def _build_internal_encoder_query_maps(self, inference_state, output_shape):
        hidden = inference_state.get('encoder_hidden_states')
        prompt = inference_state.get('encoder_prompt_after')
        prompt_mask = inference_state.get('encoder_prompt_mask')
        spatial_shapes = inference_state.get('encoder_spatial_shapes')
        level_starts = inference_state.get('encoder_level_start_index')
        if not isinstance(hidden, torch.Tensor) or not isinstance(prompt, torch.Tensor):
            return {}
        if not isinstance(spatial_shapes, torch.Tensor):
            return {}
        hidden = hidden.detach().float()
        prompt = prompt.detach().float()
        if hidden.ndim != 3 or prompt.ndim < 2:
            return {}

        channel_dim = int(hidden.shape[-1])
        if hidden.shape[1] == 1:
            tokens = hidden[:, 0, :]
        elif hidden.shape[0] == 1:
            tokens = hidden[0]
        else:
            tokens = hidden.reshape(-1, channel_dim)
        if tokens.shape[-1] != channel_dim:
            return {}

        if prompt.shape[-1] != channel_dim:
            return {}
        prompt_tokens = prompt.reshape(-1, channel_dim)
        if isinstance(prompt_mask, torch.Tensor) and prompt.ndim == 3:
            mask = prompt_mask.detach().bool()
            valid_prompt = None
            if (prompt.shape[0] == mask.shape[1]
                    and prompt.shape[1] == mask.shape[0]):
                valid_prompt = (~mask).transpose(0, 1).reshape(-1)
            elif (prompt.shape[0] == mask.shape[0]
                    and prompt.shape[1] == mask.shape[1]):
                valid_prompt = (~mask).reshape(-1)
            if (valid_prompt is not None
                    and valid_prompt.numel() == prompt_tokens.shape[0]
                    and valid_prompt.any()):
                prompt_tokens = prompt_tokens[valid_prompt]
        if prompt_tokens.numel() == 0:
            return {}
        prompt_vector = F.normalize(
            prompt_tokens.mean(dim=0),
            dim=0,
            eps=float(self.seed_similarity_eps),
        )
        tokens = F.normalize(
            tokens,
            dim=-1,
            eps=float(self.seed_similarity_eps),
        )
        similarity = tokens @ prompt_vector

        shapes = spatial_shapes.detach().long().cpu().tolist()
        if isinstance(level_starts, torch.Tensor):
            starts = level_starts.detach().long().cpu().tolist()
        else:
            starts = []
            offset = 0
            for height, width in shapes:
                starts.append(offset)
                offset += int(height) * int(width)

        maps = {}
        for level_idx, shape in enumerate(shapes):
            if len(shape) < 2 or level_idx >= len(starts):
                continue
            height, width = int(shape[0]), int(shape[1])
            start = int(starts[level_idx])
            end = start + height * width
            if height <= 0 or width <= 0 or start < 0 or end > similarity.numel():
                continue
            level_map = similarity[start:end].view(1, 1, height, width)
            level_map = self._interpolate_float32(
                level_map,
                output_shape,
            ).squeeze()
            maps[f'encoder_level_{level_idx}'] = level_map
        return maps

    def _build_internal_view_source_maps(
            self, final_query_logits, semantic_query_logits,
            instance_query_logits, presence_query_scores,
            raw_query_maps, encoder_query_maps, vision_features, pe_layers,
            position_layers, output_shape):
        if not self._uses_internal_evidence_diagnostics():
            return {}, [], []

        final_query = self._interpolate_float32(
            final_query_logits.detach().unsqueeze(0),
            output_shape,
        ).squeeze(0)
        semantic_query = self._interpolate_float32(
            semantic_query_logits.detach().unsqueeze(0),
            output_shape,
        ).squeeze(0)
        instance_query = self._interpolate_float32(
            instance_query_logits.detach().unsqueeze(0),
            output_shape,
        ).squeeze(0)
        presence = presence_query_scores.detach().float().view(-1, 1, 1)

        class_final = self._aggregate_query_logits_to_classes(final_query)
        class_semantic = self._aggregate_query_logits_to_classes(semantic_query)
        class_instance = self._aggregate_query_logits_to_classes(instance_query)
        source_maps = {}
        if self._internal_selection_wants('fusion_no_presence'):
            source_maps['fusion_no_presence'] = self._aggregate_query_logits_to_classes(
                torch.maximum(semantic_query, instance_query))
        if self._internal_selection_wants('semantic_presence'):
            source_maps['semantic_presence'] = self._aggregate_query_logits_to_classes(
                semantic_query * presence)
        if self._internal_selection_wants('instance_presence'):
            source_maps['instance_presence'] = self._aggregate_query_logits_to_classes(
                instance_query * presence)

        for source_name, query_maps in (raw_query_maps or {}).items():
            if query_maps is not None and self._internal_selection_wants(source_name):
                query_maps = query_maps.detach().float()
                class_maps = self._aggregate_query_logits_to_classes(query_maps)
                query_available = query_maps.abs().flatten(1).sum(dim=1) > 0
                class_available = torch.zeros(
                    self.num_cls, device=self.device, dtype=torch.bool)
                for query_idx in range(self.num_queries):
                    if query_available[query_idx]:
                        class_available[int(self.query_idx[query_idx].item())] = True
                class_maps[~class_available] = float('nan')
                source_maps[source_name] = class_maps
        for source_name, query_maps in (encoder_query_maps or {}).items():
            if query_maps is not None and self._internal_selection_wants(source_name):
                source_maps[source_name] = self._aggregate_query_logits_to_classes(
                    query_maps.detach().float())

        local_pred = torch.argmax(class_final, dim=0)
        local_pred[class_final.max(dim=0)[0] < float(self.prob_thd)] = int(self.bg_idx)
        local_components = dict(
            semantic_logits=class_semantic,
            instance_logits=class_instance,
        )
        seed_context = self._build_seed_rule_context(
            class_final, local_pred, local_components)
        seed_mask = None
        seed_class = None
        if seed_context is not None:
            seed_class = seed_context['final_top1_idx']
            rule_defs = seed_context['rule_defs']
            selected_rule = None
            if self.dump_scene_common_bias_stats:
                requested_seed_rule = self.scene_common_seed_rule
            elif (
                    self.dump_candidate_residual_trajectory_stats
                    or self.dump_candidate_residual_miou_stats):
                requested_seed_rule = self.candidate_residual_seed_rule
            elif self.dump_position_bias_stats:
                requested_seed_rule = self.position_bias_seed_rule
            else:
                requested_seed_rule = self.internal_selection_seed_rule
            for rule_name, rule_mask in rule_defs:
                if rule_name == str(requested_seed_rule):
                    selected_rule = rule_mask
                    break
            if selected_rule is None and rule_defs:
                selected_rule = rule_defs[-1][1]
            seed_mask = selected_rule

        if seed_mask is not None and seed_class is not None:
            feature_components = {}
            if vision_features is not None:
                feature_components['vision_features'] = vision_features
            if isinstance(pe_layers, (list, tuple)):
                feature_components['pe_layers'] = pe_layers
            feature_spaces = []
            if self._internal_selection_wants('vision'):
                feature_spaces.append('vision')
            active_internal_sources = self._get_active_internal_sources()
            if 'pe_all' in active_internal_sources:
                feature_spaces.extend([
                    f'pe_layer_{idx}' for idx in range(len(pe_layers or []))
                ])
            else:
                feature_spaces.extend([
                    source for source in active_internal_sources
                    if source.startswith('pe_layer_')
                ])
            for space in feature_spaces:
                similarity_maps, has_seed = self._build_feature_seed_similarity_maps(
                    space,
                    class_final,
                    feature_components,
                    seed_mask,
                    seed_class,
                )
                if similarity_maps is None or has_seed is None:
                    continue
                similarity_maps = similarity_maps.detach().float()
                similarity_maps[~has_seed, :, :] = float('nan')
                source_maps[space] = similarity_maps

        position_maps, position_stats = self._build_position_bias_source_maps(
            class_final,
            class_semantic,
            class_instance,
            pe_layers,
            position_layers,
            output_shape,
            seed_mask,
            seed_class,
        )
        source_maps.update(position_maps)
        scene_common_maps, scene_common_stats = (
            self._build_scene_common_bias_source_maps(
                class_final,
                pe_layers,
                output_shape,
                seed_mask,
                seed_class,
            )
        )
        source_maps.update(scene_common_maps)
        return source_maps, position_stats, scene_common_stats

    def _collect_raw_mask_oracle_candidates(self, query_idx, query_word, inference_state):
        raw_masks = inference_state.get('raw_masks_logits_lowres')
        raw_scores = inference_state.get('raw_object_score')
        presence_scores = inference_state.get('raw_object_score_presence')
        keep_mask = inference_state.get('raw_keep_mask')
        if raw_masks is None or raw_scores is None or presence_scores is None or keep_mask is None:
            return None
        if raw_masks.numel() == 0 or raw_scores.numel() == 0:
            return None

        num_candidates = int(raw_scores.numel())
        topk = max(1, min(int(self.raw_mask_oracle_topk), num_candidates))
        selected = [
            torch.topk(raw_scores.detach().float(), k=topk).indices,
            torch.topk(presence_scores.detach().float(), k=topk).indices,
        ]
        kept_indices = torch.nonzero(keep_mask.detach().bool(), as_tuple=False).flatten()
        if kept_indices.numel() > 0:
            if self._uses_query_topology_diagnostic():
                # Exact instance-head provenance requires every query that
                # survived SAM3's confidence/presence filter.  Unkept queries
                # remain bounded by raw_mask_oracle_topk.
                selected.append(kept_indices)
            else:
                kept_presence = presence_scores.detach().float()[
                    kept_indices]
                kept_topk = min(topk, int(kept_indices.numel()))
                selected.append(kept_indices[
                    torch.topk(
                        kept_presence, k=kept_topk).indices])
        selected_indices = torch.unique(torch.cat(selected)).to(torch.long)

        return dict(
            query_index=int(query_idx),
            class_index=int(self.query_idx[query_idx].item()),
            query_word=str(query_word),
            selected_indices=selected_indices.detach().cpu(),
            raw_masks_lowres=raw_masks.detach()[selected_indices].float().cpu(),
            raw_scores=raw_scores.detach()[selected_indices].float().cpu(),
            raw_presence_scores=presence_scores.detach()[selected_indices].float().cpu(),
            raw_keep_mask=keep_mask.detach()[selected_indices].bool().cpu(),
            raw_candidate_count=num_candidates,
            raw_kept_count=int(keep_mask.detach().bool().sum().item()),
        )

    def _inference_single_view(self, image, return_stats=False, return_components=False,
                               view_id=None, crop_box=None):
        """Inference on a single PIL image or crop patch."""
        w, h = image.size
        seg_logits = torch.zeros((self.num_queries, h, w), device=self.device)
        prompt_stats = [] if return_stats else None
        semantic_logits_all = (
            torch.zeros((self.num_queries, h, w), device=self.device)
            if return_components else None)
        instance_logits_all = (
            torch.zeros((self.num_queries, h, w), device=self.device)
            if return_components else None)
        internal_diag_shape = (
            self._internal_selection_diag_shape(h, w)
            if return_components and self._uses_internal_evidence_diagnostics()
            else None)
        presence_query_scores = (
            torch.zeros(self.num_queries, device=self.device)
            if return_components and (
                internal_diag_shape is not None
                or self._uses_region_contrastive_readout()
                or self._uses_region_hypothesis_v2()
                or self._uses_active_concept_diagnostic()
                or self._uses_concept_specificity_diagnostic()
                or self._uses_residual_background_modeling()
                or self.dump_prompt_winner_attribution_stats
                or self._uses_cross_image_bank_features()
                or self._uses_ontology_self_verification())
            else None)
        raw_query_maps = {}
        if internal_diag_shape is not None:
            for source_name in ('raw_object', 'raw_presence'):
                if self._internal_selection_wants(source_name):
                    raw_query_maps[source_name] = torch.zeros(
                        (self.num_queries, *internal_diag_shape),
                        device=self.device,
                        dtype=torch.float32,
                    )
        encoder_query_maps = {}
        vision_features = None
        pe_layers = None
        position_layers = None
        needs_vision_features = (
            self._uses_internal_evidence_diagnostics()
            or self._uses_pe_layer_similarity()
            or self.dump_seed_separability_stats
            or self.use_coco_sec_fusion
                or self.dump_coco_sec_stats
                or self._uses_cross_image_bank_features()
                or self._uses_ontology_self_verification()
        )
        raw_mask_candidates = (
            [] if return_components and (
                self.dump_raw_mask_oracle_stats
                or self.dump_candidate_quality_stats
                or self.dump_region_prompt_identity_stats
                or self._candidate_residual_miou_uses_raw_masks()
                or self._uses_region_contrastive_readout()
                or self._uses_region_hypothesis_v2()
                or self._uses_candidate_region_quality_diagnostic()
                or self._uses_query_topology_diagnostic())
            else None)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = self.processor.set_image(image)
            if (
                    return_components
                    and needs_vision_features
                    and 'vision_features' in inference_state):
                vision_features = inference_state['vision_features'].detach()
            if return_components and self._uses_pe_layer_similarity():
                backbone_out = inference_state.get('backbone_out') or {}
                backbone_fpn = backbone_out.get('backbone_fpn')
                if isinstance(backbone_fpn, (list, tuple)):
                    pe_layers = [
                        feat.detach()
                        for feat in backbone_fpn
                        if isinstance(feat, torch.Tensor)
                    ]
                vision_pos_enc = backbone_out.get('vision_pos_enc')
                if isinstance(vision_pos_enc, (list, tuple)):
                    position_layers = [
                        pos.detach()
                        for pos in vision_pos_enc
                        if isinstance(pos, torch.Tensor)
                    ]
            
            for query_idx, query_word in enumerate(self.query_words):
                self.processor.reset_all_prompts(inference_state)
                inference_state = self.processor.set_text_prompt(state=inference_state, prompt=query_word)
                if presence_query_scores is not None:
                    presence_value = inference_state.get('presence_score')
                    if isinstance(presence_value, torch.Tensor) and presence_value.numel() > 0:
                        presence_query_scores[query_idx] = presence_value.detach().float().mean()
                if internal_diag_shape is not None:
                    for source_name, query_maps in raw_query_maps.items():
                        raw_map = self._build_internal_raw_query_map(
                            inference_state,
                            internal_diag_shape,
                            source_name,
                        )
                        if raw_map is not None:
                            query_maps[query_idx] = raw_map
                    requested_internal_sources = (
                        self._get_active_internal_sources())
                    if (
                            'encoder_all' in requested_internal_sources
                            or any(
                                source.startswith('encoder_level_')
                                for source in requested_internal_sources)):
                        encoder_maps = self._build_internal_encoder_query_maps(
                            inference_state,
                            internal_diag_shape,
                        )
                        for source_name, source_map in encoder_maps.items():
                            if not self._internal_selection_wants(source_name):
                                continue
                            if source_name not in encoder_query_maps:
                                encoder_query_maps[source_name] = torch.zeros(
                                    (self.num_queries, *internal_diag_shape),
                                    device=self.device,
                                    dtype=torch.float32,
                                )
                            encoder_query_maps[source_name][query_idx] = source_map
                if raw_mask_candidates is not None:
                    query_candidates = self._collect_raw_mask_oracle_candidates(
                        query_idx=query_idx,
                        query_word=query_word,
                        inference_state=inference_state,
                    )
                    if query_candidates is not None:
                        query_candidates['crop_box'] = crop_box
                        query_candidates['view_id'] = view_id
                        query_candidates['source_image_size'] = [w, h]
                        raw_mask_candidates.append(query_candidates)
                instance_component = (
                    torch.zeros((h, w), device=self.device)
                    if (return_stats or return_components) else None)
                semantic_component = None

                if self.use_transformer_decoder:
                    if inference_state['masks_logits'].shape[0] > 0:
                        inst_len = inference_state['masks_logits'].shape[0]
                        for inst_id in range(inst_len):
                            instance_logits = inference_state['masks_logits'][inst_id].squeeze()
                            instance_score = self._get_instance_score(inference_state, inst_id)
                            # instance_mask = inference_state['masks'][inst_id].squeeze()
                            
                            # Handle potential dimension mismatch if SAM3 output differs slightly
                            if instance_logits.shape != (h, w):
                                instance_logits = F.interpolate(
                                    instance_logits.view(1, 1, *instance_logits.shape), 
                                    size=(h, w), 
                                    mode='bilinear', 
                                    align_corners=False
                                ).squeeze()

                            instance_weighted = instance_logits * instance_score
                            seg_logits[query_idx] = torch.max(seg_logits[query_idx], instance_weighted)
                            if return_stats or return_components:
                                instance_component = torch.max(instance_component, instance_weighted)
                    
                if self.use_sem_seg:
                    semantic_logits = inference_state['semantic_mask_logits']
                    if semantic_logits.shape != (h, w):
                            semantic_logits = F.interpolate(
                                semantic_logits, 
                                size=(h, w), 
                                mode='bilinear', 
                                align_corners=False
                            ).squeeze()
                    
                    seg_logits[query_idx] = torch.max(seg_logits[query_idx], semantic_logits)
                    if return_stats:
                        semantic_component = semantic_logits
                    if return_components:
                        semantic_logits_all[query_idx] = semantic_logits

                if return_components:
                    instance_logits_all[query_idx] = instance_component
                
                if self.use_presence_score:
                    seg_logits[query_idx] = seg_logits[query_idx] * inference_state["presence_score"]

                if return_stats:
                    prompt_stats.append(self._build_prompt_evidence_stats(
                        query_idx=query_idx,
                        query_word=query_word,
                        inference_state=inference_state,
                        semantic_component=semantic_component,
                        instance_component=instance_component,
                        final_component=seg_logits[query_idx],
                    ))
                
        if not return_stats and not return_components:
            return seg_logits

        stats = dict(
            view_id=view_id,
            crop_box=crop_box,
            image_size=[w, h],
            prompt_stats=prompt_stats,
        ) if return_stats else None
        components = dict(
            semantic_logits=semantic_logits_all,
            instance_logits=instance_logits_all,
        ) if return_components else None
        if return_components and vision_features is not None:
            components['vision_features'] = vision_features
        if return_components and pe_layers is not None:
            components['pe_layers'] = pe_layers
        if return_components and position_layers is not None:
            components['position_layers'] = position_layers
        if return_components and raw_mask_candidates is not None:
            components['raw_mask_candidates'] = raw_mask_candidates
        if return_components and presence_query_scores is not None:
            components['presence_query_scores'] = (
                presence_query_scores.detach())
        if return_components and internal_diag_shape is not None:
            (
                internal_maps,
                position_stats,
                scene_common_stats,
            ) = self._build_internal_view_source_maps(
                seg_logits,
                semantic_logits_all,
                instance_logits_all,
                presence_query_scores,
                raw_query_maps,
                encoder_query_maps,
                vision_features,
                pe_layers,
                position_layers,
                internal_diag_shape,
            )
            components['internal_source_maps'] = internal_maps
            if position_stats:
                components['position_bias_feature_stats'] = position_stats
            if scene_common_stats:
                components['scene_common_feature_stats'] = (
                    scene_common_stats)
            components['internal_diag_shape'] = list(internal_diag_shape)
        if return_stats and return_components:
            return seg_logits, stats, components
        if return_stats:
            return seg_logits, stats
        return seg_logits, components

    def slide_inference(self, image, stride, crop_size, return_stats=False, return_components=False):
        """Inference by sliding-window with overlap using PIL cropping."""
        w_img, h_img = image.size
        
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        
        # Initialize accumulators
        preds = torch.zeros((self.num_queries, h_img, w_img), device=self.device)
        count_mat = torch.zeros((1, h_img, w_img), device=self.device)
        crop_stats = [] if return_stats else None
        semantic_preds = (
            torch.zeros((self.num_queries, h_img, w_img), device=self.device)
            if return_components else None)
        instance_preds = (
            torch.zeros((self.num_queries, h_img, w_img), device=self.device)
            if return_components else None)
        coco_sec_prior_preds = (
            torch.zeros((self.num_cls, h_img, w_img), device=self.device)
            if return_components
            and (self.use_coco_sec_fusion or self.dump_coco_sec_stats)
            else None)
        internal_diag_shape = (
            self._internal_selection_diag_shape(h_img, w_img)
            if return_components and self._uses_internal_evidence_diagnostics()
            else None)
        internal_source_preds = {}
        internal_source_counts = {}
        position_bias_feature_stats = []
        scene_common_feature_stats = []
        raw_mask_candidates = (
            [] if return_components
            and (
                self._uses_region_contrastive_readout()
                or self._uses_region_hypothesis_v2()
                or self._uses_candidate_region_quality_diagnostic()
                or self._uses_query_topology_diagnostic()
                or self.dump_region_prompt_identity_stats)
            else None)
        presence_query_sum = (
            torch.zeros(
                self.num_queries,
                device=self.device,
                dtype=torch.float32,
            )
            if return_components
            and (
                self._uses_region_contrastive_readout()
                or internal_diag_shape is not None
                or self._uses_concept_specificity_diagnostic()
                or self._uses_residual_background_modeling()
                or self.dump_prompt_winner_attribution_stats
                or self._uses_cross_image_bank_features()
                or self._uses_ontology_self_verification())
            else None)
        presence_query_views = 0
        
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                
                # Adjust start points to ensure crop size is valid at boundaries
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                
                # Crop via PIL
                crop_img = image.crop((x1, y1, x2, y2))
                
                # Inference on crop
                if return_stats and return_components:
                    crop_seg_logit, crop_stat, crop_components = self._inference_single_view(
                        crop_img,
                        return_stats=True,
                        return_components=True,
                        view_id=f'crop_{h_idx}_{w_idx}',
                        crop_box=[x1, y1, x2, y2],
                    )
                    crop_stats.append(crop_stat)
                elif return_stats:
                    crop_seg_logit, crop_stat = self._inference_single_view(
                        crop_img,
                        return_stats=True,
                        view_id=f'crop_{h_idx}_{w_idx}',
                        crop_box=[x1, y1, x2, y2],
                    )
                    crop_stats.append(crop_stat)
                    crop_components = None
                elif return_components:
                    crop_seg_logit, crop_components = self._inference_single_view(
                        crop_img,
                        return_components=True,
                        view_id=f'crop_{h_idx}_{w_idx}',
                        crop_box=[x1, y1, x2, y2],
                    )
                else:
                    crop_seg_logit = self._inference_single_view(crop_img)
                    crop_components = None
                
                # Accumulate results
                preds[:, y1:y2, x1:x2] += crop_seg_logit
                if return_components:
                    semantic_preds[:, y1:y2, x1:x2] += crop_components['semantic_logits']
                    instance_preds[:, y1:y2, x1:x2] += crop_components['instance_logits']
                    if raw_mask_candidates is not None:
                        raw_mask_candidates.extend(
                            crop_components.get(
                                'raw_mask_candidates', []))
                    if presence_query_sum is not None:
                        crop_presence = crop_components.get(
                            'presence_query_scores')
                        if isinstance(crop_presence, torch.Tensor):
                            presence_query_sum += crop_presence.float()
                            presence_query_views += 1
                    if coco_sec_prior_preds is not None:
                        crop_prior = self._build_coco_sec_prior(
                            crop_components, crop_seg_logit.shape[-2:])
                        if crop_prior is not None:
                            coco_sec_prior_preds[:, y1:y2, x1:x2] += crop_prior
                    if internal_diag_shape is not None:
                        position_bias_feature_stats.extend(
                            crop_components.get(
                                'position_bias_feature_stats', []))
                        scene_common_feature_stats.extend(
                            crop_components.get(
                                'scene_common_feature_stats', []))
                        crop_internal_maps = crop_components.get(
                            'internal_source_maps', {})
                        diag_h, diag_w = internal_diag_shape
                        gx1 = max(0, min(diag_w - 1, int(round(x1 * diag_w / w_img))))
                        gy1 = max(0, min(diag_h - 1, int(round(y1 * diag_h / h_img))))
                        gx2 = max(gx1 + 1, min(diag_w, int(round(x2 * diag_w / w_img))))
                        gy2 = max(gy1 + 1, min(diag_h, int(round(y2 * diag_h / h_img))))
                        target_shape = (gy2 - gy1, gx2 - gx1)
                        for source_name, source_map in crop_internal_maps.items():
                            if not isinstance(source_map, torch.Tensor):
                                continue
                            resized = self._interpolate_float32(
                                source_map.detach().unsqueeze(0),
                                target_shape,
                            ).squeeze(0)
                            if source_name not in internal_source_preds:
                                internal_source_preds[source_name] = torch.zeros(
                                    (self.num_cls, diag_h, diag_w),
                                    device=self.device,
                                    dtype=torch.float32,
                                )
                                internal_source_counts[source_name] = torch.zeros(
                                    (self.num_cls, diag_h, diag_w),
                                    device=self.device,
                                    dtype=torch.float32,
                                )
                            finite = torch.isfinite(resized)
                            internal_source_preds[source_name][:, gy1:gy2, gx1:gx2] += (
                                torch.nan_to_num(resized, nan=0.0))
                            internal_source_counts[source_name][:, gy1:gy2, gx1:gx2] += (
                                finite.float())
                count_mat[:, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0, "Error: Sparse sliding window coverage."
        
        preds = preds / count_mat
        if return_components:
            semantic_preds = semantic_preds / count_mat
            instance_preds = instance_preds / count_mat
            if coco_sec_prior_preds is not None:
                coco_sec_prior_preds = coco_sec_prior_preds / count_mat
                coco_sec_prior_preds = coco_sec_prior_preds.clamp_min(
                    float(self.coco_sec_eps))
                coco_sec_prior_preds = coco_sec_prior_preds / coco_sec_prior_preds.sum(
                    dim=0, keepdim=True).clamp_min(float(self.coco_sec_eps))
        if not return_stats and not return_components:
            return preds
        stats = dict(
            view_id='slide',
            image_size=[w_img, h_img],
            stride=[w_stride, h_stride],
            crop_size=[w_crop, h_crop],
            num_crops=len(crop_stats),
            crop_stats=crop_stats,
        ) if return_stats else None
        components = dict(
            semantic_logits=semantic_preds,
            instance_logits=instance_preds,
        ) if return_components else None
        if return_components and coco_sec_prior_preds is not None:
            components['coco_sec_prior'] = coco_sec_prior_preds
        if return_components and raw_mask_candidates is not None:
            components['raw_mask_candidates'] = raw_mask_candidates
        if (
                return_components
                and presence_query_sum is not None
                and presence_query_views > 0):
            components['presence_query_scores'] = (
                presence_query_sum / float(presence_query_views))
        if return_components and internal_diag_shape is not None:
            internal_source_maps = {}
            for source_name, source_map in internal_source_preds.items():
                count = internal_source_counts[source_name]
                averaged = source_map / count.clamp_min(1.0)
                averaged[count <= 0] = float('nan')
                internal_source_maps[source_name] = averaged
            components['internal_source_maps'] = internal_source_maps
            components['internal_diag_shape'] = list(internal_diag_shape)
            if position_bias_feature_stats:
                components['position_bias_feature_stats'] = (
                    position_bias_feature_stats)
            if scene_common_feature_stats:
                components['scene_common_feature_stats'] = (
                    scene_common_feature_stats)
        if return_stats and return_components:
            return preds, stats, components
        if return_stats:
            return preds, stats
        return preds, components

    def _build_prompt_evidence_stats(self, query_idx, query_word, inference_state,
                                     semantic_component, instance_component,
                                     final_component):
        presence_score = _tensor_scalar(inference_state.get('presence_score'))
        raw_scores_all = inference_state.get('raw_object_score')
        presence_scores_all = inference_state.get('raw_object_score_presence')
        raw_keep_mask = inference_state.get('raw_keep_mask')
        kept_raw_scores = inference_state.get('object_score_raw')
        kept_presence_scores = inference_state.get('object_score_presence')
        masks = inference_state.get('masks_logits')

        semantic_stats = _mask_score_stats(semantic_component)
        instance_stats = _mask_score_stats(instance_component)
        final_stats = _mask_score_stats(final_component, area_threshold=self.prob_thd)

        sem_inst_iou = None
        fusion_semantic_win_ratio = None
        if semantic_component is not None and instance_component is not None:
            semantic_bin = semantic_component > 0.5
            instance_bin = instance_component > 0.5
            sem_inst_iou = _binary_iou(semantic_bin, instance_bin)
            fusion_semantic_win_ratio = _tensor_scalar(
                (semantic_component >= instance_component).float().mean())

        instance_area_union_05 = 0.0
        instance_area_max_05 = 0.0
        if masks is not None and masks.numel() > 0:
            mask_bin = masks.squeeze(1) > 0.5
            instance_area_union_05 = _tensor_scalar(mask_bin.any(dim=0).float().mean())
            instance_area_max_05 = _tensor_scalar(mask_bin.float().flatten(1).mean(dim=1).max())

        raw_candidate_count = int(raw_scores_all.numel()) if raw_scores_all is not None else 0
        kept_count = int(kept_raw_scores.numel()) if kept_raw_scores is not None else 0
        raw_keep_count = int(raw_keep_mask.sum().item()) if raw_keep_mask is not None else kept_count

        return dict(
            query_index=int(query_idx),
            class_index=int(self.query_idx[query_idx].item()),
            class_name=self.class_names[int(self.query_idx[query_idx].item())],
            prompt=query_word,
            presence_score=presence_score,
            raw_candidate_count=raw_candidate_count,
            raw_keep_count=raw_keep_count,
            kept_instance_count=kept_count,
            raw_keep_ratio=_safe_div(raw_keep_count, raw_candidate_count),
            raw_score_max=_tensor_reduce(raw_scores_all, 'max'),
            raw_score_mean=_tensor_reduce(raw_scores_all, 'mean'),
            presence_score_max=_tensor_reduce(presence_scores_all, 'max'),
            presence_score_mean=_tensor_reduce(presence_scores_all, 'mean'),
            kept_raw_score_max=_tensor_reduce(kept_raw_scores, 'max'),
            kept_raw_score_mean=_tensor_reduce(kept_raw_scores, 'mean'),
            kept_presence_score_max=_tensor_reduce(kept_presence_scores, 'max'),
            kept_presence_score_mean=_tensor_reduce(kept_presence_scores, 'mean'),
            instance_area_union_05=instance_area_union_05,
            instance_area_max_05=instance_area_max_05,
            semantic=semantic_stats,
            instance=instance_stats,
            final=final_stats,
            sem_inst_iou_05=sem_inst_iou,
            fusion_semantic_win_ratio=fusion_semantic_win_ratio,
        )

    def _build_class_evidence_stats(self, seg_logits, seg_pred, data_sample):
        max_vals = seg_logits.max(0)[0]
        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)

        valid_mask = torch.ones_like(seg_pred, dtype=torch.bool, device=self.device)
        if gt_data is not None:
            valid_mask = gt_data != 255
        valid_count = int(valid_mask.sum().item())

        class_stats = []
        for class_idx in range(self.num_cls):
            class_logit = seg_logits[class_idx]
            pred_mask = (seg_pred == class_idx) & valid_mask
            stat = dict(
                class_index=class_idx,
                class_name=self.class_names[class_idx],
                logit_max=_tensor_reduce(class_logit, 'max'),
                logit_mean=_tensor_reduce(class_logit, 'mean'),
                logit_area_prob_thd=_tensor_scalar((class_logit >= self.prob_thd).float().mean()),
                pred_area_ratio=_safe_div(int(pred_mask.sum().item()), valid_count),
                mean_winning_score=_tensor_reduce(max_vals[pred_mask], 'mean') if pred_mask.any() else None,
            )
            if gt_data is not None:
                gt_mask = (gt_data == class_idx) & valid_mask
                intersection = int((pred_mask & gt_mask).sum().item())
                pred_area = int(pred_mask.sum().item())
                gt_area = int(gt_mask.sum().item())
                fp = pred_area - intersection
                fn = gt_area - intersection
                union = pred_area + gt_area - intersection
                stat.update(dict(
                    gt_area_ratio=_safe_div(gt_area, valid_count),
                    tp=intersection,
                    fp=fp,
                    fn=fn,
                    iou=_safe_div(intersection, union),
                    precision=_safe_div(intersection, pred_area),
                    recall=_safe_div(intersection, gt_area),
                ))
            class_stats.append(stat)
        return class_stats

    def _write_evidence_stats(self, record):
        if not self.dump_evidence_stats:
            return
        if self._evidence_stats_file is None:
            path = self.evidence_stats_path or './work_dirs/evidence_stats/evidence_stats.jsonl'
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._evidence_stats_file = open(path, 'a', buffering=1)
        self._evidence_stats_file.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _build_competition_stats(self, seg_logits, seg_pred, data_sample, components):
        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)
        if gt_data is None:
            return None

        valid_mask = gt_data != 255
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return []

        top2_vals, top2_idx = torch.topk(seg_logits, k=min(2, self.num_cls), dim=0)
        top1_score = top2_vals[0]
        top1_idx = top2_idx[0]
        if self.num_cls > 1:
            top2_score = top2_vals[1]
            top2_class = top2_idx[1]
        else:
            top2_score = torch.zeros_like(top1_score)
            top2_class = torch.zeros_like(top1_idx)
        margin = top1_score - top2_score

        semantic_logits = components.get('semantic_logits') if components is not None else None
        instance_logits = components.get('instance_logits') if components is not None else None

        records = []
        for gt_class in range(self.num_cls):
            gt_mask = (gt_data == gt_class) & valid_mask
            gt_area = int(gt_mask.sum().item())
            if gt_area == 0:
                continue

            target_scores = seg_logits[gt_class]
            correct_mask = gt_mask & (seg_pred == gt_class)
            fn_mask = gt_mask & (seg_pred != gt_class)
            correct_count = int(correct_mask.sum().item())
            fn_count = int(fn_mask.sum().item())

            record = dict(
                gt_class_index=gt_class,
                gt_class_name=self.class_names[gt_class],
                gt_pixels=gt_area,
                gt_area_ratio=_safe_div(gt_area, valid_count),
                correct_pixels=correct_count,
                fn_pixels=fn_count,
                recall=_safe_div(correct_count, gt_area),
                target_final_mean_on_gt=_masked_mean(target_scores, gt_mask),
                target_final_mean_on_correct=_masked_mean(target_scores, correct_mask),
                target_final_mean_on_fn=_masked_mean(target_scores, fn_mask),
                top1_score_mean_on_fn=_masked_mean(top1_score, fn_mask),
                top2_score_mean_on_fn=_masked_mean(top2_score, fn_mask),
                top1_top2_margin_mean_on_fn=_masked_mean(margin, fn_mask),
                target_gap_mean_on_fn=_masked_mean(top1_score - target_scores, fn_mask),
                threshold_reject_ratio_on_fn=_safe_div(
                    int((fn_mask & (top1_score < self.prob_thd)).sum().item()),
                    fn_count),
                target_semantic_mean_on_fn=_masked_mean(semantic_logits[gt_class], fn_mask)
                    if semantic_logits is not None else None,
                target_instance_mean_on_fn=_masked_mean(instance_logits[gt_class], fn_mask)
                    if instance_logits is not None else None,
                competitors=[],
            )

            for pred_class in range(self.num_cls):
                if pred_class == gt_class:
                    continue
                comp_mask = gt_mask & (seg_pred == pred_class)
                comp_pixels = int(comp_mask.sum().item())
                if comp_pixels == 0:
                    continue
                comp_scores = seg_logits[pred_class]
                comp_record = dict(
                    pred_class_index=pred_class,
                    pred_class_name=self.class_names[pred_class],
                    pixels=comp_pixels,
                    pixel_ratio_in_gt=_safe_div(comp_pixels, gt_area),
                    target_final_mean=_masked_mean(target_scores, comp_mask),
                    competitor_final_mean=_masked_mean(comp_scores, comp_mask),
                    target_gap_mean=_masked_mean(comp_scores - target_scores, comp_mask),
                    top1_top2_margin_mean=_masked_mean(margin, comp_mask),
                    top2_class_mode=_mode_class(top2_class, comp_mask),
                    threshold_reject_ratio=_safe_div(
                        int((comp_mask & (top1_score < self.prob_thd)).sum().item()),
                        comp_pixels),
                    target_semantic_mean=_masked_mean(semantic_logits[gt_class], comp_mask)
                        if semantic_logits is not None else None,
                    competitor_semantic_mean=_masked_mean(semantic_logits[pred_class], comp_mask)
                        if semantic_logits is not None else None,
                    target_instance_mean=_masked_mean(instance_logits[gt_class], comp_mask)
                        if instance_logits is not None else None,
                    competitor_instance_mean=_masked_mean(instance_logits[pred_class], comp_mask)
                        if instance_logits is not None else None,
                )
                record['competitors'].append(comp_record)

            record['competitors'].sort(key=lambda item: item['pixels'], reverse=True)
            records.append(record)
        return records

    def _write_competition_stats(self, record):
        if not self.dump_competition_stats:
            return
        if self._competition_stats_file is None:
            path = self.competition_stats_path or './work_dirs/evidence_stats/competition_stats.jsonl'
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._competition_stats_file = open(path, 'a', buffering=1)
        self._competition_stats_file.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _build_oracle_stats(self, seg_logits, seg_pred, data_sample, components):
        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)
        if gt_data is None or components is None:
            return None

        valid_mask = gt_data != 255
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return dict(valid_pixels=0, heads=[], class_stats=[], pair_stats=[])

        topk = min(max(int(self.oracle_topk), 1), self.num_cls)
        score_maps = dict(
            final=seg_logits,
            semantic=components.get('semantic_logits'),
            instance=components.get('instance_logits'),
        )
        score_maps = {
            name: value for name, value in score_maps.items()
            if value is not None
        }

        head_topk = {}
        head_top1 = {}
        head_contains = {}
        head_stats = []
        gt_expanded = gt_data.unsqueeze(0)
        for name, scores in score_maps.items():
            topk_idx = torch.topk(scores, k=topk, dim=0).indices
            head_topk[name] = topk_idx
            head_top1[name] = topk_idx[0]
            contains_by_k = []
            for k_idx in range(1, topk + 1):
                contains = (topk_idx[:k_idx] == gt_expanded).any(dim=0) & valid_mask
                contains_by_k.append(contains)
            head_contains[name] = contains_by_k
            top1_correct = (topk_idx[0] == gt_data) & valid_mask
            stat = dict(
                head=name,
                top1_correct_pixels=int(top1_correct.sum().item()),
                top1_recall=_safe_div(int(top1_correct.sum().item()), valid_count),
            )
            for k_idx, contains in enumerate(contains_by_k, start=1):
                correct = int(contains.sum().item())
                stat[f'top{k_idx}_contains_gt_pixels'] = correct
                stat[f'top{k_idx}_contains_gt_ratio'] = _safe_div(correct, valid_count)
            if name == 'final':
                rejected = valid_mask & (seg_logits.max(0)[0] < self.prob_thd)
                threshold_correct = (seg_pred == gt_data) & valid_mask
                stat.update(dict(
                    threshold_correct_pixels=int(threshold_correct.sum().item()),
                    threshold_recall=_safe_div(int(threshold_correct.sum().item()), valid_count),
                    threshold_reject_pixels=int(rejected.sum().item()),
                    threshold_reject_ratio=_safe_div(int(rejected.sum().item()), valid_count),
                ))
            head_stats.append(stat)

        any_head_by_k = []
        for k_idx in range(topk):
            contains = torch.zeros_like(valid_mask, dtype=torch.bool)
            for per_head in head_contains.values():
                contains |= per_head[k_idx]
            any_head_by_k.append(contains & valid_mask)

        any_head_top1 = torch.zeros_like(valid_mask, dtype=torch.bool)
        for top1_idx in head_top1.values():
            any_head_top1 |= (top1_idx == gt_data) & valid_mask

        oracle_stats = dict(
            valid_pixels=valid_count,
            oracle_topk=topk,
            any_head_top1_correct_pixels=int(any_head_top1.sum().item()),
            any_head_top1_recall=_safe_div(int(any_head_top1.sum().item()), valid_count),
        )
        for k_idx, contains in enumerate(any_head_by_k, start=1):
            correct = int(contains.sum().item())
            oracle_stats[f'any_head_top{k_idx}_contains_gt_pixels'] = correct
            oracle_stats[f'any_head_top{k_idx}_contains_gt_ratio'] = _safe_div(correct, valid_count)

        class_stats = []
        final_rejected = valid_mask & (seg_logits.max(0)[0] < self.prob_thd)
        final_threshold_correct = (seg_pred == gt_data) & valid_mask
        for class_idx in range(self.num_cls):
            gt_mask = (gt_data == class_idx) & valid_mask
            gt_pixels = int(gt_mask.sum().item())
            if gt_pixels == 0:
                continue
            class_record = dict(
                class_index=class_idx,
                class_name=self.class_names[class_idx],
                gt_pixels=gt_pixels,
                final_threshold_correct_pixels=int((final_threshold_correct & gt_mask).sum().item()),
                final_threshold_recall=_safe_div(
                    int((final_threshold_correct & gt_mask).sum().item()), gt_pixels),
                final_threshold_reject_pixels=int((final_rejected & gt_mask).sum().item()),
                final_threshold_reject_ratio=_safe_div(
                    int((final_rejected & gt_mask).sum().item()), gt_pixels),
                any_head_top1_correct_pixels=int((any_head_top1 & gt_mask).sum().item()),
                any_head_top1_recall=_safe_div(
                    int((any_head_top1 & gt_mask).sum().item()), gt_pixels),
            )
            for k_idx, contains in enumerate(any_head_by_k, start=1):
                correct = int((contains & gt_mask).sum().item())
                class_record[f'any_head_top{k_idx}_contains_gt_pixels'] = correct
                class_record[f'any_head_top{k_idx}_contains_gt_ratio'] = _safe_div(correct, gt_pixels)
            for name, contains_by_k in head_contains.items():
                top1_correct = (head_top1[name] == gt_data) & gt_mask
                class_record[f'{name}_top1_correct_pixels'] = int(top1_correct.sum().item())
                class_record[f'{name}_top1_recall'] = _safe_div(
                    int(top1_correct.sum().item()), gt_pixels)
                for k_idx, contains in enumerate(contains_by_k, start=1):
                    correct = int((contains & gt_mask).sum().item())
                    class_record[f'{name}_top{k_idx}_contains_gt_pixels'] = correct
                    class_record[f'{name}_top{k_idx}_contains_gt_ratio'] = _safe_div(correct, gt_pixels)
            class_stats.append(class_record)

        pair_stats = []
        for gt_class in range(self.num_cls):
            gt_mask = (gt_data == gt_class) & valid_mask
            gt_pixels = int(gt_mask.sum().item())
            if gt_pixels == 0:
                continue
            for pred_class in range(self.num_cls):
                if pred_class == gt_class:
                    continue
                pair_mask = gt_mask & (seg_pred == pred_class)
                pixels = int(pair_mask.sum().item())
                if pixels == 0:
                    continue
                pair_record = dict(
                    gt_class_index=gt_class,
                    gt_class_name=self.class_names[gt_class],
                    pred_class_index=pred_class,
                    pred_class_name=self.class_names[pred_class],
                    pixels=pixels,
                    pixel_ratio_in_gt=_safe_div(pixels, gt_pixels),
                    any_head_top1_recoverable_pixels=int((any_head_top1 & pair_mask).sum().item()),
                    any_head_top1_recoverable_ratio=_safe_div(
                        int((any_head_top1 & pair_mask).sum().item()), pixels),
                )
                for k_idx, contains in enumerate(any_head_by_k, start=1):
                    recoverable = int((contains & pair_mask).sum().item())
                    pair_record[f'any_head_top{k_idx}_recoverable_pixels'] = recoverable
                    pair_record[f'any_head_top{k_idx}_recoverable_ratio'] = _safe_div(recoverable, pixels)
                for name, contains_by_k in head_contains.items():
                    top1_recoverable = (head_top1[name] == gt_data) & pair_mask
                    pair_record[f'{name}_top1_recoverable_pixels'] = int(top1_recoverable.sum().item())
                    pair_record[f'{name}_top1_recoverable_ratio'] = _safe_div(
                        int(top1_recoverable.sum().item()), pixels)
                    pair_record[f'{name}_top1_mode_on_pair'] = _mode_class(head_top1[name], pair_mask)
                    for k_idx, contains in enumerate(contains_by_k, start=1):
                        recoverable = int((contains & pair_mask).sum().item())
                        pair_record[f'{name}_top{k_idx}_recoverable_pixels'] = recoverable
                        pair_record[f'{name}_top{k_idx}_recoverable_ratio'] = _safe_div(recoverable, pixels)
                pair_stats.append(pair_record)
        pair_stats.sort(key=lambda item: item['pixels'], reverse=True)

        return dict(
            valid_pixels=valid_count,
            oracle_topk=topk,
            heads=head_stats,
            oracle=oracle_stats,
            class_stats=class_stats,
            pair_stats=pair_stats,
        )

    def _write_oracle_stats(self, record):
        if not self.dump_oracle_stats:
            return
        if self._oracle_stats_file is None:
            path = self.oracle_stats_path or './work_dirs/evidence_stats/oracle_stats.jsonl'
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._oracle_stats_file = open(path, 'a', buffering=1)
        self._oracle_stats_file.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _build_topk_rank_maps(self, score_maps, topk):
        rank_maps = {}
        topk_indices = {}
        for name, scores in score_maps.items():
            topk_idx = torch.topk(scores, k=topk, dim=0).indices
            topk_indices[name] = topk_idx
            rank_map = torch.zeros_like(scores)
            _, h, w = scores.shape
            for rank in range(topk):
                weight = float(topk - rank) / float(topk)
                src = torch.full((1, h, w), weight, device=scores.device, dtype=scores.dtype)
                rank_map.scatter_(0, topk_idx[rank].unsqueeze(0), src)
            rank_maps[name] = rank_map
        return rank_maps, topk_indices

    def _build_topk_candidate_verifier_logits(self, seg_logits, components):
        if components is None or self.num_cls <= 1:
            return seg_logits, None

        semantic_logits = components.get('semantic_logits')
        instance_logits = components.get('instance_logits')
        if semantic_logits is None or instance_logits is None:
            return seg_logits, None

        topk = min(max(int(self.topk_verifier_k), 1), self.num_cls)
        score_maps = dict(
            final=seg_logits,
            semantic=semantic_logits,
            instance=instance_logits,
        )
        rank_maps, topk_indices = self._build_topk_rank_maps(score_maps, topk)

        final_rank = rank_maps['final']
        semantic_rank = rank_maps['semantic']
        instance_rank = rank_maps['instance']
        aux_rank = torch.maximum(semantic_rank, instance_rank)
        vote_count = (
            (final_rank > 0).float()
            + (semantic_rank > 0).float()
            + (instance_rank > 0).float()
        )

        kernel = max(int(self.topk_verifier_local_kernel), 1)
        if kernel % 2 == 0:
            kernel += 1
        local_aux_rank = F.avg_pool2d(
            aux_rank.unsqueeze(0),
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        ).squeeze(0)

        candidate_mask = aux_rank >= self.topk_verifier_min_aux_rank
        if self.topk_verifier_require_final_candidate:
            candidate_mask = candidate_mask & (final_rank > 0)

        final_top2_vals, final_top2_idx = torch.topk(seg_logits, k=min(2, self.num_cls), dim=0)
        final_top1_score = final_top2_vals[0]
        final_top1_idx = final_top2_idx[0]
        if self.num_cls > 1:
            final_top2_score = final_top2_vals[1]
        else:
            final_top2_score = torch.zeros_like(final_top1_score)
        final_margin = final_top1_score - final_top2_score
        semantic_top1_idx = topk_indices['semantic'][0]
        instance_top1_idx = topk_indices['instance'][0]
        head_disagree = (
            (semantic_top1_idx != final_top1_idx)
            | (instance_top1_idx != final_top1_idx)
        )
        threshold_reject = final_top1_score < self.prob_thd
        low_margin = final_margin <= self.topk_verifier_low_margin
        confident_margin = final_margin >= self.topk_verifier_confident_margin

        mode = str(self.topk_verifier_apply_mode).lower()
        if mode == 'all':
            apply_mask = torch.ones_like(final_top1_score, dtype=torch.bool)
        elif mode == 'uncertain':
            apply_mask = threshold_reject | low_margin
        elif mode in ('legacy_uncertain', 'uncertain_with_disagree'):
            apply_mask = threshold_reject | low_margin | head_disagree
        elif mode == 'confident_disagree':
            apply_mask = (~threshold_reject) & confident_margin & head_disagree
        elif mode == 'head_disagree':
            apply_mask = head_disagree
        elif mode == 'threshold_reject':
            apply_mask = threshold_reject
        elif mode == 'low_margin':
            apply_mask = low_margin
        else:
            raise ValueError(
                "topk_verifier_apply_mode must be one of 'all', 'uncertain', "
                "'legacy_uncertain', 'uncertain_with_disagree', "
                "'confident_disagree', 'head_disagree', 'threshold_reject', "
                "or 'low_margin', "
                f"but got {self.topk_verifier_apply_mode!r}")

        vote_norm = vote_count / float(len(rank_maps))
        verifier_bonus = (
            self.topk_verifier_aux_rank_weight * aux_rank
            + self.topk_verifier_vote_weight * vote_norm
            + self.topk_verifier_local_weight * local_aux_rank
            + self.topk_verifier_final_weight * final_rank
        )
        verifier_bonus = verifier_bonus * candidate_mask.float()
        verifier_bonus = verifier_bonus * apply_mask.unsqueeze(0).float()

        verified_logits = seg_logits + verifier_bonus
        context = dict(
            topk=topk,
            topk_indices=topk_indices,
            rank_maps=rank_maps,
            candidate_mask=candidate_mask,
            apply_mask=apply_mask,
            threshold_reject=threshold_reject,
            low_margin=low_margin,
            confident_margin=confident_margin,
            head_disagree=head_disagree,
            final_margin=final_margin,
            final_top1_score=final_top1_score,
            verifier_bonus=verifier_bonus,
            verified_logits=verified_logits,
        )
        return verified_logits, context

    def _build_topk_verifier_stats(self, base_logits, verifier_logits, base_pred,
                                   verifier_pred, data_sample, components, context):
        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)
        if gt_data is None or context is None or components is None:
            return None

        valid_mask = gt_data != 255
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return dict(valid_pixels=0, class_stats=[], pair_stats=[])

        changed = (base_pred != verifier_pred) & valid_mask
        base_correct = (base_pred == gt_data) & valid_mask
        verifier_correct = (verifier_pred == gt_data) & valid_mask
        improved = (~base_correct) & verifier_correct
        harmed = base_correct & (~verifier_correct)

        apply_mask = context['apply_mask'] & valid_mask
        candidate_mask = context['candidate_mask']
        rank_maps = context['rank_maps']
        gt_gather_idx = gt_data.clamp(min=0, max=self.num_cls - 1)
        gt_rank_hits = {}
        for head, rank_map in rank_maps.items():
            gt_rank_hits[head] = (_gather_class_map(rank_map, gt_gather_idx) > 0) & valid_mask
        any_head_candidate = torch.zeros_like(valid_mask, dtype=torch.bool)
        for hit in gt_rank_hits.values():
            any_head_candidate |= hit
        verifier_candidate_hit = (
            _gather_class_map(candidate_mask.float(), gt_gather_idx) > 0
        ) & valid_mask
        bonus_for_gt = _gather_class_map(context['verifier_bonus'], gt_gather_idx)

        overall = dict(
            valid_pixels=valid_count,
            base_correct_pixels=int(base_correct.sum().item()),
            base_correct_ratio=_safe_div(int(base_correct.sum().item()), valid_count),
            verifier_correct_pixels=int(verifier_correct.sum().item()),
            verifier_correct_ratio=_safe_div(int(verifier_correct.sum().item()), valid_count),
            changed_pixels=int(changed.sum().item()),
            changed_ratio=_safe_div(int(changed.sum().item()), valid_count),
            improved_pixels=int(improved.sum().item()),
            improved_ratio=_safe_div(int(improved.sum().item()), valid_count),
            harmed_pixels=int(harmed.sum().item()),
            harmed_ratio=_safe_div(int(harmed.sum().item()), valid_count),
            net_improved_pixels=int(improved.sum().item()) - int(harmed.sum().item()),
            apply_pixels=int(apply_mask.sum().item()),
            apply_ratio=_safe_div(int(apply_mask.sum().item()), valid_count),
            threshold_reject_pixels=int((context['threshold_reject'] & valid_mask).sum().item()),
            low_margin_pixels=int((context['low_margin'] & valid_mask).sum().item()),
            confident_margin_pixels=int((context['confident_margin'] & valid_mask).sum().item()),
            head_disagree_pixels=int((context['head_disagree'] & valid_mask).sum().item()),
        )

        regime_defs = [
            ('threshold_reject', context['threshold_reject']),
            ('low_margin_non_reject', context['low_margin'] & (~context['threshold_reject'])),
            ('confident_head_agree',
             context['confident_margin'] & (~context['threshold_reject']) & (~context['head_disagree'])),
            ('confident_head_disagree',
             context['confident_margin'] & (~context['threshold_reject']) & context['head_disagree']),
            ('mid_margin_head_agree',
             (~context['threshold_reject']) & (~context['low_margin'])
             & (~context['confident_margin']) & (~context['head_disagree'])),
            ('mid_margin_head_disagree',
             (~context['threshold_reject']) & (~context['low_margin'])
             & (~context['confident_margin']) & context['head_disagree']),
        ]
        regime_stats = []
        for regime_name, regime_mask_raw in regime_defs:
            regime_mask = regime_mask_raw & valid_mask
            pixels = int(regime_mask.sum().item())
            if pixels == 0:
                continue
            regime_improved = improved & regime_mask
            regime_harmed = harmed & regime_mask
            regime_stats.append(dict(
                regime=regime_name,
                pixels=pixels,
                pixel_ratio=_safe_div(pixels, valid_count),
                base_correct_pixels=int((base_correct & regime_mask).sum().item()),
                base_correct_ratio=_safe_div(int((base_correct & regime_mask).sum().item()), pixels),
                verifier_correct_pixels=int((verifier_correct & regime_mask).sum().item()),
                verifier_correct_ratio=_safe_div(int((verifier_correct & regime_mask).sum().item()), pixels),
                changed_pixels=int((changed & regime_mask).sum().item()),
                changed_ratio=_safe_div(int((changed & regime_mask).sum().item()), pixels),
                improved_pixels=int(regime_improved.sum().item()),
                improved_ratio=_safe_div(int(regime_improved.sum().item()), pixels),
                harmed_pixels=int(regime_harmed.sum().item()),
                harmed_ratio=_safe_div(int(regime_harmed.sum().item()), pixels),
                net_improved_pixels=int(regime_improved.sum().item()) - int(regime_harmed.sum().item()),
                any_head_candidate_pixels=int((any_head_candidate & regime_mask).sum().item()),
                any_head_candidate_ratio=_safe_div(int((any_head_candidate & regime_mask).sum().item()), pixels),
                verifier_candidate_pixels=int((verifier_candidate_hit & regime_mask).sum().item()),
                verifier_candidate_ratio=_safe_div(int((verifier_candidate_hit & regime_mask).sum().item()), pixels),
                mean_final_margin=_masked_mean(context['final_margin'], regime_mask),
                mean_final_top1_score=_masked_mean(context['final_top1_score'], regime_mask),
                mean_gt_bonus=_masked_mean(bonus_for_gt, regime_mask),
            ))

        class_stats = []
        for class_idx in range(self.num_cls):
            gt_mask = (gt_data == class_idx) & valid_mask
            gt_pixels = int(gt_mask.sum().item())
            if gt_pixels == 0:
                continue
            class_improved = improved & gt_mask
            class_harmed = harmed & gt_mask
            class_changed = changed & gt_mask
            record = dict(
                class_index=class_idx,
                class_name=self.class_names[class_idx],
                gt_pixels=gt_pixels,
                base_correct_pixels=int((base_correct & gt_mask).sum().item()),
                base_recall=_safe_div(int((base_correct & gt_mask).sum().item()), gt_pixels),
                verifier_correct_pixels=int((verifier_correct & gt_mask).sum().item()),
                verifier_recall=_safe_div(int((verifier_correct & gt_mask).sum().item()), gt_pixels),
                changed_pixels=int(class_changed.sum().item()),
                changed_ratio=_safe_div(int(class_changed.sum().item()), gt_pixels),
                improved_pixels=int(class_improved.sum().item()),
                improved_ratio=_safe_div(int(class_improved.sum().item()), gt_pixels),
                harmed_pixels=int(class_harmed.sum().item()),
                harmed_ratio=_safe_div(int(class_harmed.sum().item()), gt_pixels),
                net_improved_pixels=int(class_improved.sum().item()) - int(class_harmed.sum().item()),
                any_head_candidate_pixels=int((any_head_candidate & gt_mask).sum().item()),
                any_head_candidate_ratio=_safe_div(int((any_head_candidate & gt_mask).sum().item()), gt_pixels),
                verifier_candidate_pixels=int((verifier_candidate_hit & gt_mask).sum().item()),
                verifier_candidate_ratio=_safe_div(int((verifier_candidate_hit & gt_mask).sum().item()), gt_pixels),
                mean_gt_bonus=_masked_mean(bonus_for_gt, gt_mask),
            )
            class_stats.append(record)

        pair_stats = []
        for gt_class in range(self.num_cls):
            gt_mask = (gt_data == gt_class) & valid_mask
            gt_pixels = int(gt_mask.sum().item())
            if gt_pixels == 0:
                continue
            for pred_class in range(self.num_cls):
                if pred_class == gt_class:
                    continue
                pair_mask = gt_mask & (base_pred == pred_class)
                pixels = int(pair_mask.sum().item())
                if pixels == 0:
                    continue
                recovered = pair_mask & verifier_correct
                same_wrong = pair_mask & (verifier_pred == pred_class)
                other_wrong = pair_mask & (~verifier_correct) & (verifier_pred != pred_class)
                pair_record = dict(
                    gt_class_index=gt_class,
                    gt_class_name=self.class_names[gt_class],
                    base_pred_class_index=pred_class,
                    base_pred_class_name=self.class_names[pred_class],
                    pixels=pixels,
                    pixel_ratio_in_gt=_safe_div(pixels, gt_pixels),
                    recovered_pixels=int(recovered.sum().item()),
                    recovered_ratio=_safe_div(int(recovered.sum().item()), pixels),
                    same_wrong_pixels=int(same_wrong.sum().item()),
                    same_wrong_ratio=_safe_div(int(same_wrong.sum().item()), pixels),
                    other_wrong_pixels=int(other_wrong.sum().item()),
                    other_wrong_ratio=_safe_div(int(other_wrong.sum().item()), pixels),
                    verifier_candidate_pixels=int((verifier_candidate_hit & pair_mask).sum().item()),
                    verifier_candidate_ratio=_safe_div(
                        int((verifier_candidate_hit & pair_mask).sum().item()), pixels),
                    any_head_candidate_pixels=int((any_head_candidate & pair_mask).sum().item()),
                    any_head_candidate_ratio=_safe_div(
                        int((any_head_candidate & pair_mask).sum().item()), pixels),
                    mean_gt_bonus=_masked_mean(bonus_for_gt, pair_mask),
                )
                pair_stats.append(pair_record)
        pair_stats.sort(key=lambda item: item['pixels'], reverse=True)

        return dict(
            valid_pixels=valid_count,
            topk=int(context['topk']),
            apply_mode=self.topk_verifier_apply_mode,
            overall=overall,
            regime_stats=regime_stats,
            class_stats=class_stats,
            pair_stats=pair_stats,
        )

    def _write_topk_verifier_stats(self, record):
        if not self.dump_topk_verifier_stats:
            return
        if self._topk_verifier_stats_file is None:
            path = self.topk_verifier_stats_path or './work_dirs/evidence_stats/topk_verifier_stats.jsonl'
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._topk_verifier_stats_file = open(path, 'a', buffering=1)
        self._topk_verifier_stats_file.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _build_error_rank_stats(self, base_logits, base_pred, data_sample, components):
        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)
        if gt_data is None or components is None:
            return None

        valid_mask = gt_data != 255
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return dict(valid_pixels=0, error_pixels=0, topk=0,
                        overall=None, regime_stats=[], class_stats=[], pair_stats=[])

        semantic_logits = components.get('semantic_logits')
        instance_logits = components.get('instance_logits')
        if semantic_logits is None or instance_logits is None:
            return None

        base_wrong = (base_pred != gt_data) & valid_mask
        error_count = int(base_wrong.sum().item())
        topk = min(max(int(self.error_rank_topk), 1), self.num_cls)
        score_maps = dict(
            final=base_logits,
            semantic=semantic_logits,
            instance=instance_logits,
        )
        _, topk_indices = self._build_topk_rank_maps(score_maps, topk)

        final_top2_vals, final_top2_idx = torch.topk(base_logits, k=min(2, self.num_cls), dim=0)
        final_top1_score = final_top2_vals[0]
        final_top1_idx = final_top2_idx[0]
        if self.num_cls > 1:
            final_top2_score = final_top2_vals[1]
        else:
            final_top2_score = torch.zeros_like(final_top1_score)
        final_margin = final_top1_score - final_top2_score
        gt_gather_idx = gt_data.clamp(min=0, max=self.num_cls - 1)
        final_gt_score = _gather_class_map(base_logits, gt_gather_idx)
        top1_gt_gap = final_top1_score - final_gt_score

        semantic_top1_idx = topk_indices['semantic'][0]
        instance_top1_idx = topk_indices['instance'][0]
        threshold_reject = final_top1_score < self.prob_thd
        low_margin = final_margin <= self.topk_verifier_low_margin
        confident_margin = final_margin >= self.topk_verifier_confident_margin
        head_disagree = (
            (semantic_top1_idx != final_top1_idx)
            | (instance_top1_idx != final_top1_idx)
        )

        head_hits = {}
        for head, indices in topk_indices.items():
            rank_hits = []
            contains = torch.zeros_like(valid_mask, dtype=torch.bool)
            for rank in range(topk):
                hit = (indices[rank] == gt_gather_idx) & valid_mask
                rank_hits.append(hit)
                contains |= hit
            head_hits[head] = dict(rank_hits=rank_hits, contains=contains)
        any_head_contains = torch.zeros_like(valid_mask, dtype=torch.bool)
        for hit_info in head_hits.values():
            any_head_contains |= hit_info['contains']

        def build_subset(name, mask, denominator):
            pixels = int(mask.sum().item())
            if pixels == 0:
                return None
            row = dict(
                name=name,
                pixels=pixels,
                pixel_ratio=_safe_div(pixels, denominator),
                threshold_reject_pixels=int((threshold_reject & mask).sum().item()),
                threshold_reject_ratio=_safe_div(int((threshold_reject & mask).sum().item()), pixels),
                low_margin_pixels=int((low_margin & mask).sum().item()),
                low_margin_ratio=_safe_div(int((low_margin & mask).sum().item()), pixels),
                confident_margin_pixels=int((confident_margin & mask).sum().item()),
                confident_margin_ratio=_safe_div(int((confident_margin & mask).sum().item()), pixels),
                head_disagree_pixels=int((head_disagree & mask).sum().item()),
                head_disagree_ratio=_safe_div(int((head_disagree & mask).sum().item()), pixels),
                final_top1_is_gt_pixels=int(((final_top1_idx == gt_gather_idx) & mask).sum().item()),
                final_top1_is_gt_ratio=_safe_div(
                    int(((final_top1_idx == gt_gather_idx) & mask).sum().item()), pixels),
                any_head_topk_contains_gt_pixels=int((any_head_contains & mask).sum().item()),
                any_head_topk_contains_gt_ratio=_safe_div(
                    int((any_head_contains & mask).sum().item()), pixels),
                mean_final_top1_score=_masked_mean(final_top1_score, mask),
                mean_final_gt_score=_masked_mean(final_gt_score, mask),
                mean_top1_gt_gap=_masked_mean(top1_gt_gap, mask),
                mean_final_margin=_masked_mean(final_margin, mask),
            )

            gap_defs = [
                ('gap_le_0', top1_gt_gap <= 0.0),
                ('gap_0_0p02', (top1_gt_gap > 0.0) & (top1_gt_gap <= 0.02)),
                ('gap_0p02_0p05', (top1_gt_gap > 0.02) & (top1_gt_gap <= 0.05)),
                ('gap_0p05_0p10', (top1_gt_gap > 0.05) & (top1_gt_gap <= 0.10)),
                ('gap_0p10_0p20', (top1_gt_gap > 0.10) & (top1_gt_gap <= 0.20)),
                ('gap_0p20_0p50', (top1_gt_gap > 0.20) & (top1_gt_gap <= 0.50)),
                ('gap_gt_0p50', top1_gt_gap > 0.50),
            ]
            for bucket_name, bucket_mask_raw in gap_defs:
                bucket_pixels = int((bucket_mask_raw & mask).sum().item())
                row[f'{bucket_name}_pixels'] = bucket_pixels
                row[f'{bucket_name}_ratio'] = _safe_div(bucket_pixels, pixels)

            margin_defs = [
                ('margin_le_0p02', final_margin <= 0.02),
                ('margin_0p02_0p05', (final_margin > 0.02) & (final_margin <= 0.05)),
                ('margin_0p05_0p10', (final_margin > 0.05) & (final_margin <= 0.10)),
                ('margin_0p10_0p25', (final_margin > 0.10) & (final_margin <= 0.25)),
                ('margin_gt_0p25', final_margin > 0.25),
            ]
            for bucket_name, bucket_mask_raw in margin_defs:
                bucket_pixels = int((bucket_mask_raw & mask).sum().item())
                row[f'{bucket_name}_pixels'] = bucket_pixels
                row[f'{bucket_name}_ratio'] = _safe_div(bucket_pixels, pixels)

            for head, hit_info in head_hits.items():
                contains_pixels = int((hit_info['contains'] & mask).sum().item())
                row[f'{head}_topk_contains_gt_pixels'] = contains_pixels
                row[f'{head}_topk_contains_gt_ratio'] = _safe_div(contains_pixels, pixels)
                ranked_pixels = 0
                for rank, hit in enumerate(hit_info['rank_hits'], start=1):
                    rank_pixels = int((hit & mask).sum().item())
                    ranked_pixels += rank_pixels
                    row[f'{head}_gt_rank{rank}_pixels'] = rank_pixels
                    row[f'{head}_gt_rank{rank}_ratio'] = _safe_div(rank_pixels, pixels)
                outside_pixels = pixels - ranked_pixels
                row[f'{head}_gt_rank_gt{topk}_pixels'] = outside_pixels
                row[f'{head}_gt_rank_gt{topk}_ratio'] = _safe_div(outside_pixels, pixels)
            return row

        regime_defs = [
            ('threshold_reject', threshold_reject),
            ('low_margin_non_reject', low_margin & (~threshold_reject)),
            ('confident_head_agree',
             confident_margin & (~threshold_reject) & (~head_disagree)),
            ('confident_head_disagree',
             confident_margin & (~threshold_reject) & head_disagree),
            ('mid_margin_head_agree',
             (~threshold_reject) & (~low_margin) & (~confident_margin) & (~head_disagree)),
            ('mid_margin_head_disagree',
             (~threshold_reject) & (~low_margin) & (~confident_margin) & head_disagree),
        ]

        overall = build_subset('all_base_wrong', base_wrong, error_count) if error_count > 0 else None
        regime_stats = []
        for regime_name, regime_mask_raw in regime_defs:
            row = build_subset(regime_name, base_wrong & regime_mask_raw, error_count)
            if row is not None:
                row['regime'] = row.pop('name')
                regime_stats.append(row)

        class_stats = []
        for class_idx in range(self.num_cls):
            gt_mask = base_wrong & (gt_data == class_idx)
            row = build_subset(self.class_names[class_idx], gt_mask, error_count)
            if row is not None:
                row['class_index'] = class_idx
                row['class_name'] = row.pop('name')
                class_stats.append(row)

        pair_stats = []
        for gt_class in range(self.num_cls):
            gt_mask = base_wrong & (gt_data == gt_class)
            if not gt_mask.any():
                continue
            for pred_class in range(self.num_cls):
                if pred_class == gt_class:
                    continue
                pair_mask = gt_mask & (base_pred == pred_class)
                row = build_subset(f'{self.class_names[gt_class]}->{self.class_names[pred_class]}',
                                   pair_mask, error_count)
                if row is not None:
                    row.pop('name')
                    row['gt_class_index'] = gt_class
                    row['gt_class_name'] = self.class_names[gt_class]
                    row['base_pred_class_index'] = pred_class
                    row['base_pred_class_name'] = self.class_names[pred_class]
                    pair_stats.append(row)
        pair_stats.sort(key=lambda item: item['pixels'], reverse=True)

        return dict(
            valid_pixels=valid_count,
            error_pixels=error_count,
            error_ratio=_safe_div(error_count, valid_count),
            topk=topk,
            overall=overall,
            regime_stats=regime_stats,
            class_stats=class_stats,
            pair_stats=pair_stats,
        )

    def _write_error_rank_stats(self, record):
        if not self.dump_error_rank_stats:
            return
        if self._error_rank_stats_file is None:
            path = self.error_rank_stats_path or './work_dirs/evidence_stats/error_rank_stats.jsonl'
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._error_rank_stats_file = open(path, 'a', buffering=1)
        self._error_rank_stats_file.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _build_top2_risk_context(self, seg_logits, components):
        if components is None or self.num_cls <= 1:
            return None
        semantic_logits = components.get('semantic_logits')
        instance_logits = components.get('instance_logits')
        if semantic_logits is None or instance_logits is None:
            return None

        topk = min(max(int(self.top2_risk_topk), 2), self.num_cls)
        score_maps = dict(
            final=seg_logits,
            semantic=semantic_logits,
            instance=instance_logits,
        )
        rank_maps, topk_indices = self._build_topk_rank_maps(score_maps, topk)

        final_top2_vals, final_top2_idx = torch.topk(seg_logits, k=2, dim=0)
        top1_score = final_top2_vals[0]
        top2_score = final_top2_vals[1]
        top1_idx = final_top2_idx[0]
        top2_idx = final_top2_idx[1]
        final_margin = top1_score - top2_score

        top1_semantic = _gather_class_map(semantic_logits, top1_idx)
        top2_semantic = _gather_class_map(semantic_logits, top2_idx)
        top1_instance = _gather_class_map(instance_logits, top1_idx)
        top2_instance = _gather_class_map(instance_logits, top2_idx)
        semantic_advantage = top2_semantic - top1_semantic
        instance_advantage = top2_instance - top1_instance
        top1_agreement = torch.minimum(top1_semantic, top1_instance)
        top2_agreement = torch.minimum(top2_semantic, top2_instance)
        agreement_advantage = top2_agreement - top1_agreement
        top1_sem_only = (top1_semantic - top1_instance).clamp(min=0.0)
        top1_inst_only = (top1_instance - top1_semantic).clamp(min=0.0)
        top2_sem_only = (top2_semantic - top2_instance).clamp(min=0.0)
        top2_inst_only = (top2_instance - top2_semantic).clamp(min=0.0)

        semantic_top1_idx = topk_indices['semantic'][0]
        instance_top1_idx = topk_indices['instance'][0]
        head_disagree = (
            (semantic_top1_idx != top1_idx)
            | (instance_top1_idx != top1_idx)
        )
        top2_sem_rank = _gather_class_map(rank_maps['semantic'], top2_idx)
        top2_inst_rank = _gather_class_map(rank_maps['instance'], top2_idx)
        top2_aux_rank = torch.maximum(top2_sem_rank, top2_inst_rank)

        threshold_reject = top1_score < self.prob_thd
        low_margin = final_margin <= self.topk_verifier_low_margin
        confident_margin = final_margin >= self.topk_verifier_confident_margin

        sem_support = (
            (top2_sem_rank > 0)
            & (semantic_advantage >= self.top2_risk_min_sem_adv)
        )
        inst_support = (
            (top2_inst_rank > 0)
            & (instance_advantage >= self.top2_risk_min_inst_adv)
        )
        agreement_support = agreement_advantage >= self.top2_risk_min_agreement_adv
        top1_sem_overexpands = top1_sem_only >= self.top2_risk_min_top1_sem_only

        mode = str(self.top2_risk_mode).lower()
        if mode == 'source_advantage':
            evidence_support = sem_support | inst_support | agreement_support
        elif mode == 'dual_advantage':
            evidence_support = (sem_support & inst_support) | agreement_support
        elif mode == 'semantic_advantage':
            evidence_support = sem_support
        elif mode == 'instance_advantage':
            evidence_support = inst_support
        elif mode == 'agreement_advantage':
            evidence_support = agreement_support
        elif mode == 'semantic_over_instance':
            evidence_support = top1_sem_overexpands & (sem_support | agreement_support)
        else:
            raise ValueError(
                "top2_risk_mode must be one of 'source_advantage', "
                "'dual_advantage', 'semantic_advantage', 'instance_advantage', "
                "'agreement_advantage', or 'semantic_over_instance', "
                f"but got {self.top2_risk_mode!r}")

        margin_gate = (
            (final_margin >= self.top2_risk_min_margin)
            & (final_margin <= self.top2_risk_max_margin)
            & (~threshold_reject)
        )
        if self.top2_risk_require_head_disagree:
            evidence_support = evidence_support & head_disagree

        apply_mask = margin_gate & evidence_support
        bg_policy = str(self.top2_risk_bg_policy).lower()
        if bg_policy == 'none':
            pass
        elif bg_policy == 'avoid_top2_bg':
            apply_mask = apply_mask & (top2_idx != self.bg_idx)
        elif bg_policy == 'avoid_any_bg':
            apply_mask = apply_mask & (top1_idx != self.bg_idx) & (top2_idx != self.bg_idx)
        elif bg_policy == 'only_top2_bg':
            apply_mask = apply_mask & (top2_idx == self.bg_idx)
        else:
            raise ValueError(
                "top2_risk_bg_policy must be one of 'none', 'avoid_top2_bg', "
                "'avoid_any_bg', or 'only_top2_bg', "
                f"but got {self.top2_risk_bg_policy!r}")

        return dict(
            topk=topk,
            topk_indices=topk_indices,
            rank_maps=rank_maps,
            top1_idx=top1_idx,
            top2_idx=top2_idx,
            top1_score=top1_score,
            top2_score=top2_score,
            final_margin=final_margin,
            top1_semantic=top1_semantic,
            top2_semantic=top2_semantic,
            top1_instance=top1_instance,
            top2_instance=top2_instance,
            semantic_advantage=semantic_advantage,
            instance_advantage=instance_advantage,
            top1_agreement=top1_agreement,
            top2_agreement=top2_agreement,
            agreement_advantage=agreement_advantage,
            top1_sem_only=top1_sem_only,
            top1_inst_only=top1_inst_only,
            top2_sem_only=top2_sem_only,
            top2_inst_only=top2_inst_only,
            top2_sem_rank=top2_sem_rank,
            top2_inst_rank=top2_inst_rank,
            top2_aux_rank=top2_aux_rank,
            sem_support=sem_support,
            inst_support=inst_support,
            agreement_support=agreement_support,
            top1_sem_overexpands=top1_sem_overexpands,
            threshold_reject=threshold_reject,
            low_margin=low_margin,
            confident_margin=confident_margin,
            head_disagree=head_disagree,
            apply_mask=apply_mask,
        )

    def _build_top2_risk_arbitration_logits(self, seg_logits, components):
        context = self._build_top2_risk_context(seg_logits, components)
        if context is None:
            return seg_logits, None
        apply_mask = context['apply_mask']
        if not apply_mask.any():
            return seg_logits, context
        arbitrated = seg_logits.clone()
        current_top2_score = _gather_class_map(arbitrated, context['top2_idx'])
        forced_top2_score = torch.where(
            apply_mask,
            context['top1_score'] + 1e-4,
            current_top2_score,
        )
        arbitrated.scatter_(0, context['top2_idx'].unsqueeze(0), forced_top2_score.unsqueeze(0))
        context['arbitrated_logits'] = arbitrated
        return arbitrated, context

    def _build_top2_risk_stats(self, base_logits, arbitrated_logits, base_pred,
                               arbitrated_pred, data_sample, context):
        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)
        if gt_data is None or context is None:
            return None

        valid_mask = gt_data != 255
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return dict(valid_pixels=0, overall=None, regime_stats=[], class_stats=[], pair_stats=[])

        base_correct = (base_pred == gt_data) & valid_mask
        arbitrated_correct = (arbitrated_pred == gt_data) & valid_mask
        changed = (base_pred != arbitrated_pred) & valid_mask
        improved = (~base_correct) & arbitrated_correct
        harmed = base_correct & (~arbitrated_correct)
        top2_is_gt = (context['top2_idx'] == gt_data) & valid_mask
        apply_mask = context['apply_mask'] & valid_mask
        oracle_improve = (~base_correct) & top2_is_gt
        oracle_harm = base_correct & (~top2_is_gt)
        gt_idx = gt_data.clamp(min=0, max=self.num_cls - 1)
        final_gt_score = _gather_class_map(base_logits, gt_idx)
        top1_gt_gap = context['top1_score'] - final_gt_score

        def build_subset(name, raw_mask, denominator):
            mask = raw_mask & valid_mask
            pixels = int(mask.sum().item())
            if pixels == 0:
                return None
            applied = apply_mask & mask
            applied_pixels = int(applied.sum().item())
            row = dict(
                name=name,
                pixels=pixels,
                pixel_ratio=_safe_div(pixels, denominator),
                base_correct_pixels=int((base_correct & mask).sum().item()),
                base_correct_ratio=_safe_div(int((base_correct & mask).sum().item()), pixels),
                top2_is_gt_pixels=int((top2_is_gt & mask).sum().item()),
                top2_is_gt_ratio=_safe_div(int((top2_is_gt & mask).sum().item()), pixels),
                apply_pixels=applied_pixels,
                apply_ratio=_safe_div(applied_pixels, pixels),
                applied_top2_is_gt_pixels=int((top2_is_gt & applied).sum().item()),
                applied_top2_is_gt_ratio=_safe_div(int((top2_is_gt & applied).sum().item()), applied_pixels),
                changed_pixels=int((changed & mask).sum().item()),
                changed_ratio=_safe_div(int((changed & mask).sum().item()), pixels),
                improved_pixels=int((improved & mask).sum().item()),
                improved_ratio=_safe_div(int((improved & mask).sum().item()), pixels),
                harmed_pixels=int((harmed & mask).sum().item()),
                harmed_ratio=_safe_div(int((harmed & mask).sum().item()), pixels),
                net_improved_pixels=int((improved & mask).sum().item()) - int((harmed & mask).sum().item()),
                oracle_flip_improve_pixels=int((oracle_improve & mask).sum().item()),
                oracle_flip_improve_ratio=_safe_div(int((oracle_improve & mask).sum().item()), pixels),
                oracle_flip_harm_pixels=int((oracle_harm & mask).sum().item()),
                oracle_flip_harm_ratio=_safe_div(int((oracle_harm & mask).sum().item()), pixels),
                oracle_flip_net_pixels=int((oracle_improve & mask).sum().item()) - int((oracle_harm & mask).sum().item()),
                mean_top1_gt_gap=_masked_mean(top1_gt_gap, mask),
                mean_final_margin=_masked_mean(context['final_margin'], mask),
                mean_semantic_advantage=_masked_mean(context['semantic_advantage'], mask),
                mean_instance_advantage=_masked_mean(context['instance_advantage'], mask),
                mean_agreement_advantage=_masked_mean(context['agreement_advantage'], mask),
                mean_top1_sem_only=_masked_mean(context['top1_sem_only'], mask),
                threshold_reject_pixels=int((context['threshold_reject'] & mask).sum().item()),
                low_margin_pixels=int((context['low_margin'] & mask).sum().item()),
                confident_margin_pixels=int((context['confident_margin'] & mask).sum().item()),
                head_disagree_pixels=int((context['head_disagree'] & mask).sum().item()),
                sem_support_pixels=int((context['sem_support'] & mask).sum().item()),
                inst_support_pixels=int((context['inst_support'] & mask).sum().item()),
                agreement_support_pixels=int((context['agreement_support'] & mask).sum().item()),
                top1_sem_overexpands_pixels=int((context['top1_sem_overexpands'] & mask).sum().item()),
            )
            for key in [
                    'threshold_reject', 'low_margin', 'confident_margin',
                    'head_disagree', 'sem_support', 'inst_support',
                    'agreement_support', 'top1_sem_overexpands']:
                pixels_key = f'{key}_pixels'
                row[f'{key}_ratio'] = _safe_div(row[pixels_key], pixels)
            return row

        regime_defs = [
            ('all_pixels', valid_mask),
            ('proposed_apply', context['apply_mask']),
            ('threshold_reject', context['threshold_reject']),
            ('low_margin_non_reject', context['low_margin'] & (~context['threshold_reject'])),
            ('confident_head_agree',
             context['confident_margin'] & (~context['threshold_reject']) & (~context['head_disagree'])),
            ('confident_head_disagree',
             context['confident_margin'] & (~context['threshold_reject']) & context['head_disagree']),
            ('mid_margin_head_agree',
             (~context['threshold_reject']) & (~context['low_margin'])
             & (~context['confident_margin']) & (~context['head_disagree'])),
            ('mid_margin_head_disagree',
             (~context['threshold_reject']) & (~context['low_margin'])
             & (~context['confident_margin']) & context['head_disagree']),
            ('sem_support', context['sem_support']),
            ('inst_support', context['inst_support']),
            ('agreement_support', context['agreement_support']),
            ('top1_sem_overexpands', context['top1_sem_overexpands']),
        ]

        overall = build_subset('all_pixels', valid_mask, valid_count)
        regime_stats = []
        for regime_name, regime_mask in regime_defs:
            row = build_subset(regime_name, regime_mask, valid_count)
            if row is not None:
                row['regime'] = row.pop('name')
                regime_stats.append(row)

        class_stats = []
        for class_idx in range(self.num_cls):
            gt_mask = gt_data == class_idx
            row = build_subset(self.class_names[class_idx], gt_mask, valid_count)
            if row is not None:
                row['class_index'] = class_idx
                row['class_name'] = row.pop('name')
                class_stats.append(row)

        pair_stats = []
        for gt_class in range(self.num_cls):
            gt_mask = (gt_data == gt_class) & valid_mask
            if not gt_mask.any():
                continue
            for pred_class in range(self.num_cls):
                if pred_class == gt_class:
                    continue
                pair_mask = gt_mask & (base_pred == pred_class)
                row = build_subset(f'{self.class_names[gt_class]}->{self.class_names[pred_class]}',
                                   pair_mask, valid_count)
                if row is not None:
                    row.pop('name')
                    row['gt_class_index'] = gt_class
                    row['gt_class_name'] = self.class_names[gt_class]
                    row['base_pred_class_index'] = pred_class
                    row['base_pred_class_name'] = self.class_names[pred_class]
                    pair_stats.append(row)
        pair_stats.sort(key=lambda item: item['pixels'], reverse=True)

        return dict(
            valid_pixels=valid_count,
            topk=int(context['topk']),
            overall=overall,
            regime_stats=regime_stats,
            class_stats=class_stats,
            pair_stats=pair_stats,
        )

    def _write_top2_risk_stats(self, record):
        if not self.dump_top2_risk_stats:
            return
        if self._top2_risk_stats_file is None:
            path = self.top2_risk_stats_path or './work_dirs/evidence_stats/top2_risk_stats.jsonl'
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._top2_risk_stats_file = open(path, 'a', buffering=1)
        self._top2_risk_stats_file.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _infer_class_components_for_image(self, image, ori_shape,
                                          slide_crop=None, slide_stride=None,
                                          view_id=None):
        slide_crop = self.slide_crop if slide_crop is None else slide_crop
        slide_stride = self.slide_stride if slide_stride is None else slide_stride
        if slide_crop > 0 and (slide_crop < image.size[0] or slide_crop < image.size[1]):
            seg_logits, components = self.slide_inference(
                image, slide_stride, slide_crop, return_components=True)
        else:
            seg_logits, components = self._inference_single_view(
                image, return_components=True, view_id=view_id)

        if seg_logits.shape[-2:] != ori_shape:
            seg_logits = F.interpolate(
                seg_logits.unsqueeze(0),
                size=ori_shape,
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
            for key in ['semantic_logits', 'instance_logits']:
                components[key] = F.interpolate(
                    components[key].unsqueeze(0),
                    size=ori_shape,
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0)

        seg_logits = self._aggregate_query_logits_to_classes(seg_logits)
        components = {
            key: self._aggregate_query_logits_to_classes(value)
            for key, value in components.items()
        }
        seg_logits = self._apply_evidence_competition_graph(seg_logits, components)
        seg_logits = self._apply_reject_aware_calibration(seg_logits, components)
        return seg_logits, components

    def _parse_multiview_list(self, value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            items = value
        else:
            items = str(value).split(',')
        return [str(item).strip() for item in items if str(item).strip()]

    def _make_transformed_image(self, image, view_name):
        view = str(view_name).lower()
        if view == 'hflip':
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        if view == 'vflip':
            return image.transpose(Image.FLIP_TOP_BOTTOM)
        if view == 'rot90':
            return image.transpose(Image.ROTATE_90)
        if view == 'rot180':
            return image.transpose(Image.ROTATE_180)
        if view == 'rot270':
            return image.transpose(Image.ROTATE_270)
        raise ValueError(
            "multiview_oracle_views only supports 'hflip', 'vflip', "
            "'rot90', 'rot180', and 'rot270', "
            f"but got {view_name!r}")

    def _invert_transformed_tensor(self, tensor, view_name):
        view = str(view_name).lower()
        if view == 'hflip':
            return torch.flip(tensor, dims=[2])
        if view == 'vflip':
            return torch.flip(tensor, dims=[1])
        if view == 'rot90':
            return torch.rot90(tensor, k=-1, dims=(1, 2))
        if view == 'rot180':
            return torch.rot90(tensor, k=2, dims=(1, 2))
        if view == 'rot270':
            return torch.rot90(tensor, k=1, dims=(1, 2))
        raise ValueError(f'Unsupported inverse view: {view_name!r}')

    def _invert_transformed_components(self, components, view_name):
        return {
            key: self._invert_transformed_tensor(value, view_name)
            for key, value in components.items()
        }

    def _parse_multiview_slide_views(self):
        slide_views = []
        for item in self._parse_multiview_list(self.multiview_oracle_slide_views):
            normalized = item.lower().replace('slide', '')
            if ':' in normalized:
                crop_str, stride_str = normalized.split(':', 1)
            elif '/' in normalized:
                crop_str, stride_str = normalized.split('/', 1)
            else:
                raise ValueError(
                    "Each multiview_oracle_slide_views item must be 'crop:stride', "
                    f"but got {item!r}")
            crop = int(crop_str)
            stride = int(stride_str)
            if crop <= 0 or stride <= 0:
                raise ValueError(
                    f'Invalid slide view crop/stride in {item!r}: both must be positive.')
            slide_views.append((f'slide{crop}_stride{stride}', crop, stride))
        return slide_views

    def _compute_multiview_evidence(self, view_name, seg_logits, components, gt_idx,
                                    valid_mask, topk):
        score_maps = dict(
            final=seg_logits,
            semantic=components['semantic_logits'],
            instance=components['instance_logits'],
        )
        topk_indices = {}
        contains = {}
        for head, scores in score_maps.items():
            topk_idx = torch.topk(scores, k=topk, dim=0).indices
            topk_indices[head] = topk_idx
            head_contains = torch.zeros_like(valid_mask, dtype=torch.bool)
            for rank in range(topk):
                head_contains |= ((topk_idx[rank] == gt_idx) & valid_mask)
            contains[head] = head_contains

        any_head_contains = contains['final'] | contains['semantic'] | contains['instance']
        final_top2_vals, final_top2_idx = torch.topk(seg_logits, k=min(2, self.num_cls), dim=0)
        final_top1_idx = final_top2_idx[0]
        final_top1_score = final_top2_vals[0]
        if self.num_cls > 1:
            final_top2_score = final_top2_vals[1]
        else:
            final_top2_score = torch.zeros_like(final_top1_score)
        final_margin = final_top1_score - final_top2_score
        final_gt_score = _gather_class_map(seg_logits, gt_idx)
        top1_gt_gap = final_top1_score - final_gt_score
        final_top1_gt = (final_top1_idx == gt_idx) & valid_mask
        return dict(
            view_name=view_name,
            contains=contains,
            any_head_contains=any_head_contains,
            final_top1_gt=final_top1_gt,
            final_margin=final_margin,
            final_top1_score=final_top1_score,
            final_gt_score=final_gt_score,
            top1_gt_gap=top1_gt_gap,
            final_top1_idx=final_top1_idx,
            topk_indices=topk_indices,
        )

    def _build_multiview_oracle_stats(self, base_logits, base_pred, data_sample,
                                      base_components, image, ori_shape):
        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)
        if gt_data is None or base_components is None:
            return None

        valid_mask = gt_data != 255
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return dict(valid_pixels=0, topk=0, view_names=[], view_stats=[],
                        regime_stats=[], class_stats=[], pair_stats=[])

        topk = min(max(int(self.multiview_oracle_topk), 1), self.num_cls)
        gt_idx = gt_data.clamp(min=0, max=self.num_cls - 1)
        base_wrong = (base_pred != gt_data) & valid_mask
        base_evidence = self._compute_multiview_evidence(
            'base', base_logits, base_components, gt_idx, valid_mask, topk)
        base_gap = base_evidence['top1_gt_gap']
        base_large_gap = base_gap > 0.20
        threshold_reject = base_evidence['final_top1_score'] < self.prob_thd
        low_margin = base_evidence['final_margin'] <= self.topk_verifier_low_margin
        confident_margin = base_evidence['final_margin'] >= self.topk_verifier_confident_margin

        view_names = []
        view_stats = []
        any_final_contains = torch.zeros_like(valid_mask, dtype=torch.bool)
        any_semantic_contains = torch.zeros_like(valid_mask, dtype=torch.bool)
        any_instance_contains = torch.zeros_like(valid_mask, dtype=torch.bool)
        any_any_head_contains = torch.zeros_like(valid_mask, dtype=torch.bool)
        any_final_top1_gt = torch.zeros_like(valid_mask, dtype=torch.bool)
        final_contains_count = torch.zeros_like(gt_data, dtype=torch.float32, device=self.device)
        any_head_contains_count = torch.zeros_like(gt_data, dtype=torch.float32, device=self.device)
        best_gap = torch.full_like(base_gap.detach().float(), float('inf'))

        def update_from_view(evidence):
            nonlocal any_final_contains, any_semantic_contains, any_instance_contains
            nonlocal any_any_head_contains, any_final_top1_gt, best_gap
            view_names.append(evidence['view_name'])
            final_hit = evidence['contains']['final']
            sem_hit = evidence['contains']['semantic']
            inst_hit = evidence['contains']['instance']
            any_hit = evidence['any_head_contains']
            any_final_contains |= final_hit
            any_semantic_contains |= sem_hit
            any_instance_contains |= inst_hit
            any_any_head_contains |= any_hit
            any_final_top1_gt |= evidence['final_top1_gt']
            final_contains_count.add_(final_hit.float())
            any_head_contains_count.add_(any_hit.float())
            best_gap = torch.minimum(best_gap, evidence['top1_gt_gap'].detach().float())
            view_stats.append(self._build_multiview_view_row(
                evidence, base_wrong, valid_mask, valid_count))

        if self.multiview_oracle_include_base:
            update_from_view(base_evidence)

        for view_name in self._parse_multiview_list(self.multiview_oracle_views):
            transformed = self._make_transformed_image(image, view_name)
            view_ori_shape = (transformed.size[1], transformed.size[0])
            view_logits, view_components = self._infer_class_components_for_image(
                transformed, view_ori_shape, view_id=f'multiview_{view_name}')
            view_logits = self._invert_transformed_tensor(view_logits, view_name)
            view_components = self._invert_transformed_components(view_components, view_name)
            if view_logits.shape[-2:] != ori_shape:
                view_logits = F.interpolate(
                    view_logits.unsqueeze(0),
                    size=ori_shape,
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0)
                for key in ['semantic_logits', 'instance_logits']:
                    view_components[key] = F.interpolate(
                        view_components[key].unsqueeze(0),
                        size=ori_shape,
                        mode='bilinear',
                        align_corners=False,
                    ).squeeze(0)
            evidence = self._compute_multiview_evidence(
                view_name, view_logits, view_components, gt_idx, valid_mask, topk)
            update_from_view(evidence)
            del transformed, view_logits, view_components, evidence
            if self.multiview_oracle_empty_cache and self.device.type == 'cuda':
                torch.cuda.empty_cache()

        for slide_name, crop, stride in self._parse_multiview_slide_views():
            slide_logits, slide_components = self._infer_class_components_for_image(
                image, ori_shape, slide_crop=crop, slide_stride=stride,
                view_id=f'multiview_{slide_name}')
            evidence = self._compute_multiview_evidence(
                slide_name, slide_logits, slide_components, gt_idx, valid_mask, topk)
            update_from_view(evidence)
            del slide_logits, slide_components, evidence
            if self.multiview_oracle_empty_cache and self.device.type == 'cuda':
                torch.cuda.empty_cache()

        view_count = max(len(view_names), 1)
        gap_reduction = base_gap.detach().float() - best_gap
        large_gap_resolved = base_large_gap & (best_gap <= 0.05)
        gap_reduced_0p10 = gap_reduction >= 0.10
        stable_final_topk = final_contains_count >= min(2, view_count)
        stable_any_head_topk = any_head_contains_count >= min(2, view_count)

        def build_subset(name, raw_mask, denominator):
            mask = raw_mask & valid_mask
            pixels = int(mask.sum().item())
            if pixels == 0:
                return None
            row = dict(
                name=name,
                pixels=pixels,
                pixel_ratio=_safe_div(pixels, denominator),
                base_wrong_pixels=int((base_wrong & mask).sum().item()),
                base_wrong_ratio=_safe_div(int((base_wrong & mask).sum().item()), pixels),
                threshold_reject_pixels=int((threshold_reject & mask).sum().item()),
                threshold_reject_ratio=_safe_div(int((threshold_reject & mask).sum().item()), pixels),
                low_margin_pixels=int((low_margin & mask).sum().item()),
                low_margin_ratio=_safe_div(int((low_margin & mask).sum().item()), pixels),
                confident_margin_pixels=int((confident_margin & mask).sum().item()),
                confident_margin_ratio=_safe_div(int((confident_margin & mask).sum().item()), pixels),
                base_final_topk_contains_gt_pixels=int((base_evidence['contains']['final'] & mask).sum().item()),
                base_final_topk_contains_gt_ratio=_safe_div(
                    int((base_evidence['contains']['final'] & mask).sum().item()), pixels),
                base_any_head_topk_contains_gt_pixels=int((base_evidence['any_head_contains'] & mask).sum().item()),
                base_any_head_topk_contains_gt_ratio=_safe_div(
                    int((base_evidence['any_head_contains'] & mask).sum().item()), pixels),
                any_view_final_topk_contains_gt_pixels=int((any_final_contains & mask).sum().item()),
                any_view_final_topk_contains_gt_ratio=_safe_div(
                    int((any_final_contains & mask).sum().item()), pixels),
                any_view_semantic_topk_contains_gt_pixels=int((any_semantic_contains & mask).sum().item()),
                any_view_semantic_topk_contains_gt_ratio=_safe_div(
                    int((any_semantic_contains & mask).sum().item()), pixels),
                any_view_instance_topk_contains_gt_pixels=int((any_instance_contains & mask).sum().item()),
                any_view_instance_topk_contains_gt_ratio=_safe_div(
                    int((any_instance_contains & mask).sum().item()), pixels),
                any_view_any_head_topk_contains_gt_pixels=int((any_any_head_contains & mask).sum().item()),
                any_view_any_head_topk_contains_gt_ratio=_safe_div(
                    int((any_any_head_contains & mask).sum().item()), pixels),
                any_view_final_top1_gt_pixels=int((any_final_top1_gt & mask).sum().item()),
                any_view_final_top1_gt_ratio=_safe_div(
                    int((any_final_top1_gt & mask).sum().item()), pixels),
                stable_final_topk_pixels=int((stable_final_topk & mask).sum().item()),
                stable_final_topk_ratio=_safe_div(int((stable_final_topk & mask).sum().item()), pixels),
                stable_any_head_topk_pixels=int((stable_any_head_topk & mask).sum().item()),
                stable_any_head_topk_ratio=_safe_div(int((stable_any_head_topk & mask).sum().item()), pixels),
                base_large_gap_pixels=int((base_large_gap & mask).sum().item()),
                base_large_gap_ratio=_safe_div(int((base_large_gap & mask).sum().item()), pixels),
                large_gap_resolved_pixels=int((large_gap_resolved & mask).sum().item()),
                large_gap_resolved_ratio=_safe_div(int((large_gap_resolved & mask).sum().item()), pixels),
                gap_reduced_0p10_pixels=int((gap_reduced_0p10 & mask).sum().item()),
                gap_reduced_0p10_ratio=_safe_div(int((gap_reduced_0p10 & mask).sum().item()), pixels),
                mean_base_top1_gt_gap=_masked_mean(base_gap, mask),
                mean_best_final_top1_gt_gap=_masked_mean(best_gap, mask),
                mean_gap_reduction=_masked_mean(gap_reduction, mask),
                mean_base_final_margin=_masked_mean(base_evidence['final_margin'], mask),
                mean_final_topk_view_count=_masked_mean(final_contains_count, mask),
                mean_any_head_topk_view_count=_masked_mean(any_head_contains_count, mask),
            )
            return row

        regime_defs = [
            ('all_valid', valid_mask),
            ('base_wrong', base_wrong),
            ('base_wrong_threshold_reject', base_wrong & threshold_reject),
            ('base_wrong_low_margin', base_wrong & low_margin),
            ('base_wrong_large_gap', base_wrong & base_large_gap),
            ('base_wrong_large_gap_final_topk',
             base_wrong & base_large_gap & base_evidence['contains']['final']),
            ('base_wrong_confident_margin', base_wrong & confident_margin),
        ]
        regime_stats = []
        for regime_name, regime_mask in regime_defs:
            row = build_subset(regime_name, regime_mask, valid_count)
            if row is not None:
                row['regime'] = row.pop('name')
                regime_stats.append(row)

        class_stats = []
        for class_idx in range(self.num_cls):
            gt_mask = base_wrong & (gt_data == class_idx)
            row = build_subset(self.class_names[class_idx], gt_mask, valid_count)
            if row is not None:
                row['class_index'] = class_idx
                row['class_name'] = row.pop('name')
                class_stats.append(row)

        pair_stats = []
        for gt_class in range(self.num_cls):
            gt_mask = base_wrong & (gt_data == gt_class)
            if not gt_mask.any():
                continue
            for pred_class in range(self.num_cls):
                if pred_class == gt_class:
                    continue
                pair_mask = gt_mask & (base_pred == pred_class)
                row = build_subset(
                    f'{self.class_names[gt_class]}->{self.class_names[pred_class]}',
                    pair_mask,
                    valid_count)
                if row is not None:
                    row.pop('name')
                    row['gt_class_index'] = gt_class
                    row['gt_class_name'] = self.class_names[gt_class]
                    row['base_pred_class_index'] = pred_class
                    row['base_pred_class_name'] = self.class_names[pred_class]
                    pair_stats.append(row)
        pair_stats.sort(key=lambda item: item['pixels'], reverse=True)

        return dict(
            valid_pixels=valid_count,
            topk=topk,
            view_names=view_names,
            view_count=view_count,
            view_stats=view_stats,
            regime_stats=regime_stats,
            class_stats=class_stats,
            pair_stats=pair_stats,
        )

    def _build_multiview_view_row(self, evidence, base_wrong, valid_mask, valid_count):
        mask = base_wrong & valid_mask
        pixels = int(mask.sum().item())
        if pixels == 0:
            mask = valid_mask
            pixels = int(mask.sum().item())
        return dict(
            view_name=evidence['view_name'],
            pixels=pixels,
            pixel_ratio=_safe_div(pixels, valid_count),
            final_topk_contains_gt_pixels=int((evidence['contains']['final'] & mask).sum().item()),
            final_topk_contains_gt_ratio=_safe_div(
                int((evidence['contains']['final'] & mask).sum().item()), pixels),
            semantic_topk_contains_gt_pixels=int((evidence['contains']['semantic'] & mask).sum().item()),
            semantic_topk_contains_gt_ratio=_safe_div(
                int((evidence['contains']['semantic'] & mask).sum().item()), pixels),
            instance_topk_contains_gt_pixels=int((evidence['contains']['instance'] & mask).sum().item()),
            instance_topk_contains_gt_ratio=_safe_div(
                int((evidence['contains']['instance'] & mask).sum().item()), pixels),
            any_head_topk_contains_gt_pixels=int((evidence['any_head_contains'] & mask).sum().item()),
            any_head_topk_contains_gt_ratio=_safe_div(
                int((evidence['any_head_contains'] & mask).sum().item()), pixels),
            final_top1_gt_pixels=int((evidence['final_top1_gt'] & mask).sum().item()),
            final_top1_gt_ratio=_safe_div(int((evidence['final_top1_gt'] & mask).sum().item()), pixels),
            mean_top1_gt_gap=_masked_mean(evidence['top1_gt_gap'], mask),
            mean_final_margin=_masked_mean(evidence['final_margin'], mask),
        )

    def _write_multiview_oracle_stats(self, record):
        if not self.dump_multiview_oracle_stats:
            return
        if self._multiview_oracle_stats_file is None:
            path = self.multiview_oracle_stats_path or './work_dirs/evidence_stats/multiview_oracle_stats.jsonl'
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._multiview_oracle_stats_file = open(path, 'a', buffering=1)
        self._multiview_oracle_stats_file.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _build_seed_rule_context(self, base_logits, base_pred, components):
        if components is None or self.num_cls <= 0:
            return None
        semantic_logits = components.get('semantic_logits')
        instance_logits = components.get('instance_logits')
        if semantic_logits is None or instance_logits is None:
            return None

        top2_vals, top2_idx = torch.topk(base_logits, k=min(2, self.num_cls), dim=0)
        final_top1_score = top2_vals[0]
        final_top1_idx = top2_idx[0]
        if self.num_cls > 1:
            final_top2_score = top2_vals[1]
        else:
            final_top2_score = torch.zeros_like(final_top1_score)
        final_margin = final_top1_score - final_top2_score

        semantic_top1_idx = torch.argmax(semantic_logits, dim=0)
        instance_top1_idx = torch.argmax(instance_logits, dim=0)
        sem_final_agree = semantic_top1_idx == final_top1_idx
        inst_final_agree = instance_top1_idx == final_top1_idx
        any_head_agree = sem_final_agree | inst_final_agree
        assigned_top1 = base_pred == final_top1_idx

        local_consistency = self._same_class_local_consistency(
            final_top1_idx, self.seed_local_kernel)
        core_consistency = self._same_class_local_consistency(
            final_top1_idx, self.seed_core_kernel)

        score_mask = (
            assigned_top1
            & (final_top1_score >= float(self.seed_final_score_thd))
        )
        margin_mask = final_margin >= float(self.seed_margin_thd)
        local_mask = local_consistency >= float(self.seed_local_consistency_thd)
        core_mask = core_consistency >= float(self.seed_core_consistency_thd)

        rule_defs = [
            ('final_score', score_mask),
            ('final_score_margin', score_mask & margin_mask),
            ('final_score_margin_sem_final_agree',
             score_mask & margin_mask & sem_final_agree),
            ('final_score_margin_sem_inst_final_agree',
             score_mask & margin_mask & sem_final_agree & inst_final_agree),
            ('final_score_margin_sem_inst_final_agree_local',
             score_mask & margin_mask & sem_final_agree & inst_final_agree & local_mask),
            ('final_score_margin_any_head_agree',
             score_mask & margin_mask & any_head_agree),
            ('final_score_margin_any_head_agree_local',
             score_mask & margin_mask & any_head_agree & local_mask),
            ('final_score_margin_any_head_agree_local_core',
             score_mask & margin_mask & any_head_agree
             & local_mask & core_mask),
            ('final_score_margin_sem_inst_final_agree_local_core',
             score_mask & margin_mask & sem_final_agree & inst_final_agree
             & local_mask & core_mask),
        ]

        return dict(
            rule_defs=rule_defs,
            final_top1_idx=final_top1_idx,
            final_top1_score=final_top1_score,
            final_margin=final_margin,
            semantic_top1_idx=semantic_top1_idx,
            instance_top1_idx=instance_top1_idx,
            sem_final_agree=sem_final_agree,
            inst_final_agree=inst_final_agree,
            any_head_agree=any_head_agree,
            local_consistency=local_consistency,
            core_consistency=core_consistency,
        )

    @staticmethod
    def _select_seed_rule(seed_context, rule_name):
        if seed_context is None:
            return None
        rule_defs = seed_context.get('rule_defs') or []
        for candidate_name, candidate_mask in rule_defs:
            if candidate_name == str(rule_name):
                return candidate_mask
        if rule_defs:
            return rule_defs[-1][1]
        return None

    def _same_class_local_consistency(self, class_idx_map, kernel):
        kernel = max(int(kernel), 1)
        if kernel % 2 == 0:
            kernel += 1
        if kernel <= 1:
            return torch.ones_like(class_idx_map, dtype=torch.float32)

        one_hot = F.one_hot(
            class_idx_map.clamp(min=0, max=self.num_cls - 1),
            num_classes=self.num_cls,
        ).permute(2, 0, 1).float().unsqueeze(0)
        local_ratio = F.avg_pool2d(
            one_hot,
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        ).squeeze(0)
        return _gather_class_map(local_ratio, class_idx_map)

    def _build_seed_similarity_maps(self, base_logits, components, seed_mask, seed_class):
        semantic_logits = components.get('semantic_logits')
        instance_logits = components.get('instance_logits')
        similarity_maps = torch.full_like(
            base_logits.detach().float(),
            float('nan'),
        )
        has_seed = torch.zeros(self.num_cls, device=base_logits.device, dtype=torch.bool)
        eps = float(self.seed_similarity_eps)
        for class_idx in range(self.num_cls):
            class_seed = seed_mask & (seed_class == class_idx)
            if class_seed.any():
                final_map = base_logits[class_idx].detach().float().clamp(min=0.0)
                semantic_map = semantic_logits[class_idx].detach().float().clamp(min=0.0)
                instance_map = instance_logits[class_idx].detach().float().clamp(min=0.0)
                proto = torch.stack([
                    final_map[class_seed].mean(),
                    semantic_map[class_seed].mean(),
                    instance_map[class_seed].mean(),
                ])
                proto_norm = torch.linalg.vector_norm(proto).clamp_min(eps)
                pixel_norm = torch.sqrt(
                    final_map.square()
                    + semantic_map.square()
                    + instance_map.square()
                ).clamp_min(eps)
                similarity_maps[class_idx] = (
                    proto[0] * final_map
                    + proto[1] * semantic_map
                    + proto[2] * instance_map
                ) / (proto_norm * pixel_norm)
                has_seed[class_idx] = True

        return similarity_maps, has_seed

    def _get_feature_map_for_similarity_space(self, components, space):
        if components is None:
            return None
        if space == 'vision':
            return components.get('vision_features')
        if space.startswith('pe_layer_'):
            layer_id = space[len('pe_layer_'):]
            if not layer_id.isdigit():
                return None
            pe_layers = components.get('pe_layers')
            if not isinstance(pe_layers, (list, tuple)):
                return None
            layer_idx = int(layer_id)
            if layer_idx < 0 or layer_idx >= len(pe_layers):
                return None
            return pe_layers[layer_idx]
        return None

    def _build_feature_seed_similarity_maps(self, space, base_logits, components,
                                            seed_mask, seed_class):
        vision_features = self._get_feature_map_for_similarity_space(
            components, space)
        if vision_features is None:
            return None, None
        return self._build_feature_seed_similarity_from_map(
            vision_features,
            base_logits,
            seed_mask,
            seed_class,
        )

    def _build_feature_seed_similarity_from_map(
            self, vision_features, base_logits, seed_mask, seed_class):
        if vision_features.ndim == 4:
            vision_features = vision_features.squeeze(0)
        if vision_features.ndim != 3:
            return None, None

        feature_map = vision_features.detach().float()
        feat_h, feat_w = feature_map.shape[-2:]
        seed_mask_low = F.interpolate(
            seed_mask.float().view(1, 1, *seed_mask.shape),
            size=(feat_h, feat_w),
            mode='nearest',
        ).squeeze().bool()
        seed_class_low = F.interpolate(
            seed_class.float().view(1, 1, *seed_class.shape),
            size=(feat_h, feat_w),
            mode='nearest',
        ).squeeze().long().clamp(min=0, max=self.num_cls - 1)

        norm_features = F.normalize(
            feature_map,
            dim=0,
            eps=float(self.seed_similarity_eps),
        )
        similarity_low = torch.full(
            (self.num_cls, feat_h, feat_w),
            float('nan'),
            device=base_logits.device,
            dtype=torch.float32,
        )
        has_seed = torch.zeros(self.num_cls, device=base_logits.device, dtype=torch.bool)
        for class_idx in range(self.num_cls):
            class_seed = seed_mask_low & (seed_class_low == class_idx)
            if class_seed.any():
                proto = feature_map[:, class_seed].mean(dim=1)
                proto = F.normalize(
                    proto,
                    dim=0,
                    eps=float(self.seed_similarity_eps),
                )
                similarity_low[class_idx] = (norm_features * proto[:, None, None]).sum(dim=0)
                has_seed[class_idx] = True

        similarity_maps = F.interpolate(
            similarity_low.unsqueeze(0),
            size=base_logits.shape[-2:],
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)
        return similarity_maps, has_seed

    def _get_position_bias_layers(self, layer_count):
        requested = {
            int(round(value))
            for value in self._parse_float_list(
                self.position_bias_layers, '0,1,2')
        }
        return [
            layer_idx for layer_idx in sorted(requested)
            if 0 <= layer_idx < int(layer_count)
        ]

    def _get_position_bias_ranks(self, channel_count):
        requested = {
            max(1, int(round(value)))
            for value in self._parse_float_list(
                self.position_bias_ranks, '2,4,8')
        }
        return [
            rank for rank in sorted(requested)
            if rank <= int(channel_count)
        ]

    def _get_position_bias_basis(self, position_map, layer_idx, max_rank):
        if position_map.ndim == 4:
            position_map = position_map.squeeze(0)
        if position_map.ndim != 3:
            return None, None
        channels, height, width = position_map.shape
        rank = min(int(max_rank), int(channels))
        cache_key = (
            int(layer_idx),
            int(channels),
            int(height),
            int(width),
            int(rank),
            str(position_map.device),
        )
        cached = self._position_bias_basis_cache.get(cache_key)
        if cached is not None:
            return cached

        positions = position_map.detach().float().reshape(channels, -1).t()
        positions = positions - positions.mean(dim=0, keepdim=True)
        covariance = positions.t().matmul(positions)
        covariance = covariance / max(1, int(positions.shape[0]) - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order].clamp_min(0.0)
        basis = eigenvectors[:, order[:rank]]
        cached = (basis, eigenvalues)
        self._position_bias_basis_cache[cache_key] = cached
        return cached

    def _get_position_bias_control_basis(
            self, basis, layer_idx, rank, control_name, trial):
        channels = int(basis.shape[0])
        cache_key = (
            int(layer_idx),
            int(channels),
            int(rank),
            str(control_name),
            int(trial),
            str(basis.device),
        )
        cached = self._position_bias_control_basis_cache.get(cache_key)
        if cached is not None:
            return cached

        if control_name == 'permuted':
            generator = torch.Generator(device='cpu')
            generator.manual_seed(
                17011 + 1009 * int(layer_idx) + 97 * int(rank)
                + 13 * int(trial))
            permutation = torch.randperm(
                channels, generator=generator).to(basis.device)
            control_basis = basis[permutation]
        elif control_name == 'random':
            generator = torch.Generator(device='cpu')
            generator.manual_seed(
                29009 + 1013 * int(layer_idx) + 101 * int(rank)
                + 17 * int(trial))
            random_matrix = torch.randn(
                channels,
                int(rank),
                generator=generator,
                dtype=torch.float32,
            ).to(basis.device)
            control_basis = torch.linalg.qr(
                random_matrix, mode='reduced')[0]
        else:
            return None
        self._position_bias_control_basis_cache[cache_key] = control_basis
        return control_basis

    @staticmethod
    def _remove_feature_subspace(centered_feature, basis):
        flat = centered_feature.reshape(centered_feature.shape[0], -1)
        removed = basis.matmul(basis.t().matmul(flat))
        residual = flat - removed
        total_energy = float(flat.square().sum().item())
        removed_energy = float(removed.square().sum().item())
        return (
            residual.view_as(centered_feature),
            _safe_div(removed_energy, total_energy),
        )

    def _position_bias_head_alignment(
            self, head_maps, position_map, basis):
        if position_map.ndim == 4:
            position_map = position_map.squeeze(0)
        if position_map.ndim != 3:
            return None
        height, width = position_map.shape[-2:]
        resized = self._interpolate_float32(
            head_maps.detach().float().unsqueeze(0),
            (height, width),
        ).squeeze(0)
        values = resized.flatten(1)
        values = values - values.mean(dim=1, keepdim=True)
        total = values.square().sum(dim=1)

        positions = position_map.detach().float().flatten(1).t()
        positions = positions - positions.mean(dim=0, keepdim=True)
        spatial_scores = positions.matmul(basis)
        spatial_basis = torch.linalg.qr(
            spatial_scores, mode='reduced')[0]
        explained = values.matmul(spatial_basis).square().sum(dim=1)
        valid = total > 1e-8
        if not valid.any():
            return None
        ratios = explained[valid] / total[valid]
        return dict(
            valid_classes=int(valid.sum().item()),
            mean_explained_ratio=float(ratios.mean().item()),
            max_explained_ratio=float(ratios.max().item()),
        )

    def _build_position_bias_source_maps(
            self, class_final, class_semantic, class_instance,
            feature_layers, position_layers, output_shape,
            seed_mask, seed_class):
        if not self.dump_position_bias_stats:
            return {}, []
        if not isinstance(feature_layers, (list, tuple)):
            return {}, []
        if not isinstance(position_layers, (list, tuple)):
            return {}, []
        if seed_mask is None or seed_class is None:
            return {}, []

        layer_count = min(len(feature_layers), len(position_layers))
        selected_layers = self._get_position_bias_layers(layer_count)
        controls = set(self._parse_name_list(
            self.position_bias_controls))
        random_trials = max(1, int(self.position_bias_random_trials))
        source_maps = {}
        feature_stats = []
        head_maps = {
            'final': class_final,
            'semantic': class_semantic,
            'instance': class_instance,
        }

        def add_similarity(name, feature):
            similarity, has_seed = (
                self._build_feature_seed_similarity_from_map(
                    feature,
                    class_final,
                    seed_mask,
                    seed_class,
                )
            )
            if similarity is None or has_seed is None:
                return
            similarity = self._interpolate_float32(
                similarity.unsqueeze(0),
                output_shape,
            ).squeeze(0)
            similarity[~has_seed, :, :] = float('nan')
            source_maps[name] = similarity

        for layer_idx in selected_layers:
            feature = feature_layers[layer_idx]
            position = position_layers[layer_idx]
            if feature.ndim == 4:
                feature = feature.squeeze(0)
            if position.ndim == 4:
                position = position.squeeze(0)
            if feature.ndim != 3 or position.ndim != 3:
                continue
            feature = feature.detach().float()
            position = position.detach().float()
            if feature.shape[0] != position.shape[0]:
                continue
            if feature.shape[-2:] != position.shape[-2:]:
                position = self._interpolate_float32(
                    position.unsqueeze(0),
                    feature.shape[-2:],
                ).squeeze(0)

            ranks = self._get_position_bias_ranks(feature.shape[0])
            if not ranks:
                continue
            basis, eigenvalues = self._get_position_bias_basis(
                position, layer_idx, max(ranks))
            if basis is None:
                continue
            centered = (
                feature - feature.mean(dim=(-2, -1), keepdim=True))
            total_feature_energy = float(centered.square().sum().item())
            total_position_energy = float(eigenvalues.sum().item())
            spatial_pixels = int(feature.shape[-2] * feature.shape[-1])

            add_similarity(f'posbias_l{layer_idx}_raw', feature)
            add_similarity(f'posbias_l{layer_idx}_centered', centered)

            for rank in ranks:
                rank_basis = basis[:, :rank]
                residual, removed_ratio = self._remove_feature_subspace(
                    centered, rank_basis)
                variant_name = f'posbias_l{layer_idx}_pe_r{rank}'
                add_similarity(variant_name, residual)
                feature_stats.append(dict(
                    stat_type='feature_projection',
                    layer_index=int(layer_idx),
                    rank=int(rank),
                    variant='pe',
                    trial=0,
                    spatial_pixels=spatial_pixels,
                    feature_channels=int(feature.shape[0]),
                    total_feature_energy=total_feature_energy,
                    removed_feature_energy_ratio=removed_ratio,
                    position_variance_explained_ratio=_safe_div(
                        float(eigenvalues[:rank].sum().item()),
                        total_position_energy),
                ))
                for head_name, head_map in head_maps.items():
                    alignment = self._position_bias_head_alignment(
                        head_map, position, rank_basis)
                    if alignment is not None:
                        feature_stats.append(dict(
                            stat_type='head_alignment',
                            layer_index=int(layer_idx),
                            rank=int(rank),
                            variant='pe',
                            trial=0,
                            head_name=head_name,
                            spatial_pixels=spatial_pixels,
                            **alignment,
                        ))

                for control_name in ('permuted', 'random'):
                    if control_name not in controls:
                        continue
                    trial_count = (
                        random_trials if control_name == 'random' else 1)
                    for trial in range(trial_count):
                        control_basis = (
                            self._get_position_bias_control_basis(
                                rank_basis,
                                layer_idx,
                                rank,
                                control_name,
                                trial,
                            )
                        )
                        if control_basis is None:
                            continue
                        control_residual, control_removed_ratio = (
                            self._remove_feature_subspace(
                                centered, control_basis)
                        )
                        short_name = (
                            'perm' if control_name == 'permuted'
                            else 'rand')
                        control_variant = (
                            f'posbias_l{layer_idx}_{short_name}_r{rank}'
                            f'_t{trial}')
                        add_similarity(control_variant, control_residual)
                        feature_stats.append(dict(
                            stat_type='feature_projection',
                            layer_index=int(layer_idx),
                            rank=int(rank),
                            variant=control_name,
                            trial=int(trial),
                            spatial_pixels=spatial_pixels,
                            feature_channels=int(feature.shape[0]),
                            total_feature_energy=total_feature_energy,
                            removed_feature_energy_ratio=(
                                control_removed_ratio),
                            position_variance_explained_ratio=None,
                        ))
                        for head_name, head_map in head_maps.items():
                            alignment = self._position_bias_head_alignment(
                                head_map, position, control_basis)
                            if alignment is not None:
                                feature_stats.append(dict(
                                    stat_type='head_alignment',
                                    layer_index=int(layer_idx),
                                    rank=int(rank),
                                    variant=control_name,
                                    trial=int(trial),
                                    head_name=head_name,
                                    spatial_pixels=spatial_pixels,
                                    **alignment,
                                ))

        return source_maps, feature_stats

    def _get_scene_common_layers(self, layer_count):
        requested = {
            int(round(value))
            for value in self._parse_float_list(
                self.scene_common_layers, '0,2')
        }
        return [
            layer_idx for layer_idx in sorted(requested)
            if 0 <= layer_idx < int(layer_count)
        ]

    def _get_candidate_residual_layers(self, layer_count):
        requested = {
            int(round(value))
            for value in self._parse_float_list(
                self.candidate_residual_layers, '0,2')
        }
        return [
            layer_idx for layer_idx in sorted(requested)
            if 0 <= layer_idx < int(layer_count)
        ]

    @staticmethod
    def _scene_common_strength_token(strength):
        return f'{float(strength):g}'.replace('.', 'p')

    @staticmethod
    def _scene_common_subtract_vector(feature, vector, strength=1.0):
        return (
            feature
            - float(strength) * vector[:, None, None]
        )

    def _scene_common_feature_descriptor(
            self, feature, compute_spectrum=False):
        flat = feature.reshape(feature.shape[0], -1).float()
        token_count = int(flat.shape[1])
        total_energy = float(flat.square().sum().item())
        mean_vector = flat.mean(dim=1)
        mean_norm = float(torch.linalg.vector_norm(mean_vector).item())
        common_energy = float(token_count) * float(
            mean_vector.square().sum().item())
        token_norm = torch.linalg.vector_norm(flat, dim=0)
        mean_token_norm = float(token_norm.mean().item())
        normalized_mean = F.normalize(
            mean_vector, dim=0, eps=float(self.seed_similarity_eps))
        normalized_tokens = F.normalize(
            flat, dim=0, eps=float(self.seed_similarity_eps))
        mean_token_cosine = float(
            (normalized_tokens * normalized_mean[:, None])
            .sum(dim=0).mean().item())
        descriptor = dict(
            spatial_pixels=token_count,
            feature_channels=int(flat.shape[0]),
            total_feature_energy=total_energy,
            mean_vector_norm=mean_norm,
            mean_token_norm=mean_token_norm,
            mean_norm_ratio=_safe_div(mean_norm, mean_token_norm),
            common_energy_ratio=_safe_div(common_energy, total_energy),
            mean_token_cosine=mean_token_cosine,
            effective_rank=None,
            top1_variance_ratio=None,
            top4_variance_ratio=None,
            top8_variance_ratio=None,
        )
        if not compute_spectrum or token_count <= 1:
            return descriptor

        centered = flat - mean_vector[:, None]
        covariance = centered.matmul(centered.t())
        covariance = covariance / max(1, token_count - 1)
        try:
            eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        except RuntimeError:
            return descriptor
        eigenvalues = torch.flip(eigenvalues, dims=(0,))
        total_variance = float(eigenvalues.sum().item())
        if total_variance <= 1e-12:
            return descriptor
        probability = eigenvalues / total_variance
        positive = probability > 0
        entropy = -(
            probability[positive]
            * probability[positive].log()
        ).sum()
        descriptor.update(
            effective_rank=float(torch.exp(entropy).item()),
            top1_variance_ratio=float(probability[:1].sum().item()),
            top4_variance_ratio=float(probability[:4].sum().item()),
            top8_variance_ratio=float(probability[:8].sum().item()),
        )
        return descriptor

    def _scene_common_seed_stats(
            self, feature, seed_mask, seed_class):
        height, width = feature.shape[-2:]
        seed_mask_low = F.interpolate(
            seed_mask.float().unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode='nearest',
        ).squeeze().bool()
        seed_class_low = F.interpolate(
            seed_class.float().unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode='nearest',
        ).squeeze().long().clamp(0, self.num_cls - 1)
        prototypes = []
        for class_idx in range(self.num_cls):
            class_seed = seed_mask_low & (seed_class_low == class_idx)
            if class_seed.any():
                prototype = feature[:, class_seed].mean(dim=1)
                prototypes.append(F.normalize(
                    prototype,
                    dim=0,
                    eps=float(self.seed_similarity_eps),
                ))
        mean_interclass_cosine = None
        prototype_dispersion = None
        if len(prototypes) >= 2:
            prototype_stack = torch.stack(prototypes)
            similarity = prototype_stack.matmul(prototype_stack.t())
            off_diagonal = ~torch.eye(
                len(prototypes),
                device=similarity.device,
                dtype=torch.bool,
            )
            mean_interclass_cosine = float(
                similarity[off_diagonal].mean().item())
            prototype_dispersion = 1.0 - mean_interclass_cosine
        return dict(
            seed_pixels=int(seed_mask_low.sum().item()),
            seed_ratio=_safe_div(
                int(seed_mask_low.sum().item()), height * width),
            seed_class_count=len(prototypes),
            mean_interclass_seed_cosine=mean_interclass_cosine,
            seed_prototype_dispersion=prototype_dispersion,
        )

    def _scene_common_robust_mean(self, feature):
        flat = feature.reshape(feature.shape[0], -1)
        token_norm = torch.linalg.vector_norm(flat, dim=0)
        quantile = min(
            max(float(self.scene_common_trim_quantile), 0.0),
            0.49,
        )
        if quantile <= 0.0 or token_norm.numel() < 4:
            return flat.mean(dim=1), int(token_norm.numel())
        lower = torch.quantile(token_norm, quantile)
        upper = torch.quantile(token_norm, 1.0 - quantile)
        keep = (token_norm >= lower) & (token_norm <= upper)
        if not keep.any():
            return flat.mean(dim=1), int(token_norm.numel())
        return flat[:, keep].mean(dim=1), int(keep.sum().item())

    def _scene_common_class_balanced_mean(
            self, feature, seed_mask, seed_class):
        height, width = feature.shape[-2:]
        seed_mask_low = F.interpolate(
            seed_mask.float().unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode='nearest',
        ).squeeze().bool()
        seed_class_low = F.interpolate(
            seed_class.float().unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode='nearest',
        ).squeeze().long().clamp(0, self.num_cls - 1)
        class_means = []
        for class_idx in range(self.num_cls):
            class_seed = seed_mask_low & (seed_class_low == class_idx)
            if class_seed.any():
                class_means.append(feature[:, class_seed].mean(dim=1))
        if not class_means:
            return feature.mean(dim=(-2, -1)), 0
        return torch.stack(class_means).mean(dim=0), len(class_means)

    def _build_scene_common_bias_source_maps(
            self, class_final, feature_layers, output_shape,
            seed_mask, seed_class):
        if not (
                self.dump_scene_common_bias_stats
                or self.dump_candidate_residual_trajectory_stats
                or self.dump_candidate_residual_miou_stats):
            return {}, []
        if not isinstance(feature_layers, (list, tuple)):
            return {}, []
        if seed_mask is None or seed_class is None:
            return {}, []

        selected_layers = set()
        variants = set()
        strengths = set()
        if self.dump_scene_common_bias_stats:
            selected_layers.update(self._get_scene_common_layers(
                len(feature_layers)))
            variants.update(self._parse_name_list(
                self.scene_common_variants))
            strengths.update(
                min(max(float(value), 0.0), 1.5)
                for value in self._parse_float_list(
                    self.scene_common_strengths,
                    '0.25,0.50,0.75,1.00',
                )
            )
        if (
                self.dump_candidate_residual_trajectory_stats
                or self.dump_candidate_residual_miou_stats):
            selected_layers.update(self._get_candidate_residual_layers(
                len(feature_layers)))
            variants.update(self._parse_name_list(
                self.candidate_residual_variants))
            strengths.update(
                min(max(float(value), 0.0), 1.5)
                for value in self._parse_float_list(
                    self.candidate_residual_strengths,
                    '0.00,0.25,0.50,0.75,1.00',
                )
            )
        selected_layers = sorted(selected_layers)
        strengths = sorted(strengths)
        local_kernels = sorted({
            max(1, int(round(value)))
            for value in self._parse_float_list(
                self.scene_common_local_kernels, '7,15')
        })
        random_trials = max(
            1, int(self.scene_common_random_trials))
        source_maps = {}
        feature_stats = []
        prepared = {}
        mean_directions = {}

        for layer_idx in selected_layers:
            feature = feature_layers[layer_idx]
            if feature.ndim == 4:
                feature = feature.squeeze(0)
            if feature.ndim != 3:
                continue
            feature = feature.detach().float()
            prepared[layer_idx] = feature
            mean_vector = feature.mean(dim=(-2, -1))
            if float(torch.linalg.vector_norm(mean_vector).item()) > 1e-8:
                mean_directions.setdefault(
                    int(feature.shape[0]), []).append(
                        F.normalize(
                            mean_vector,
                            dim=0,
                            eps=float(self.seed_similarity_eps),
                        ))

        shared_directions = {}
        for channels, directions in mean_directions.items():
            shared_directions[channels] = F.normalize(
                torch.stack(directions).mean(dim=0),
                dim=0,
                eps=float(self.seed_similarity_eps),
            )

        def add_variant(
                layer_idx, name, feature, raw_feature, variant_kind,
                strength=None, extra=None):
            similarity, has_seed = (
                self._build_feature_seed_similarity_from_map(
                    feature,
                    class_final,
                    seed_mask,
                    seed_class,
                )
            )
            if similarity is None or has_seed is None:
                return
            similarity = self._interpolate_float32(
                similarity.unsqueeze(0),
                output_shape,
            ).squeeze(0)
            similarity[~has_seed, :, :] = float('nan')
            source_maps[name] = similarity

            total_energy = float(raw_feature.square().sum().item())
            changed_energy = float(
                (raw_feature - feature).square().sum().item())
            raw_mean_norm = float(torch.linalg.vector_norm(
                raw_feature.mean(dim=(-2, -1))).item())
            residual_mean_norm = float(torch.linalg.vector_norm(
                feature.mean(dim=(-2, -1))).item())
            row = dict(
                stat_type='variant',
                layer_index=int(layer_idx),
                variant_name=name,
                variant_kind=str(variant_kind),
                strength=(
                    None if strength is None else float(strength)),
                removed_feature_energy_ratio=_safe_div(
                    changed_energy, total_energy),
                residual_mean_norm_ratio=_safe_div(
                    residual_mean_norm, raw_mean_norm),
            )
            row.update(self._scene_common_feature_descriptor(
                feature, compute_spectrum=False))
            row.update(self._scene_common_seed_stats(
                feature, seed_mask, seed_class))
            if extra:
                row.update(extra)
            feature_stats.append(row)

        for layer_idx, feature in prepared.items():
            raw_descriptor = self._scene_common_feature_descriptor(
                feature,
                compute_spectrum=bool(
                    self.scene_common_compute_spectrum),
            )
            raw_descriptor.update(dict(
                stat_type='layer_descriptor',
                layer_index=int(layer_idx),
                variant_name=f'scenecommon_l{layer_idx}_raw',
                variant_kind='raw',
                strength=0.0,
                removed_feature_energy_ratio=0.0,
                residual_mean_norm_ratio=1.0,
            ))
            raw_descriptor.update(self._scene_common_seed_stats(
                feature, seed_mask, seed_class))
            feature_stats.append(raw_descriptor)
            add_variant(
                layer_idx,
                f'scenecommon_l{layer_idx}_raw',
                feature,
                feature,
                'raw',
                strength=0.0,
            )

            mean_vector = feature.mean(dim=(-2, -1))
            if 'global' in variants:
                for strength in strengths:
                    if strength <= 0.0:
                        continue
                    token = self._scene_common_strength_token(strength)
                    add_variant(
                        layer_idx,
                        f'scenecommon_l{layer_idx}_global_a{token}',
                        self._scene_common_subtract_vector(
                            feature, mean_vector, strength),
                        feature,
                        'global',
                        strength=strength,
                    )

            if 'robust' in variants:
                robust_mean, robust_tokens = (
                    self._scene_common_robust_mean(feature))
                add_variant(
                    layer_idx,
                    f'scenecommon_l{layer_idx}_robust',
                    self._scene_common_subtract_vector(
                        feature, robust_mean),
                    feature,
                    'robust',
                    strength=1.0,
                    extra=dict(robust_tokens=robust_tokens),
                )

            if 'class_balanced' in variants:
                balanced_mean, balanced_classes = (
                    self._scene_common_class_balanced_mean(
                        feature, seed_mask, seed_class))
                add_variant(
                    layer_idx,
                    f'scenecommon_l{layer_idx}_balanced',
                    self._scene_common_subtract_vector(
                        feature, balanced_mean),
                    feature,
                    'class_balanced',
                    strength=1.0,
                    extra=dict(
                        balanced_seed_classes=balanced_classes),
                )
                if (
                        self.dump_candidate_residual_trajectory_stats
                        or self.dump_candidate_residual_miou_stats):
                    for strength in strengths:
                        if strength <= 0.0:
                            continue
                        token = self._scene_common_strength_token(
                            strength)
                        add_variant(
                            layer_idx,
                            f'scenecommon_l{layer_idx}_balanced_a{token}',
                            self._scene_common_subtract_vector(
                                feature, balanced_mean, strength),
                            feature,
                            'class_balanced_trajectory',
                            strength=strength,
                            extra=dict(
                                balanced_seed_classes=balanced_classes),
                        )
                    if 'controls' in variants:
                        generator = torch.Generator(device='cpu')
                        generator.manual_seed(
                            53003 + 1009 * int(layer_idx))
                        permutation = torch.randperm(
                            int(feature.shape[0]),
                            generator=generator,
                        ).to(feature.device)
                        balanced_norm = torch.linalg.vector_norm(
                            balanced_mean)
                        random_balanced = torch.randn(
                            int(feature.shape[0]),
                            generator=generator,
                            dtype=torch.float32,
                        ).to(feature.device)
                        random_balanced = F.normalize(
                            random_balanced,
                            dim=0,
                            eps=float(self.seed_similarity_eps),
                        ) * balanced_norm
                        for strength in strengths:
                            if strength <= 0.0:
                                continue
                            token = self._scene_common_strength_token(
                                strength)
                            add_variant(
                                layer_idx,
                                f'scenecommon_l{layer_idx}_balanced_'
                                f'perm_t0_a{token}',
                                self._scene_common_subtract_vector(
                                    feature,
                                    balanced_mean[permutation],
                                    strength),
                                feature,
                                'balanced_permuted_control_trajectory',
                                strength=strength,
                                extra=dict(
                                    trial=0,
                                    balanced_seed_classes=(
                                        balanced_classes)),
                            )
                            add_variant(
                                layer_idx,
                                f'scenecommon_l{layer_idx}_balanced_'
                                f'rand_t0_a{token}',
                                self._scene_common_subtract_vector(
                                    feature,
                                    random_balanced,
                                    strength),
                                feature,
                                'balanced_random_control_trajectory',
                                strength=strength,
                                extra=dict(
                                    trial=0,
                                    balanced_seed_classes=(
                                        balanced_classes)),
                            )

            if 'local' in variants:
                max_kernel = max(
                    1,
                    2 * min(feature.shape[-2:]) - 1,
                )
                for requested_kernel in local_kernels:
                    kernel = min(requested_kernel, max_kernel)
                    if kernel % 2 == 0:
                        kernel = max(1, kernel - 1)
                    radius = kernel // 2
                    local_mean = F.avg_pool2d(
                        F.pad(
                            feature.unsqueeze(0),
                            (radius, radius, radius, radius),
                            mode='replicate',
                        ),
                        kernel_size=kernel,
                        stride=1,
                        padding=0,
                    ).squeeze(0)
                    add_variant(
                        layer_idx,
                        f'scenecommon_l{layer_idx}_local_k{kernel}',
                        feature - local_mean,
                        feature,
                        'local',
                        strength=1.0,
                        extra=dict(local_kernel=int(kernel)),
                    )

            if 'shared' in variants:
                shared_direction = shared_directions.get(
                    int(feature.shape[0]))
                if shared_direction is not None:
                    flat = feature.reshape(feature.shape[0], -1)
                    projection = (
                        shared_direction[:, None]
                        * shared_direction.matmul(flat)[None, :]
                    ).view_as(feature)
                    add_variant(
                        layer_idx,
                        f'scenecommon_l{layer_idx}_shared',
                        feature - projection,
                        feature,
                        'shared',
                        strength=1.0,
                        extra=dict(
                            shared_layer_count=len(
                                mean_directions[int(feature.shape[0])]),
                        ),
                    )

            if 'controls' in variants and 'global' in variants:
                generator = torch.Generator(device='cpu')
                generator.manual_seed(43003 + 1009 * int(layer_idx))
                permutation = torch.randperm(
                    int(feature.shape[0]),
                    generator=generator,
                ).to(feature.device)
                add_variant(
                    layer_idx,
                    f'scenecommon_l{layer_idx}_perm_t0',
                    self._scene_common_subtract_vector(
                        feature, mean_vector[permutation]),
                    feature,
                    'permuted_control',
                    strength=1.0,
                    extra=dict(trial=0),
                )
                if self.dump_candidate_residual_trajectory_stats:
                    for strength in strengths:
                        if strength <= 0.0:
                            continue
                        token = self._scene_common_strength_token(
                            strength)
                        add_variant(
                            layer_idx,
                            f'scenecommon_l{layer_idx}_perm_t0_a{token}',
                            self._scene_common_subtract_vector(
                                feature,
                                mean_vector[permutation],
                                strength),
                            feature,
                            'permuted_control_trajectory',
                            strength=strength,
                            extra=dict(trial=0),
                        )
                mean_norm = torch.linalg.vector_norm(mean_vector)
                for trial in range(random_trials):
                    generator = torch.Generator(device='cpu')
                    generator.manual_seed(
                        47017 + 1013 * int(layer_idx)
                        + 97 * int(trial))
                    random_vector = torch.randn(
                        int(feature.shape[0]),
                        generator=generator,
                        dtype=torch.float32,
                    ).to(feature.device)
                    random_vector = F.normalize(
                        random_vector,
                        dim=0,
                        eps=float(self.seed_similarity_eps),
                    ) * mean_norm
                    add_variant(
                        layer_idx,
                        f'scenecommon_l{layer_idx}_rand_t{trial}',
                        self._scene_common_subtract_vector(
                            feature, random_vector),
                        feature,
                        'random_control',
                        strength=1.0,
                        extra=dict(trial=int(trial)),
                    )
                    if self.dump_candidate_residual_trajectory_stats:
                        for strength in strengths:
                            if strength <= 0.0:
                                continue
                            token = self._scene_common_strength_token(
                                strength)
                            add_variant(
                                layer_idx,
                                f'scenecommon_l{layer_idx}_rand_t'
                                f'{trial}_a{token}',
                                self._scene_common_subtract_vector(
                                    feature,
                                    random_vector,
                                    strength),
                                feature,
                                'random_control_trajectory',
                                strength=strength,
                                extra=dict(trial=int(trial)),
                            )

        return source_maps, feature_stats

    def _build_seed_similarity_maps_by_space(self, space, base_logits, components,
                                             seed_mask, seed_class):
        if space == 'evidence':
            return self._build_seed_similarity_maps(
                base_logits, components, seed_mask, seed_class)
        if space == 'vision' or space.startswith('pe_layer_'):
            return self._build_feature_seed_similarity_maps(
                space, base_logits, components, seed_mask, seed_class)
        return None, None

    def _get_seed_similarity_spaces(self):
        spaces = self.seed_similarity_spaces
        if spaces is None:
            return ['evidence']
        if isinstance(spaces, str):
            spaces = [item.strip() for item in spaces.split(',')]
        elif isinstance(spaces, (list, tuple)):
            spaces = [str(item).strip() for item in spaces]
        else:
            spaces = [str(spaces).strip()]
        normalized = []
        for space in spaces:
            if not space:
                continue
            if space == 'pe_all':
                expanded = ['pe_layer_0', 'pe_layer_1', 'pe_layer_2', 'pe_layer_3']
                for item in expanded:
                    if item not in normalized:
                        normalized.append(item)
                continue
            is_pe_layer = (
                space.startswith('pe_layer_')
                and space[len('pe_layer_'):].isdigit()
            )
            if space not in ('evidence', 'vision') and not is_pe_layer:
                raise ValueError(
                    "seed_similarity_spaces supports 'evidence', 'vision', "
                    "'pe_all', and 'pe_layer_{idx}', "
                    f'but got {space!r}')
            if space not in normalized:
                normalized.append(space)
        return normalized or ['evidence']

    def _uses_pe_layer_similarity(self):
        if (
                self.dump_position_bias_stats
                or self.dump_scene_common_bias_stats
                or self.dump_candidate_residual_trajectory_stats
                or self.dump_candidate_residual_miou_stats
                or self._ontology_self_verification_wants_pe()
                or (
                    self._uses_cross_image_bank_features()
                    and self._cross_image_bank_uses_pe())):
            return True
        if self._uses_internal_evidence_diagnostics():
            sources = self._get_internal_selection_sources()
            if self._uses_ontology_readout_oracle():
                sources = list(sources) + list(
                    self._ontology_readout_internal_sources())
            if 'pe_all' in sources or any(
                    source.startswith('pe_layer_') for source in sources):
                return True
        if (self.dump_reject_recovery_precision_stats
                and bool(self.reject_recovery_precision_use_pe)):
            return True
        if not self.dump_seed_separability_stats:
            return False
        try:
            spaces = self._get_seed_similarity_spaces()
        except ValueError:
            return False
        return any(space.startswith('pe_layer_') for space in spaces)

    def _build_seed_separability_stats(self, base_logits, base_pred, data_sample, components):
        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)
        context = self._build_seed_rule_context(base_logits, base_pred, components)
        if gt_data is None or context is None:
            return None

        valid_mask = gt_data != 255
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return dict(valid_pixels=0, rule_stats=[], class_stats=[],
                        pair_stats=[], region_stats=[])

        gt_idx = gt_data.clamp(min=0, max=self.num_cls - 1)
        pred_idx = base_pred.clamp(min=0, max=self.num_cls - 1)
        base_correct = (base_pred == gt_data) & valid_mask
        base_wrong = (~base_correct) & valid_mask
        wrong_count = int(base_wrong.sum().item())
        seed_class = context['final_top1_idx']

        gt_present = []
        gt_pixels_by_class = []
        pred_pixels_by_class = []
        for class_idx in range(self.num_cls):
            gt_mask = (gt_data == class_idx) & valid_mask
            pred_mask = (base_pred == class_idx) & valid_mask
            gt_pixels = int(gt_mask.sum().item())
            gt_present.append(gt_pixels > 0)
            gt_pixels_by_class.append(gt_pixels)
            pred_pixels_by_class.append(int(pred_mask.sum().item()))

        rule_stats = []
        class_stats = []
        pair_stats = []
        region_stats = []

        for rule_order, (rule_name, raw_seed_mask) in enumerate(context['rule_defs']):
            seed_mask = raw_seed_mask & valid_mask
            seed_pixels = int(seed_mask.sum().item())
            seed_correct = seed_mask & base_correct
            seed_correct_pixels = int(seed_correct.sum().item())

            seed_class_count = 0
            covered_gt_class_count = 0
            for class_idx in range(self.num_cls):
                class_seed = seed_mask & (seed_class == class_idx)
                class_seed_pixels = int(class_seed.sum().item())
                if class_seed_pixels > 0:
                    seed_class_count += 1
                if gt_present[class_idx] and class_seed_pixels > 0:
                    covered_gt_class_count += 1
                class_seed_correct = int((class_seed & (gt_data == class_idx)).sum().item())
                class_stats.append(dict(
                    rule_order=rule_order,
                    rule_name=rule_name,
                    class_index=class_idx,
                    class_name=self.class_names[class_idx],
                    gt_pixels=gt_pixels_by_class[class_idx],
                    pred_pixels=pred_pixels_by_class[class_idx],
                    seed_pixels=class_seed_pixels,
                    seed_ratio=_safe_div(class_seed_pixels, valid_count),
                    seed_ratio_in_pred=_safe_div(class_seed_pixels, pred_pixels_by_class[class_idx]),
                    seed_correct_pixels=class_seed_correct,
                    seed_purity=_safe_div(class_seed_correct, class_seed_pixels),
                    gt_class_has_seed=bool(class_seed_pixels > 0),
                ))
                if class_seed_pixels > 0:
                    region_row = self._build_seed_region_row(
                        rule_order,
                        rule_name,
                        class_idx,
                        class_seed,
                        (gt_data == class_idx) & valid_mask,
                    )
                    if region_row is not None:
                        region_stats.append(region_row)

            for similarity_space in self._get_seed_similarity_spaces():
                similarity_maps, has_seed = self._build_seed_similarity_maps_by_space(
                    similarity_space, base_logits, components, seed_mask, seed_class)
                if similarity_maps is None or has_seed is None:
                    continue
                gt_has_seed = has_seed[gt_idx] & valid_mask
                pred_has_seed = has_seed[pred_idx] & valid_mask
                wrong_gt_has_seed = base_wrong & gt_has_seed
                wrong_pred_has_seed = base_wrong & pred_has_seed
                wrong_both_seed = wrong_gt_has_seed & wrong_pred_has_seed
                sim_to_pred_seed = _gather_class_map(similarity_maps, pred_idx)
                sim_to_gt_seed = _gather_class_map(similarity_maps, gt_idx)
                sim_margin = sim_to_gt_seed - sim_to_pred_seed
                sim_favors_gt = wrong_both_seed & (sim_margin > 0)

                wrong_pred_seed_pixels = int(wrong_pred_has_seed.sum().item())
                wrong_gt_seed_pixels = int(wrong_gt_has_seed.sum().item())
                wrong_both_seed_pixels = int(wrong_both_seed.sum().item())
                pred_sim_sum = _masked_sum(sim_to_pred_seed, wrong_pred_has_seed)
                gt_sim_sum = _masked_sum(sim_to_gt_seed, wrong_gt_has_seed)
                margin_sum = _masked_sum(sim_margin, wrong_both_seed)
                rule_stats.append(dict(
                    rule_order=rule_order,
                    rule_name=rule_name,
                    similarity_space=similarity_space,
                    valid_pixels=valid_count,
                    baseline_correct_pixels=int(base_correct.sum().item()),
                    baseline_wrong_pixels=wrong_count,
                    seed_pixels=seed_pixels,
                    seed_ratio=_safe_div(seed_pixels, valid_count),
                    seed_correct_pixels=seed_correct_pixels,
                    seed_purity=_safe_div(seed_correct_pixels, seed_pixels),
                    gt_present_class_count=int(sum(gt_present)),
                    seed_class_count=seed_class_count,
                    covered_gt_class_count=covered_gt_class_count,
                    class_coverage_ratio=_safe_div(covered_gt_class_count, int(sum(gt_present))),
                    wrong_gt_class_has_seed_pixels=wrong_gt_seed_pixels,
                    wrong_gt_class_has_seed_ratio=_safe_div(wrong_gt_seed_pixels, wrong_count),
                    wrong_pred_class_has_seed_pixels=wrong_pred_seed_pixels,
                    wrong_pred_class_has_seed_ratio=_safe_div(wrong_pred_seed_pixels, wrong_count),
                    wrong_both_seed_available_pixels=wrong_both_seed_pixels,
                    wrong_both_seed_available_ratio=_safe_div(wrong_both_seed_pixels, wrong_count),
                    wrong_similarity_to_pred_seed_sum=pred_sim_sum,
                    wrong_similarity_to_pred_seed_pixels=wrong_pred_seed_pixels,
                    mean_wrong_similarity_to_pred_seed=_safe_div(pred_sim_sum, wrong_pred_seed_pixels),
                    wrong_similarity_to_gt_seed_sum=gt_sim_sum,
                    wrong_similarity_to_gt_seed_pixels=wrong_gt_seed_pixels,
                    mean_wrong_similarity_to_gt_seed=_safe_div(gt_sim_sum, wrong_gt_seed_pixels),
                    wrong_similarity_margin_sum=margin_sum,
                    wrong_similarity_margin_pixels=wrong_both_seed_pixels,
                    mean_wrong_similarity_margin=_safe_div(margin_sum, wrong_both_seed_pixels),
                    wrong_seed_similarity_favors_gt_pixels=int(sim_favors_gt.sum().item()),
                    wrong_seed_similarity_favors_gt_ratio=_safe_div(
                        int(sim_favors_gt.sum().item()), wrong_both_seed_pixels),
                ))

                for gt_class in range(self.num_cls):
                    gt_wrong_mask = base_wrong & (gt_data == gt_class)
                    gt_wrong_pixels = int(gt_wrong_mask.sum().item())
                    if gt_wrong_pixels == 0:
                        continue
                    for pred_class in range(self.num_cls):
                        if pred_class == gt_class:
                            continue
                        pair_mask = gt_wrong_mask & (base_pred == pred_class)
                        pixels = int(pair_mask.sum().item())
                        if pixels == 0:
                            continue
                        pair_pred_seed = pair_mask & pred_has_seed
                        pair_gt_seed = pair_mask & gt_has_seed
                        pair_both_seed = pair_pred_seed & pair_gt_seed
                        pair_sim_favors_gt = pair_both_seed & (sim_margin > 0)
                        pair_pred_sim_sum = _masked_sum(sim_to_pred_seed, pair_pred_seed)
                        pair_gt_sim_sum = _masked_sum(sim_to_gt_seed, pair_gt_seed)
                        pair_margin_sum = _masked_sum(sim_margin, pair_both_seed)
                        pair_stats.append(dict(
                            rule_order=rule_order,
                            rule_name=rule_name,
                            similarity_space=similarity_space,
                            gt_class_index=gt_class,
                            gt_class_name=self.class_names[gt_class],
                            base_pred_class_index=pred_class,
                            base_pred_class_name=self.class_names[pred_class],
                            pixels=pixels,
                            pixel_ratio_in_error=_safe_div(pixels, wrong_count),
                            pixel_ratio_in_gt_wrong=_safe_div(pixels, gt_wrong_pixels),
                            gt_class_has_seed_pixels=int(pair_gt_seed.sum().item()),
                            gt_class_has_seed_ratio=_safe_div(int(pair_gt_seed.sum().item()), pixels),
                            pred_class_has_seed_pixels=int(pair_pred_seed.sum().item()),
                            pred_class_has_seed_ratio=_safe_div(int(pair_pred_seed.sum().item()), pixels),
                            both_seed_available_pixels=int(pair_both_seed.sum().item()),
                            both_seed_available_ratio=_safe_div(int(pair_both_seed.sum().item()), pixels),
                            similarity_to_pred_seed_sum=pair_pred_sim_sum,
                            similarity_to_pred_seed_pixels=int(pair_pred_seed.sum().item()),
                            mean_similarity_to_pred_seed=_safe_div(
                                pair_pred_sim_sum, int(pair_pred_seed.sum().item())),
                            similarity_to_gt_seed_sum=pair_gt_sim_sum,
                            similarity_to_gt_seed_pixels=int(pair_gt_seed.sum().item()),
                            mean_similarity_to_gt_seed=_safe_div(
                                pair_gt_sim_sum, int(pair_gt_seed.sum().item())),
                            similarity_margin_sum=pair_margin_sum,
                            similarity_margin_pixels=int(pair_both_seed.sum().item()),
                            mean_similarity_margin=_safe_div(
                                pair_margin_sum, int(pair_both_seed.sum().item())),
                            seed_similarity_favors_gt_pixels=int(pair_sim_favors_gt.sum().item()),
                            seed_similarity_favors_gt_ratio=_safe_div(
                                int(pair_sim_favors_gt.sum().item()),
                                int(pair_both_seed.sum().item())),
                        ))

        pair_stats.sort(key=lambda item: item['pixels'], reverse=True)
        return dict(
            dataset_name=self.seed_dataset_name,
            valid_pixels=valid_count,
            baseline_correct_pixels=int(base_correct.sum().item()),
            baseline_wrong_pixels=wrong_count,
            seed_final_score_thd=float(self.seed_final_score_thd),
            seed_margin_thd=float(self.seed_margin_thd),
            seed_local_kernel=int(self.seed_local_kernel),
            seed_local_consistency_thd=float(self.seed_local_consistency_thd),
            seed_core_kernel=int(self.seed_core_kernel),
            seed_core_consistency_thd=float(self.seed_core_consistency_thd),
            seed_similarity_spaces=self._get_seed_similarity_spaces(),
            multiview_rule_available=False,
            rule_stats=rule_stats,
            class_stats=class_stats,
            pair_stats=pair_stats,
            region_stats=region_stats,
        )

    def _build_seed_region_row(self, rule_order, rule_name, class_idx,
                               seed_mask, correct_mask):
        stats = _connected_component_summary(
            seed_mask,
            correct_mask,
            min_pixels=int(self.seed_region_min_pixels),
            purity_threshold=float(self.seed_region_purity_thd),
        )
        if stats['region_count'] == 0:
            return None
        row = dict(
            rule_order=rule_order,
            rule_name=rule_name,
            class_index=class_idx,
            class_name=self.class_names[class_idx],
        )
        row.update(stats)
        return row

    def _write_seed_separability_stats(self, record):
        if not self.dump_seed_separability_stats:
            return
        if self._seed_separability_stats_file is None:
            path = (self.seed_separability_stats_path
                    or './work_dirs/evidence_stats/seed_separability_stats.jsonl')
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._seed_separability_stats_file = open(path, 'a', buffering=1)
        self._seed_separability_stats_file.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _build_raw_mask_oracle_stats(self, base_logits, base_pred, data_sample, components):
        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)
        raw_candidates = None if components is None else components.get('raw_mask_candidates')
        if gt_data is None or not raw_candidates:
            return None

        valid_mask = gt_data != 255
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return dict(valid_pixels=0, class_stats=[], pair_stats=[], candidate_stats=[])

        gt_idx = gt_data.clamp(min=0, max=self.num_cls - 1)
        pred_idx = base_pred.clamp(min=0, max=self.num_cls - 1)
        base_correct = (base_pred == gt_data) & valid_mask
        base_wrong = (~base_correct) & valid_mask
        wrong_count = int(base_wrong.sum().item())

        seed_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        seed_class = base_pred.clamp(min=0, max=self.num_cls - 1)
        seed_context = self._build_seed_rule_context(base_logits, base_pred, components)
        if seed_context is not None:
            seed_mask = seed_context['rule_defs'][-1][1] & valid_mask
            seed_class = seed_context['final_top1_idx'].clamp(min=0, max=self.num_cls - 1)

        gt_pixels_by_class = []
        pred_pixels_by_class = []
        wrong_pixels_by_class = []
        seed_pixels_by_class = []
        for class_idx in range(self.num_cls):
            gt_mask = (gt_data == class_idx) & valid_mask
            pred_mask = (base_pred == class_idx) & valid_mask
            wrong_mask = base_wrong & (gt_data == class_idx)
            class_seed = seed_mask & (seed_class == class_idx)
            gt_pixels_by_class.append(int(gt_mask.sum().item()))
            pred_pixels_by_class.append(int(pred_mask.sum().item()))
            wrong_pixels_by_class.append(int(wrong_mask.sum().item()))
            seed_pixels_by_class.append(int(class_seed.sum().item()))

        class_rows = {}
        candidate_stats = []
        candidates_by_class = {idx: [] for idx in range(self.num_cls)}
        for prompt_record in raw_candidates:
            class_idx = int(prompt_record['class_index'])
            if class_idx < 0 or class_idx >= self.num_cls:
                continue
            selected_indices = prompt_record['selected_indices']
            raw_masks = prompt_record['raw_masks_lowres']
            raw_scores = prompt_record['raw_scores']
            presence_scores = prompt_record['raw_presence_scores']
            keep_mask = prompt_record['raw_keep_mask']
            for local_idx in range(int(raw_masks.shape[0])):
                mask_prob = torch.sigmoid(
                    F.interpolate(
                        raw_masks[local_idx].view(1, 1, *raw_masks[local_idx].shape).to(self.device),
                        size=gt_data.shape[-2:],
                        mode='bilinear',
                        align_corners=False,
                    ).squeeze())
                mask = (mask_prob >= float(self.raw_mask_oracle_bin_thd)) & valid_mask
                mask_pixels = int(mask.sum().item())
                if mask_pixels < int(self.raw_mask_oracle_min_pixels):
                    continue
                gt_mask = (gt_data == class_idx) & valid_mask
                class_wrong = base_wrong & (gt_data == class_idx)
                class_seed = seed_mask & (seed_class == class_idx)
                correct_pixels = int((mask & gt_mask).sum().item())
                wrong_cover_pixels = int((mask & class_wrong).sum().item())
                seed_cover_pixels = int((mask & class_seed).sum().item())
                row = dict(
                    class_index=class_idx,
                    class_name=self.class_names[class_idx],
                    query_index=int(prompt_record['query_index']),
                    query_word=prompt_record['query_word'],
                    raw_candidate_index=int(selected_indices[local_idx].item()),
                    raw_candidate_count=int(prompt_record.get('raw_candidate_count', 0)),
                    raw_kept_count=int(prompt_record.get('raw_kept_count', 0)),
                    selected_local_index=int(local_idx),
                    raw_score=float(raw_scores[local_idx].item()),
                    raw_presence_score=float(presence_scores[local_idx].item()),
                    kept=bool(keep_mask[local_idx].item()),
                    mask_pixels=mask_pixels,
                    mask_ratio=_safe_div(mask_pixels, valid_count),
                    gt_pixels=gt_pixels_by_class[class_idx],
                    pred_pixels=pred_pixels_by_class[class_idx],
                    wrong_pixels=wrong_pixels_by_class[class_idx],
                    seed_pixels=seed_pixels_by_class[class_idx],
                    correct_pixels=correct_pixels,
                    purity=_safe_div(correct_pixels, mask_pixels),
                    gt_recall=_safe_div(correct_pixels, gt_pixels_by_class[class_idx]),
                    wrong_cover_pixels=wrong_cover_pixels,
                    wrong_coverage=_safe_div(wrong_cover_pixels, wrong_pixels_by_class[class_idx]),
                    seed_cover_pixels=seed_cover_pixels,
                    seed_coverage=_safe_div(seed_cover_pixels, seed_pixels_by_class[class_idx]),
                )
                candidate_stats.append(row)
                candidates_by_class[class_idx].append(dict(row=row, mask=mask))

        for class_idx in range(self.num_cls):
            candidates = candidates_by_class[class_idx]
            best_purity = _best_candidate_row(candidates, 'purity')
            best_gt_recall = _best_candidate_row(candidates, 'gt_recall')
            best_wrong_coverage = _best_candidate_row(candidates, 'wrong_coverage')
            best_seed_coverage = _best_candidate_row(candidates, 'seed_coverage')
            class_rows[class_idx] = dict(
                class_index=class_idx,
                class_name=self.class_names[class_idx],
                gt_pixels=gt_pixels_by_class[class_idx],
                pred_pixels=pred_pixels_by_class[class_idx],
                wrong_pixels=wrong_pixels_by_class[class_idx],
                seed_pixels=seed_pixels_by_class[class_idx],
                candidate_count=len(candidates),
                kept_candidate_count=sum(1 for item in candidates if item['row']['kept']),
                best_purity=None if best_purity is None else best_purity['row']['purity'],
                best_purity_pixels=0 if best_purity is None else best_purity['row']['mask_pixels'],
                best_purity_kept=None if best_purity is None else best_purity['row']['kept'],
                best_gt_recall=None if best_gt_recall is None else best_gt_recall['row']['gt_recall'],
                best_gt_recall_pixels=0 if best_gt_recall is None else best_gt_recall['row']['correct_pixels'],
                best_gt_recall_kept=None if best_gt_recall is None else best_gt_recall['row']['kept'],
                best_wrong_coverage=None if best_wrong_coverage is None else best_wrong_coverage['row']['wrong_coverage'],
                best_wrong_cover_pixels=0 if best_wrong_coverage is None else best_wrong_coverage['row']['wrong_cover_pixels'],
                best_wrong_coverage_kept=None if best_wrong_coverage is None else best_wrong_coverage['row']['kept'],
                best_seed_coverage=None if best_seed_coverage is None else best_seed_coverage['row']['seed_coverage'],
                best_seed_cover_pixels=0 if best_seed_coverage is None else best_seed_coverage['row']['seed_cover_pixels'],
                best_seed_coverage_kept=None if best_seed_coverage is None else best_seed_coverage['row']['kept'],
            )

        pair_stats = []
        for gt_class in range(self.num_cls):
            gt_wrong_mask = base_wrong & (gt_data == gt_class)
            gt_wrong_pixels = int(gt_wrong_mask.sum().item())
            if gt_wrong_pixels == 0:
                continue
            for pred_class in range(self.num_cls):
                if pred_class == gt_class:
                    continue
                pair_mask = gt_wrong_mask & (pred_idx == pred_class)
                pair_pixels = int(pair_mask.sum().item())
                if pair_pixels == 0:
                    continue
                gt_best = self._best_raw_candidate_for_mask(
                    candidates_by_class[gt_class], pair_mask)
                pred_best = self._best_raw_candidate_for_mask(
                    candidates_by_class[pred_class], pair_mask)
                gt_cover_pixels = 0 if gt_best is None else gt_best['cover_pixels']
                pred_cover_pixels = 0 if pred_best is None else pred_best['cover_pixels']
                pair_stats.append(dict(
                    gt_class_index=gt_class,
                    gt_class_name=self.class_names[gt_class],
                    base_pred_class_index=pred_class,
                    base_pred_class_name=self.class_names[pred_class],
                    pixels=pair_pixels,
                    pixel_ratio_in_error=_safe_div(pair_pixels, wrong_count),
                    pixel_ratio_in_gt_wrong=_safe_div(pair_pixels, gt_wrong_pixels),
                    gt_candidate_count=len(candidates_by_class[gt_class]),
                    pred_candidate_count=len(candidates_by_class[pred_class]),
                    gt_best_pair_cover_pixels=gt_cover_pixels,
                    pred_best_pair_cover_pixels=pred_cover_pixels,
                    gt_best_pair_coverage=_safe_div(gt_cover_pixels, pair_pixels),
                    pred_best_pair_coverage=_safe_div(pred_cover_pixels, pair_pixels),
                    pair_coverage_margin=_safe_div(gt_cover_pixels - pred_cover_pixels, pair_pixels),
                    raw_support_favors_gt=bool(gt_cover_pixels > pred_cover_pixels),
                    gt_best_kept=None if gt_best is None else gt_best['row']['kept'],
                    pred_best_kept=None if pred_best is None else pred_best['row']['kept'],
                    gt_best_raw_score=None if gt_best is None else gt_best['row']['raw_score'],
                    pred_best_raw_score=None if pred_best is None else pred_best['row']['raw_score'],
                    gt_best_presence_score=None if gt_best is None else gt_best['row']['raw_presence_score'],
                    pred_best_presence_score=None if pred_best is None else pred_best['row']['raw_presence_score'],
                    gt_best_purity=None if gt_best is None else gt_best['row']['purity'],
                    pred_best_purity=None if pred_best is None else pred_best['row']['purity'],
                ))

        pair_stats.sort(key=lambda item: item['pixels'], reverse=True)
        return dict(
            dataset_name=self.seed_dataset_name,
            valid_pixels=valid_count,
            baseline_correct_pixels=int(base_correct.sum().item()),
            baseline_wrong_pixels=wrong_count,
            raw_mask_oracle_topk=int(self.raw_mask_oracle_topk),
            raw_mask_oracle_bin_thd=float(self.raw_mask_oracle_bin_thd),
            raw_mask_oracle_min_pixels=int(self.raw_mask_oracle_min_pixels),
            seed_rule_for_core='final_score_margin_sem_inst_final_agree_local_core',
            candidate_stats=candidate_stats,
            class_stats=list(class_rows.values()),
            pair_stats=pair_stats,
        )

    def _best_raw_candidate_for_mask(self, candidates, target_mask):
        target_pixels = int(target_mask.sum().item())
        if target_pixels == 0:
            return None
        best = None
        best_pixels = -1
        for candidate in candidates:
            cover_pixels = int((candidate['mask'] & target_mask).sum().item())
            if cover_pixels > best_pixels:
                best_pixels = cover_pixels
                best = dict(candidate)
                best['cover_pixels'] = cover_pixels
        return best

    def _write_raw_mask_oracle_stats(self, record):
        if not self.dump_raw_mask_oracle_stats:
            return
        if self._raw_mask_oracle_stats_file is None:
            path = (self.raw_mask_oracle_stats_path
                    or './work_dirs/evidence_stats/raw_mask_oracle_stats.jsonl')
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._raw_mask_oracle_stats_file = open(path, 'a', buffering=1)
        self._raw_mask_oracle_stats_file.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _build_candidate_quality_stats(self, base_logits, base_pred, data_sample, components):
        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)
        raw_candidates = None if components is None else components.get('raw_mask_candidates')
        semantic_logits = None if components is None else components.get('semantic_logits')
        instance_logits = None if components is None else components.get('instance_logits')
        if (gt_data is None or not raw_candidates
                or semantic_logits is None or instance_logits is None):
            return None

        valid_mask = gt_data != 255
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return dict(valid_pixels=0, candidate_stats=[], pair_quality_stats=[])

        gt_idx = gt_data.clamp(min=0, max=self.num_cls - 1)
        pred_idx = base_pred.clamp(min=0, max=self.num_cls - 1)
        base_correct = (base_pred == gt_data) & valid_mask
        base_wrong = (~base_correct) & valid_mask
        wrong_count = int(base_wrong.sum().item())

        top2_vals, top2_idx = torch.topk(base_logits, k=min(2, self.num_cls), dim=0)
        final_top1_score = top2_vals[0]
        final_top1_idx = top2_idx[0]
        if self.num_cls > 1:
            final_top2_score = top2_vals[1]
        else:
            final_top2_score = torch.zeros_like(final_top1_score)
        final_margin = final_top1_score - final_top2_score
        semantic_top1_idx = torch.argmax(semantic_logits, dim=0)
        instance_top1_idx = torch.argmax(instance_logits, dim=0)
        local_consistency = self._same_class_local_consistency(
            final_top1_idx, self.seed_local_kernel)

        seed_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        seed_class = final_top1_idx.clamp(min=0, max=self.num_cls - 1)
        seed_context = self._build_seed_rule_context(base_logits, base_pred, components)
        if seed_context is not None:
            seed_mask = seed_context['rule_defs'][-1][1] & valid_mask
            seed_class = seed_context['final_top1_idx'].clamp(min=0, max=self.num_cls - 1)

        gt_pixels_by_class = []
        wrong_pixels_by_class = []
        seed_pixels_by_class = []
        for class_idx in range(self.num_cls):
            gt_mask = (gt_data == class_idx) & valid_mask
            wrong_mask = base_wrong & (gt_data == class_idx)
            class_seed = seed_mask & (seed_class == class_idx)
            gt_pixels_by_class.append(int(gt_mask.sum().item()))
            wrong_pixels_by_class.append(int(wrong_mask.sum().item()))
            seed_pixels_by_class.append(int(class_seed.sum().item()))

        candidate_stats = []
        candidates_by_class = {idx: [] for idx in range(self.num_cls)}
        for prompt_record in raw_candidates:
            class_idx = int(prompt_record['class_index'])
            if class_idx < 0 or class_idx >= self.num_cls:
                continue
            selected_indices = prompt_record['selected_indices']
            raw_masks = prompt_record['raw_masks_lowres']
            raw_scores = prompt_record['raw_scores']
            presence_scores = prompt_record['raw_presence_scores']
            keep_mask = prompt_record['raw_keep_mask']
            for local_idx in range(int(raw_masks.shape[0])):
                mask_prob = torch.sigmoid(
                    F.interpolate(
                        raw_masks[local_idx].view(1, 1, *raw_masks[local_idx].shape).to(self.device),
                        size=gt_data.shape[-2:],
                        mode='bilinear',
                        align_corners=False,
                    ).squeeze())
                mask = (mask_prob >= float(self.raw_mask_oracle_bin_thd)) & valid_mask
                mask_pixels = int(mask.sum().item())
                if mask_pixels < int(self.raw_mask_oracle_min_pixels):
                    continue

                gt_class_mask = (gt_data == class_idx) & valid_mask
                class_wrong = base_wrong & (gt_data == class_idx)
                own_seed_mask = seed_mask & (seed_class == class_idx)
                other_seed_mask = seed_mask & (seed_class != class_idx)
                correct_pixels = int((mask & gt_class_mask).sum().item())
                wrong_cover_pixels = int((mask & class_wrong).sum().item())
                own_seed_pixels = int((mask & own_seed_mask).sum().item())
                other_seed_pixels = int((mask & other_seed_mask).sum().item())

                final_class_map = base_logits[class_idx]
                semantic_class_map = semantic_logits[class_idx]
                instance_class_map = instance_logits[class_idx]
                final_agree = final_top1_idx == class_idx
                semantic_agree = semantic_top1_idx == class_idx
                instance_agree = instance_top1_idx == class_idx
                tri_agree = final_agree & semantic_agree & instance_agree
                sem_final_agree = semantic_agree & final_agree
                inst_final_agree = instance_agree & final_agree
                pred_mode = _mode_class(pred_idx, mask)
                gt_mode = _mode_class(gt_idx, mask)
                core_mask = _binary_erode(mask, int(self.seed_core_kernel))
                core_pixels = int((core_mask & mask).sum().item())
                bbox_fill_ratio = _bbox_fill_ratio(mask, mask_pixels)

                final_class_mean = _masked_mean(final_class_map, mask)
                semantic_class_mean = _masked_mean(semantic_class_map, mask)
                instance_class_mean = _masked_mean(instance_class_map, mask)
                final_margin_mean = _masked_mean(final_margin, mask)
                final_agreement_ratio = _masked_mean(final_agree.float(), mask)
                semantic_agreement_ratio = _masked_mean(semantic_agree.float(), mask)
                instance_agreement_ratio = _masked_mean(instance_agree.float(), mask)
                tri_agreement_ratio = _masked_mean(tri_agree.float(), mask)
                sem_final_agreement_ratio = _masked_mean(sem_final_agree.float(), mask)
                inst_final_agreement_ratio = _masked_mean(inst_final_agree.float(), mask)
                local_consistency_mean = _masked_mean(local_consistency, mask)
                own_seed_overlap = _safe_div(own_seed_pixels, mask_pixels)
                other_seed_overlap = _safe_div(other_seed_pixels, mask_pixels)
                own_seed_coverage = _safe_div(own_seed_pixels, seed_pixels_by_class[class_idx])
                purity = _safe_div(correct_pixels, mask_pixels)

                evidence_mean = _mean_existing([
                    final_class_mean,
                    semantic_class_mean,
                    instance_class_mean,
                ])
                agreement_mean = _mean_existing([
                    final_agreement_ratio,
                    semantic_agreement_ratio,
                    instance_agreement_ratio,
                ])
                seed_clean_score = (
                    _zero_if_none(own_seed_overlap)
                    - _zero_if_none(other_seed_overlap)
                    + 0.5 * _zero_if_none(_safe_div(core_pixels, mask_pixels))
                    + 0.25 * _zero_if_none(bbox_fill_ratio)
                    + 0.5 * _zero_if_none(tri_agreement_ratio)
                )
                evidence_consistency_score = (
                    _zero_if_none(evidence_mean)
                    + 0.5 * _zero_if_none(agreement_mean)
                    + 0.25 * _zero_if_none(final_margin_mean)
                    - 0.25 * _zero_if_none(_safe_div(mask_pixels, valid_count))
                )
                hybrid_quality_score = (
                    seed_clean_score
                    + evidence_consistency_score
                    - 0.25 * _zero_if_none(_safe_div(mask_pixels, valid_count))
                )

                row = dict(
                    class_index=class_idx,
                    class_name=self.class_names[class_idx],
                    query_index=int(prompt_record['query_index']),
                    query_word=prompt_record['query_word'],
                    raw_candidate_index=int(selected_indices[local_idx].item()),
                    selected_local_index=int(local_idx),
                    raw_candidate_count=int(prompt_record.get('raw_candidate_count', 0)),
                    raw_kept_count=int(prompt_record.get('raw_kept_count', 0)),
                    raw_score=float(raw_scores[local_idx].item()),
                    raw_presence_score=float(presence_scores[local_idx].item()),
                    kept=bool(keep_mask[local_idx].item()),
                    mask_pixels=mask_pixels,
                    mask_ratio=_safe_div(mask_pixels, valid_count),
                    mask_prob_mean=_masked_mean(mask_prob, mask),
                    core_pixels=core_pixels,
                    core_ratio=_safe_div(core_pixels, mask_pixels),
                    bbox_fill_ratio=bbox_fill_ratio,
                    final_class_mean=final_class_mean,
                    semantic_class_mean=semantic_class_mean,
                    instance_class_mean=instance_class_mean,
                    evidence_mean=evidence_mean,
                    final_margin_mean=final_margin_mean,
                    final_agreement_ratio=final_agreement_ratio,
                    semantic_agreement_ratio=semantic_agreement_ratio,
                    instance_agreement_ratio=instance_agreement_ratio,
                    tri_agreement_ratio=tri_agreement_ratio,
                    sem_final_agreement_ratio=sem_final_agreement_ratio,
                    inst_final_agreement_ratio=inst_final_agreement_ratio,
                    local_consistency_mean=local_consistency_mean,
                    own_seed_overlap=own_seed_overlap,
                    other_seed_overlap=other_seed_overlap,
                    own_seed_coverage=own_seed_coverage,
                    seed_clean_score=seed_clean_score,
                    evidence_consistency_score=evidence_consistency_score,
                    hybrid_quality_score=hybrid_quality_score,
                    pred_mode_class_index=pred_mode,
                    pred_mode_class_name=None if pred_mode is None else self.class_names[pred_mode],
                    gt_mode_class_index=gt_mode,
                    gt_mode_class_name=None if gt_mode is None else self.class_names[gt_mode],
                    pred_mode_is_candidate=bool(pred_mode == class_idx) if pred_mode is not None else None,
                    gt_mode_is_candidate=bool(gt_mode == class_idx) if gt_mode is not None else None,
                    gt_pixels=gt_pixels_by_class[class_idx],
                    wrong_pixels=wrong_pixels_by_class[class_idx],
                    seed_pixels=seed_pixels_by_class[class_idx],
                    correct_pixels=correct_pixels,
                    purity=purity,
                    is_clean=bool(purity is not None and purity >= float(self.candidate_quality_clean_purity_thd)),
                    gt_recall=_safe_div(correct_pixels, gt_pixels_by_class[class_idx]),
                    wrong_cover_pixels=wrong_cover_pixels,
                    wrong_coverage=_safe_div(wrong_cover_pixels, wrong_pixels_by_class[class_idx]),
                )
                candidate_stats.append(row)
                candidates_by_class[class_idx].append(dict(row=row, mask=mask))

        pair_quality_stats = []
        score_names = [
            'raw_score',
            'raw_presence_score',
            'seed_clean_score',
            'evidence_consistency_score',
            'hybrid_quality_score',
            'own_seed_overlap',
            'tri_agreement_ratio',
        ]
        for gt_class in range(self.num_cls):
            gt_wrong_mask = base_wrong & (gt_data == gt_class)
            gt_wrong_pixels = int(gt_wrong_mask.sum().item())
            if gt_wrong_pixels == 0:
                continue
            for pred_class in range(self.num_cls):
                if pred_class == gt_class:
                    continue
                pair_mask = gt_wrong_mask & (pred_idx == pred_class)
                pair_pixels = int(pair_mask.sum().item())
                if pair_pixels == 0:
                    continue
                for score_name in score_names:
                    gt_best = _best_candidate_row(candidates_by_class[gt_class], score_name)
                    pred_best = _best_candidate_row(candidates_by_class[pred_class], score_name)
                    gt_cover_pixels = (
                        0 if gt_best is None else int((gt_best['mask'] & pair_mask).sum().item()))
                    pred_cover_pixels = (
                        0 if pred_best is None else int((pred_best['mask'] & pair_mask).sum().item()))
                    gt_score = None if gt_best is None else gt_best['row'].get(score_name)
                    pred_score = None if pred_best is None else pred_best['row'].get(score_name)
                    gt_purity = None if gt_best is None else gt_best['row'].get('purity')
                    pred_purity = None if pred_best is None else pred_best['row'].get('purity')
                    pair_quality_stats.append(dict(
                        score_name=score_name,
                        gt_class_index=gt_class,
                        gt_class_name=self.class_names[gt_class],
                        base_pred_class_index=pred_class,
                        base_pred_class_name=self.class_names[pred_class],
                        pixels=pair_pixels,
                        pixel_ratio_in_error=_safe_div(pair_pixels, wrong_count),
                        pixel_ratio_in_gt_wrong=_safe_div(pair_pixels, gt_wrong_pixels),
                        gt_candidate_count=len(candidates_by_class[gt_class]),
                        pred_candidate_count=len(candidates_by_class[pred_class]),
                        gt_selected_score=gt_score,
                        pred_selected_score=pred_score,
                        score_margin=None if gt_score is None or pred_score is None else gt_score - pred_score,
                        score_favors_gt=(
                            None if gt_score is None or pred_score is None else bool(gt_score > pred_score)),
                        gt_selected_purity=gt_purity,
                        pred_selected_purity=pred_purity,
                        selected_purity_margin=(
                            None if gt_purity is None or pred_purity is None else gt_purity - pred_purity),
                        selected_purity_favors_gt=(
                            None if gt_purity is None or pred_purity is None else bool(gt_purity > pred_purity)),
                        gt_selected_clean=None if gt_best is None else gt_best['row'].get('is_clean'),
                        pred_selected_clean=None if pred_best is None else pred_best['row'].get('is_clean'),
                        gt_selected_pair_cover_pixels=gt_cover_pixels,
                        pred_selected_pair_cover_pixels=pred_cover_pixels,
                        gt_selected_pair_coverage=_safe_div(gt_cover_pixels, pair_pixels),
                        pred_selected_pair_coverage=_safe_div(pred_cover_pixels, pair_pixels),
                        selected_pair_coverage_margin=_safe_div(
                            gt_cover_pixels - pred_cover_pixels, pair_pixels),
                        gt_selected_kept=None if gt_best is None else gt_best['row'].get('kept'),
                        pred_selected_kept=None if pred_best is None else pred_best['row'].get('kept'),
                        gt_selected_mask_ratio=None if gt_best is None else gt_best['row'].get('mask_ratio'),
                        pred_selected_mask_ratio=None if pred_best is None else pred_best['row'].get('mask_ratio'),
                        gt_selected_own_seed_overlap=None if gt_best is None else gt_best['row'].get('own_seed_overlap'),
                        pred_selected_own_seed_overlap=None if pred_best is None else pred_best['row'].get('own_seed_overlap'),
                        gt_selected_other_seed_overlap=None if gt_best is None else gt_best['row'].get('other_seed_overlap'),
                        pred_selected_other_seed_overlap=None if pred_best is None else pred_best['row'].get('other_seed_overlap'),
                    ))

        pair_quality_stats.sort(key=lambda item: item['pixels'], reverse=True)
        return dict(
            dataset_name=self.seed_dataset_name,
            valid_pixels=valid_count,
            baseline_correct_pixels=int(base_correct.sum().item()),
            baseline_wrong_pixels=wrong_count,
            raw_mask_oracle_topk=int(self.raw_mask_oracle_topk),
            raw_mask_oracle_bin_thd=float(self.raw_mask_oracle_bin_thd),
            raw_mask_oracle_min_pixels=int(self.raw_mask_oracle_min_pixels),
            candidate_quality_clean_purity_thd=float(self.candidate_quality_clean_purity_thd),
            score_names=score_names,
            candidate_stats=candidate_stats,
            pair_quality_stats=pair_quality_stats,
        )

    def _write_candidate_quality_stats(self, record):
        if not self.dump_candidate_quality_stats:
            return
        if self._candidate_quality_stats_file is None:
            path = (self.candidate_quality_stats_path
                    or './work_dirs/evidence_stats/candidate_quality_stats.jsonl')
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._candidate_quality_stats_file = open(path, 'a', buffering=1)
        self._candidate_quality_stats_file.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _compute_prompt_variant_logit(self, inference_state, h, w):
        variant_logit = torch.zeros((h, w), device=self.device)
        semantic_component = None
        instance_component = torch.zeros((h, w), device=self.device)

        if self.use_transformer_decoder and inference_state['masks_logits'].shape[0] > 0:
            inst_len = inference_state['masks_logits'].shape[0]
            for inst_id in range(inst_len):
                instance_logits = inference_state['masks_logits'][inst_id].squeeze()
                instance_score = self._get_instance_score(inference_state, inst_id)
                if instance_logits.shape != (h, w):
                    instance_logits = F.interpolate(
                        instance_logits.view(1, 1, *instance_logits.shape),
                        size=(h, w),
                        mode='bilinear',
                        align_corners=False,
                    ).squeeze()
                instance_weighted = instance_logits * instance_score
                instance_component = torch.max(instance_component, instance_weighted)
                variant_logit = torch.max(variant_logit, instance_weighted)

        if self.use_sem_seg:
            semantic_logits = inference_state['semantic_mask_logits']
            if semantic_logits.shape != (h, w):
                semantic_logits = F.interpolate(
                    semantic_logits,
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze()
            semantic_component = semantic_logits
            variant_logit = torch.max(variant_logit, semantic_logits)

        if self.use_presence_score:
            variant_logit = variant_logit * inference_state['presence_score']

        return variant_logit, semantic_component, instance_component

    def _build_prompt_competition_stats(self, base_logits, base_pred, data_sample,
                                        image, ori_shape):
        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)
        if gt_data is None:
            return None

        valid_mask = gt_data != 255
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return dict(valid_pixels=0, class_variant_stats=[], pair_variant_stats=[])

        selected_classes = self._get_prompt_competition_class_indices()
        if not selected_classes:
            return dict(valid_pixels=valid_count, class_variant_stats=[], pair_variant_stats=[])

        gt_idx = gt_data.clamp(min=0, max=self.num_cls - 1)
        pred_idx = base_pred.clamp(min=0, max=self.num_cls - 1)
        base_correct = (base_pred == gt_data) & valid_mask
        base_wrong = (~base_correct) & valid_mask
        wrong_count = int(base_wrong.sum().item())

        if base_logits.shape[-2:] != gt_data.shape[-2:]:
            base_logits = F.interpolate(
                base_logits.unsqueeze(0),
                size=gt_data.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

        class_variant_stats = []
        pair_variant_stats = []
        w, h = image.size

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = self.processor.set_image(image)
            for class_idx in selected_classes:
                class_idx = int(class_idx)
                class_name = self.class_names[class_idx]
                prompts = self._build_prompt_competition_variants(class_name)
                if not prompts:
                    continue
                other_indices = [idx for idx in range(self.num_cls) if idx != class_idx]
                if other_indices:
                    base_other_max = base_logits[other_indices].max(dim=0)[0]
                else:
                    base_other_max = torch.zeros_like(base_logits[class_idx])
                class_gt_mask = (gt_idx == class_idx) & valid_mask
                class_gt_pixels = int(class_gt_mask.sum().item())
                class_base_correct = class_gt_mask & (base_pred == class_idx)
                class_wrong_mask = base_wrong & (gt_idx == class_idx)
                class_wrong_pixels = int(class_wrong_mask.sum().item())
                base_class_score = base_logits[class_idx]

                for variant_order, prompt in enumerate(prompts):
                    self.processor.reset_all_prompts(inference_state)
                    inference_state = self.processor.set_text_prompt(
                        state=inference_state,
                        prompt=prompt,
                    )
                    variant_logit, semantic_component, instance_component = (
                        self._compute_prompt_variant_logit(inference_state, h, w))
                    if variant_logit.shape[-2:] != gt_data.shape[-2:]:
                        variant_logit = F.interpolate(
                            variant_logit.unsqueeze(0).unsqueeze(0),
                            size=gt_data.shape[-2:],
                            mode='bilinear',
                            align_corners=False,
                        ).squeeze()
                    variant_gain = variant_logit - base_class_score
                    variant_beats_base = variant_logit > base_class_score
                    variant_beats_all = variant_logit > base_other_max

                    class_variant_stats.append(dict(
                        class_index=class_idx,
                        class_name=class_name,
                        variant_order=int(variant_order),
                        variant_prompt=prompt,
                        gt_pixels=class_gt_pixels,
                        baseline_correct_pixels=int(class_base_correct.sum().item()),
                        baseline_recall=_safe_div(
                            int(class_base_correct.sum().item()), class_gt_pixels),
                        class_wrong_pixels=class_wrong_pixels,
                        base_class_score_sum=_masked_sum(base_class_score, class_gt_mask),
                        variant_score_sum=_masked_sum(variant_logit, class_gt_mask),
                        variant_gain_sum=_masked_sum(variant_gain, class_gt_mask),
                        variant_gain_pixels=class_gt_pixels,
                        mean_base_class_score=_masked_mean(base_class_score, class_gt_mask),
                        mean_variant_score=_masked_mean(variant_logit, class_gt_mask),
                        mean_variant_gain=_masked_mean(variant_gain, class_gt_mask),
                        variant_beats_base_pixels=int((variant_beats_base & class_gt_mask).sum().item()),
                        variant_beats_base_ratio=_safe_div(
                            int((variant_beats_base & class_gt_mask).sum().item()),
                            class_gt_pixels),
                        variant_beats_all_pixels=int((variant_beats_all & class_gt_mask).sum().item()),
                        variant_beats_all_ratio=_safe_div(
                            int((variant_beats_all & class_gt_mask).sum().item()),
                            class_gt_pixels),
                        semantic_mean=_masked_mean(semantic_component, class_gt_mask),
                        instance_mean=_masked_mean(instance_component, class_gt_mask),
                    ))

                    for pred_class in range(self.num_cls):
                        if pred_class == class_idx:
                            continue
                        pair_mask = class_wrong_mask & (pred_idx == pred_class)
                        pair_pixels = int(pair_mask.sum().item())
                        if pair_pixels == 0:
                            continue
                        base_pred_score = base_logits[pred_class]
                        variant_vs_pred = variant_logit - base_pred_score
                        pair_variant_stats.append(dict(
                            gt_class_index=class_idx,
                            gt_class_name=class_name,
                            base_pred_class_index=pred_class,
                            base_pred_class_name=self.class_names[pred_class],
                            variant_order=int(variant_order),
                            variant_prompt=prompt,
                            pixels=pair_pixels,
                            pixel_ratio_in_error=_safe_div(pair_pixels, wrong_count),
                            pixel_ratio_in_gt_wrong=_safe_div(pair_pixels, class_wrong_pixels),
                            base_gt_score_sum=_masked_sum(base_class_score, pair_mask),
                            base_pred_score_sum=_masked_sum(base_pred_score, pair_mask),
                            variant_score_sum=_masked_sum(variant_logit, pair_mask),
                            variant_gain_sum=_masked_sum(variant_gain, pair_mask),
                            variant_vs_pred_margin_sum=_masked_sum(variant_vs_pred, pair_mask),
                            mean_base_gt_score=_masked_mean(base_class_score, pair_mask),
                            mean_base_pred_score=_masked_mean(base_pred_score, pair_mask),
                            mean_variant_score=_masked_mean(variant_logit, pair_mask),
                            mean_variant_gain=_masked_mean(variant_gain, pair_mask),
                            mean_variant_vs_pred_margin=_masked_mean(variant_vs_pred, pair_mask),
                            variant_beats_base_pixels=int((variant_beats_base & pair_mask).sum().item()),
                            variant_beats_base_ratio=_safe_div(
                                int((variant_beats_base & pair_mask).sum().item()),
                                pair_pixels),
                            variant_beats_pred_pixels=int(
                                ((variant_logit > base_pred_score) & pair_mask).sum().item()),
                            variant_beats_pred_ratio=_safe_div(
                                int(((variant_logit > base_pred_score) & pair_mask).sum().item()),
                                pair_pixels),
                            variant_beats_all_pixels=int((variant_beats_all & pair_mask).sum().item()),
                            variant_beats_all_ratio=_safe_div(
                                int((variant_beats_all & pair_mask).sum().item()),
                                pair_pixels),
                            semantic_mean=_masked_mean(semantic_component, pair_mask),
                            instance_mean=_masked_mean(instance_component, pair_mask),
                        ))

        pair_variant_stats.sort(key=lambda item: item['pixels'], reverse=True)
        return dict(
            dataset_name=self.seed_dataset_name,
            valid_pixels=valid_count,
            baseline_correct_pixels=int(base_correct.sum().item()),
            baseline_wrong_pixels=wrong_count,
            selected_class_indices=selected_classes,
            selected_class_names=[self.class_names[idx] for idx in selected_classes],
            prompt_competition_templates=self._get_prompt_competition_templates(),
            prompt_competition_max_variants=int(self.prompt_competition_max_variants),
            prompt_competition_use_builtin_variants=bool(
                self.prompt_competition_use_builtin_variants),
            class_variant_stats=class_variant_stats,
            pair_variant_stats=pair_variant_stats,
        )

    def _write_prompt_competition_stats(self, record):
        if not self.dump_prompt_competition_stats:
            return
        if self._prompt_competition_stats_file is None:
            path = (self.prompt_competition_stats_path
                    or './work_dirs/evidence_stats/prompt_competition_stats.jsonl')
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._prompt_competition_stats_file = open(path, 'a', buffering=1)
        self._prompt_competition_stats_file.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _get_prompt_competition_templates(self):
        templates = self.prompt_competition_templates
        if templates is None:
            return ['{class}']
        if isinstance(templates, str):
            values = [item.strip() for item in templates.split('|')]
        elif isinstance(templates, (list, tuple)):
            values = [str(item).strip() for item in templates]
        else:
            values = [str(templates).strip()]
        return [item for item in values if item]

    def _get_prompt_competition_class_indices(self):
        names = self.prompt_competition_class_names
        selected = []
        if names is not None and str(names).strip():
            if isinstance(names, str):
                requested = [item.strip() for item in names.split(',') if item.strip()]
            elif isinstance(names, (list, tuple)):
                requested = [str(item).strip() for item in names if str(item).strip()]
            else:
                requested = [str(names).strip()]
            for item in requested:
                if item.isdigit():
                    idx = int(item)
                    if 0 <= idx < self.num_cls and idx not in selected:
                        selected.append(idx)
                    continue
                item_tokens = _class_name_tokens(item)
                for idx, class_name in enumerate(self.class_names):
                    class_tokens = _class_name_tokens(class_name)
                    if (item.lower() == class_name.lower()
                            or item.lower() in class_name.lower()
                            or bool(item_tokens & class_tokens)):
                        if idx not in selected:
                            selected.append(idx)
            return selected

        dataset = (self.seed_dataset_name or '').lower()
        focus_tokens = []
        if dataset == 'vdd':
            focus_tokens = ['roof', 'facade']
        elif dataset == 'vaihingen':
            focus_tokens = ['grass', 'tree']
        elif dataset == 'potsdam':
            focus_tokens = ['tree', 'grass']
        elif dataset == 'udd5':
            focus_tokens = ['background', 'road', 'vegetation', 'building']
        elif dataset == 'openearthmap':
            focus_tokens = ['pavement', 'building', 'tree', 'grass']
        elif dataset == 'loveda':
            focus_tokens = ['forest', 'agricultural', 'background']
        for token in focus_tokens:
            for idx, class_name in enumerate(self.class_names):
                if token in _class_name_tokens(class_name) and idx not in selected:
                    selected.append(idx)
        return selected

    def _build_prompt_competition_variants(self, class_name):
        prompts = []
        if bool(self.prompt_competition_use_builtin_variants):
            prompts.extend(_builtin_prompt_variants(class_name))
        for template in self._get_prompt_competition_templates():
            prompts.append(template.replace('{class}', class_name))
        deduped = []
        seen = set()
        for prompt in prompts:
            prompt = str(prompt).strip()
            if not prompt:
                continue
            key = prompt.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(prompt)
            if len(deduped) >= int(self.prompt_competition_max_variants):
                break
        return deduped

    def _build_pair_prompt_competition_stats(self, base_logits, base_pred,
                                             data_sample, image, ori_shape):
        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)
        if gt_data is None:
            return None

        valid_mask = gt_data != 255
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return dict(valid_pixels=0, pair_variant_stats=[])

        if base_logits.shape[-2:] != gt_data.shape[-2:]:
            base_logits = F.interpolate(
                base_logits.unsqueeze(0),
                size=gt_data.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

        gt_idx = gt_data.clamp(min=0, max=self.num_cls - 1)
        pred_idx = base_pred.clamp(min=0, max=self.num_cls - 1)
        base_correct = (base_pred == gt_data) & valid_mask
        base_wrong = (~base_correct) & valid_mask
        wrong_count = int(base_wrong.sum().item())

        selected_pairs = self._get_pair_prompt_competition_pairs()
        if not selected_pairs:
            return dict(valid_pixels=valid_count, pair_variant_stats=[])

        pair_variant_stats = []
        w, h = image.size

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            inference_state = self.processor.set_image(image)
            for gt_class, pred_class in selected_pairs:
                gt_class = int(gt_class)
                pred_class = int(pred_class)
                if gt_class == pred_class:
                    continue
                if not (0 <= gt_class < self.num_cls and 0 <= pred_class < self.num_cls):
                    continue
                pair_mask = base_wrong & (gt_idx == gt_class) & (pred_idx == pred_class)
                pair_pixels = int(pair_mask.sum().item())
                if pair_pixels == 0:
                    continue

                gt_class_name = self.class_names[gt_class]
                pred_class_name = self.class_names[pred_class]
                base_gt_score = base_logits[gt_class]
                base_pred_score = base_logits[pred_class]
                other_indices = [idx for idx in range(self.num_cls) if idx != gt_class]
                if other_indices:
                    base_other_max = base_logits[other_indices].max(dim=0)[0]
                else:
                    base_other_max = torch.zeros_like(base_gt_score)

                prompts = self._build_pair_prompt_competition_variants(
                    gt_class_name, pred_class_name)
                for variant_order, prompt in enumerate(prompts):
                    self.processor.reset_all_prompts(inference_state)
                    inference_state = self.processor.set_text_prompt(
                        state=inference_state,
                        prompt=prompt,
                    )
                    variant_logit, semantic_component, instance_component = (
                        self._compute_prompt_variant_logit(inference_state, h, w))
                    if variant_logit.shape[-2:] != gt_data.shape[-2:]:
                        variant_logit = F.interpolate(
                            variant_logit.unsqueeze(0).unsqueeze(0),
                            size=gt_data.shape[-2:],
                            mode='bilinear',
                            align_corners=False,
                        ).squeeze()

                    variant_gain = variant_logit - base_gt_score
                    variant_vs_pred = variant_logit - base_pred_score
                    variant_beats_base = variant_logit > base_gt_score
                    variant_beats_pred = variant_logit > base_pred_score
                    variant_beats_all = variant_logit > base_other_max

                    pair_variant_stats.append(dict(
                        gt_class_index=gt_class,
                        gt_class_name=gt_class_name,
                        base_pred_class_index=pred_class,
                        base_pred_class_name=pred_class_name,
                        variant_order=int(variant_order),
                        variant_prompt=prompt,
                        pixels=pair_pixels,
                        pixel_ratio_in_error=_safe_div(pair_pixels, wrong_count),
                        base_gt_score_sum=_masked_sum(base_gt_score, pair_mask),
                        base_pred_score_sum=_masked_sum(base_pred_score, pair_mask),
                        variant_score_sum=_masked_sum(variant_logit, pair_mask),
                        variant_gain_sum=_masked_sum(variant_gain, pair_mask),
                        variant_vs_pred_margin_sum=_masked_sum(variant_vs_pred, pair_mask),
                        mean_base_gt_score=_masked_mean(base_gt_score, pair_mask),
                        mean_base_pred_score=_masked_mean(base_pred_score, pair_mask),
                        mean_variant_score=_masked_mean(variant_logit, pair_mask),
                        mean_variant_gain=_masked_mean(variant_gain, pair_mask),
                        mean_variant_vs_pred_margin=_masked_mean(variant_vs_pred, pair_mask),
                        variant_beats_base_pixels=int((variant_beats_base & pair_mask).sum().item()),
                        variant_beats_base_ratio=_safe_div(
                            int((variant_beats_base & pair_mask).sum().item()),
                            pair_pixels),
                        variant_beats_pred_pixels=int((variant_beats_pred & pair_mask).sum().item()),
                        variant_beats_pred_ratio=_safe_div(
                            int((variant_beats_pred & pair_mask).sum().item()),
                            pair_pixels),
                        variant_beats_all_pixels=int((variant_beats_all & pair_mask).sum().item()),
                        variant_beats_all_ratio=_safe_div(
                            int((variant_beats_all & pair_mask).sum().item()),
                            pair_pixels),
                        semantic_mean=_masked_mean(semantic_component, pair_mask),
                        instance_mean=_masked_mean(instance_component, pair_mask),
                    ))
                    del variant_logit
                    del semantic_component
                    del instance_component
                    del variant_gain
                    del variant_vs_pred
                    del variant_beats_base
                    del variant_beats_pred
                    del variant_beats_all
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()

        pair_variant_stats.sort(key=lambda item: item['pixels'], reverse=True)
        return dict(
            dataset_name=self.seed_dataset_name,
            valid_pixels=valid_count,
            baseline_correct_pixels=int(base_correct.sum().item()),
            baseline_wrong_pixels=wrong_count,
            selected_pairs=selected_pairs,
            selected_pair_names=[
                [self.class_names[gt_class], self.class_names[pred_class]]
                for gt_class, pred_class in selected_pairs
                if 0 <= gt_class < self.num_cls and 0 <= pred_class < self.num_cls
            ],
            pair_prompt_competition_templates=self._get_pair_prompt_competition_templates(),
            pair_prompt_competition_max_variants=int(
                self.pair_prompt_competition_max_variants),
            pair_prompt_competition_use_builtin_variants=bool(
                self.pair_prompt_competition_use_builtin_variants),
            pair_variant_stats=pair_variant_stats,
        )

    def _write_pair_prompt_competition_stats(self, record):
        if not self.dump_pair_prompt_competition_stats:
            return
        if self._pair_prompt_competition_stats_file is None:
            path = (self.pair_prompt_competition_stats_path
                    or './work_dirs/evidence_stats/pair_prompt_competition_stats.jsonl')
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._pair_prompt_competition_stats_file = open(path, 'a', buffering=1)
        self._pair_prompt_competition_stats_file.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _get_pair_prompt_competition_templates(self):
        templates = self.pair_prompt_competition_templates
        if templates is None:
            return ['{class}, not {competitor}']
        if isinstance(templates, str):
            values = [item.strip() for item in templates.split('|')]
        elif isinstance(templates, (list, tuple)):
            values = [str(item).strip() for item in templates]
        else:
            values = [str(templates).strip()]
        return [item for item in values if item]

    def _get_pair_prompt_competition_pairs(self):
        pairs = self.pair_prompt_competition_pairs
        if pairs is not None and str(pairs).strip():
            return self._parse_pair_prompt_competition_pairs(pairs)

        dataset = (self.seed_dataset_name or '').lower()
        if dataset == 'vdd':
            names = [('roof', 'facade'), ('roof', 'background')]
        elif dataset == 'vaihingen':
            names = [('grass', 'tree'), ('grass', 'clutter')]
        elif dataset == 'potsdam':
            names = [('tree', 'grass'), ('grass', 'clutter')]
        elif dataset == 'udd5':
            names = [
                ('road', 'background'),
                ('vegetation', 'background'),
                ('building', 'background'),
                ('background', 'road'),
            ]
        elif dataset == 'openearthmap':
            names = [('pavement', 'building'), ('grass', 'tree')]
        elif dataset == 'loveda':
            names = [('forest', 'background'), ('agricultural', 'background')]
        else:
            names = []
        selected = []
        for gt_name, pred_name in names:
            gt_idx = self._find_class_index_by_name(gt_name)
            pred_idx = self._find_class_index_by_name(pred_name)
            if gt_idx is not None and pred_idx is not None:
                selected.append((gt_idx, pred_idx))
        return selected

    def _parse_pair_prompt_competition_pairs(self, pairs):
        if isinstance(pairs, str):
            items = [item.strip() for item in pairs.split(';') if item.strip()]
        elif isinstance(pairs, (list, tuple)):
            items = [str(item).strip() for item in pairs if str(item).strip()]
        else:
            items = [str(pairs).strip()]
        selected = []
        for item in items:
            if '->' in item:
                left, right = item.split('->', 1)
            elif ':' in item:
                left, right = item.split(':', 1)
            elif ',' in item:
                left, right = item.split(',', 1)
            else:
                continue
            gt_idx = self._find_class_index_by_name(left.strip())
            pred_idx = self._find_class_index_by_name(right.strip())
            if gt_idx is not None and pred_idx is not None:
                pair = (gt_idx, pred_idx)
                if pair not in selected:
                    selected.append(pair)
        return selected

    def _find_class_index_by_name(self, name):
        name = str(name).strip()
        if not name:
            return None
        if name.isdigit():
            idx = int(name)
            return idx if 0 <= idx < self.num_cls else None
        target_tokens = _class_name_tokens(name)
        for idx, class_name in enumerate(self.class_names):
            class_tokens = _class_name_tokens(class_name)
            if (name.lower() == class_name.lower()
                    or name.lower() in class_name.lower()
                    or bool(target_tokens & class_tokens)):
                return idx
        return None

    def _build_pair_prompt_competition_variants(self, class_name, competitor_name):
        prompts = []
        if bool(self.pair_prompt_competition_use_builtin_variants):
            prompts.extend(_builtin_pair_prompt_variants(class_name, competitor_name))
        for template in self._get_pair_prompt_competition_templates():
            prompt = template.replace('{class}', class_name)
            prompt = prompt.replace('{competitor}', competitor_name)
            prompts.append(prompt)
        deduped = []
        seen = set()
        for prompt in prompts:
            prompt = str(prompt).strip()
            if not prompt:
                continue
            key = prompt.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(prompt)
            if len(deduped) >= int(self.pair_prompt_competition_max_variants):
                    break
        return deduped

    def _get_context_requery_pairs(self):
        pairs = self.context_requery_pairs
        if pairs is not None and str(pairs).strip():
            return self._parse_pair_prompt_competition_pairs(pairs)
        return self._get_pair_prompt_competition_pairs()

    def _get_context_requery_modes(self):
        modes = self.context_requery_modes
        if modes is None:
            return ['full', 'bbox1.5', 'bbox2.0']
        if isinstance(modes, str):
            values = [item.strip().lower() for item in modes.split(',') if item.strip()]
        elif isinstance(modes, (list, tuple)):
            values = [str(item).strip().lower() for item in modes if str(item).strip()]
        else:
            values = [str(modes).strip().lower()]
        return values or ['full']

    def _context_requery_crop_box(self, pair_mask, image_size, mode):
        w_img, h_img = image_size
        if mode in ('full', 'full_image', 'image'):
            return [0, 0, w_img, h_img], 'full'
        if not pair_mask.any():
            return None, None

        scale_text = mode
        for prefix in ('bbox', 'box', 'context'):
            if scale_text.startswith(prefix):
                scale_text = scale_text[len(prefix):]
                break
        try:
            scale = float(scale_text) if scale_text else 1.0
        except ValueError:
            scale = 1.5
        scale = max(scale, 1.0)

        coords = torch.nonzero(pair_mask.detach(), as_tuple=False)
        if coords.numel() == 0:
            return None, None
        y_min = float(coords[:, 0].min().item())
        y_max = float(coords[:, 0].max().item()) + 1.0
        x_min = float(coords[:, 1].min().item())
        x_max = float(coords[:, 1].max().item()) + 1.0
        cx = (x_min + x_max) * 0.5
        cy = (y_min + y_max) * 0.5
        bw = max(x_max - x_min, float(self.context_requery_min_crop_size))
        bh = max(y_max - y_min, float(self.context_requery_min_crop_size))
        side = max(bw, bh) * scale
        side = max(side, float(self.context_requery_min_crop_size))
        side = min(side, float(self.context_requery_max_crop_size))
        side = min(side, float(max(w_img, h_img)))
        crop_w = min(side, float(w_img))
        crop_h = min(side, float(h_img))
        x1 = int(round(cx - crop_w * 0.5))
        y1 = int(round(cy - crop_h * 0.5))
        x1 = max(0, min(x1, max(w_img - int(round(crop_w)), 0)))
        y1 = max(0, min(y1, max(h_img - int(round(crop_h)), 0)))
        x2 = min(w_img, x1 + int(round(crop_w)))
        y2 = min(h_img, y1 + int(round(crop_h)))
        return [x1, y1, x2, y2], f'bbox{scale:g}'

    def _context_requery_prompt_logit(self, image, prompt):
        w, h = image.size
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = self.processor.set_image(image)
            self.processor.reset_all_prompts(inference_state)
            inference_state = self.processor.set_text_prompt(
                state=inference_state,
                prompt=prompt,
            )
            logit, semantic_component, instance_component = (
                self._compute_prompt_variant_logit(inference_state, h, w))
        return logit, semantic_component, instance_component

    def _build_context_requery_stats(self, base_logits, base_pred, data_sample,
                                     image, ori_shape):
        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)
        if gt_data is None:
            return None
        if base_logits.shape[-2:] != gt_data.shape[-2:]:
            base_logits = F.interpolate(
                base_logits.unsqueeze(0),
                size=gt_data.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

        valid_mask = gt_data != 255
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return dict(valid_pixels=0, pair_stats=[])

        gt_idx = gt_data.clamp(min=0, max=self.num_cls - 1)
        pred_idx = base_pred.clamp(min=0, max=self.num_cls - 1)
        base_wrong = (base_pred != gt_data) & valid_mask
        wrong_count = int(base_wrong.sum().item())
        selected_pairs = self._get_context_requery_pairs()
        thresholds = self._parse_float_list(
            self.context_requery_thresholds,
            '0.05,0.10,0.20,0.50')
        modes = self._get_context_requery_modes()
        w_img, h_img = image.size
        pair_stats = []

        for gt_class, pred_class in selected_pairs:
            gt_class = int(gt_class)
            pred_class = int(pred_class)
            if gt_class == pred_class:
                continue
            if not (0 <= gt_class < self.num_cls and 0 <= pred_class < self.num_cls):
                continue
            pair_mask = (
                base_wrong
                & (gt_idx == gt_class)
                & (pred_idx == pred_class)
            )
            pair_pixels_full = int(pair_mask.sum().item())
            if pair_pixels_full < int(self.context_requery_min_pair_pixels):
                continue

            gt_name = self.class_names[gt_class]
            pred_name = self.class_names[pred_class]
            base_gt_score = base_logits[gt_class]
            base_pred_score = base_logits[pred_class]
            base_margin = base_gt_score - base_pred_score

            for mode in modes:
                crop_box, normalized_mode = self._context_requery_crop_box(
                    pair_mask,
                    (w_img, h_img),
                    mode,
                )
                if crop_box is None:
                    continue
                x1, y1, x2, y2 = crop_box
                crop_pair_mask = pair_mask[y1:y2, x1:x2]
                covered_pixels = int(crop_pair_mask.sum().item())
                if covered_pixels < int(self.context_requery_min_pair_pixels):
                    continue
                crop_image = image.crop((x1, y1, x2, y2))

                gt_logit, gt_semantic, gt_instance = self._context_requery_prompt_logit(
                    crop_image, gt_name)
                pred_logit, pred_semantic, pred_instance = self._context_requery_prompt_logit(
                    crop_image, pred_name)
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

                requery_margin = gt_logit - pred_logit
                base_gt_crop = base_gt_score[y1:y2, x1:x2]
                base_pred_crop = base_pred_score[y1:y2, x1:x2]
                base_margin_crop = base_margin[y1:y2, x1:x2]
                margin_gain = requery_margin - base_margin_crop
                gt_beats_pred = requery_margin > 0
                base_gt_beats_pred = base_margin_crop > 0

                for threshold in thresholds:
                    gt_support = gt_logit >= float(threshold)
                    pred_support = pred_logit >= float(threshold)
                    gt_on_pair = crop_pair_mask & gt_support
                    pred_on_pair = crop_pair_mask & pred_support
                    both = gt_on_pair & pred_support
                    gt_only = gt_on_pair & (~pred_support)
                    pred_only = pred_on_pair & (~gt_support)
                    neither = crop_pair_mask & (~gt_support) & (~pred_support)

                    pair_stats.append(dict(
                        gt_class_index=gt_class,
                        gt_class_name=gt_name,
                        base_pred_class_index=pred_class,
                        base_pred_class_name=pred_name,
                        mode=normalized_mode,
                        requested_mode=mode,
                        crop_box=crop_box,
                        threshold=float(threshold),
                        full_pair_pixels=pair_pixels_full,
                        covered_pair_pixels=covered_pixels,
                        covered_pair_ratio=_safe_div(covered_pixels, pair_pixels_full),
                        pixel_ratio_in_error=_safe_div(pair_pixels_full, wrong_count),
                        base_gt_score_sum=_masked_sum(base_gt_crop, crop_pair_mask),
                        base_pred_score_sum=_masked_sum(base_pred_crop, crop_pair_mask),
                        base_margin_sum=_masked_sum(base_margin_crop, crop_pair_mask),
                        requery_gt_score_sum=_masked_sum(gt_logit, crop_pair_mask),
                        requery_pred_score_sum=_masked_sum(pred_logit, crop_pair_mask),
                        requery_margin_sum=_masked_sum(requery_margin, crop_pair_mask),
                        margin_gain_sum=_masked_sum(margin_gain, crop_pair_mask),
                        base_gt_beats_pred_pixels=int((base_gt_beats_pred & crop_pair_mask).sum().item()),
                        requery_gt_beats_pred_pixels=int((gt_beats_pred & crop_pair_mask).sum().item()),
                        gt_support_pixels=int(gt_on_pair.sum().item()),
                        pred_support_pixels=int(pred_on_pair.sum().item()),
                        gt_only_support_pixels=int(gt_only.sum().item()),
                        pred_only_support_pixels=int(pred_only.sum().item()),
                        both_support_pixels=int(both.sum().item()),
                        neither_support_pixels=int(neither.sum().item()),
                        gt_semantic_sum=_masked_sum(gt_semantic, crop_pair_mask),
                        pred_semantic_sum=_masked_sum(pred_semantic, crop_pair_mask),
                        gt_instance_sum=_masked_sum(gt_instance, crop_pair_mask),
                        pred_instance_sum=_masked_sum(pred_instance, crop_pair_mask),
                    ))

                del gt_logit
                del pred_logit
                del gt_semantic
                del pred_semantic
                del gt_instance
                del pred_instance

        pair_stats.sort(key=lambda item: item['full_pair_pixels'], reverse=True)
        return dict(
            dataset_name=self.seed_dataset_name,
            valid_pixels=valid_count,
            baseline_wrong_pixels=wrong_count,
            selected_pairs=selected_pairs,
            context_requery_modes=modes,
            context_requery_thresholds=thresholds,
            context_requery_min_pair_pixels=int(self.context_requery_min_pair_pixels),
            context_requery_min_crop_size=int(self.context_requery_min_crop_size),
            context_requery_max_crop_size=int(self.context_requery_max_crop_size),
            pair_stats=pair_stats,
        )

    def _write_context_requery_stats(self, record):
        if not self.dump_context_requery_stats:
            return
        if self._context_requery_stats_file is None:
            path = (
                self.context_requery_stats_path
                or './work_dirs/evidence_stats/context_requery_stats.jsonl'
            )
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._context_requery_stats_file = open(path, 'a', buffering=1)
        self._context_requery_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    def _get_scale_requery_pairs(self):
        pairs = self.scale_requery_pairs
        if pairs is not None and str(pairs).strip():
            return self._parse_pair_prompt_competition_pairs(pairs)
        return self._get_pair_prompt_competition_pairs()

    def _get_scale_requery_scales(self):
        scales = self._parse_float_list(
            self.scale_requery_scales,
            '0.50,0.75,1.00,1.25,1.50')
        return [float(scale) for scale in scales if float(scale) > 0]

    def _resize_for_scale_requery(self, image, requested_scale):
        w, h = image.size
        requested_scale = float(requested_scale)
        new_w = max(1, int(round(w * requested_scale)))
        new_h = max(1, int(round(h * requested_scale)))
        max_side = int(self.scale_requery_max_side)
        if max_side > 0:
            current_max = max(new_w, new_h)
            if current_max > max_side:
                cap = float(max_side) / float(current_max)
                new_w = max(1, int(round(new_w * cap)))
                new_h = max(1, int(round(new_h * cap)))
        effective_scale_x = _safe_div(new_w, w)
        effective_scale_y = _safe_div(new_h, h)
        if new_w == w and new_h == h:
            return image, effective_scale_x, effective_scale_y
        return (
            image.resize((new_w, new_h), resample=Image.BICUBIC),
            effective_scale_x,
            effective_scale_y,
        )

    def _resize_scale_requery_map(self, score_map, target_size):
        score_map = score_map.float()
        if score_map.shape[-2:] == target_size:
            return score_map
        return F.interpolate(
            score_map.unsqueeze(0).unsqueeze(0),
            size=target_size,
            mode='bilinear',
            align_corners=False,
        ).squeeze()

    def _build_scale_requery_stats(self, base_logits, base_pred, data_sample,
                                   image):
        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)
        if gt_data is None:
            return None
        if base_logits.shape[-2:] != gt_data.shape[-2:]:
            base_logits = F.interpolate(
                base_logits.unsqueeze(0),
                size=gt_data.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

        valid_mask = gt_data != 255
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return dict(valid_pixels=0, pair_stats=[])

        gt_idx = gt_data.clamp(min=0, max=self.num_cls - 1)
        pred_idx = base_pred.clamp(min=0, max=self.num_cls - 1)
        base_wrong = (base_pred != gt_data) & valid_mask
        wrong_count = int(base_wrong.sum().item())
        selected_pairs = self._get_scale_requery_pairs()
        thresholds = self._parse_float_list(
            self.scale_requery_thresholds,
            '0.05,0.10,0.20,0.50')
        scales = self._get_scale_requery_scales()
        pair_stats = []
        orig_w, orig_h = image.size

        for gt_class, pred_class in selected_pairs:
            gt_class = int(gt_class)
            pred_class = int(pred_class)
            if gt_class == pred_class:
                continue
            if not (0 <= gt_class < self.num_cls and 0 <= pred_class < self.num_cls):
                continue
            pair_mask = (
                base_wrong
                & (gt_idx == gt_class)
                & (pred_idx == pred_class)
            )
            pair_pixels = int(pair_mask.sum().item())
            if pair_pixels < int(self.scale_requery_min_pair_pixels):
                continue

            gt_name = self.class_names[gt_class]
            pred_name = self.class_names[pred_class]
            base_gt_score = base_logits[gt_class]
            base_pred_score = base_logits[pred_class]
            base_margin = base_gt_score - base_pred_score

            for scale in scales:
                scaled_image, effective_scale_x, effective_scale_y = (
                    self._resize_for_scale_requery(image, scale))
                scaled_w, scaled_h = scaled_image.size
                gt_logit, gt_semantic, gt_instance = self._context_requery_prompt_logit(
                    scaled_image, gt_name)
                pred_logit, pred_semantic, pred_instance = self._context_requery_prompt_logit(
                    scaled_image, pred_name)
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

                target_size = gt_data.shape[-2:]
                if gt_logit.shape[-2:] != target_size:
                    gt_logit = self._resize_scale_requery_map(
                        gt_logit, target_size)
                    pred_logit = self._resize_scale_requery_map(
                        pred_logit, target_size)
                    gt_semantic = self._resize_scale_requery_map(
                        gt_semantic, target_size)
                    pred_semantic = self._resize_scale_requery_map(
                        pred_semantic, target_size)
                    gt_instance = self._resize_scale_requery_map(
                        gt_instance, target_size)
                    pred_instance = self._resize_scale_requery_map(
                        pred_instance, target_size)
                else:
                    gt_logit = gt_logit.float()
                    pred_logit = pred_logit.float()
                    gt_semantic = gt_semantic.float()
                    pred_semantic = pred_semantic.float()
                    gt_instance = gt_instance.float()
                    pred_instance = pred_instance.float()

                requery_margin = gt_logit - pred_logit
                margin_gain = requery_margin - base_margin
                gt_beats_pred = requery_margin > 0
                base_gt_beats_pred = base_margin > 0

                for threshold in thresholds:
                    gt_support = gt_logit >= float(threshold)
                    pred_support = pred_logit >= float(threshold)
                    gt_on_pair = pair_mask & gt_support
                    pred_on_pair = pair_mask & pred_support
                    both = gt_on_pair & pred_support
                    gt_only = gt_on_pair & (~pred_support)
                    pred_only = pred_on_pair & (~gt_support)
                    neither = pair_mask & (~gt_support) & (~pred_support)

                    pair_stats.append(dict(
                        gt_class_index=gt_class,
                        gt_class_name=gt_name,
                        base_pred_class_index=pred_class,
                        base_pred_class_name=pred_name,
                        requested_scale=float(scale),
                        effective_scale_x=effective_scale_x,
                        effective_scale_y=effective_scale_y,
                        original_size=[orig_w, orig_h],
                        scaled_size=[scaled_w, scaled_h],
                        threshold=float(threshold),
                        full_pair_pixels=pair_pixels,
                        covered_pair_pixels=pair_pixels,
                        covered_pair_ratio=1.0,
                        pixel_ratio_in_error=_safe_div(pair_pixels, wrong_count),
                        base_gt_score_sum=_masked_sum(base_gt_score, pair_mask),
                        base_pred_score_sum=_masked_sum(base_pred_score, pair_mask),
                        base_margin_sum=_masked_sum(base_margin, pair_mask),
                        requery_gt_score_sum=_masked_sum(gt_logit, pair_mask),
                        requery_pred_score_sum=_masked_sum(pred_logit, pair_mask),
                        requery_margin_sum=_masked_sum(requery_margin, pair_mask),
                        margin_gain_sum=_masked_sum(margin_gain, pair_mask),
                        base_gt_beats_pred_pixels=int((base_gt_beats_pred & pair_mask).sum().item()),
                        requery_gt_beats_pred_pixels=int((gt_beats_pred & pair_mask).sum().item()),
                        gt_support_pixels=int(gt_on_pair.sum().item()),
                        pred_support_pixels=int(pred_on_pair.sum().item()),
                        gt_only_support_pixels=int(gt_only.sum().item()),
                        pred_only_support_pixels=int(pred_only.sum().item()),
                        both_support_pixels=int(both.sum().item()),
                        neither_support_pixels=int(neither.sum().item()),
                        gt_semantic_sum=_masked_sum(gt_semantic, pair_mask),
                        pred_semantic_sum=_masked_sum(pred_semantic, pair_mask),
                        gt_instance_sum=_masked_sum(gt_instance, pair_mask),
                        pred_instance_sum=_masked_sum(pred_instance, pair_mask),
                    ))

                del gt_logit
                del pred_logit
                del gt_semantic
                del pred_semantic
                del gt_instance
                del pred_instance

        pair_stats.sort(key=lambda item: item['full_pair_pixels'], reverse=True)
        return dict(
            dataset_name=self.seed_dataset_name,
            valid_pixels=valid_count,
            baseline_wrong_pixels=wrong_count,
            selected_pairs=selected_pairs,
            scale_requery_scales=scales,
            scale_requery_thresholds=thresholds,
            scale_requery_min_pair_pixels=int(self.scale_requery_min_pair_pixels),
            scale_requery_max_side=int(self.scale_requery_max_side),
            pair_stats=pair_stats,
        )

    def _write_scale_requery_stats(self, record):
        if not self.dump_scale_requery_stats:
            return
        if self._scale_requery_stats_file is None:
            path = (
                self.scale_requery_stats_path
                or './work_dirs/evidence_stats/scale_requery_stats.jsonl'
            )
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._scale_requery_stats_file = open(path, 'a', buffering=1)
        self._scale_requery_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    def _get_scale_stability_pairs(self):
        pairs = self.scale_stability_pairs
        if pairs is not None and str(pairs).strip():
            return self._parse_pair_prompt_competition_pairs(pairs)
        return self._get_pair_prompt_competition_pairs()

    def _resize_for_scale_stability(self, image, requested_scale):
        old_max_side = self.scale_requery_max_side
        self.scale_requery_max_side = self.scale_stability_max_side
        try:
            return self._resize_for_scale_requery(image, requested_scale)
        finally:
            self.scale_requery_max_side = old_max_side

    def _compute_scaled_class_logits(self, image, target_size):
        w, h = image.size
        class_logits = []
        class_semantic = []
        class_instance = []
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = self.processor.set_image(image)
            for class_idx in range(self.num_cls):
                self.processor.reset_all_prompts(inference_state)
                inference_state = self.processor.set_text_prompt(
                    state=inference_state,
                    prompt=self.class_names[class_idx],
                )
                logit, semantic_component, instance_component = (
                    self._compute_prompt_variant_logit(inference_state, h, w))
                class_logits.append(self._resize_scale_requery_map(logit, target_size))
                class_semantic.append(self._resize_scale_requery_map(
                    semantic_component, target_size))
                class_instance.append(self._resize_scale_requery_map(
                    instance_component, target_size))
                del logit
                del semantic_component
                del instance_component
        return (
            torch.stack(class_logits, dim=0),
            torch.stack(class_semantic, dim=0),
            torch.stack(class_instance, dim=0),
        )

    def _compute_scaled_selected_class_logits(self, image, target_size,
                                             class_indices):
        class_indices = sorted({
            int(class_idx)
            for class_idx in class_indices
            if 0 <= int(class_idx) < self.num_cls
        })
        full_logits = torch.full(
            (self.num_cls, int(target_size[0]), int(target_size[1])),
            float('-inf'),
            device=self.device,
            dtype=torch.float32,
        )
        full_semantic = torch.zeros_like(full_logits)
        full_instance = torch.zeros_like(full_logits)
        if not class_indices:
            return full_logits, full_semantic, full_instance

        w, h = image.size
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = self.processor.set_image(image)
            for class_idx in class_indices:
                self.processor.reset_all_prompts(inference_state)
                inference_state = self.processor.set_text_prompt(
                    state=inference_state,
                    prompt=self.class_names[class_idx],
                )
                logit, semantic_component, instance_component = (
                    self._compute_prompt_variant_logit(inference_state, h, w))
                full_logits[class_idx] = self._resize_scale_requery_map(
                    logit, target_size).to(torch.float32)
                full_semantic[class_idx] = self._resize_scale_requery_map(
                    semantic_component, target_size).to(torch.float32)
                full_instance[class_idx] = self._resize_scale_requery_map(
                    instance_component, target_size).to(torch.float32)
                del logit
                del semantic_component
                del instance_component
        return full_logits, full_semantic, full_instance

    def _build_scale_stability_stats(self, base_logits, base_pred, data_sample,
                                     image):
        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)
        if gt_data is None:
            return None
        if base_logits.shape[-2:] != gt_data.shape[-2:]:
            base_logits = F.interpolate(
                base_logits.unsqueeze(0),
                size=gt_data.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

        valid_mask = gt_data != 255
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return dict(valid_pixels=0, pair_stats=[])

        topk = max(1, min(int(self.scale_stability_topk), self.num_cls))
        topk_values, topk_indices = torch.topk(base_logits, k=topk, dim=0)
        gt_idx = gt_data.clamp(min=0, max=self.num_cls - 1)
        pred_idx = base_pred.clamp(min=0, max=self.num_cls - 1)
        base_wrong = (base_pred != gt_data) & valid_mask
        wrong_count = int(base_wrong.sum().item())
        selected_pairs = self._get_scale_stability_pairs()
        drop_thresholds = self._parse_float_list(
            self.scale_stability_drop_thresholds,
            '0.02,0.05,0.10')

        scaled_image, effective_scale_x, effective_scale_y = (
            self._resize_for_scale_stability(
                image,
                float(self.scale_stability_scale)))
        scaled_w, scaled_h = scaled_image.size
        target_size = gt_data.shape[-2:]
        scaled_logits, scaled_semantic, scaled_instance = (
            self._compute_scaled_class_logits(scaled_image, target_size))
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        scaled_topk_scores = torch.gather(scaled_logits, 0, topk_indices)
        scaled_choice_score, scaled_choice_rank = scaled_topk_scores.max(dim=0)
        scaled_choice_class = torch.gather(
            topk_indices,
            0,
            scaled_choice_rank.unsqueeze(0),
        ).squeeze(0)
        scaled_choice_base_score = torch.gather(
            topk_values,
            0,
            scaled_choice_rank.unsqueeze(0),
        ).squeeze(0)
        scaled_choice_drop = scaled_choice_base_score - scaled_choice_score

        pair_stats = []
        for gt_class, pred_class in selected_pairs:
            gt_class = int(gt_class)
            pred_class = int(pred_class)
            if gt_class == pred_class:
                continue
            if not (0 <= gt_class < self.num_cls and 0 <= pred_class < self.num_cls):
                continue
            pair_mask = (
                base_wrong
                & (gt_idx == gt_class)
                & (pred_idx == pred_class)
            )
            pair_pixels = int(pair_mask.sum().item())
            if pair_pixels < int(self.scale_stability_min_pair_pixels):
                continue

            gt_name = self.class_names[gt_class]
            pred_name = self.class_names[pred_class]
            base_gt_score = base_logits[gt_class]
            base_pred_score = base_logits[pred_class]
            scaled_gt_score = scaled_logits[gt_class]
            scaled_pred_score = scaled_logits[pred_class]
            base_margin = base_gt_score - base_pred_score
            scaled_margin = scaled_gt_score - scaled_pred_score
            margin_gain = scaled_margin - base_margin
            pred_drop = base_pred_score - scaled_pred_score
            gt_drop = base_gt_score - scaled_gt_score
            relative_stability = pred_drop - gt_drop
            topk_contains_gt = (topk_indices == gt_class).any(dim=0)
            scaled_gt_beats_pred = scaled_margin > 0
            scaled_choice_is_gt = scaled_choice_class == gt_class
            scaled_choice_is_pred = scaled_choice_class == pred_class
            scaled_choice_not_pred = ~scaled_choice_is_pred
            scaled_choice_margin = scaled_choice_score - scaled_pred_score
            choice_relative_stability = pred_drop - scaled_choice_drop

            base_counts = dict(
                gt_class_index=gt_class,
                gt_class_name=gt_name,
                base_pred_class_index=pred_class,
                base_pred_class_name=pred_name,
                requested_scale=float(self.scale_stability_scale),
                effective_scale_x=effective_scale_x,
                effective_scale_y=effective_scale_y,
                original_size=list(image.size),
                scaled_size=[scaled_w, scaled_h],
                scale_stability_topk=topk,
                full_pair_pixels=pair_pixels,
                covered_pair_pixels=pair_pixels,
                covered_pair_ratio=1.0,
                pixel_ratio_in_error=_safe_div(pair_pixels, wrong_count),
                base_margin_sum=_masked_sum(base_margin, pair_mask),
                scaled_margin_sum=_masked_sum(scaled_margin, pair_mask),
                margin_gain_sum=_masked_sum(margin_gain, pair_mask),
                base_gt_score_sum=_masked_sum(base_gt_score, pair_mask),
                base_pred_score_sum=_masked_sum(base_pred_score, pair_mask),
                scaled_gt_score_sum=_masked_sum(scaled_gt_score, pair_mask),
                scaled_pred_score_sum=_masked_sum(scaled_pred_score, pair_mask),
                pred_drop_sum=_masked_sum(pred_drop, pair_mask),
                gt_drop_sum=_masked_sum(gt_drop, pair_mask),
                relative_stability_sum=_masked_sum(relative_stability, pair_mask),
                scaled_choice_score_sum=_masked_sum(scaled_choice_score, pair_mask),
                scaled_choice_margin_sum=_masked_sum(scaled_choice_margin, pair_mask),
                scaled_choice_drop_sum=_masked_sum(scaled_choice_drop, pair_mask),
                choice_relative_stability_sum=_masked_sum(
                    choice_relative_stability, pair_mask),
                gt_semantic_sum=_masked_sum(scaled_semantic[gt_class], pair_mask),
                pred_semantic_sum=_masked_sum(scaled_semantic[pred_class], pair_mask),
                gt_instance_sum=_masked_sum(scaled_instance[gt_class], pair_mask),
                pred_instance_sum=_masked_sum(scaled_instance[pred_class], pair_mask),
                topk_contains_gt_pixels=int((topk_contains_gt & pair_mask).sum().item()),
                scaled_gt_beats_pred_pixels=int(
                    (scaled_gt_beats_pred & pair_mask).sum().item()),
                scaled_choice_is_gt_pixels=int(
                    (scaled_choice_is_gt & pair_mask).sum().item()),
                scaled_choice_is_pred_pixels=int(
                    (scaled_choice_is_pred & pair_mask).sum().item()),
                scaled_choice_not_pred_pixels=int(
                    (scaled_choice_not_pred & pair_mask).sum().item()),
            )
            for threshold in drop_thresholds:
                threshold = float(threshold)
                top1_drop_gate = (
                    pair_mask
                    & (pred_drop >= threshold)
                    & scaled_choice_not_pred
                )
                stability_gate = (
                    pair_mask
                    & (choice_relative_stability >= threshold)
                    & scaled_choice_not_pred
                )
                row = dict(base_counts)
                row.update(dict(
                    drop_threshold=threshold,
                    top1_drop_gate_pixels=int(top1_drop_gate.sum().item()),
                    top1_drop_gate_gt_pixels=int(
                        (top1_drop_gate & scaled_choice_is_gt).sum().item()),
                    top1_drop_gate_gt_beats_pred_pixels=int(
                        (top1_drop_gate & scaled_gt_beats_pred).sum().item()),
                    stability_gate_pixels=int(stability_gate.sum().item()),
                    stability_gate_gt_pixels=int(
                        (stability_gate & scaled_choice_is_gt).sum().item()),
                    stability_gate_gt_beats_pred_pixels=int(
                        (stability_gate & scaled_gt_beats_pred).sum().item()),
                ))
                pair_stats.append(row)

        pair_stats.sort(key=lambda item: item['full_pair_pixels'], reverse=True)
        return dict(
            dataset_name=self.seed_dataset_name,
            valid_pixels=valid_count,
            baseline_wrong_pixels=wrong_count,
            selected_pairs=selected_pairs,
            scale_stability_scale=float(self.scale_stability_scale),
            effective_scale_x=effective_scale_x,
            effective_scale_y=effective_scale_y,
            scale_stability_topk=topk,
            scale_stability_drop_thresholds=drop_thresholds,
            scale_stability_min_pair_pixels=int(self.scale_stability_min_pair_pixels),
            scale_stability_max_side=int(self.scale_stability_max_side),
            pair_stats=pair_stats,
        )

    def _write_scale_stability_stats(self, record):
        if not self.dump_scale_stability_stats:
            return
        if self._scale_stability_stats_file is None:
            path = (
                self.scale_stability_stats_path
                or './work_dirs/evidence_stats/scale_stability_stats.jsonl'
            )
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._scale_stability_stats_file = open(path, 'a', buffering=1)
        self._scale_stability_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    def _get_scale_stability_rerank_pairs(self):
        pair_file = self.scale_stability_rerank_pair_file
        if pair_file is not None and str(pair_file).strip():
            items = []
            with open(str(pair_file).strip(), 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    items.append(line)
            if items:
                return self._parse_pair_prompt_competition_pairs(items)
        pairs = self.scale_stability_rerank_pairs
        if pairs is not None and str(pairs).strip():
            return self._parse_pair_prompt_competition_pairs(pairs)
        return self._get_scale_stability_pairs()

    def _scale_stability_rerank_exclude_bg(self):
        value = self.scale_stability_rerank_auto_exclude_bg
        if isinstance(value, str):
            return value.strip().lower() not in {'0', 'false', 'no', 'off'}
        return bool(value)

    def _scale_stability_rerank_bg_indices(self):
        bg_indices = set()
        if 0 <= int(self.bg_idx) < self.num_cls:
            bg_indices.add(int(self.bg_idx))
        bg_names = set(self._parse_name_list(
            self.scale_stability_rerank_bg_names))
        for class_idx, class_name in enumerate(self.class_names):
            tokens = _class_name_tokens(class_name)
            if tokens & bg_names:
                bg_indices.add(class_idx)
        return bg_indices

    def _scale_stability_class_roles(self, class_idx):
        tokens = _class_name_tokens(self.class_names[int(class_idx)])
        roles = set()
        if tokens & {
                'tree', 'forest', 'grass', 'vegetation', 'low', 'agriculture',
                'agricultural', 'farmland', 'cropland', 'crop', 'field',
                'orchard', 'woodland'}:
            roles.add('vegetation')
        if tokens & {
                'building', 'roof', 'facade', 'house', 'road', 'pavement',
                'paved', 'sidewalk', 'impervious', 'surface', 'parking',
                'runway', 'bridge'}:
            roles.add('built_surface')
        if tokens & {
                'vehicle', 'car', 'ship', 'plane', 'airplane', 'aircraft',
                'harbor', 'storage', 'tank', 'baseball', 'tennis', 'court',
                'ground', 'track', 'field'}:
            roles.add('object')
        if tokens & {'water', 'river', 'lake', 'sea', 'pond', 'flood'}:
            roles.add('water')
        return roles

    def _scale_stability_role_compatible(self, target_class,
                                         competitor_class):
        target_roles = self._scale_stability_class_roles(target_class)
        competitor_roles = self._scale_stability_class_roles(competitor_class)
        return bool(target_roles & competitor_roles)

    def _scale_stability_rerank_pair_mode_name(self):
        mode = str(self.scale_stability_rerank_pair_mode or 'manual').lower()
        if mode not in {'manual', 'role_auto', 'dynamic_topk'}:
            raise ValueError(
                "scale_stability_rerank_pair_mode must be one of "
                "'manual', 'role_auto', or 'dynamic_topk', "
                f"but got {self.scale_stability_rerank_pair_mode!r}")
        return mode

    def _scale_stability_rerank_query_mode_name(self):
        mode = str(self.scale_stability_rerank_query_mode
                   or 'all_classes').lower()
        if mode not in {'all_classes', 'risk_classes'}:
            raise ValueError(
                "scale_stability_rerank_query_mode must be one of "
                "'all_classes' or 'risk_classes', "
                f"but got {self.scale_stability_rerank_query_mode!r}")
        return mode

    def _scale_stability_pair_candidate_mask(self, base_logits, topk_indices,
                                             top1_idx, target_class,
                                             competitor_class, topk,
                                             rank_template):
        max_target_rank = max(1, int(self.scale_stability_rerank_max_target_rank))
        min_base_margin = float(self.scale_stability_rerank_min_base_margin)
        max_base_margin = float(self.scale_stability_rerank_max_base_margin)
        target_class = int(target_class)
        competitor_class = int(competitor_class)
        target_match = topk_indices == target_class
        target_rank = torch.where(
            target_match,
            rank_template.expand_as(topk_indices),
            torch.full_like(topk_indices, topk + 1),
        ).min(dim=0).values
        target_in_topk = target_rank < topk
        base_margin = base_logits[target_class] - base_logits[competitor_class]
        competitor_top1 = top1_idx == competitor_class
        return (
            competitor_top1
            & target_in_topk
            & (target_rank < max_target_rank)
            & (base_margin >= min_base_margin)
            & (base_margin <= max_base_margin)
        )

    def _prepare_scale_stability_rerank_pairs_and_queries(
            self, base_logits, topk_values, topk_indices, top1_idx, topk,
            rank_template):
        pair_mode = self._scale_stability_rerank_pair_mode_name()
        query_mode = self._scale_stability_rerank_query_mode_name()
        min_pixels = max(1, int(self.scale_stability_rerank_auto_min_pixels))
        bg_indices = self._scale_stability_rerank_bg_indices()
        exclude_bg = self._scale_stability_rerank_exclude_bg()
        selected_pairs = []
        pair_pixel_counts = {}

        if pair_mode == 'manual':
            selected_pairs = [
                (int(target), int(competitor))
                for target, competitor in self._get_scale_stability_rerank_pairs()
            ]
        else:
            max_rank = max(
                1,
                min(int(self.scale_stability_rerank_max_target_rank), topk))
            for rank in range(1, max_rank):
                target_idx = topk_indices[rank]
                competitor_idx = topk_indices[0]
                base_margin = topk_values[rank] - topk_values[0]
                candidate = (
                    (target_idx != competitor_idx)
                    & (base_margin >= float(
                        self.scale_stability_rerank_min_base_margin))
                    & (base_margin <= float(
                        self.scale_stability_rerank_max_base_margin))
                )
                if exclude_bg and bg_indices:
                    bg_tensor = torch.zeros(
                        self.num_cls, device=base_logits.device,
                        dtype=torch.bool)
                    bg_tensor[list(bg_indices)] = True
                    candidate = (
                        candidate
                        & (~bg_tensor[target_idx])
                        & (~bg_tensor[competitor_idx])
                    )
                if not candidate.any():
                    continue
                pair_ids = torch.stack(
                    [target_idx[candidate], competitor_idx[candidate]], dim=1)
                for target_class, competitor_class in torch.unique(
                        pair_ids, dim=0).tolist():
                    target_class = int(target_class)
                    competitor_class = int(competitor_class)
                    if pair_mode == 'role_auto' and not (
                            self._scale_stability_role_compatible(
                                target_class, competitor_class)):
                        continue
                    pair_mask = (
                        candidate
                        & (target_idx == target_class)
                        & (competitor_idx == competitor_class)
                    )
                    pixels = int(pair_mask.sum().item())
                    if pixels < min_pixels:
                        continue
                    key = (target_class, competitor_class)
                    pair_pixel_counts[key] = pair_pixel_counts.get(key, 0) + pixels
            selected_pairs = [
                pair
                for pair, _ in sorted(
                    pair_pixel_counts.items(),
                    key=lambda item: item[1],
                    reverse=True)
            ]

        valid_pairs = []
        for target_class, competitor_class in selected_pairs:
            target_class = int(target_class)
            competitor_class = int(competitor_class)
            if target_class == competitor_class:
                continue
            if not (0 <= target_class < self.num_cls
                    and 0 <= competitor_class < self.num_cls):
                continue
            if exclude_bg and pair_mode != 'manual':
                if target_class in bg_indices or competitor_class in bg_indices:
                    continue
            if (target_class, competitor_class) not in valid_pairs:
                valid_pairs.append((target_class, competitor_class))

        if query_mode == 'all_classes':
            query_classes = list(range(self.num_cls))
        else:
            query_classes = set()
            required_query_classes = set()
            for target_class, competitor_class in valid_pairs:
                query_classes.add(target_class)
                query_classes.add(competitor_class)
                required_query_classes.add(target_class)
                required_query_classes.add(competitor_class)
                candidate_mask = self._scale_stability_pair_candidate_mask(
                    base_logits, topk_indices, top1_idx, target_class,
                    competitor_class, topk, rank_template)
                if candidate_mask.any():
                    for class_idx in torch.unique(
                            topk_indices[:, candidate_mask]).tolist():
                        query_classes.add(int(class_idx))
            max_risk_classes = int(
                self.scale_stability_rerank_max_risk_classes)
            if max_risk_classes > 0 and len(query_classes) > max_risk_classes:
                ranked = []
                for class_idx in query_classes:
                    if class_idx in required_query_classes:
                        continue
                    count = int((topk_indices == int(class_idx)).sum().item())
                    ranked.append((count, int(class_idx)))
                ranked.sort(reverse=True)
                keep = set(required_query_classes)
                remaining_budget = max(0, max_risk_classes - len(keep))
                keep.update(
                    class_idx for _, class_idx in ranked[:remaining_budget])
                query_classes = keep
            query_classes = sorted(query_classes)

        return valid_pairs, query_classes, pair_mode, query_mode

    def _resize_for_scale_stability_rerank(self, image, requested_scale):
        old_max_side = self.scale_requery_max_side
        self.scale_requery_max_side = self.scale_stability_rerank_max_side
        try:
            return self._resize_for_scale_requery(image, requested_scale)
        finally:
            self.scale_requery_max_side = old_max_side

    def _build_scale_stability_rerank_logits(self, base_logits, image):
        topk = max(1, min(int(self.scale_stability_rerank_topk), self.num_cls))
        topk_values, topk_indices = torch.topk(base_logits, k=topk, dim=0)
        top1_idx = topk_indices[0]
        rank_template = torch.arange(
            topk,
            device=base_logits.device,
            dtype=topk_indices.dtype).view(topk, 1, 1)
        selected_pairs, query_classes, pair_mode, query_mode = (
            self._prepare_scale_stability_rerank_pairs_and_queries(
                base_logits, topk_values, topk_indices, top1_idx, topk,
                rank_template))
        context = dict(
            dataset_name=self.seed_dataset_name,
            selected_pairs=selected_pairs,
            queried_class_indices=query_classes,
            queried_class_names=[
                self.class_names[class_idx] for class_idx in query_classes],
            queried_class_count=len(query_classes),
            scale_stability_rerank_pair_mode=pair_mode,
            scale_stability_rerank_query_mode=query_mode,
            scale_stability_rerank_auto_min_pixels=int(
                self.scale_stability_rerank_auto_min_pixels),
            scale_stability_rerank_auto_exclude_bg=bool(
                self._scale_stability_rerank_exclude_bg()),
            scale_stability_rerank_scale=float(
                self.scale_stability_rerank_scale),
            scale_stability_rerank_topk=topk,
            scale_stability_rerank_gate=str(
                self.scale_stability_rerank_gate),
            scale_stability_rerank_drop_threshold=float(
                self.scale_stability_rerank_drop_threshold),
            pair_stats=[],
        )
        if not selected_pairs or not query_classes:
            return base_logits, context

        scaled_image, effective_scale_x, effective_scale_y = (
            self._resize_for_scale_stability_rerank(
                image,
                float(self.scale_stability_rerank_scale)))
        scaled_w, scaled_h = scaled_image.size
        target_size = base_logits.shape[-2:]
        if query_mode == 'all_classes':
            scaled_logits, scaled_semantic, scaled_instance = (
                self._compute_scaled_class_logits(scaled_image, target_size))
        else:
            scaled_logits, scaled_semantic, scaled_instance = (
                self._compute_scaled_selected_class_logits(
                    scaled_image, target_size, query_classes))
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        scaled_topk_scores = torch.gather(scaled_logits, 0, topk_indices)
        scaled_choice_score, scaled_choice_rank = scaled_topk_scores.max(dim=0)
        scaled_choice_class = torch.gather(
            topk_indices,
            0,
            scaled_choice_rank.unsqueeze(0),
        ).squeeze(0)
        scaled_choice_base_score = torch.gather(
            topk_values,
            0,
            scaled_choice_rank.unsqueeze(0),
        ).squeeze(0)
        scaled_choice_drop = scaled_choice_base_score - scaled_choice_score
        reranked = base_logits.clone()

        gate_name = str(self.scale_stability_rerank_gate).lower()
        drop_threshold = float(self.scale_stability_rerank_drop_threshold)
        max_target_rank = max(1, int(self.scale_stability_rerank_max_target_rank))
        min_scaled_margin = float(self.scale_stability_rerank_min_scaled_margin)
        min_base_margin = float(self.scale_stability_rerank_min_base_margin)
        max_base_margin = float(self.scale_stability_rerank_max_base_margin)
        boost = float(self.scale_stability_rerank_margin_boost)

        for target_class, competitor_class in selected_pairs:
            target_class = int(target_class)
            competitor_class = int(competitor_class)
            if target_class == competitor_class:
                continue
            if not (0 <= target_class < self.num_cls
                    and 0 <= competitor_class < self.num_cls):
                continue

            target_match = topk_indices == target_class
            target_rank = torch.where(
                target_match,
                rank_template.expand_as(topk_indices),
                torch.full_like(topk_indices, topk + 1),
            ).min(dim=0).values
            target_in_topk = target_rank < topk

            base_target = base_logits[target_class]
            base_competitor = base_logits[competitor_class]
            scaled_target = scaled_logits[target_class]
            scaled_competitor = scaled_logits[competitor_class]
            base_margin = base_target - base_competitor
            scaled_margin = scaled_target - scaled_competitor
            margin_gain = scaled_margin - base_margin
            competitor_drop = base_competitor - scaled_competitor
            target_drop = base_target - scaled_target
            relative_stability = competitor_drop - target_drop
            choice_relative_stability = competitor_drop - scaled_choice_drop
            scaled_choice_is_target = scaled_choice_class == target_class
            competitor_top1 = top1_idx == competitor_class
            candidate_mask = (
                competitor_top1
                & target_in_topk
                & (target_rank < max_target_rank)
                & (base_margin >= min_base_margin)
                & (base_margin <= max_base_margin)
                & (scaled_margin >= min_scaled_margin)
            )

            if gate_name == 'choice':
                gate_mask = candidate_mask & scaled_choice_is_target
            elif gate_name == 'drop':
                gate_mask = (
                    candidate_mask
                    & scaled_choice_is_target
                    & (competitor_drop >= drop_threshold)
                )
            elif gate_name == 'relative':
                gate_mask = (
                    candidate_mask
                    & scaled_choice_is_target
                    & (relative_stability >= drop_threshold)
                )
            elif gate_name == 'choice_relative':
                gate_mask = (
                    candidate_mask
                    & scaled_choice_is_target
                    & (choice_relative_stability >= drop_threshold)
                )
            else:
                raise ValueError(
                    "scale_stability_rerank_gate must be one of "
                    "'choice', 'drop', 'relative', or 'choice_relative', "
                    f"but got {self.scale_stability_rerank_gate!r}")

            if gate_mask.any():
                target_score = reranked[target_class]
                competitor_score = reranked[competitor_class]
                target_score = torch.where(
                    gate_mask,
                    torch.maximum(target_score, competitor_score + boost),
                    target_score,
                )
                reranked[target_class] = target_score

            candidate_pixels = int(candidate_mask.sum().item())
            gate_pixels = int(gate_mask.sum().item())
            row = dict(
                target_class_index=target_class,
                target_class_name=self.class_names[target_class],
                competitor_class_index=competitor_class,
                competitor_class_name=self.class_names[competitor_class],
                pair_mode=pair_mode,
                query_mode=query_mode,
                queried_class_count=len(query_classes),
                auto_min_pixels=int(self.scale_stability_rerank_auto_min_pixels),
                requested_scale=float(self.scale_stability_rerank_scale),
                effective_scale_x=effective_scale_x,
                effective_scale_y=effective_scale_y,
                original_size=list(image.size),
                scaled_size=[scaled_w, scaled_h],
                scale_stability_rerank_topk=topk,
                gate_name=gate_name,
                drop_threshold=drop_threshold,
                min_scaled_margin=min_scaled_margin,
                min_base_margin=min_base_margin,
                max_base_margin=max_base_margin,
                max_target_rank=max_target_rank,
                candidate_pixels=candidate_pixels,
                target_in_topk_pixels=int(
                    (competitor_top1 & target_in_topk).sum().item()),
                scaled_choice_is_target_pixels=int(
                    (competitor_top1 & target_in_topk
                     & scaled_choice_is_target).sum().item()),
                gate_pixels=gate_pixels,
                gate_base_margin_sum=_masked_sum(base_margin, gate_mask),
                gate_scaled_margin_sum=_masked_sum(scaled_margin, gate_mask),
                gate_margin_gain_sum=_masked_sum(margin_gain, gate_mask),
                gate_competitor_drop_sum=_masked_sum(
                    competitor_drop, gate_mask),
                gate_target_drop_sum=_masked_sum(target_drop, gate_mask),
                gate_relative_stability_sum=_masked_sum(
                    relative_stability, gate_mask),
                gate_choice_relative_stability_sum=_masked_sum(
                    choice_relative_stability, gate_mask),
                gate_target_semantic_sum=_masked_sum(
                    scaled_semantic[target_class], gate_mask),
                gate_competitor_semantic_sum=_masked_sum(
                    scaled_semantic[competitor_class], gate_mask),
                gate_target_instance_sum=_masked_sum(
                    scaled_instance[target_class], gate_mask),
                gate_competitor_instance_sum=_masked_sum(
                    scaled_instance[competitor_class], gate_mask),
                gate_mask=gate_mask,
            )
            context['pair_stats'].append(row)

        context.update(dict(
            effective_scale_x=effective_scale_x,
            effective_scale_y=effective_scale_y,
            original_size=list(image.size),
            scaled_size=[scaled_w, scaled_h],
            scale_stability_rerank_max_side=int(
                self.scale_stability_rerank_max_side),
            scale_stability_rerank_min_scaled_margin=min_scaled_margin,
            scale_stability_rerank_min_base_margin=min_base_margin,
            scale_stability_rerank_max_base_margin=max_base_margin,
            scale_stability_rerank_max_target_rank=max_target_rank,
            scale_stability_rerank_margin_boost=boost,
            scale_stability_rerank_pair_mode=pair_mode,
            scale_stability_rerank_query_mode=query_mode,
            scale_stability_rerank_auto_min_pixels=int(
                self.scale_stability_rerank_auto_min_pixels),
            scale_stability_rerank_auto_exclude_bg=bool(
                self._scale_stability_rerank_exclude_bg()),
        ))
        return reranked, context

    def _build_scale_stability_rerank_stats(self, base_logits, reranked_logits,
                                            base_pred, reranked_pred,
                                            data_sample, context):
        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)
        if gt_data is None or context is None:
            return None

        if base_pred.shape[-2:] != gt_data.shape[-2:]:
            base_pred = F.interpolate(
                base_pred.float().unsqueeze(0).unsqueeze(0),
                size=gt_data.shape[-2:],
                mode='nearest').squeeze(0).squeeze(0).long()
        if reranked_pred.shape[-2:] != gt_data.shape[-2:]:
            reranked_pred = F.interpolate(
                reranked_pred.float().unsqueeze(0).unsqueeze(0),
                size=gt_data.shape[-2:],
                mode='nearest').squeeze(0).squeeze(0).long()
        if base_logits.shape[-2:] != gt_data.shape[-2:]:
            base_logits = F.interpolate(
                base_logits.unsqueeze(0),
                size=gt_data.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
        if reranked_logits.shape[-2:] != gt_data.shape[-2:]:
            reranked_logits = F.interpolate(
                reranked_logits.unsqueeze(0),
                size=gt_data.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

        valid_mask = gt_data != 255
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return dict(valid_pixels=0, overall=None, pair_stats=[])

        base_correct = (base_pred == gt_data) & valid_mask
        reranked_correct = (reranked_pred == gt_data) & valid_mask
        changed = (base_pred != reranked_pred) & valid_mask
        improved = (~base_correct) & reranked_correct
        harmed = base_correct & (~reranked_correct)
        overall = dict(
            valid_pixels=valid_count,
            base_correct_pixels=int(base_correct.sum().item()),
            reranked_correct_pixels=int(reranked_correct.sum().item()),
            changed_pixels=int(changed.sum().item()),
            improved_pixels=int(improved.sum().item()),
            harmed_pixels=int(harmed.sum().item()),
            net_improved_pixels=(
                int(improved.sum().item()) - int(harmed.sum().item())),
            base_accuracy=_safe_div(int(base_correct.sum().item()), valid_count),
            reranked_accuracy=_safe_div(
                int(reranked_correct.sum().item()), valid_count),
        )

        pair_stats = []
        for raw_row in context.get('pair_stats') or []:
            gate_mask = raw_row.get('gate_mask')
            if gate_mask is None:
                continue
            gate_mask = gate_mask & valid_mask
            target_class = int(raw_row['target_class_index'])
            competitor_class = int(raw_row['competitor_class_index'])
            pair_error_mask = (
                valid_mask
                & (gt_data == target_class)
                & (base_pred == competitor_class)
            )
            gate_pixels = int(gate_mask.sum().item())
            pair_error_pixels = int(pair_error_mask.sum().item())
            pair_error_gate = gate_mask & pair_error_mask
            changed_gate = gate_mask & changed
            improved_gate = gate_mask & improved
            harmed_gate = gate_mask & harmed

            row = {
                key: value
                for key, value in raw_row.items()
                if key != 'gate_mask'
            }
            row.update(dict(
                valid_pixels=valid_count,
                pair_error_pixels=pair_error_pixels,
                pair_error_gate_pixels=int(pair_error_gate.sum().item()),
                pair_error_gate_coverage=_safe_div(
                    int(pair_error_gate.sum().item()), pair_error_pixels),
                gate_valid_pixels=gate_pixels,
                gate_true_target_pixels=int(
                    (gate_mask & (gt_data == target_class)).sum().item()),
                gate_true_competitor_pixels=int(
                    (gate_mask & (gt_data == competitor_class)).sum().item()),
                gate_true_other_pixels=int(
                    (gate_mask
                     & (gt_data != target_class)
                     & (gt_data != competitor_class)).sum().item()),
                gate_counterfactual_improved_pixels=int(
                    (gate_mask & (gt_data == target_class)).sum().item()),
                gate_counterfactual_harmed_pixels=int(
                    (gate_mask & (gt_data == competitor_class)).sum().item()),
                gate_counterfactual_other_pixels=int(
                    (gate_mask
                     & (gt_data != target_class)
                     & (gt_data != competitor_class)).sum().item()),
                gate_counterfactual_net_pixels=(
                    int((gate_mask & (gt_data == target_class)).sum().item())
                    - int((gate_mask & (gt_data == competitor_class)).sum().item())),
                gate_true_target_ratio=_safe_div(
                    int((gate_mask & (gt_data == target_class)).sum().item()),
                    gate_pixels),
                gate_counterfactual_precision=_safe_div(
                    int((gate_mask & (gt_data == target_class)).sum().item()),
                    (int((gate_mask & (gt_data == target_class)).sum().item())
                     + int((gate_mask & (gt_data == competitor_class)).sum().item()))),
                changed_gate_pixels=int(changed_gate.sum().item()),
                improved_gate_pixels=int(improved_gate.sum().item()),
                harmed_gate_pixels=int(harmed_gate.sum().item()),
                net_improved_gate_pixels=(
                    int(improved_gate.sum().item())
                    - int(harmed_gate.sum().item())),
                changed_gate_ratio=_safe_div(
                    int(changed_gate.sum().item()), gate_pixels),
                improved_gate_ratio=_safe_div(
                    int(improved_gate.sum().item()), gate_pixels),
                harmed_gate_ratio=_safe_div(
                    int(harmed_gate.sum().item()), gate_pixels),
            ))
            pair_stats.append(row)

        pair_stats.sort(key=lambda item: item.get('gate_valid_pixels', 0),
                        reverse=True)
        return dict(
            dataset_name=self.seed_dataset_name,
            valid_pixels=valid_count,
            overall=overall,
            pair_stats=pair_stats,
        )

    def _write_scale_stability_rerank_stats(self, record):
        if not self.dump_scale_stability_rerank_stats:
            return
        if self._scale_stability_rerank_stats_file is None:
            path = (
                self.scale_stability_rerank_stats_path
                or './work_dirs/evidence_stats/scale_stability_rerank_stats.jsonl'
            )
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._scale_stability_rerank_stats_file = open(
                path, 'a', buffering=1)
        self._scale_stability_rerank_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    def _build_expert_reliability_stats(self, base_logits, base_pred,
                                        data_sample, components,
                                        scale_context):
        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)
        if gt_data is None or components is None or scale_context is None:
            return None
        semantic_logits = components.get('semantic_logits')
        instance_logits = components.get('instance_logits')
        if semantic_logits is None or instance_logits is None:
            return None

        if base_pred.shape[-2:] != gt_data.shape[-2:]:
            base_pred = F.interpolate(
                base_pred.float().unsqueeze(0).unsqueeze(0),
                size=gt_data.shape[-2:],
                mode='nearest').squeeze(0).squeeze(0).long()
        if base_logits.shape[-2:] != gt_data.shape[-2:]:
            base_logits = F.interpolate(
                base_logits.unsqueeze(0),
                size=gt_data.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
        if semantic_logits.shape[-2:] != gt_data.shape[-2:]:
            semantic_logits = F.interpolate(
                semantic_logits.unsqueeze(0),
                size=gt_data.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
        if instance_logits.shape[-2:] != gt_data.shape[-2:]:
            instance_logits = F.interpolate(
                instance_logits.unsqueeze(0),
                size=gt_data.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

        valid_mask = gt_data != 255
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return dict(valid_pixels=0, pair_expert_stats=[])

        gt_idx = gt_data.clamp(min=0, max=self.num_cls - 1)
        final_top1_idx = base_logits.argmax(dim=0)
        semantic_top1_idx = semantic_logits.argmax(dim=0)
        instance_top1_idx = instance_logits.argmax(dim=0)
        local_consistency = self._same_class_local_consistency(
            final_top1_idx, self.expert_reliability_local_kernel)

        seed_context = self._build_seed_rule_context(
            base_logits, base_pred, components)
        target_seed_counts = {}
        if seed_context is not None:
            seed_mask = seed_context['rule_defs'][-1][1] & valid_mask
            seed_class = seed_context['final_top1_idx'].clamp(
                min=0, max=self.num_cls - 1)
            for class_idx in range(self.num_cls):
                target_seed_counts[class_idx] = int(
                    (seed_mask & (seed_class == class_idx)).sum().item())
        else:
            target_seed_counts = {class_idx: 0 for class_idx in range(self.num_cls)}

        min_gate_pixels = max(1, int(self.expert_reliability_min_gate_pixels))
        rows = []

        def row_for_mask(raw_row, expert_name, mask, target_class,
                         competitor_class, extra=None):
            mask = mask & valid_mask
            pixels = int(mask.sum().item())
            if pixels < min_gate_pixels:
                return None
            gt_target = mask & (gt_idx == target_class)
            gt_competitor = mask & (gt_idx == competitor_class)
            gt_other = mask & (gt_idx != target_class) & (gt_idx != competitor_class)
            pair_error = (
                valid_mask
                & (gt_idx == target_class)
                & (base_pred == competitor_class)
            )
            improved_pixels = int(gt_target.sum().item())
            harmed_pixels = int(gt_competitor.sum().item())
            row = dict(
                dataset_name=self.seed_dataset_name,
                target_class_index=int(target_class),
                target_class_name=self.class_names[int(target_class)],
                competitor_class_index=int(competitor_class),
                competitor_class_name=self.class_names[int(competitor_class)],
                expert_name=expert_name,
                pair_mode=scale_context.get(
                    'scale_stability_rerank_pair_mode'),
                query_mode=scale_context.get(
                    'scale_stability_rerank_query_mode'),
                requested_scale=scale_context.get(
                    'scale_stability_rerank_scale'),
                gate_name=scale_context.get(
                    'scale_stability_rerank_gate'),
                drop_threshold=scale_context.get(
                    'scale_stability_rerank_drop_threshold'),
                valid_pixels=valid_count,
                expert_pixels=pixels,
                pair_error_pixels=int(pair_error.sum().item()),
                pair_error_expert_pixels=int((pair_error & mask).sum().item()),
                pair_error_expert_coverage=_safe_div(
                    int((pair_error & mask).sum().item()),
                    int(pair_error.sum().item())),
                gt_target_pixels=improved_pixels,
                gt_competitor_pixels=harmed_pixels,
                gt_other_pixels=int(gt_other.sum().item()),
                gt_target_ratio=_safe_div(improved_pixels, pixels),
                gt_competitor_ratio=_safe_div(harmed_pixels, pixels),
                gt_other_ratio=_safe_div(int(gt_other.sum().item()), pixels),
                counterfactual_net_pixels=improved_pixels - harmed_pixels,
                counterfactual_precision=_safe_div(
                    improved_pixels, improved_pixels + harmed_pixels),
                mean_final_margin=_masked_mean(
                    base_logits[target_class] - base_logits[competitor_class],
                    mask),
                mean_semantic_margin=_masked_mean(
                    semantic_logits[target_class] - semantic_logits[competitor_class],
                    mask),
                mean_instance_margin=_masked_mean(
                    instance_logits[target_class] - instance_logits[competitor_class],
                    mask),
                mean_local_consistency=_masked_mean(local_consistency, mask),
                final_top1_target_pixels=int(
                    (mask & (final_top1_idx == target_class)).sum().item()),
                final_top1_competitor_pixels=int(
                    (mask & (final_top1_idx == competitor_class)).sum().item()),
                semantic_top1_target_pixels=int(
                    (mask & (semantic_top1_idx == target_class)).sum().item()),
                semantic_top1_competitor_pixels=int(
                    (mask & (semantic_top1_idx == competitor_class)).sum().item()),
                instance_top1_target_pixels=int(
                    (mask & (instance_top1_idx == target_class)).sum().item()),
                instance_top1_competitor_pixels=int(
                    (mask & (instance_top1_idx == competitor_class)).sum().item()),
                target_seed_pixels=int(target_seed_counts.get(
                    int(target_class), 0)),
                competitor_seed_pixels=int(target_seed_counts.get(
                    int(competitor_class), 0)),
            )
            if extra:
                row.update(extra)
            return row

        for raw_row in scale_context.get('pair_stats') or []:
            gate_mask = raw_row.get('gate_mask')
            if gate_mask is None:
                continue
            target_class = int(raw_row['target_class_index'])
            competitor_class = int(raw_row['competitor_class_index'])
            gate_mask = gate_mask & valid_mask
            if int(gate_mask.sum().item()) < min_gate_pixels:
                continue

            final_margin = base_logits[target_class] - base_logits[competitor_class]
            semantic_margin = (
                semantic_logits[target_class] - semantic_logits[competitor_class])
            instance_margin = (
                instance_logits[target_class] - instance_logits[competitor_class])
            final_favors = final_margin > 0
            semantic_favors = semantic_margin > 0
            instance_favors = instance_margin > 0
            vote_count = (
                final_favors.long()
                + semantic_favors.long()
                + instance_favors.long())
            local_strong = local_consistency >= 0.70
            base_extra = dict(
                scale_gate_pixels=int(raw_row.get('gate_pixels', 0)),
                scale_candidate_pixels=int(raw_row.get('candidate_pixels', 0)),
                scale_target_in_topk_pixels=int(raw_row.get(
                    'target_in_topk_pixels', 0)),
                scale_scaled_choice_is_target_pixels=int(raw_row.get(
                    'scaled_choice_is_target_pixels', 0)),
            )
            expert_defs = [
                ('scale_gate', gate_mask),
                ('scale_and_final_margin_pos', gate_mask & final_favors),
                ('scale_and_semantic_margin_pos', gate_mask & semantic_favors),
                ('scale_and_instance_margin_pos', gate_mask & instance_favors),
                ('scale_and_semantic_instance_pos',
                 gate_mask & semantic_favors & instance_favors),
                ('scale_and_head_vote_ge2', gate_mask & (vote_count >= 2)),
                ('scale_and_head_vote_ge1', gate_mask & (vote_count >= 1)),
                ('scale_and_local_ge0p70', gate_mask & local_strong),
                ('scale_semantic_or_instance_pos',
                 gate_mask & (semantic_favors | instance_favors)),
                ('scale_semantic_pos_instance_neg',
                 gate_mask & semantic_favors & (~instance_favors)),
                ('scale_instance_pos_semantic_neg',
                 gate_mask & instance_favors & (~semantic_favors)),
            ]
            for expert_name, expert_mask in expert_defs:
                row = row_for_mask(
                    raw_row,
                    expert_name,
                    expert_mask,
                    target_class,
                    competitor_class,
                    base_extra)
                if row is not None:
                    rows.append(row)

        rows.sort(key=lambda item: (
            item.get('dataset_name') or '',
            item.get('target_class_name') or '',
            item.get('competitor_class_name') or '',
            item.get('expert_name') or '',
        ))
        return dict(
            dataset_name=self.seed_dataset_name,
            valid_pixels=valid_count,
            expert_reliability_min_gate_pixels=min_gate_pixels,
            expert_reliability_local_kernel=int(
                self.expert_reliability_local_kernel),
            scale_pair_mode=scale_context.get(
                'scale_stability_rerank_pair_mode'),
            scale_query_mode=scale_context.get(
                'scale_stability_rerank_query_mode'),
            pair_expert_stats=rows,
        )

    def _write_expert_reliability_stats(self, record):
        if not self.dump_expert_reliability_stats:
            return
        if self._expert_reliability_stats_file is None:
            path = (
                self.expert_reliability_stats_path
                or './work_dirs/evidence_stats/expert_reliability_stats.jsonl'
            )
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._expert_reliability_stats_file = open(path, 'a', buffering=1)
        self._expert_reliability_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    def _get_evidence_bias_heads(self):
        heads = self._parse_name_list(self.evidence_bias_heads)
        allowed = {'final', 'semantic', 'instance'}
        return [head for head in heads if head in allowed] or [
            'final', 'semantic', 'instance']

    def _get_evidence_bias_probes(self):
        raw = self.evidence_bias_probes
        if raw is None:
            return []
        if isinstance(raw, str):
            items = [item.strip() for item in raw.split(',')]
        elif isinstance(raw, (list, tuple)):
            items = [str(item).strip() for item in raw]
        else:
            items = [str(raw).strip()]
        return [item for item in items if item]

    def _get_evidence_bias_manual_pairs(self):
        pairs = self.evidence_bias_pairs
        if pairs is not None and str(pairs).strip():
            return self._parse_pair_prompt_competition_pairs(pairs)
        return self._get_pair_prompt_competition_pairs()

    def _get_evidence_bias_pairs(self, base_pred, gt_data, valid_mask):
        mode = str(self.evidence_bias_pair_mode or 'manual').lower()
        if mode == 'manual':
            return self._get_evidence_bias_manual_pairs()
        if mode not in ('gt_top_errors', 'gt_errors'):
            raise ValueError(
                "evidence_bias_pair_mode must be 'manual', 'gt_top_errors', "
                f"or 'gt_errors', but got {self.evidence_bias_pair_mode!r}")
        min_pixels = max(1, int(self.evidence_bias_min_pair_pixels))
        max_pairs = int(self.evidence_bias_max_gt_pairs)
        pairs = []
        for gt_class in range(self.num_cls):
            gt_mask = valid_mask & (gt_data == gt_class)
            if not gt_mask.any():
                continue
            for pred_class in range(self.num_cls):
                if pred_class == gt_class:
                    continue
                pixels = int((gt_mask & (base_pred == pred_class)).sum().item())
                if pixels >= min_pixels:
                    pairs.append((pixels, gt_class, pred_class))
        pairs.sort(reverse=True)
        if mode == 'gt_top_errors' and max_pairs > 0:
            pairs = pairs[:max_pairs]
        return [(gt_class, pred_class) for _, gt_class, pred_class in pairs]

    def _compute_selected_class_prompt_evidence(self, image, class_indices):
        h, w = image.size[1], image.size[0]
        class_indices = sorted({
            int(class_idx)
            for class_idx in class_indices
            if 0 <= int(class_idx) < self.num_cls
        })
        final_logits = torch.full(
            (self.num_cls, h, w),
            float('-inf'),
            dtype=torch.float32,
            device=self.device,
        )
        semantic_logits = torch.full_like(final_logits, float('-inf'))
        instance_logits = torch.full_like(final_logits, float('-inf'))
        presence_scores = {
            int(class_idx): None
            for class_idx in class_indices
        }
        if not class_indices:
            return dict(
                final=final_logits,
                semantic=semantic_logits,
                instance=instance_logits,
                presence=presence_scores,
            )

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = self.processor.set_image(image)
            for class_idx in class_indices:
                self.processor.reset_all_prompts(inference_state)
                inference_state = self.processor.set_text_prompt(
                    state=inference_state,
                    prompt=self.class_names[int(class_idx)],
                )
                logit, semantic_component, instance_component = (
                    self._compute_prompt_variant_logit(inference_state, h, w))
                final_logits[int(class_idx)] = logit.float()
                if semantic_component is not None:
                    semantic_logits[int(class_idx)] = semantic_component.float()
                if instance_component is not None:
                    instance_logits[int(class_idx)] = instance_component.float()
                presence_scores[int(class_idx)] = _tensor_scalar(
                    inference_state.get('presence_score'))
                del logit
                del semantic_component
                del instance_component
        return dict(
            final=final_logits,
            semantic=semantic_logits,
            instance=instance_logits,
            presence=presence_scores,
        )

    def _resize_evidence_maps(self, evidence, target_size):
        resized = {'presence': evidence.get('presence', {})}
        for key in ('final', 'semantic', 'instance'):
            value = evidence.get(key)
            if value is None:
                continue
            if value.shape[-2:] == target_size:
                resized[key] = value.float()
            else:
                resized[key] = F.interpolate(
                    value.float().unsqueeze(0),
                    size=target_size,
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0)
        return resized

    def _edge_pad_image(self, image, pad):
        pad = int(pad)
        if pad <= 0:
            return image, [0, 0, image.size[0], image.size[1]]
        array = np.asarray(image)
        padded = np.pad(
            array,
            ((pad, pad), (pad, pad), (0, 0)),
            mode='edge',
        )
        return (
            Image.fromarray(padded),
            [pad, pad, pad + image.size[0], pad + image.size[1]],
        )

    def _build_evidence_bias_probe(self, image, probe_name, class_indices,
                                   target_size):
        probe = str(probe_name).strip()
        probe_lower = probe.lower()
        original_w, original_h = image.size
        crop_box = None
        effective_scale_x = 1.0
        effective_scale_y = 1.0

        if probe_lower in ('identity', 'base', 'original'):
            evidence = self._compute_selected_class_prompt_evidence(
                image, class_indices)
        elif probe_lower.startswith('scale:'):
            scale = float(probe_lower.split(':', 1)[1])
            old_max_side = self.scale_requery_max_side
            self.scale_requery_max_side = int(self.evidence_bias_max_side)
            try:
                probe_image, effective_scale_x, effective_scale_y = (
                    self._resize_for_scale_requery(image, scale))
            finally:
                self.scale_requery_max_side = old_max_side
            evidence = self._compute_selected_class_prompt_evidence(
                probe_image, class_indices)
        elif probe_lower.startswith('blur:'):
            radius = float(probe_lower.split(':', 1)[1])
            probe_image = image.filter(ImageFilter.GaussianBlur(radius=radius))
            evidence = self._compute_selected_class_prompt_evidence(
                probe_image, class_indices)
        elif probe_lower in ('sharpen', 'unsharp'):
            probe_image = image.filter(ImageFilter.UnsharpMask(
                radius=2, percent=150, threshold=3))
            evidence = self._compute_selected_class_prompt_evidence(
                probe_image, class_indices)
        elif probe_lower in ('hflip', 'vflip', 'rot90', 'rot180', 'rot270'):
            probe_image = self._make_transformed_image(image, probe_lower)
            evidence = self._compute_selected_class_prompt_evidence(
                probe_image, class_indices)
            for key in ('final', 'semantic', 'instance'):
                evidence[key] = self._invert_transformed_tensor(
                    evidence[key], probe_lower)
        elif probe_lower.startswith('pad:'):
            pad = int(float(probe_lower.split(':', 1)[1]))
            probe_image, crop_box = self._edge_pad_image(image, pad)
            evidence = self._compute_selected_class_prompt_evidence(
                probe_image, class_indices)
            x1, y1, x2, y2 = crop_box
            for key in ('final', 'semantic', 'instance'):
                evidence[key] = evidence[key][:, y1:y2, x1:x2]
        else:
            raise ValueError(
                "Unsupported evidence_bias probe. Use items such as "
                "'scale:0.5', 'blur:1.5', 'sharpen', 'hflip', 'rot90', "
                f"or 'pad:128', but got {probe_name!r}")

        evidence = self._resize_evidence_maps(evidence, target_size)
        return dict(
            probe_name=probe,
            original_size=[original_w, original_h],
            effective_scale_x=effective_scale_x,
            effective_scale_y=effective_scale_y,
            crop_box=crop_box,
            evidence=evidence,
        )

    def _build_evidence_bias_stats(self, base_logits, base_pred, data_sample,
                                   image):
        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)
        if gt_data is None:
            return None
        if base_pred.shape[-2:] != gt_data.shape[-2:]:
            base_pred = F.interpolate(
                base_pred.float().unsqueeze(0).unsqueeze(0),
                size=gt_data.shape[-2:],
                mode='nearest').squeeze(0).squeeze(0).long()
        if base_logits.shape[-2:] != gt_data.shape[-2:]:
            base_logits = F.interpolate(
                base_logits.unsqueeze(0),
                size=gt_data.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

        valid_mask = gt_data != 255
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return dict(valid_pixels=0, pair_stats=[])

        base_wrong = (base_pred != gt_data) & valid_mask
        wrong_count = int(base_wrong.sum().item())
        selected_pairs = self._get_evidence_bias_pairs(
            base_pred, gt_data, valid_mask)
        if not selected_pairs:
            return dict(
                dataset_name=self.seed_dataset_name,
                valid_pixels=valid_count,
                baseline_wrong_pixels=wrong_count,
                selected_pairs=[],
                probes=[],
                pair_stats=[],
            )

        min_pair_pixels = max(1, int(self.evidence_bias_min_pair_pixels))
        class_indices = sorted({
            int(class_idx)
            for pair in selected_pairs
            for class_idx in pair
            if 0 <= int(class_idx) < self.num_cls
        })
        heads = self._get_evidence_bias_heads()
        support_thresholds = self._parse_float_list(
            self.evidence_bias_support_thresholds,
            '0.05,0.10,0.20')
        target_size = gt_data.shape[-2:]
        base_direct = self._compute_selected_class_prompt_evidence(
            image, class_indices)
        base_direct = self._resize_evidence_maps(base_direct, target_size)
        if self.device.type == 'cuda' and bool(self.evidence_bias_empty_cache):
            torch.cuda.empty_cache()

        probe_rows = []
        pair_rows = []
        for probe_name in self._get_evidence_bias_probes():
            probe_context = self._build_evidence_bias_probe(
                image, probe_name, class_indices, target_size)
            probe_evidence = probe_context['evidence']
            probe_rows.append(dict(
                probe_name=probe_context['probe_name'],
                effective_scale_x=probe_context['effective_scale_x'],
                effective_scale_y=probe_context['effective_scale_y'],
                crop_box=probe_context['crop_box'],
            ))

            for target_class, competitor_class in selected_pairs:
                target_class = int(target_class)
                competitor_class = int(competitor_class)
                if target_class == competitor_class:
                    continue
                if not (0 <= target_class < self.num_cls
                        and 0 <= competitor_class < self.num_cls):
                    continue
                pair_mask = (
                    base_wrong
                    & (gt_data == target_class)
                    & (base_pred == competitor_class)
                )
                pair_pixels = int(pair_mask.sum().item())
                if pair_pixels < min_pair_pixels:
                    continue
                base_aggregate_target = base_logits[target_class]
                base_aggregate_competitor = base_logits[competitor_class]
                base_aggregate_margin = (
                    base_aggregate_target - base_aggregate_competitor)

                base_presence = base_direct.get('presence', {})
                probe_presence = probe_evidence.get('presence', {})
                target_presence_base = base_presence.get(target_class)
                competitor_presence_base = base_presence.get(competitor_class)
                target_presence_probe = probe_presence.get(target_class)
                competitor_presence_probe = probe_presence.get(competitor_class)
                target_presence_drop = (
                    None if target_presence_base is None
                    or target_presence_probe is None
                    else target_presence_base - target_presence_probe)
                competitor_presence_drop = (
                    None if competitor_presence_base is None
                    or competitor_presence_probe is None
                    else competitor_presence_base - competitor_presence_probe)
                presence_relative_stability = (
                    None if target_presence_drop is None
                    or competitor_presence_drop is None
                    else competitor_presence_drop - target_presence_drop)

                for head in heads:
                    base_head = base_direct.get(head)
                    probe_head = probe_evidence.get(head)
                    if base_head is None or probe_head is None:
                        continue
                    base_target = base_head[target_class]
                    base_competitor = base_head[competitor_class]
                    probe_target = probe_head[target_class]
                    probe_competitor = probe_head[competitor_class]
                    base_margin = base_target - base_competitor
                    probe_margin = probe_target - probe_competitor
                    margin_gain = probe_margin - base_margin
                    target_drop = base_target - probe_target
                    competitor_drop = base_competitor - probe_competitor
                    relative_stability = competitor_drop - target_drop
                    base_target_wins = base_margin > 0
                    probe_target_wins = probe_margin > 0

                    base_row = dict(
                        dataset_name=self.seed_dataset_name,
                        pair_mode=str(self.evidence_bias_pair_mode),
                        probe_name=probe_context['probe_name'],
                        head=head,
                        target_class_index=target_class,
                        target_class_name=self.class_names[target_class],
                        competitor_class_index=competitor_class,
                        competitor_class_name=self.class_names[competitor_class],
                        valid_pixels=valid_count,
                        baseline_wrong_pixels=wrong_count,
                        pair_pixels=pair_pixels,
                        pair_ratio_in_error=_safe_div(pair_pixels, wrong_count),
                        base_aggregate_margin_sum=_masked_sum(
                            base_aggregate_margin, pair_mask),
                        base_direct_margin_sum=_masked_sum(base_margin, pair_mask),
                        probe_margin_sum=_masked_sum(probe_margin, pair_mask),
                        margin_gain_sum=_masked_sum(margin_gain, pair_mask),
                        target_drop_sum=_masked_sum(target_drop, pair_mask),
                        competitor_drop_sum=_masked_sum(
                            competitor_drop, pair_mask),
                        relative_stability_sum=_masked_sum(
                            relative_stability, pair_mask),
                        base_target_score_sum=_masked_sum(
                            base_target, pair_mask),
                        base_competitor_score_sum=_masked_sum(
                            base_competitor, pair_mask),
                        probe_target_score_sum=_masked_sum(
                            probe_target, pair_mask),
                        probe_competitor_score_sum=_masked_sum(
                            probe_competitor, pair_mask),
                        base_target_wins_pixels=int(
                            (base_target_wins & pair_mask).sum().item()),
                        probe_target_wins_pixels=int(
                            (probe_target_wins & pair_mask).sum().item()),
                        target_presence_base=target_presence_base,
                        competitor_presence_base=competitor_presence_base,
                        target_presence_probe=target_presence_probe,
                        competitor_presence_probe=competitor_presence_probe,
                        target_presence_drop=target_presence_drop,
                        competitor_presence_drop=competitor_presence_drop,
                        presence_relative_stability=(
                            presence_relative_stability),
                    )
                    pair_rows.append(base_row)

                    for threshold in support_thresholds:
                        threshold = float(threshold)
                        base_target_support = base_target >= threshold
                        base_competitor_support = base_competitor >= threshold
                        probe_target_support = probe_target >= threshold
                        probe_competitor_support = probe_competitor >= threshold
                        target_gain_support = (
                            probe_target_support & (~base_target_support))
                        competitor_lost_support = (
                            base_competitor_support & (~probe_competitor_support))
                        support_row = dict(base_row)
                        support_row.update(dict(
                            support_threshold=threshold,
                            base_target_support_pixels=int(
                                (base_target_support & pair_mask).sum().item()),
                            base_competitor_support_pixels=int(
                                (base_competitor_support & pair_mask).sum().item()),
                            probe_target_support_pixels=int(
                                (probe_target_support & pair_mask).sum().item()),
                            probe_competitor_support_pixels=int(
                                (probe_competitor_support & pair_mask).sum().item()),
                            probe_target_only_support_pixels=int(
                                (probe_target_support & (~probe_competitor_support)
                                 & pair_mask).sum().item()),
                            probe_competitor_only_support_pixels=int(
                                (probe_competitor_support & (~probe_target_support)
                                 & pair_mask).sum().item()),
                            probe_both_support_pixels=int(
                                (probe_target_support & probe_competitor_support
                                 & pair_mask).sum().item()),
                            probe_neither_support_pixels=int(
                                ((~probe_target_support) & (~probe_competitor_support)
                                 & pair_mask).sum().item()),
                            target_gain_support_pixels=int(
                                (target_gain_support & pair_mask).sum().item()),
                            competitor_lost_support_pixels=int(
                                (competitor_lost_support & pair_mask).sum().item()),
                        ))
                        pair_rows.append(support_row)

            del probe_evidence
            if self.device.type == 'cuda' and bool(self.evidence_bias_empty_cache):
                torch.cuda.empty_cache()

        pair_rows.sort(key=lambda item: (
            item.get('probe_name') or '',
            item.get('head') or '',
            -(item.get('pair_pixels') or 0),
        ))
        return dict(
            dataset_name=self.seed_dataset_name,
            valid_pixels=valid_count,
            baseline_wrong_pixels=wrong_count,
            selected_pairs=selected_pairs,
            selected_pair_names=[
                [self.class_names[target], self.class_names[competitor]]
                for target, competitor in selected_pairs
                if 0 <= int(target) < self.num_cls
                and 0 <= int(competitor) < self.num_cls
            ],
            selected_class_indices=class_indices,
            selected_class_names=[self.class_names[idx] for idx in class_indices],
            probes=probe_rows,
            heads=heads,
            support_thresholds=support_thresholds,
            evidence_bias_min_pair_pixels=min_pair_pixels,
            evidence_bias_max_gt_pairs=int(self.evidence_bias_max_gt_pairs),
            evidence_bias_max_side=int(self.evidence_bias_max_side),
            pair_stats=pair_rows,
        )

    def _write_evidence_bias_stats(self, record):
        if not self.dump_evidence_bias_stats:
            return
        if self._evidence_bias_stats_file is None:
            path = (
                self.evidence_bias_stats_path
                or './work_dirs/evidence_stats/evidence_bias_stats.jsonl'
            )
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._evidence_bias_stats_file = open(path, 'a', buffering=1)
        self._evidence_bias_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    def _get_topk_conflict_heads(self):
        heads = self._parse_name_list(self.topk_conflict_heads)
        allowed = {'final', 'semantic', 'instance'}
        return [head for head in heads if head in allowed] or [
            'final', 'semantic', 'instance']

    def _get_topk_conflict_probes(self):
        raw = self.topk_conflict_probes
        if raw is None:
            return []
        if isinstance(raw, str):
            items = [item.strip() for item in raw.split(',')]
        elif isinstance(raw, (list, tuple)):
            items = [str(item).strip() for item in raw]
        else:
            items = [str(raw).strip()]
        return [item for item in items if item]

    def _topk_conflict_pair_counts(self, topk_idx, valid_mask):
        topk = int(topk_idx.shape[0])
        if topk <= 1:
            return {}
        top1 = topk_idx[0]
        counts = {}
        for rank in range(1, topk):
            candidate = topk_idx[rank]
            encoded = (candidate.long() * int(self.num_cls) + top1.long())
            encoded = encoded[valid_mask]
            if encoded.numel() == 0:
                continue
            values, value_counts = torch.unique(encoded, return_counts=True)
            for value, count in zip(values.tolist(), value_counts.tolist()):
                target = int(value // int(self.num_cls))
                competitor = int(value % int(self.num_cls))
                if target == competitor:
                    continue
                counts[(target, competitor)] = (
                    counts.get((target, competitor), 0) + int(count))
        return counts

    def _topk_conflict_pair_mask(self, topk_idx, target_class, competitor_class,
                                 valid_mask):
        if topk_idx.shape[0] <= 1:
            return torch.zeros_like(valid_mask, dtype=torch.bool)
        target_hits = (topk_idx[1:] == int(target_class)).any(dim=0)
        return valid_mask & (topk_idx[0] == int(competitor_class)) & target_hits

    def _topk_conflict_selected_pairs(self, pair_counts):
        min_pixels = max(1, int(self.topk_conflict_min_pair_pixels))
        max_pairs = int(self.topk_conflict_max_pairs)
        pairs = [
            (pixels, target, competitor)
            for (target, competitor), pixels in pair_counts.items()
            if int(pixels) >= min_pixels
        ]
        pairs.sort(reverse=True)
        if max_pairs > 0:
            pairs = pairs[:max_pairs]
        return [(target, competitor, pixels) for pixels, target, competitor in pairs]

    def _build_topk_conflict_bias_stats(self, base_logits, base_pred,
                                        data_sample, components, image):
        if components is None or self.num_cls <= 1:
            return None
        semantic_logits = components.get('semantic_logits')
        instance_logits = components.get('instance_logits')
        if semantic_logits is None or instance_logits is None:
            return None

        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)

        target_size = base_logits.shape[-2:]
        if gt_data is not None and gt_data.shape[-2:] != target_size:
            gt_data = F.interpolate(
                gt_data.float().unsqueeze(0).unsqueeze(0),
                size=target_size,
                mode='nearest',
            ).squeeze(0).squeeze(0).long()
        if base_pred.shape[-2:] != target_size:
            base_pred = F.interpolate(
                base_pred.float().unsqueeze(0).unsqueeze(0),
                size=target_size,
                mode='nearest',
            ).squeeze(0).squeeze(0).long()
        if semantic_logits.shape[-2:] != target_size:
            semantic_logits = F.interpolate(
                semantic_logits.unsqueeze(0),
                size=target_size,
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
        if instance_logits.shape[-2:] != target_size:
            instance_logits = F.interpolate(
                instance_logits.unsqueeze(0),
                size=target_size,
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

        if gt_data is None:
            valid_mask = torch.ones(target_size, dtype=torch.bool, device=self.device)
        else:
            valid_mask = gt_data != 255
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return dict(valid_pixels=0, pair_stats=[], probe_stats=[])

        topk = max(2, min(int(self.topk_conflict_topk), int(self.num_cls)))
        topk_vals, topk_idx = torch.topk(base_logits, k=topk, dim=0)
        top1_idx = topk_idx[0]
        top1_score = topk_vals[0]
        top2_score = topk_vals[1]
        top1_margin = top1_score - top2_score
        local_consistency = self._same_class_local_consistency(
            top1_idx, int(self.topk_conflict_local_kernel))
        pair_counts = self._topk_conflict_pair_counts(topk_idx, valid_mask)
        selected_pairs = self._topk_conflict_selected_pairs(pair_counts)
        if not selected_pairs:
            return dict(
                dataset_name=self.seed_dataset_name,
                valid_pixels=valid_count,
                topk=topk,
                selected_pairs=[],
                pair_stats=[],
                probe_stats=[],
            )

        semantic_top1 = semantic_logits.argmax(dim=0)
        instance_top1 = instance_logits.argmax(dim=0)
        semantic_topk_idx = torch.topk(
            semantic_logits, k=topk, dim=0).indices
        instance_topk_idx = torch.topk(
            instance_logits, k=topk, dim=0).indices

        pair_rows = []
        class_indices = set()
        for target_class, competitor_class, conflict_pixels in selected_pairs:
            target_class = int(target_class)
            competitor_class = int(competitor_class)
            conflict_mask = self._topk_conflict_pair_mask(
                topk_idx, target_class, competitor_class, valid_mask)
            pixels = int(conflict_mask.sum().item())
            if pixels < int(self.topk_conflict_min_pair_pixels):
                continue
            class_indices.add(target_class)
            class_indices.add(competitor_class)

            reverse_pixels = int(pair_counts.get(
                (competitor_class, target_class), 0))
            target_score = base_logits[target_class]
            competitor_score = base_logits[competitor_class]
            margin = target_score - competitor_score
            semantic_margin = (
                semantic_logits[target_class] - semantic_logits[competitor_class])
            instance_margin = (
                instance_logits[target_class] - instance_logits[competitor_class])
            semantic_target_topk = (
                semantic_topk_idx == target_class).any(dim=0)
            semantic_competitor_topk = (
                semantic_topk_idx == competitor_class).any(dim=0)
            instance_target_topk = (
                instance_topk_idx == target_class).any(dim=0)
            instance_competitor_topk = (
                instance_topk_idx == competitor_class).any(dim=0)

            target_rank_pixels = {}
            for rank in range(1, topk):
                target_rank_pixels[f'target_rank{rank + 1}_pixels'] = int(
                    (conflict_mask & (topk_idx[rank] == target_class)).sum().item())

            row = dict(
                dataset_name=self.seed_dataset_name,
                target_class_index=target_class,
                target_class_name=self.class_names[target_class],
                competitor_class_index=competitor_class,
                competitor_class_name=self.class_names[competitor_class],
                valid_pixels=valid_count,
                topk=topk,
                conflict_pixels=pixels,
                conflict_ratio=_safe_div(pixels, valid_count),
                raw_conflict_count=int(conflict_pixels),
                reverse_conflict_pixels=reverse_pixels,
                conflict_ratio_vs_reverse=_safe_div(
                    pixels, reverse_pixels),
                log_conflict_ratio_vs_reverse=float(
                    np.log((pixels + 1.0) / (reverse_pixels + 1.0))),
                mean_top1_score=_masked_mean(top1_score, conflict_mask),
                mean_top1_margin=_masked_mean(top1_margin, conflict_mask),
                mean_target_score=_masked_mean(target_score, conflict_mask),
                mean_competitor_score=_masked_mean(
                    competitor_score, conflict_mask),
                mean_target_vs_competitor_margin=_masked_mean(
                    margin, conflict_mask),
                target_score_sum=_masked_sum(target_score, conflict_mask),
                competitor_score_sum=_masked_sum(
                    competitor_score, conflict_mask),
                target_vs_competitor_margin_sum=_masked_sum(
                    margin, conflict_mask),
                semantic_target_favors_pixels=int(
                    ((semantic_margin > 0) & conflict_mask).sum().item()),
                instance_target_favors_pixels=int(
                    ((instance_margin > 0) & conflict_mask).sum().item()),
                semantic_top1_target_pixels=int(
                    ((semantic_top1 == target_class) & conflict_mask).sum().item()),
                semantic_top1_competitor_pixels=int(
                    ((semantic_top1 == competitor_class) & conflict_mask).sum().item()),
                instance_top1_target_pixels=int(
                    ((instance_top1 == target_class) & conflict_mask).sum().item()),
                instance_top1_competitor_pixels=int(
                    ((instance_top1 == competitor_class) & conflict_mask).sum().item()),
                semantic_topk_target_pixels=int(
                    (semantic_target_topk & conflict_mask).sum().item()),
                semantic_topk_competitor_pixels=int(
                    (semantic_competitor_topk & conflict_mask).sum().item()),
                instance_topk_target_pixels=int(
                    (instance_target_topk & conflict_mask).sum().item()),
                instance_topk_competitor_pixels=int(
                    (instance_competitor_topk & conflict_mask).sum().item()),
                aux_both_favor_target_pixels=int(
                    ((semantic_margin > 0) & (instance_margin > 0)
                     & conflict_mask).sum().item()),
                aux_any_favor_target_pixels=int(
                    (((semantic_margin > 0) | (instance_margin > 0))
                     & conflict_mask).sum().item()),
                mean_semantic_margin=_masked_mean(
                    semantic_margin, conflict_mask),
                mean_instance_margin=_masked_mean(
                    instance_margin, conflict_mask),
                mean_top1_local_consistency=_masked_mean(
                    local_consistency, conflict_mask),
                base_pred_target_pixels=int(
                    ((base_pred == target_class) & conflict_mask).sum().item()),
                base_pred_competitor_pixels=int(
                    ((base_pred == competitor_class) & conflict_mask).sum().item()),
                base_pred_bg_pixels=int(
                    ((base_pred == int(self.bg_idx)) & conflict_mask).sum().item()),
            )
            row.update(target_rank_pixels)
            if gt_data is not None:
                gt_target = (gt_data == target_class) & conflict_mask
                gt_competitor = (gt_data == competitor_class) & conflict_mask
                gt_other = conflict_mask & (~gt_target) & (~gt_competitor)
                row.update(dict(
                    gt_target_pixels=int(gt_target.sum().item()),
                    gt_competitor_pixels=int(gt_competitor.sum().item()),
                    gt_other_pixels=int(gt_other.sum().item()),
                    gt_target_ratio=_safe_div(int(gt_target.sum().item()), pixels),
                    gt_competitor_ratio=_safe_div(
                        int(gt_competitor.sum().item()), pixels),
                    gt_other_ratio=_safe_div(int(gt_other.sum().item()), pixels),
                    gt_pair_precision=_safe_div(
                        int(gt_target.sum().item()),
                        int(gt_target.sum().item()) + int(gt_competitor.sum().item())),
                    gt_counterfactual_net_pixels=(
                        int(gt_target.sum().item())
                        - int(gt_competitor.sum().item())),
                    gt_base_wrong_target_pixels=int(
                        (gt_target & (base_pred != target_class)).sum().item()),
                    gt_base_pred_competitor_pixels=int(
                        (gt_target & (base_pred == competitor_class)).sum().item()),
                ))
            pair_rows.append(row)

        class_indices = sorted(class_indices)
        if not pair_rows or not class_indices:
            return dict(
                dataset_name=self.seed_dataset_name,
                valid_pixels=valid_count,
                topk=topk,
                selected_pairs=selected_pairs,
                pair_stats=pair_rows,
                probe_stats=[],
            )

        heads = self._get_topk_conflict_heads()
        support_thresholds = self._parse_float_list(
            self.topk_conflict_support_thresholds,
            '0.05,0.10,0.20')
        base_direct = self._compute_selected_class_prompt_evidence(
            image, class_indices)
        base_direct = self._resize_evidence_maps(base_direct, target_size)
        if self.device.type == 'cuda' and bool(self.topk_conflict_empty_cache):
            torch.cuda.empty_cache()

        probe_rows = []
        for probe_name in self._get_topk_conflict_probes():
            probe_context = self._build_evidence_bias_probe(
                image, probe_name, class_indices, target_size)
            probe_evidence = probe_context['evidence']
            for target_class, competitor_class, _ in selected_pairs:
                conflict_mask = self._topk_conflict_pair_mask(
                    topk_idx, int(target_class), int(competitor_class), valid_mask)
                pixels = int(conflict_mask.sum().item())
                if pixels < int(self.topk_conflict_min_pair_pixels):
                    continue

                base_presence = base_direct.get('presence', {})
                probe_presence = probe_evidence.get('presence', {})
                target_presence_base = base_presence.get(int(target_class))
                competitor_presence_base = base_presence.get(int(competitor_class))
                target_presence_probe = probe_presence.get(int(target_class))
                competitor_presence_probe = probe_presence.get(int(competitor_class))
                target_presence_drop = (
                    None if target_presence_base is None
                    or target_presence_probe is None
                    else target_presence_base - target_presence_probe)
                competitor_presence_drop = (
                    None if competitor_presence_base is None
                    or competitor_presence_probe is None
                    else competitor_presence_base - competitor_presence_probe)
                presence_relative_stability = (
                    None if target_presence_drop is None
                    or competitor_presence_drop is None
                    else competitor_presence_drop - target_presence_drop)

                for head in heads:
                    base_head = base_direct.get(head)
                    probe_head = probe_evidence.get(head)
                    if base_head is None or probe_head is None:
                        continue
                    base_target = base_head[int(target_class)]
                    base_competitor = base_head[int(competitor_class)]
                    probe_target = probe_head[int(target_class)]
                    probe_competitor = probe_head[int(competitor_class)]
                    base_margin = base_target - base_competitor
                    probe_margin = probe_target - probe_competitor
                    margin_gain = probe_margin - base_margin
                    target_drop = base_target - probe_target
                    competitor_drop = base_competitor - probe_competitor
                    relative_stability = competitor_drop - target_drop
                    base_target_wins = base_margin > 0
                    probe_target_wins = probe_margin > 0

                    base_row = dict(
                        dataset_name=self.seed_dataset_name,
                        probe_name=probe_context['probe_name'],
                        head=head,
                        target_class_index=int(target_class),
                        target_class_name=self.class_names[int(target_class)],
                        competitor_class_index=int(competitor_class),
                        competitor_class_name=self.class_names[int(competitor_class)],
                        valid_pixels=valid_count,
                        conflict_pixels=pixels,
                        reverse_conflict_pixels=int(pair_counts.get(
                            (int(competitor_class), int(target_class)), 0)),
                        effective_scale_x=probe_context['effective_scale_x'],
                        effective_scale_y=probe_context['effective_scale_y'],
                        base_direct_margin_sum=_masked_sum(
                            base_margin, conflict_mask),
                        probe_margin_sum=_masked_sum(
                            probe_margin, conflict_mask),
                        margin_gain_sum=_masked_sum(
                            margin_gain, conflict_mask),
                        target_drop_sum=_masked_sum(
                            target_drop, conflict_mask),
                        competitor_drop_sum=_masked_sum(
                            competitor_drop, conflict_mask),
                        relative_stability_sum=_masked_sum(
                            relative_stability, conflict_mask),
                        base_target_score_sum=_masked_sum(
                            base_target, conflict_mask),
                        base_competitor_score_sum=_masked_sum(
                            base_competitor, conflict_mask),
                        probe_target_score_sum=_masked_sum(
                            probe_target, conflict_mask),
                        probe_competitor_score_sum=_masked_sum(
                            probe_competitor, conflict_mask),
                        base_target_wins_pixels=int(
                            (base_target_wins & conflict_mask).sum().item()),
                        probe_target_wins_pixels=int(
                            (probe_target_wins & conflict_mask).sum().item()),
                        target_presence_base=target_presence_base,
                        competitor_presence_base=competitor_presence_base,
                        target_presence_probe=target_presence_probe,
                        competitor_presence_probe=competitor_presence_probe,
                        target_presence_drop=target_presence_drop,
                        competitor_presence_drop=competitor_presence_drop,
                        presence_relative_stability=presence_relative_stability,
                    )
                    probe_rows.append(base_row)

                    for threshold in support_thresholds:
                        threshold = float(threshold)
                        base_target_support = base_target >= threshold
                        base_competitor_support = base_competitor >= threshold
                        probe_target_support = probe_target >= threshold
                        probe_competitor_support = probe_competitor >= threshold
                        support_row = dict(base_row)
                        support_row.update(dict(
                            support_threshold=threshold,
                            base_target_support_pixels=int(
                                (base_target_support & conflict_mask).sum().item()),
                            base_competitor_support_pixels=int(
                                (base_competitor_support & conflict_mask).sum().item()),
                            probe_target_support_pixels=int(
                                (probe_target_support & conflict_mask).sum().item()),
                            probe_competitor_support_pixels=int(
                                (probe_competitor_support & conflict_mask).sum().item()),
                            probe_target_only_support_pixels=int(
                                (probe_target_support & (~probe_competitor_support)
                                 & conflict_mask).sum().item()),
                            probe_competitor_only_support_pixels=int(
                                (probe_competitor_support & (~probe_target_support)
                                 & conflict_mask).sum().item()),
                            probe_both_support_pixels=int(
                                (probe_target_support & probe_competitor_support
                                 & conflict_mask).sum().item()),
                            probe_neither_support_pixels=int(
                                ((~probe_target_support) & (~probe_competitor_support)
                                 & conflict_mask).sum().item()),
                            target_gain_support_pixels=int(
                                (probe_target_support & (~base_target_support)
                                 & conflict_mask).sum().item()),
                            competitor_lost_support_pixels=int(
                                (base_competitor_support & (~probe_competitor_support)
                                 & conflict_mask).sum().item()),
                        ))
                        probe_rows.append(support_row)
            del probe_evidence
            if self.device.type == 'cuda' and bool(self.topk_conflict_empty_cache):
                torch.cuda.empty_cache()

        return dict(
            dataset_name=self.seed_dataset_name,
            valid_pixels=valid_count,
            topk=topk,
            selected_pairs=[
                [int(target), int(competitor), int(pixels)]
                for target, competitor, pixels in selected_pairs
            ],
            selected_pair_names=[
                [self.class_names[int(target)], self.class_names[int(competitor)]]
                for target, competitor, _ in selected_pairs
            ],
            selected_class_indices=class_indices,
            selected_class_names=[self.class_names[idx] for idx in class_indices],
            probes=self._get_topk_conflict_probes(),
            heads=heads,
            support_thresholds=support_thresholds,
            topk_conflict_min_pair_pixels=int(
                self.topk_conflict_min_pair_pixels),
            topk_conflict_max_pairs=int(self.topk_conflict_max_pairs),
            topk_conflict_max_side=int(self.topk_conflict_max_side),
            pair_stats=pair_rows,
            probe_stats=probe_rows,
        )

    def _write_topk_conflict_bias_stats(self, record):
        if not self.dump_topk_conflict_bias_stats:
            return
        if self._topk_conflict_bias_stats_file is None:
            path = (
                self.topk_conflict_bias_stats_path
                or './work_dirs/evidence_stats/topk_conflict_bias_stats.jsonl'
            )
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._topk_conflict_bias_stats_file = open(
                path, 'a', buffering=1)
        self._topk_conflict_bias_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    def _resize_internal_source_map(self, source_map, target_shape):
        if source_map is None or not isinstance(source_map, torch.Tensor):
            return None
        source_map = source_map.detach().float().to(self.device)
        if source_map.ndim != 3 or source_map.shape[0] != self.num_cls:
            return None
        if source_map.shape[-2:] != target_shape:
            source_map = self._interpolate_float32(
                source_map.unsqueeze(0),
                target_shape,
            ).squeeze(0)
        return source_map

    def _internal_source_seed_reliability(self, source_name, source_scores,
                                          source_pred, seed_mask, seed_class):
        rows = []
        reliability = torch.full(
            (self.num_cls,),
            -1.0,
            device=self.device,
            dtype=torch.float32,
        )
        finite_scores = torch.isfinite(source_scores)
        for class_idx in range(self.num_cls):
            class_seed = seed_mask & (seed_class == class_idx)
            seed_pixels = int(class_seed.sum().item())
            class_available = bool(finite_scores[class_idx].any().item())
            if seed_pixels == 0 or not class_available:
                rows.append(dict(
                    source_name=source_name,
                    class_index=class_idx,
                    class_name=self.class_names[class_idx],
                    seed_pixels=seed_pixels,
                    class_available=class_available,
                    seed_top1_agree_pixels=0,
                    seed_top1_agree_ratio=None,
                    seed_source_margin_sum=0.0,
                    mean_seed_source_margin=None,
                    no_gt_reliability=None,
                ))
                continue

            own_score = source_scores[class_idx]
            other_scores = source_scores.clone()
            other_scores[class_idx] = float('-inf')
            best_other = other_scores.max(dim=0)[0]
            valid_seed = (
                class_seed
                & torch.isfinite(own_score)
                & torch.isfinite(best_other)
            )
            valid_seed_pixels = int(valid_seed.sum().item())
            if valid_seed_pixels == 0:
                rows.append(dict(
                    source_name=source_name,
                    class_index=class_idx,
                    class_name=self.class_names[class_idx],
                    seed_pixels=seed_pixels,
                    class_available=class_available,
                    seed_top1_agree_pixels=0,
                    seed_top1_agree_ratio=None,
                    seed_source_margin_sum=0.0,
                    mean_seed_source_margin=None,
                    no_gt_reliability=None,
                ))
                continue

            agree_pixels = int(
                ((source_pred == class_idx) & valid_seed).sum().item())
            margin = own_score - best_other
            margin_sum = _masked_sum(margin, valid_seed)
            mean_margin = _safe_div(margin_sum, valid_seed_pixels)
            agree_ratio = _safe_div(agree_pixels, valid_seed_pixels)
            score = (
                _zero_if_none(agree_ratio)
                + float(np.tanh(_zero_if_none(mean_margin)))
            )
            reliability[class_idx] = float(score)
            rows.append(dict(
                source_name=source_name,
                class_index=class_idx,
                class_name=self.class_names[class_idx],
                seed_pixels=seed_pixels,
                valid_seed_pixels=valid_seed_pixels,
                class_available=class_available,
                seed_top1_agree_pixels=agree_pixels,
                seed_top1_agree_ratio=agree_ratio,
                seed_source_margin_sum=margin_sum,
                mean_seed_source_margin=mean_margin,
                no_gt_reliability=float(score),
            ))
        return reliability, rows

    def _internal_selector_row(self, rule_name, selected_pred, base_pred,
                               gt_data, valid_mask, selected_source=None):
        base_correct = (base_pred == gt_data) & valid_mask
        selected_correct = (selected_pred == gt_data) & valid_mask
        changed = (selected_pred != base_pred) & valid_mask
        improved = changed & (~base_correct) & selected_correct
        harmed = changed & base_correct & (~selected_correct)
        row = dict(
            rule_name=rule_name,
            valid_pixels=int(valid_mask.sum().item()),
            baseline_correct_pixels=int(base_correct.sum().item()),
            selected_correct_pixels=int(selected_correct.sum().item()),
            changed_pixels=int(changed.sum().item()),
            improved_pixels=int(improved.sum().item()),
            harmed_pixels=int(harmed.sum().item()),
            net_improved_pixels=(
                int(improved.sum().item()) - int(harmed.sum().item())),
        )
        if selected_source is not None:
            row['source_choice_counts'] = {
                source_name: int(
                    ((selected_source == source_idx) & valid_mask).sum().item())
                for source_idx, source_name in enumerate(
                    self._internal_selection_active_source_names)
            }
        return row

    def _build_internal_selection_gap_stats(self, base_logits, base_pred,
                                            data_sample, components):
        if components is None or self.num_cls <= 1:
            return None
        semantic_logits = components.get('semantic_logits')
        instance_logits = components.get('instance_logits')
        if semantic_logits is None or instance_logits is None:
            return None

        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)
        if gt_data is None:
            return None

        diag_shape = self._internal_selection_diag_shape(
            int(base_logits.shape[-2]), int(base_logits.shape[-1]))
        gt_diag = F.interpolate(
            gt_data.float().view(1, 1, *gt_data.shape[-2:]),
            size=diag_shape,
            mode='nearest',
        ).squeeze().long()
        pred_diag = F.interpolate(
            base_pred.float().view(1, 1, *base_pred.shape[-2:]),
            size=diag_shape,
            mode='nearest',
        ).squeeze().long()
        valid_mask = gt_diag != 255
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return dict(
                dataset_name=self.seed_dataset_name,
                diagnostic_shape=list(diag_shape),
                valid_pixels=0,
                source_stats=[],
                class_stats=[],
                pair_stats=[],
                selector_stats=[],
                source_reliability_stats=[],
                oracle_stats={},
                oracle_source_stats=[],
                regime_stats=[],
            )

        requested = self._get_internal_selection_sources()
        source_maps = {}
        base_map = self._resize_internal_source_map(base_logits, diag_shape)
        semantic_map = self._resize_internal_source_map(
            semantic_logits, diag_shape)
        instance_map = self._resize_internal_source_map(
            instance_logits, diag_shape)
        if 'final' in requested:
            source_maps['final'] = base_map
        if 'semantic' in requested:
            source_maps['semantic'] = semantic_map
        if 'instance' in requested:
            source_maps['instance'] = instance_map

        internal_maps = components.get('internal_source_maps') or {}
        for source_name, source_map in internal_maps.items():
            if not self._internal_selection_wants(source_name):
                continue
            resized = self._resize_internal_source_map(
                source_map, diag_shape)
            if resized is not None:
                source_maps[source_name] = resized
        if not source_maps:
            return None

        topk = max(2, min(int(self.internal_selection_topk), self.num_cls))
        source_data = []
        for source_name, raw_scores in source_maps.items():
            finite = torch.isfinite(raw_scores)
            valid_source = finite.sum(dim=0) >= 2
            if not valid_source.any():
                continue
            scores = torch.nan_to_num(
                raw_scores,
                nan=-1e6,
                posinf=1e6,
                neginf=-1e6,
            )
            topk_values, topk_indices = torch.topk(
                scores, k=topk, dim=0)
            source_data.append(dict(
                name=source_name,
                scores=scores,
                finite=finite,
                valid=valid_source,
                pred=topk_indices[0],
                topk_idx=topk_indices,
                top1_score=topk_values[0],
                margin=topk_values[0] - topk_values[1],
            ))
        if not source_data:
            return None

        self._internal_selection_active_source_names = [
            item['name'] for item in source_data
        ]
        gt_idx = gt_diag.clamp(min=0, max=self.num_cls - 1)
        pred_idx = pred_diag.clamp(min=0, max=self.num_cls - 1)
        base_correct = (pred_diag == gt_diag) & valid_mask
        base_wrong = (~base_correct) & valid_mask
        wrong_count = int(base_wrong.sum().item())

        seed_components = dict(
            semantic_logits=semantic_map,
            instance_logits=instance_map,
        )
        seed_context = self._build_seed_rule_context(
            base_map, pred_diag, seed_components)
        seed_mask = torch.zeros_like(valid_mask)
        seed_class = pred_idx
        if seed_context is not None:
            seed_class = seed_context['final_top1_idx']
            selected_seed = None
            for rule_name, rule_mask in seed_context['rule_defs']:
                if rule_name == str(self.internal_selection_seed_rule):
                    selected_seed = rule_mask
                    break
            if selected_seed is None and seed_context['rule_defs']:
                selected_seed = seed_context['rule_defs'][-1][1]
            if selected_seed is not None:
                seed_mask = selected_seed & valid_mask

        source_stats = []
        class_stats = []
        pair_stats = []
        source_reliability_stats = []
        reliability_rows = []
        for source_idx, item in enumerate(source_data):
            source_name = item['name']
            source_valid = item['valid'] & valid_mask
            source_pred = item['pred']
            source_correct = (source_pred == gt_diag) & source_valid
            changed = (source_pred != pred_diag) & source_valid
            improved = changed & base_wrong & source_correct
            harmed = changed & base_correct & (~source_correct)
            gt_topk = (
                item['topk_idx'] == gt_idx.unsqueeze(0)).any(dim=0)
            gt_available = _gather_class_map(
                item['finite'].float(), gt_idx).bool()
            gt_topk = gt_topk & source_valid & gt_available
            wrong_recovered = source_correct & base_wrong
            wrong_topk = gt_topk & base_wrong
            source_stats.append(dict(
                source_name=source_name,
                valid_pixels=int(source_valid.sum().item()),
                available_class_count=int(
                    item['finite'].flatten(1).any(dim=1).sum().item()),
                top1_correct_pixels=int(source_correct.sum().item()),
                top1_accuracy=_safe_div(
                    int(source_correct.sum().item()),
                    int(source_valid.sum().item())),
                baseline_wrong_pixels=wrong_count,
                wrong_top1_recovered_pixels=int(
                    wrong_recovered.sum().item()),
                wrong_top1_recovered_ratio=_safe_div(
                    int(wrong_recovered.sum().item()), wrong_count),
                wrong_gt_topk_pixels=int(wrong_topk.sum().item()),
                wrong_gt_topk_ratio=_safe_div(
                    int(wrong_topk.sum().item()), wrong_count),
                changed_pixels=int(changed.sum().item()),
                improved_pixels=int(improved.sum().item()),
                harmed_pixels=int(harmed.sum().item()),
                net_improved_pixels=(
                    int(improved.sum().item()) - int(harmed.sum().item())),
                top1_margin_sum=_masked_sum(
                    item['margin'], source_valid),
            ))

            reliability, rel_rows = self._internal_source_seed_reliability(
                source_name,
                item['scores'],
                source_pred,
                seed_mask,
                seed_class,
            )
            reliability_rows.append(reliability)
            source_reliability_stats.extend(rel_rows)

            for class_idx in range(self.num_cls):
                gt_class = (gt_diag == class_idx) & valid_mask
                gt_class_wrong = gt_class & base_wrong
                class_source_correct = gt_class & source_correct
                class_wrong_recovered = gt_class_wrong & source_correct
                class_wrong_topk = gt_class_wrong & gt_topk
                class_stats.append(dict(
                    source_name=source_name,
                    class_index=class_idx,
                    class_name=self.class_names[class_idx],
                    gt_pixels=int(gt_class.sum().item()),
                    baseline_wrong_pixels=int(gt_class_wrong.sum().item()),
                    source_top1_correct_pixels=int(
                        class_source_correct.sum().item()),
                    source_class_recall=_safe_div(
                        int(class_source_correct.sum().item()),
                        int(gt_class.sum().item())),
                    wrong_top1_recovered_pixels=int(
                        class_wrong_recovered.sum().item()),
                    wrong_top1_recovered_ratio=_safe_div(
                        int(class_wrong_recovered.sum().item()),
                        int(gt_class_wrong.sum().item())),
                    wrong_gt_topk_pixels=int(
                        class_wrong_topk.sum().item()),
                    wrong_gt_topk_ratio=_safe_div(
                        int(class_wrong_topk.sum().item()),
                        int(gt_class_wrong.sum().item())),
                ))

            for gt_class in range(self.num_cls):
                gt_wrong = base_wrong & (gt_diag == gt_class)
                for base_class in range(self.num_cls):
                    if gt_class == base_class:
                        continue
                    pair_mask = gt_wrong & (pred_diag == base_class)
                    pair_pixels = int(pair_mask.sum().item())
                    if pair_pixels < int(self.internal_selection_min_pair_pixels):
                        continue
                    pair_recovered = int(
                        (pair_mask & source_correct).sum().item())
                    pair_topk = int((pair_mask & gt_topk).sum().item())
                    gt_score = item['scores'][gt_class]
                    pred_score = item['scores'][base_class]
                    pair_margin_mask = (
                        pair_mask
                        & item['finite'][gt_class]
                        & item['finite'][base_class]
                    )
                    pair_stats.append(dict(
                        source_name=source_name,
                        gt_class_index=gt_class,
                        gt_class_name=self.class_names[gt_class],
                        base_pred_class_index=base_class,
                        base_pred_class_name=self.class_names[base_class],
                        pixels=pair_pixels,
                        source_top1_recovers_gt_pixels=pair_recovered,
                        source_top1_recovers_gt_ratio=_safe_div(
                            pair_recovered, pair_pixels),
                        source_topk_contains_gt_pixels=pair_topk,
                        source_topk_contains_gt_ratio=_safe_div(
                            pair_topk, pair_pixels),
                        gt_vs_base_pred_margin_sum=_masked_sum(
                            gt_score - pred_score, pair_margin_mask),
                        gt_vs_base_pred_margin_pixels=int(
                            pair_margin_mask.sum().item()),
                    ))

        pred_stack = torch.stack(
            [item['pred'] for item in source_data], dim=0)
        valid_stack = torch.stack(
            [item['valid'] for item in source_data], dim=0)
        margin_stack = torch.stack(
            [item['margin'] for item in source_data], dim=0)
        score_stack = torch.stack(
            [item['scores'] for item in source_data], dim=0)
        reliability_matrix = torch.stack(reliability_rows, dim=0)

        correct_stack = (
            (pred_stack == gt_idx.unsqueeze(0))
            & valid_stack
            & valid_mask.unsqueeze(0)
        )
        expanded_any_correct = correct_stack.any(dim=0)
        head_indices = [
            idx for idx, item in enumerate(source_data)
            if item['name'] in ('final', 'semantic', 'instance')
        ]
        if head_indices:
            head_any_correct = correct_stack[head_indices].any(dim=0)
        else:
            head_any_correct = torch.zeros_like(valid_mask)
        incremental_correct = (
            base_wrong & expanded_any_correct & (~head_any_correct))

        gt_scores = torch.gather(
            score_stack,
            1,
            gt_idx.view(1, 1, *diag_shape).expand(
                len(source_data), 1, *diag_shape),
        ).squeeze(1)
        base_pred_scores = torch.gather(
            score_stack,
            1,
            pred_idx.view(1, 1, *diag_shape).expand(
                len(source_data), 1, *diag_shape),
        ).squeeze(1)
        gt_class_available = torch.gather(
            torch.stack([item['finite'] for item in source_data], dim=0),
            1,
            gt_idx.view(1, 1, *diag_shape).expand(
                len(source_data), 1, *diag_shape),
        ).squeeze(1)
        pred_class_available = torch.gather(
            torch.stack([item['finite'] for item in source_data], dim=0),
            1,
            pred_idx.view(1, 1, *diag_shape).expand(
                len(source_data), 1, *diag_shape),
        ).squeeze(1)
        oracle_margin = gt_scores - base_pred_scores
        oracle_margin[~(gt_class_available & pred_class_available)] = float('-inf')
        best_oracle_margin, best_oracle_source = oracle_margin.max(dim=0)
        oracle_positive = base_wrong & (best_oracle_margin > 0)
        oracle_source_stats = []
        for source_idx, item in enumerate(source_data):
            selected = (
                oracle_positive
                & (best_oracle_source == source_idx)
            )
            oracle_source_stats.append(dict(
                source_name=item['name'],
                oracle_best_positive_pixels=int(selected.sum().item()),
                oracle_best_positive_ratio=_safe_div(
                    int(selected.sum().item()), wrong_count),
            ))

        consensus_stack = torch.zeros_like(
            margin_stack, dtype=torch.float32)
        for source_idx in range(len(source_data)):
            consensus_stack[source_idx] = (
                (pred_stack == pred_stack[source_idx].unsqueeze(0))
                & valid_stack
            ).sum(dim=0).float()

        normalized_margin = torch.full_like(margin_stack, -10.0)
        for source_idx in range(len(source_data)):
            mask = valid_stack[source_idx] & valid_mask
            values = margin_stack[source_idx][mask]
            if values.numel() == 0:
                continue
            mean = values.mean()
            std = values.std(unbiased=False).clamp_min(1e-6)
            normalized_margin[source_idx] = (
                margin_stack[source_idx] - mean) / std

        predicted_reliability = torch.gather(
            reliability_matrix[:, :, None, None].expand(
                -1, -1, *diag_shape),
            1,
            pred_stack.unsqueeze(1),
        ).squeeze(1)
        predicted_reliability[~valid_stack] = -10.0
        combined_score = (
            normalized_margin
            + consensus_stack / max(1, len(source_data))
            + predicted_reliability
        )
        combined_score[~valid_stack] = -10.0

        selector_defs = []
        max_margin_source = normalized_margin.argmax(dim=0)
        selector_defs.append(('max_margin_z', max_margin_source))
        max_consensus_source = (
            consensus_stack
            + 1e-3 * normalized_margin).argmax(dim=0)
        selector_defs.append(('max_consensus', max_consensus_source))
        max_reliability_source = predicted_reliability.argmax(dim=0)
        selector_defs.append(('max_seed_reliability', max_reliability_source))
        combined_source = combined_score.argmax(dim=0)
        selector_defs.append(('combined', combined_source))

        selector_stats = []
        selected_predictions = {}
        for rule_name, source_choice in selector_defs:
            selected_pred = torch.gather(
                pred_stack, 0, source_choice.unsqueeze(0)).squeeze(0)
            selected_predictions[rule_name] = selected_pred
            selector_stats.append(self._internal_selector_row(
                rule_name,
                selected_pred,
                pred_diag,
                gt_diag,
                valid_mask,
                source_choice,
            ))

        final_source_idx = next(
            (idx for idx, item in enumerate(source_data)
             if item['name'] == 'final'),
            None,
        )
        conservative_pred = pred_diag.clone()
        conservative_source = torch.full_like(
            pred_diag, -1, dtype=torch.long)
        if final_source_idx is not None:
            final_topk_idx = source_data[final_source_idx]['topk_idx']
            candidate_pred = selected_predictions['combined']
            candidate_consensus = torch.gather(
                consensus_stack,
                0,
                combined_source.unsqueeze(0),
            ).squeeze(0)
            candidate_reliability = torch.gather(
                predicted_reliability,
                0,
                combined_source.unsqueeze(0),
            ).squeeze(0)
            candidate_in_final_topk = (
                final_topk_idx == candidate_pred.unsqueeze(0)).any(dim=0)
            conservative_gate = (
                valid_mask
                & (candidate_pred != pred_diag)
                & candidate_in_final_topk
                & (candidate_consensus >= float(
                    self.internal_selection_conservative_min_consensus))
                & (candidate_reliability >= float(
                    self.internal_selection_conservative_min_reliability))
            )
            conservative_pred[conservative_gate] = candidate_pred[
                conservative_gate]
            conservative_source[conservative_gate] = combined_source[
                conservative_gate]
        selector_stats.append(self._internal_selector_row(
            'conservative_combined',
            conservative_pred,
            pred_diag,
            gt_diag,
            valid_mask,
            conservative_source,
        ))

        final_margin = source_data[
            final_source_idx]['margin'] if final_source_idx is not None else base_map.topk(
                k=2, dim=0).values.diff(dim=0).abs().squeeze(0)
        regime_stats = []
        margin_bins = [
            ('margin_0_0p05', 0.0, 0.05),
            ('margin_0p05_0p10', 0.05, 0.10),
            ('margin_0p10_0p20', 0.10, 0.20),
            ('margin_0p20_0p50', 0.20, 0.50),
            ('margin_ge_0p50', 0.50, None),
        ]
        conservative_correct = conservative_pred == gt_diag
        for bucket_name, lower, upper in margin_bins:
            bucket = base_wrong & (final_margin >= lower)
            if upper is not None:
                bucket = bucket & (final_margin < upper)
            pixels = int(bucket.sum().item())
            regime_stats.append(dict(
                regime_name=bucket_name,
                pixels=pixels,
                head_any_top1_correct_pixels=int(
                    (bucket & head_any_correct).sum().item()),
                expanded_any_top1_correct_pixels=int(
                    (bucket & expanded_any_correct).sum().item()),
                expanded_incremental_pixels=int(
                    (bucket & expanded_any_correct
                     & (~head_any_correct)).sum().item()),
                oracle_positive_margin_pixels=int(
                    (bucket & oracle_positive).sum().item()),
                conservative_recovers_gt_pixels=int(
                    (bucket & conservative_correct).sum().item()),
            ))

        return dict(
            dataset_name=self.seed_dataset_name,
            diagnostic_shape=list(diag_shape),
            valid_pixels=valid_count,
            baseline_correct_pixels=int(base_correct.sum().item()),
            baseline_wrong_pixels=wrong_count,
            seed_rule=str(self.internal_selection_seed_rule),
            seed_pixels=int(seed_mask.sum().item()),
            topk=topk,
            requested_sources=requested,
            active_sources=self._internal_selection_active_source_names,
            source_stats=source_stats,
            class_stats=class_stats,
            pair_stats=pair_stats,
            selector_stats=selector_stats,
            source_reliability_stats=source_reliability_stats,
            oracle_stats=dict(
                baseline_wrong_pixels=wrong_count,
                head_any_top1_correct_pixels=int(
                    (base_wrong & head_any_correct).sum().item()),
                expanded_any_top1_correct_pixels=int(
                    (base_wrong & expanded_any_correct).sum().item()),
                expanded_incremental_over_heads_pixels=int(
                    incremental_correct.sum().item()),
                oracle_positive_margin_pixels=int(
                    oracle_positive.sum().item()),
            ),
            oracle_source_stats=oracle_source_stats,
            regime_stats=regime_stats,
        )

    def _write_internal_selection_gap_stats(self, record):
        if not self.dump_internal_selection_gap_stats:
            return
        if self._internal_selection_gap_stats_file is None:
            path = (
                self.internal_selection_gap_stats_path
                or './work_dirs/evidence_stats/internal_selection_gap_stats.jsonl'
            )
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._internal_selection_gap_stats_file = open(
                path, 'a', buffering=1)
        self._internal_selection_gap_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    @staticmethod
    def _candidate_internal_normalize_map(score_map):
        finite = torch.isfinite(score_map)
        count = finite.sum(dim=0).clamp_min(1)
        clean = torch.nan_to_num(score_map, nan=0.0, posinf=0.0, neginf=0.0)
        mean = clean.sum(dim=0) / count
        centered = torch.where(
            finite,
            score_map - mean.unsqueeze(0),
            torch.zeros_like(score_map),
        )
        variance = centered.square().sum(dim=0) / count
        std = variance.sqrt().clamp_min(1e-6)
        normalized = centered / std.unsqueeze(0)
        normalized[~finite] = float('nan')
        return normalized

    def _build_candidate_internal_families(self, source_maps):
        family_sources = {
            'semantic': ['semantic'],
            'instance': ['instance'],
            'no_presence': ['fusion_no_presence'],
            'presence_readout': [
                'semantic_presence',
                'instance_presence',
            ],
            'raw_mask': ['raw_object', 'raw_presence'],
            'encoder': sorted([
                name for name in source_maps
                if name.startswith('encoder_level_')
            ]),
        }
        pe_sources = sorted([
            name for name in source_maps
            if name.startswith('pe_layer_')
        ])
        # The current vision feature is also exposed as the deepest FPN map.
        # Prefer explicit PE layers so the same tensor cannot receive two votes.
        family_sources['visual'] = (
            pe_sources if pe_sources else ['vision'])
        for source_name in sorted(source_maps):
            if (
                    source_name.startswith('posbias_')
                    or source_name.startswith('scenecommon_')):
                family_sources[source_name] = [source_name]

        families = {}
        active_sources = {}
        for family_name, source_names in family_sources.items():
            normalized_maps = []
            used_names = []
            for source_name in source_names:
                source_map = source_maps.get(source_name)
                if not isinstance(source_map, torch.Tensor):
                    continue
                normalized_maps.append(
                    self._candidate_internal_normalize_map(source_map))
                used_names.append(source_name)
            if not normalized_maps:
                continue
            stack = torch.stack(normalized_maps, dim=0)
            finite = torch.isfinite(stack)
            count = finite.sum(dim=0)
            family_map = torch.nan_to_num(stack, nan=0.0).sum(dim=0)
            family_map = family_map / count.clamp_min(1)
            family_map[count == 0] = float('nan')
            families[family_name] = family_map
            active_sources[family_name] = used_names
        return families, active_sources

    def _candidate_internal_selector_row(
            self, rule_name, selected_candidate, gate, base_pred, gt_data,
            valid_mask, base_correct, base_wrong):
        changed = gate & valid_mask & (selected_candidate != base_pred)
        selected_correct = selected_candidate == gt_data
        improved = changed & base_wrong & selected_correct
        harmed = changed & base_correct & (~selected_correct)
        neutral = changed & base_wrong & (~selected_correct)
        changed_count = int(changed.sum().item())
        return dict(
            rule_name=rule_name,
            valid_pixels=int(valid_mask.sum().item()),
            baseline_correct_pixels=int(base_correct.sum().item()),
            changed_pixels=changed_count,
            improved_pixels=int(improved.sum().item()),
            harmed_pixels=int(harmed.sum().item()),
            neutral_wrong_to_wrong_pixels=int(neutral.sum().item()),
            net_improved_pixels=(
                int(improved.sum().item()) - int(harmed.sum().item())),
            changed_precision=_safe_div(
                int(improved.sum().item()), changed_count),
            baseline_error_coverage=_safe_div(
                int(improved.sum().item()), int(base_wrong.sum().item())),
        )

    @staticmethod
    def _candidate_internal_metric_from_hist(
            score, positive_mask, negative_mask, bin_count, score_min,
            score_max):
        finite = torch.isfinite(score)
        positive = positive_mask & finite
        negative = negative_mask & finite
        positive_count = int(positive.sum().item())
        negative_count = int(negative.sum().item())
        result = dict(
            positive_pixels=positive_count,
            negative_pixels=negative_count,
            prior=_safe_div(
                positive_count, positive_count + negative_count),
            auroc=None,
            auprc=None,
            auprc_lift=None,
            comparison_weight=positive_count * negative_count,
            positive_clip_low_pixels=0,
            positive_clip_high_pixels=0,
            negative_clip_low_pixels=0,
            negative_clip_high_pixels=0,
        )
        if positive_count == 0 or negative_count == 0:
            return result

        bins = max(8, int(bin_count))
        lower = float(score_min)
        upper = float(score_max)
        if not upper > lower:
            raise ValueError(
                'candidate_internal_matched_auc_max must be larger than '
                'candidate_internal_matched_auc_min.')

        def make_hist(mask):
            values = score[mask].detach().float()
            low_count = int((values < lower).sum().item())
            high_count = int((values > upper).sum().item())
            scaled = (values.clamp(lower, upper) - lower)
            scaled = scaled / (upper - lower)
            indices = torch.floor(scaled * bins).long().clamp(0, bins - 1)
            hist = torch.bincount(indices, minlength=bins).double()
            return hist, low_count, high_count

        positive_hist, positive_low, positive_high = make_hist(positive)
        negative_hist, negative_low, negative_high = make_hist(negative)
        negative_below = torch.cumsum(negative_hist, dim=0) - negative_hist
        favorable_pairs = (
            positive_hist
            * (negative_below + 0.5 * negative_hist)
        ).sum().item()
        auroc = favorable_pairs / float(positive_count * negative_count)

        positive_desc = torch.flip(positive_hist, dims=(0,))
        negative_desc = torch.flip(negative_hist, dims=(0,))
        true_positive = torch.cumsum(positive_desc, dim=0)
        false_positive = torch.cumsum(negative_desc, dim=0)
        precision = true_positive / (
            true_positive + false_positive).clamp_min(1.0)
        auprc = (
            precision * positive_desc / float(positive_count)
        ).sum().item()
        prior = result['prior']
        result.update(
            auroc=float(auroc),
            auprc=float(auprc),
            auprc_lift=(
                None if prior is None else float(auprc - prior)),
            positive_clip_low_pixels=positive_low,
            positive_clip_high_pixels=positive_high,
            negative_clip_low_pixels=negative_low,
            negative_clip_high_pixels=negative_high,
        )
        return result

    @staticmethod
    def _candidate_internal_margin_name(lower, upper):
        def token(value):
            if math.isinf(value):
                return 'inf'
            text = f'{value:g}'.replace('-', 'm').replace('.', 'p')
            return text
        return f'margin_{token(lower)}_{token(upper)}'

    def _build_candidate_internal_matched_stats(
            self, candidate_idx, candidate_valid, final_margin, pred_idx,
            gt_idx, valid_mask, base_correct, base_wrong, family_maps,
            family_delta_stack, family_names, regime_masks, diag_shape):
        if not self.candidate_internal_dump_matched_stats:
            return []

        requested_ranks = {
            int(round(value))
            for value in self._parse_float_list(
                self.candidate_internal_matched_ranks, '2,3')
        }
        rank_indices = [
            rank - 1 for rank in sorted(requested_ranks)
            if 1 <= rank <= candidate_idx.shape[0]
        ]
        requested_regimes = set(self._parse_name_list(
            self.candidate_internal_matched_regimes))
        active_regimes = [
            (name, mask) for name, mask in regime_masks.items()
            if name in requested_regimes
        ]
        requested_families = set(self._parse_name_list(
            self.candidate_internal_matched_families))
        margin_edges = self._parse_float_list(
            self.candidate_internal_matched_margin_bins,
            '0.00,0.05,0.10,0.20,0.50,inf',
        )
        if len(margin_edges) < 2:
            raise ValueError(
                'candidate_internal_matched_margin_bins needs at least '
                'two edges.')
        margin_ranges = [
            (margin_edges[index], margin_edges[index + 1])
            for index in range(len(margin_edges) - 1)
        ]
        score_thresholds = self._parse_float_list(
            self.candidate_internal_matched_score_thresholds,
            '-1.00,-0.50,-0.25,0.00,0.10,0.25,0.50,1.00,2.00',
        )
        min_pair_pixels = max(
            1, int(self.candidate_internal_min_pair_pixels))
        auc_bins = max(
            8, int(self.candidate_internal_matched_auc_bins))
        auc_min = float(self.candidate_internal_matched_auc_min)
        auc_max = float(self.candidate_internal_matched_auc_max)
        matched_rows = []

        def append_control_rows(
                family_name, delta_by_rank, control_name, trial):
            for rank_idx in rank_indices:
                candidate_class_map = candidate_idx[rank_idx]
                base_pair_code = (
                    pred_idx * self.num_cls + candidate_class_map)
                score = delta_by_rank[rank_idx]
                for regime_name, regime_mask in active_regimes:
                    for margin_lower, margin_upper in margin_ranges:
                        margin_mask = (
                            (final_margin >= float(margin_lower))
                            & (final_margin < float(margin_upper))
                        )
                        stratum_mask = (
                            candidate_valid[rank_idx]
                            & regime_mask
                            & margin_mask
                            & valid_mask
                        )
                        if int(stratum_mask.sum().item()) < min_pair_pixels:
                            continue
                        pair_codes = torch.unique(
                            base_pair_code[stratum_mask])
                        for pair_code_tensor in pair_codes:
                            pair_code = int(pair_code_tensor.item())
                            base_class = pair_code // self.num_cls
                            candidate_class = pair_code % self.num_cls
                            if base_class == candidate_class:
                                continue
                            pair_mask = (
                                stratum_mask
                                & (base_pair_code == pair_code)
                            )
                            pair_pixels = int(pair_mask.sum().item())
                            if pair_pixels < min_pair_pixels:
                                continue

                            candidate_help = (
                                pair_mask
                                & base_wrong
                                & (gt_idx == candidate_class)
                            )
                            switch_harm = pair_mask & base_correct
                            neutral = (
                                pair_mask
                                & (~candidate_help)
                                & (~switch_harm)
                            )
                            task_masks = {
                                'candidate_correct': (
                                    candidate_help,
                                    pair_mask & (~candidate_help),
                                ),
                                'baseline_risk': (
                                    pair_mask & base_wrong,
                                    switch_harm,
                                ),
                                'switch_utility': (
                                    candidate_help,
                                    switch_harm,
                                ),
                            }
                            task_metrics = {
                                task_name:
                                self._candidate_internal_metric_from_hist(
                                    score,
                                    positive_mask,
                                    negative_mask,
                                    auc_bins,
                                    auc_min,
                                    auc_max,
                                )
                                for task_name, (
                                    positive_mask,
                                    negative_mask,
                                ) in task_masks.items()
                            }

                            row = dict(
                                family_name=family_name,
                                candidate_rank=rank_idx + 1,
                                regime_name=regime_name,
                                margin_name=(
                                    self._candidate_internal_margin_name(
                                        margin_lower, margin_upper)),
                                margin_lower=float(margin_lower),
                                margin_upper=(
                                    'inf' if math.isinf(margin_upper)
                                    else float(margin_upper)),
                                base_class_index=base_class,
                                base_class_name=self.class_names[base_class],
                                candidate_class_index=candidate_class,
                                candidate_class_name=(
                                    self.class_names[candidate_class]),
                                control_name=control_name,
                                trial=trial,
                                pair_pixels=pair_pixels,
                                help_pixels=int(candidate_help.sum().item()),
                                harm_pixels=int(switch_harm.sum().item()),
                                neutral_pixels=int(neutral.sum().item()),
                                task_metrics=task_metrics,
                            )
                            if control_name == 'observed':
                                threshold_stats = []
                                finite_pair = pair_mask & torch.isfinite(score)
                                for threshold in score_thresholds:
                                    selected = (
                                        finite_pair
                                        & (score >= float(threshold))
                                    )
                                    threshold_stats.append(dict(
                                        threshold=float(threshold),
                                        selected_pixels=int(
                                            selected.sum().item()),
                                        help_pixels=int(
                                            (selected
                                             & candidate_help).sum().item()),
                                        harm_pixels=int(
                                            (selected
                                             & switch_harm).sum().item()),
                                        neutral_pixels=int(
                                            (selected
                                             & neutral).sum().item()),
                                    ))
                                row['threshold_stats'] = threshold_stats
                            matched_rows.append(row)

        null_trials = max(
            0, int(self.candidate_internal_matched_null_trials))
        for family_idx, family_name in enumerate(family_names):
            if (
                    requested_families
                    and 'all' not in requested_families
                    and family_name not in requested_families):
                continue
            append_control_rows(
                family_name,
                family_delta_stack[family_idx],
                'observed',
                0,
            )
            family_map = family_maps[family_name]
            for trial in range(null_trials):
                class_shift = 1 + (
                    trial % max(1, self.num_cls - 1))
                shift_y = max(
                    1,
                    int(round(
                        diag_shape[0] * (trial + 1)
                        / float(null_trials + 1))),
                )
                shift_x = max(
                    1,
                    int(round(
                        diag_shape[1] * (null_trials - trial)
                        / float(null_trials + 1))),
                )
                for control_name, control_map in (
                        (
                            'class_permutation',
                            torch.roll(
                                family_map,
                                shifts=class_shift,
                                dims=0,
                            ),
                        ),
                        (
                            'spatial_shift',
                            torch.roll(
                                family_map,
                                shifts=(shift_y, shift_x),
                                dims=(-2, -1),
                            ),
                        )):
                    control_pred_score = _gather_class_map(
                        control_map, pred_idx)
                    control_candidate_score = torch.gather(
                        control_map, 0, candidate_idx)
                    control_delta = (
                        control_candidate_score
                        - control_pred_score.unsqueeze(0)
                    )
                    control_delta[~candidate_valid] = float('nan')
                    append_control_rows(
                        family_name,
                        control_delta,
                        control_name,
                        trial + 1,
                    )

        return matched_rows

    def _build_candidate_internal_verifier_stats(
            self, base_logits, base_pred, data_sample, components):
        if components is None or self.num_cls <= 1:
            return None
        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)
        semantic_logits = components.get('semantic_logits')
        instance_logits = components.get('instance_logits')
        if (gt_data is None or semantic_logits is None
                or instance_logits is None):
            return None

        diag_shape = self._internal_selection_diag_shape(
            int(base_logits.shape[-2]), int(base_logits.shape[-1]))
        gt_diag = F.interpolate(
            gt_data.float().view(1, 1, *gt_data.shape[-2:]),
            size=diag_shape,
            mode='nearest',
        ).squeeze().long()
        pred_diag = F.interpolate(
            base_pred.float().view(1, 1, *base_pred.shape[-2:]),
            size=diag_shape,
            mode='nearest',
        ).squeeze().long()
        valid_mask = gt_diag != 255
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return None

        source_maps = {
            'final': self._resize_internal_source_map(
                base_logits, diag_shape),
            'semantic': self._resize_internal_source_map(
                semantic_logits, diag_shape),
            'instance': self._resize_internal_source_map(
                instance_logits, diag_shape),
        }
        for source_name, source_map in (
                components.get('internal_source_maps') or {}).items():
            resized = self._resize_internal_source_map(
                source_map, diag_shape)
            if resized is not None:
                source_maps[source_name] = resized

        final_map = source_maps['final']
        family_maps, active_family_sources = (
            self._build_candidate_internal_families(source_maps))
        if not family_maps:
            return None

        topk = max(
            2,
            min(int(self.candidate_internal_topk), self.num_cls),
        )
        final_topk = torch.topk(
            torch.nan_to_num(final_map, nan=-1e6),
            k=topk,
            dim=0,
        )
        candidate_idx = final_topk.indices
        final_top1 = candidate_idx[0]
        local_kernel = max(1, int(self.candidate_internal_local_kernel))
        if local_kernel % 2 == 0:
            local_kernel += 1
        top1_one_hot = F.one_hot(
            final_top1,
            num_classes=self.num_cls,
        ).permute(2, 0, 1).float().unsqueeze(0)
        family_maps['support_topology'] = F.avg_pool2d(
            top1_one_hot,
            kernel_size=local_kernel,
            stride=1,
            padding=local_kernel // 2,
        ).squeeze(0)
        active_family_sources['support_topology'] = [
            f'final_top1_local_{local_kernel}x{local_kernel}']
        pred_idx = pred_diag.clamp(min=0, max=self.num_cls - 1)
        gt_idx = gt_diag.clamp(min=0, max=self.num_cls - 1)
        candidate_valid = (
            (candidate_idx != pred_idx.unsqueeze(0))
            & valid_mask.unsqueeze(0)
        )
        base_correct = (pred_diag == gt_diag) & valid_mask
        base_wrong = (~base_correct) & valid_mask
        threshold_reject = (
            pred_idx != final_top1) & valid_mask
        argmax_competition = (~threshold_reject) & valid_mask

        family_names = list(family_maps)
        family_delta_stack = []
        for family_name in family_names:
            family_map = family_maps[family_name]
            pred_score = _gather_class_map(family_map, pred_idx)
            candidate_score = torch.gather(
                family_map,
                0,
                candidate_idx,
            )
            delta = candidate_score - pred_score.unsqueeze(0)
            delta[~candidate_valid] = float('nan')
            family_delta_stack.append(delta)
        family_delta_stack = torch.stack(
            family_delta_stack, dim=0)
        finite_delta = torch.isfinite(family_delta_stack)

        regime_masks = {
            'all': valid_mask,
            'argmax_competition': argmax_competition,
            'threshold_reject': threshold_reject,
        }
        candidate_stats = []
        feature_stats = []
        for rank_idx in range(topk):
            rank_valid = candidate_valid[rank_idx]
            candidate_class = candidate_idx[rank_idx]
            candidate_correct = (
                rank_valid & base_wrong & (candidate_class == gt_idx))
            candidate_wrong_other = (
                rank_valid & base_wrong & (candidate_class != gt_idx))
            candidate_on_correct = rank_valid & base_correct
            for regime_name, regime_mask in regime_masks.items():
                mask = rank_valid & regime_mask
                candidate_stats.append(dict(
                    candidate_rank=rank_idx + 1,
                    regime_name=regime_name,
                    candidate_pixels=int(mask.sum().item()),
                    candidate_correct_pixels=int(
                        (candidate_correct & regime_mask).sum().item()),
                    candidate_wrong_other_pixels=int(
                        (candidate_wrong_other & regime_mask).sum().item()),
                    candidate_on_baseline_correct_pixels=int(
                        (candidate_on_correct & regime_mask).sum().item()),
                ))
                for family_idx, family_name in enumerate(family_names):
                    delta = family_delta_stack[family_idx, rank_idx]
                    finite_mask = mask & torch.isfinite(delta)
                    positive = finite_mask & candidate_correct
                    negative = finite_mask & (~candidate_correct)
                    positive_count = int(positive.sum().item())
                    negative_count = int(negative.sum().item())
                    feature_stats.append(dict(
                        family_name=family_name,
                        candidate_rank=rank_idx + 1,
                        regime_name=regime_name,
                        candidate_pixels=int(finite_mask.sum().item()),
                        positive_pixels=positive_count,
                        negative_pixels=negative_count,
                        positive_delta_sum=_masked_sum(delta, positive),
                        positive_delta_sq_sum=_masked_sum(
                            delta.square(), positive),
                        negative_delta_sum=_masked_sum(delta, negative),
                        negative_delta_sq_sum=_masked_sum(
                            delta.square(), negative),
                        positive_favored_pixels=int(
                            (positive & (delta > 0)).sum().item()),
                        negative_favored_pixels=int(
                            (negative & (delta > 0)).sum().item()),
                    ))

        selector_stats = []
        thresholds = self._parse_float_list(
            self.candidate_internal_delta_thresholds,
            '-0.50,-0.25,0.00,0.10,0.25,0.50,1.00',
        )
        for family_idx, family_name in enumerate(family_names):
            deltas = family_delta_stack[family_idx]
            scored = torch.nan_to_num(deltas, nan=-1e6)
            best_delta, best_rank = scored.max(dim=0)
            selected = torch.gather(
                candidate_idx, 0, best_rank.unsqueeze(0)).squeeze(0)
            selected_valid = torch.gather(
                candidate_valid, 0, best_rank.unsqueeze(0)).squeeze(0)
            for threshold in thresholds:
                gate = (
                    selected_valid
                    & (best_delta >= float(threshold))
                )
                selector_stats.append(
                    self._candidate_internal_selector_row(
                        f'{family_name}_delta_ge_{threshold:g}',
                        selected,
                        gate,
                        pred_idx,
                        gt_idx,
                        valid_mask,
                        base_correct,
                        base_wrong,
                    ))

        family_vote = (
            finite_delta & (family_delta_stack > 0)).sum(dim=0).float()
        family_mean_delta = torch.nan_to_num(
            family_delta_stack, nan=0.0).sum(dim=0)
        family_mean_delta = family_mean_delta / finite_delta.sum(
            dim=0).clamp_min(1)
        all_candidate_score = family_vote + 1e-3 * family_mean_delta
        all_candidate_score[~candidate_valid] = -1e6
        best_all_rank = all_candidate_score.argmax(dim=0)
        selected_all = torch.gather(
            candidate_idx, 0, best_all_rank.unsqueeze(0)).squeeze(0)
        selected_all_valid = torch.gather(
            candidate_valid, 0, best_all_rank.unsqueeze(0)).squeeze(0)
        selected_all_votes = torch.gather(
            family_vote, 0, best_all_rank.unsqueeze(0)).squeeze(0)

        vote_thresholds = sorted(set(
            max(1, int(round(value)))
            for value in self._parse_float_list(
                self.candidate_internal_vote_thresholds, '1,2,3')
        ))
        for vote_threshold in vote_thresholds:
            selector_stats.append(
                self._candidate_internal_selector_row(
                    f'all_family_vote_ge_{vote_threshold}',
                    selected_all,
                    selected_all_valid
                    & (selected_all_votes >= vote_threshold),
                    pred_idx,
                    gt_idx,
                    valid_mask,
                    base_correct,
                    base_wrong,
                ))

        family_index = {
            name: idx for idx, name in enumerate(family_names)}
        head_names = [
            name for name in ('semantic', 'instance')
            if name in family_index
        ]
        readout_names = [
            name for name in ('no_presence', 'presence_readout')
            if name in family_index
        ]
        latent_names = [
            name for name in ('raw_mask', 'encoder', 'visual')
            if name in family_index
        ]

        def vote_for(names):
            if not names:
                return torch.zeros_like(family_vote)
            indices = [family_index[name] for name in names]
            return (
                finite_delta[indices]
                & (family_delta_stack[indices] > 0)
            ).sum(dim=0).float()

        head_vote = vote_for(head_names)
        latent_vote = vote_for(latent_names)
        latent_mean = torch.zeros_like(family_mean_delta)
        if latent_names:
            latent_indices = [family_index[name] for name in latent_names]
            latent_finite = finite_delta[latent_indices]
            latent_mean = torch.nan_to_num(
                family_delta_stack[latent_indices], nan=0.0).sum(dim=0)
            latent_mean = latent_mean / latent_finite.sum(
                dim=0).clamp_min(1)
            latent_score = latent_vote + 1e-3 * latent_mean
            latent_score[~candidate_valid] = -1e6
            best_latent_rank = latent_score.argmax(dim=0)
            selected_latent = torch.gather(
                candidate_idx,
                0,
                best_latent_rank.unsqueeze(0),
            ).squeeze(0)
            selected_latent_valid = torch.gather(
                candidate_valid,
                0,
                best_latent_rank.unsqueeze(0),
            ).squeeze(0)
            selected_latent_votes = torch.gather(
                latent_vote,
                0,
                best_latent_rank.unsqueeze(0),
            ).squeeze(0)
            selected_head_votes = torch.gather(
                head_vote,
                0,
                best_latent_rank.unsqueeze(0),
            ).squeeze(0)
            for vote_threshold in vote_thresholds:
                selector_stats.append(
                    self._candidate_internal_selector_row(
                        f'latent_family_vote_ge_{vote_threshold}',
                        selected_latent,
                        selected_latent_valid
                        & (selected_latent_votes >= vote_threshold),
                        pred_idx,
                        gt_idx,
                        valid_mask,
                        base_correct,
                        base_wrong,
                    ))
            selector_stats.append(
                self._candidate_internal_selector_row(
                    'head_and_latent_support',
                    selected_latent,
                    selected_latent_valid
                    & (selected_latent_votes >= 1)
                    & (selected_head_votes >= 1),
                    pred_idx,
                    gt_idx,
                    valid_mask,
                    base_correct,
                    base_wrong,
                ))

        for first_name, second_name in (
                ('raw_mask', 'encoder'),
                ('raw_mask', 'visual'),
                ('encoder', 'visual')):
            if first_name not in family_index or second_name not in family_index:
                continue
            pair_vote = (
                (family_delta_stack[family_index[first_name]] > 0)
                & (family_delta_stack[family_index[second_name]] > 0)
                & candidate_valid
            )
            pair_score = torch.nan_to_num(
                family_delta_stack[family_index[first_name]], nan=-10.0)
            pair_score = pair_score + torch.nan_to_num(
                family_delta_stack[family_index[second_name]], nan=-10.0)
            pair_score[~pair_vote] = -1e6
            best_pair_rank = pair_score.argmax(dim=0)
            selected_pair = torch.gather(
                candidate_idx,
                0,
                best_pair_rank.unsqueeze(0),
            ).squeeze(0)
            selected_pair_gate = torch.gather(
                pair_vote,
                0,
                best_pair_rank.unsqueeze(0),
            ).squeeze(0)
            selector_stats.append(
                self._candidate_internal_selector_row(
                    f'{first_name}_and_{second_name}',
                    selected_pair,
                    selected_pair_gate,
                    pred_idx,
                    gt_idx,
                    valid_mask,
                    base_correct,
                    base_wrong,
                ))

        pair_stats = []
        min_pair_pixels = int(self.candidate_internal_min_pair_pixels)
        for rank_idx in range(topk):
            for base_class in range(self.num_cls):
                base_class_mask = (
                    candidate_valid[rank_idx]
                    & (pred_idx == base_class)
                )
                for candidate_class_id in range(self.num_cls):
                    if candidate_class_id == base_class:
                        continue
                    pair_mask = (
                        base_class_mask
                        & (candidate_idx[rank_idx] == candidate_class_id)
                    )
                    pair_pixels = int(pair_mask.sum().item())
                    if pair_pixels < min_pair_pixels:
                        continue
                    row = dict(
                        candidate_rank=rank_idx + 1,
                        base_class_index=base_class,
                        base_class_name=self.class_names[base_class],
                        candidate_class_index=candidate_class_id,
                        candidate_class_name=self.class_names[
                            candidate_class_id],
                        pixels=pair_pixels,
                        candidate_correct_pixels=int(
                            (pair_mask & base_wrong
                             & (gt_idx == candidate_class_id)).sum().item()),
                        baseline_correct_pixels=int(
                            (pair_mask & base_correct).sum().item()),
                    )
                    family_rows = {}
                    for family_idx, family_name in enumerate(family_names):
                        delta = family_delta_stack[family_idx, rank_idx]
                        finite_pair = pair_mask & torch.isfinite(delta)
                        family_rows[family_name] = dict(
                            valid_pixels=int(finite_pair.sum().item()),
                            delta_sum=_masked_sum(delta, finite_pair),
                            favored_pixels=int(
                                (finite_pair & (delta > 0)).sum().item()),
                        )
                    row['family_stats'] = family_rows
                    pair_stats.append(row)

        gt_in_candidate = (
            (candidate_idx == gt_idx.unsqueeze(0))
            & candidate_valid
        ).any(dim=0) & base_wrong

        def family_support(family_map):
            gt_score = _gather_class_map(family_map, gt_idx)
            pred_score = _gather_class_map(family_map, pred_idx)
            return (
                torch.isfinite(gt_score)
                & torch.isfinite(pred_score)
                & ((gt_score - pred_score) > 0)
            )

        family_gt_support = {
            name: family_support(family_map)
            for name, family_map in family_maps.items()
        }

        def any_support(names):
            result = torch.zeros_like(valid_mask)
            for name in names:
                if name in family_gt_support:
                    result |= family_gt_support[name]
            return result

        head_support = any_support(head_names)
        readout_support = any_support(readout_names)
        latent_support = any_support(latent_names)
        expanded_support = head_support | readout_support | latent_support
        prior_readout_support = head_support | readout_support
        latent_incremental = latent_support & (~prior_readout_support)
        oracle_stats = dict(
            valid_pixels=valid_count,
            baseline_wrong_pixels=int(base_wrong.sum().item()),
            gt_in_candidate_pixels=int(gt_in_candidate.sum().item()),
            head_support_pixels=int(
                (gt_in_candidate & head_support).sum().item()),
            readout_support_pixels=int(
                (gt_in_candidate & readout_support).sum().item()),
            latent_support_pixels=int(
                (gt_in_candidate & latent_support).sum().item()),
            expanded_support_pixels=int(
                (gt_in_candidate & expanded_support).sum().item()),
            latent_incremental_pixels=int(
                (gt_in_candidate & latent_incremental).sum().item()),
        )

        null_stats = [dict(
            control_name='observed',
            trial=0,
            gt_in_candidate_pixels=int(gt_in_candidate.sum().item()),
            latent_support_pixels=int(
                (gt_in_candidate & latent_support).sum().item()),
            latent_incremental_pixels=int(
                (gt_in_candidate & latent_incremental).sum().item()),
        )]
        null_trials = max(0, int(self.candidate_internal_null_trials))
        latent_maps = {
            name: family_maps[name]
            for name in latent_names
            if name in family_maps
        }
        for trial in range(null_trials):
            class_support = torch.zeros_like(valid_mask)
            spatial_support = torch.zeros_like(valid_mask)
            class_shift = 1 + (trial % max(1, self.num_cls - 1))
            shift_y = max(
                1, int(round(diag_shape[0] * (trial + 1)
                             / float(null_trials + 1))))
            shift_x = max(
                1, int(round(diag_shape[1] * (null_trials - trial)
                             / float(null_trials + 1))))
            for family_map in latent_maps.values():
                class_support |= family_support(
                    torch.roll(family_map, shifts=class_shift, dims=0))
                spatial_support |= family_support(
                    torch.roll(
                        family_map,
                        shifts=(shift_y, shift_x),
                        dims=(-2, -1),
                    ))
            null_stats.extend([
                dict(
                    control_name='class_permutation',
                    trial=trial + 1,
                    gt_in_candidate_pixels=int(
                        gt_in_candidate.sum().item()),
                    latent_support_pixels=int(
                        (gt_in_candidate & class_support).sum().item()),
                    latent_incremental_pixels=int(
                        (gt_in_candidate & class_support
                         & (~prior_readout_support)).sum().item()),
                ),
                dict(
                    control_name='spatial_shift',
                    trial=trial + 1,
                    gt_in_candidate_pixels=int(
                        gt_in_candidate.sum().item()),
                    latent_support_pixels=int(
                        (gt_in_candidate & spatial_support).sum().item()),
                    latent_incremental_pixels=int(
                        (gt_in_candidate & spatial_support
                         & (~prior_readout_support)).sum().item()),
                ),
            ])

        final_margin = final_topk.values[0] - final_topk.values[1]
        regime_oracle_stats = []
        oracle_regimes = [
            ('margin_0_0p05', valid_mask & (final_margin < 0.05)),
            ('margin_0p05_0p10',
             valid_mask & (final_margin >= 0.05) & (final_margin < 0.10)),
            ('margin_0p10_0p20',
             valid_mask & (final_margin >= 0.10) & (final_margin < 0.20)),
            ('margin_0p20_0p50',
             valid_mask & (final_margin >= 0.20) & (final_margin < 0.50)),
            ('margin_ge_0p50', valid_mask & (final_margin >= 0.50)),
            ('argmax_competition', argmax_competition),
            ('threshold_reject', threshold_reject),
        ]
        for regime_name, regime_mask in oracle_regimes:
            eligible = regime_mask & gt_in_candidate
            regime_oracle_stats.append(dict(
                regime_name=regime_name,
                baseline_wrong_pixels=int(
                    (regime_mask & base_wrong).sum().item()),
                gt_in_candidate_pixels=int(eligible.sum().item()),
                head_support_pixels=int(
                    (eligible & head_support).sum().item()),
                readout_support_pixels=int(
                    (eligible & readout_support).sum().item()),
                latent_support_pixels=int(
                    (eligible & latent_support).sum().item()),
                expanded_support_pixels=int(
                    (eligible & expanded_support).sum().item()),
                latent_incremental_pixels=int(
                    (eligible & latent_incremental).sum().item()),
            ))

        matched_stats = self._build_candidate_internal_matched_stats(
            candidate_idx=candidate_idx,
            candidate_valid=candidate_valid,
            final_margin=final_margin,
            pred_idx=pred_idx,
            gt_idx=gt_idx,
            valid_mask=valid_mask,
            base_correct=base_correct,
            base_wrong=base_wrong,
            family_maps=family_maps,
            family_delta_stack=family_delta_stack,
            family_names=family_names,
            regime_masks=regime_masks,
            diag_shape=diag_shape,
        )

        return dict(
            dataset_name=self.seed_dataset_name,
            diagnostic_shape=list(diag_shape),
            topk=topk,
            valid_pixels=valid_count,
            baseline_correct_pixels=int(base_correct.sum().item()),
            baseline_wrong_pixels=int(base_wrong.sum().item()),
            family_sources=active_family_sources,
            candidate_stats=candidate_stats,
            feature_stats=feature_stats,
            selector_stats=selector_stats,
            pair_stats=pair_stats,
            oracle_stats=oracle_stats,
            null_stats=null_stats,
            regime_oracle_stats=regime_oracle_stats,
            matched_stats=matched_stats,
        )

    def _candidate_residual_strengths(self):
        strengths = {
            min(max(float(value), 0.0), 1.5)
            for value in self._parse_float_list(
                self.candidate_residual_strengths,
                '0.00,0.25,0.50,0.75,1.00',
            )
        }
        strengths.add(0.0)
        return sorted(strengths)

    def _candidate_residual_source_sequences(self, components):
        source_maps = components.get('internal_source_maps') or {}
        strengths = self._candidate_residual_strengths()
        positive_strengths = [
            value for value in strengths if value > 0.0
        ]
        variants = set(self._parse_name_list(
            self.candidate_residual_variants))
        layer_count = len(components.get('pe_layers', []))
        if layer_count <= 0:
            layer_count = max(
                [
                    int(name.split('_')[1][1:]) + 1
                    for name in source_maps
                    if name.startswith('scenecommon_l')
                    and len(name.split('_')) > 1
                    and name.split('_')[1][1:].isdigit()
                ] or [0]
            )
        layers = self._get_candidate_residual_layers(layer_count)
        sequences = []

        def collect_sequence(layer_idx, variant_name, control_name,
                             trial, name_builder):
            raw_name = f'scenecommon_l{layer_idx}_raw'
            raw_map = source_maps.get(raw_name)
            if not isinstance(raw_map, torch.Tensor):
                return
            maps = [raw_map]
            alphas = [0.0]
            source_names = [raw_name]
            for strength in positive_strengths:
                source_name = name_builder(strength)
                source_map = source_maps.get(source_name)
                if not isinstance(source_map, torch.Tensor):
                    continue
                maps.append(source_map)
                alphas.append(float(strength))
                source_names.append(source_name)
            if len(maps) < 2:
                return
            sequences.append(dict(
                layer_index=int(layer_idx),
                variant_name=str(variant_name),
                control_name=str(control_name),
                trial=int(trial),
                alphas=alphas,
                source_names=source_names,
                maps=maps,
            ))

        for layer_idx in layers:
            if 'global' in variants:
                collect_sequence(
                    layer_idx,
                    'global',
                    'observed',
                    0,
                    lambda strength, layer_idx=layer_idx: (
                        f'scenecommon_l{layer_idx}_global_a'
                        f'{self._scene_common_strength_token(strength)}'
                    ),
                )
            if 'class_balanced' in variants:
                collect_sequence(
                    layer_idx,
                    'class_balanced',
                    'observed',
                    0,
                    lambda strength, layer_idx=layer_idx: (
                        f'scenecommon_l{layer_idx}_balanced_a'
                        f'{self._scene_common_strength_token(strength)}'
                    ),
                )
            if 'controls' in variants:
                if 'global' in variants:
                    collect_sequence(
                        layer_idx,
                        'global',
                        'feature_permutation',
                        0,
                        lambda strength, layer_idx=layer_idx: (
                            f'scenecommon_l{layer_idx}_perm_t0_a'
                            f'{self._scene_common_strength_token(strength)}'
                        ),
                    )
                    for trial in range(max(
                            1, int(self.scene_common_random_trials))):
                        collect_sequence(
                            layer_idx,
                            'global',
                            'random_direction',
                            trial + 1,
                            lambda strength, layer_idx=layer_idx,
                            trial=trial: (
                                f'scenecommon_l{layer_idx}_rand_t{trial}_a'
                                f'{self._scene_common_strength_token(strength)}'
                            ),
                        )
                collect_sequence(
                    layer_idx,
                    'class_balanced',
                    'feature_permutation',
                    0,
                    lambda strength, layer_idx=layer_idx: (
                        f'scenecommon_l{layer_idx}_balanced_'
                        f'perm_t0_a'
                        f'{self._scene_common_strength_token(strength)}'
                    ),
                )
                collect_sequence(
                    layer_idx,
                    'class_balanced',
                    'random_direction',
                    1,
                    lambda strength, layer_idx=layer_idx: (
                        f'scenecommon_l{layer_idx}_balanced_'
                        f'rand_t0_a'
                        f'{self._scene_common_strength_token(strength)}'
                    ),
                )
        return sequences

    def _candidate_residual_trajectory_scores(
            self, sequence_maps, alphas, candidate_idx, pred_idx,
            candidate_valid):
        normalized = [
            self._candidate_internal_normalize_map(item)
            for item in sequence_maps
        ]
        stack = torch.stack(normalized, dim=0)
        pred_scores = torch.stack([
            _gather_class_map(item, pred_idx)
            for item in normalized
        ], dim=0)
        candidate_scores = torch.stack([
            torch.gather(item, 0, candidate_idx)
            for item in normalized
        ], dim=0)
        margin = candidate_scores - pred_scores.unsqueeze(1)
        valid = (
            torch.isfinite(margin).all(dim=0)
            & candidate_valid
        )
        margin = margin.masked_fill(
            ~candidate_valid.unsqueeze(0), float('nan'))

        alpha = torch.tensor(
            alphas,
            device=margin.device,
            dtype=margin.dtype,
        )
        alpha_centered = alpha - alpha.mean()
        alpha_energy = alpha_centered.square().sum().clamp_min(1e-6)
        margin_centered = margin - margin.mean(dim=0, keepdim=True)
        margin_slope = (
            margin_centered
            * alpha_centered[:, None, None, None]
        ).sum(dim=0) / alpha_energy

        alpha_span = float(alphas[-1] - alphas[0])
        if alpha_span <= 0:
            return None
        margin_auc = torch.trapz(
            margin,
            alpha,
            dim=0,
        ) / alpha_span
        margin_auc_gain = margin_auc - margin[0]
        endpoint_margin_gain = margin[-1] - margin[0]
        candidate_gain = candidate_scores[-1] - candidate_scores[0]
        competitor_suppression = (
            pred_scores[0] - pred_scores[-1]
        ).unsqueeze(0).expand_as(candidate_gain).clone()
        candidate_gain_positive = candidate_gain.clamp_min(0.0)
        competitor_suppression_positive = (
            competitor_suppression.clamp_min(0.0))
        independent_support = (
            candidate_gain - competitor_suppression_positive)
        gain_fraction = candidate_gain_positive / (
            candidate_gain_positive
            + competitor_suppression_positive
            + 1e-6
        )

        step_gain = margin[1:] - margin[:-1]
        monotonicity = (
            step_gain >= 0.0
        ).float().mean(dim=0)
        max_margin_gain = (
            margin - margin[0:1]
        ).max(dim=0)[0]
        crossing_alpha = torch.full_like(
            margin[0], float('nan'))
        not_crossed = margin[0] <= 0.0
        for step_idx in range(1, len(alphas)):
            newly_crossed = (
                not_crossed
                & torch.isnan(crossing_alpha)
                & (margin[step_idx] > 0.0)
            )
            crossing_alpha[newly_crossed] = float(
                alphas[step_idx])
        crossing_score = torch.where(
            torch.isfinite(crossing_alpha),
            1.0 - crossing_alpha / max(float(alphas[-1]), 1e-6),
            torch.full_like(crossing_alpha, -1.0),
        )

        score_maps = {
            'margin_slope': margin_slope,
            'margin_auc_gain': margin_auc_gain,
            'endpoint_margin_gain': endpoint_margin_gain,
            'candidate_gain': candidate_gain,
            'competitor_suppression': competitor_suppression,
            'independent_support': independent_support,
            'candidate_gain_fraction': gain_fraction,
            'margin_monotonicity': monotonicity,
            'max_margin_gain': max_margin_gain,
            'crossing_score': crossing_score,
        }
        requested_scores = set(self._parse_name_list(
            self.candidate_residual_scores))
        if requested_scores and 'all' not in requested_scores:
            score_maps = {
                name: value
                for name, value in score_maps.items()
                if name in requested_scores
            }
        for score_name, score in list(score_maps.items()):
            if score.shape != valid.shape:
                raise RuntimeError(
                    'Candidate residual trajectory score shape mismatch: '
                    f'{score_name} has {tuple(score.shape)}, expected '
                    f'{tuple(valid.shape)}.')
            score_maps[score_name] = score.masked_fill(
                ~valid, float('nan'))
        return dict(
            score_maps=score_maps,
            candidate_gain=candidate_gain,
            competitor_suppression=competitor_suppression,
            endpoint_margin_gain=endpoint_margin_gain,
            crossing_alpha=crossing_alpha,
            valid=valid,
        )

    def _build_candidate_residual_trajectory_stats(
            self, base_logits, base_pred, data_sample, components):
        if (
                components is None
                or self.num_cls <= 1
                or not self.dump_candidate_residual_trajectory_stats):
            return None
        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)
        if gt_data is None:
            return None

        diag_shape = self._internal_selection_diag_shape(
            int(base_logits.shape[-2]), int(base_logits.shape[-1]))
        gt_diag = F.interpolate(
            gt_data.float().view(1, 1, *gt_data.shape[-2:]),
            size=diag_shape,
            mode='nearest',
        ).squeeze().long()
        pred_diag = F.interpolate(
            base_pred.float().view(1, 1, *base_pred.shape[-2:]),
            size=diag_shape,
            mode='nearest',
        ).squeeze().long()
        valid_mask = gt_diag != 255
        if not valid_mask.any():
            return None

        final_map = self._resize_internal_source_map(
            base_logits, diag_shape)
        semantic_map = self._resize_internal_source_map(
            components.get('semantic_logits'), diag_shape)
        instance_map = self._resize_internal_source_map(
            components.get('instance_logits'), diag_shape)
        if final_map is None:
            return None
        topk = max(
            2,
            min(
                max(
                    int(round(value))
                    for value in self._parse_float_list(
                        self.candidate_residual_ranks, '2,3')
                ),
                self.num_cls,
            ),
        )
        final_topk = torch.topk(
            torch.nan_to_num(final_map, nan=-1e6),
            k=topk,
            dim=0,
        )
        candidate_idx = final_topk.indices
        final_top1 = candidate_idx[0]
        pred_idx = pred_diag.clamp(0, self.num_cls - 1)
        gt_idx = gt_diag.clamp(0, self.num_cls - 1)
        candidate_valid = (
            (candidate_idx != pred_idx.unsqueeze(0))
            & valid_mask.unsqueeze(0)
        )
        base_correct = (pred_diag == gt_diag) & valid_mask
        base_wrong = (~base_correct) & valid_mask
        threshold_reject = (
            pred_idx != final_top1) & valid_mask
        regime_masks = {
            'all': valid_mask,
            'argmax_competition': (
                (~threshold_reject) & valid_mask),
            'threshold_reject': threshold_reject,
        }
        final_margin = (
            final_topk.values[0] - final_topk.values[1])

        requested_ranks = {
            int(round(value))
            for value in self._parse_float_list(
                self.candidate_residual_ranks, '2,3')
        }
        rank_indices = [
            rank - 1 for rank in sorted(requested_ranks)
            if 1 <= rank <= candidate_idx.shape[0]
        ]
        requested_regimes = set(self._parse_name_list(
            self.candidate_residual_regimes))
        active_regimes = [
            (name, mask) for name, mask in regime_masks.items()
            if name in requested_regimes
        ]
        margin_edges = self._parse_float_list(
            self.candidate_residual_margin_bins,
            '0.00,0.05,0.10,0.20,0.50,inf',
        )
        margin_ranges = [
            (margin_edges[index], margin_edges[index + 1])
            for index in range(len(margin_edges) - 1)
        ]
        min_pair_pixels = max(
            1, int(self.candidate_residual_min_pair_pixels))
        auc_bins = max(8, int(self.candidate_residual_auc_bins))
        auc_min = float(self.candidate_residual_auc_min)
        auc_max = float(self.candidate_residual_auc_max)
        sequences = self._candidate_residual_source_sequences(
            components)
        matched_rows = []
        mechanism_rows = []
        source_inventory = []

        def new_metric_bucket():
            return dict(
                positive_pixels=0,
                negative_pixels=0,
                comparison_weight=0,
                auroc_weighted=0.0,
                ap_weight=0,
                auprc_weighted=0.0,
                auprc_lift_weighted=0.0,
            )

        def add_metric_bucket(bucket, metric):
            positive = int(metric.get('positive_pixels') or 0)
            negative = int(metric.get('negative_pixels') or 0)
            comparison_weight = int(
                metric.get('comparison_weight') or 0)
            bucket['positive_pixels'] += positive
            bucket['negative_pixels'] += negative
            bucket['comparison_weight'] += comparison_weight
            if metric.get('auroc') is not None:
                bucket['auroc_weighted'] += (
                    float(metric['auroc']) * comparison_weight)
            if metric.get('auprc') is not None:
                bucket['auprc_weighted'] += (
                    float(metric['auprc']) * positive)
                bucket['auprc_lift_weighted'] += (
                    float(metric.get('auprc_lift') or 0.0)
                    * positive)
                bucket['ap_weight'] += positive

        def metric_bucket_values(bucket):
            positive = int(bucket['positive_pixels'])
            negative = int(bucket['negative_pixels'])
            prior = _safe_div(
                positive, positive + negative)
            auprc = _safe_div(
                bucket['auprc_weighted'],
                bucket['ap_weight'],
            )
            return dict(
                positive_pixels=positive,
                negative_pixels=negative,
                prior=prior,
                auroc=_safe_div(
                    bucket['auroc_weighted'],
                    bucket['comparison_weight'],
                ),
                auprc=auprc,
                auprc_lift=_safe_div(
                    bucket['auprc_lift_weighted'],
                    bucket['ap_weight'],
                ),
                comparison_weight=int(
                    bucket['comparison_weight']),
            )

        def evaluate_sequence(sequence, control_name, trial, maps):
            trajectory = self._candidate_residual_trajectory_scores(
                maps,
                sequence['alphas'],
                candidate_idx,
                pred_idx,
                candidate_valid,
            )
            if trajectory is None:
                return
            source_inventory.append(dict(
                layer_index=sequence['layer_index'],
                variant_name=sequence['variant_name'],
                control_name=control_name,
                trial=int(trial),
                strengths=list(sequence['alphas']),
                source_names=list(sequence['source_names']),
            ))
            dataset_metric_buckets = defaultdict(
                lambda: defaultdict(new_metric_bucket))
            for rank_idx in rank_indices:
                candidate_class_map = candidate_idx[rank_idx]
                rank_valid = trajectory['valid'][rank_idx]
                candidate_help = (
                    rank_valid
                    & base_wrong
                    & (gt_idx == candidate_class_map)
                )
                switch_harm = rank_valid & base_correct
                positive_gain = (
                    trajectory['candidate_gain'][rank_idx] > 0.0)
                positive_suppression = (
                    trajectory['competitor_suppression'][rank_idx] > 0.0)
                positive_margin_gain = (
                    trajectory['endpoint_margin_gain'][rank_idx] > 0.0)
                candidate_dominant = (
                    positive_gain
                    & (
                        trajectory['candidate_gain'][rank_idx]
                        > trajectory[
                            'competitor_suppression'][rank_idx].clamp_min(0.0)
                    )
                )
                competitor_only = (
                    (~positive_gain)
                    & positive_suppression
                    & positive_margin_gain
                )
                crossing = torch.isfinite(
                    trajectory['crossing_alpha'][rank_idx])
                mechanism_rows.append(dict(
                    layer_index=sequence['layer_index'],
                    variant_name=sequence['variant_name'],
                    control_name=control_name,
                    trial=int(trial),
                    candidate_rank=rank_idx + 1,
                    eligible_pixels=int(rank_valid.sum().item()),
                    help_pixels=int(candidate_help.sum().item()),
                    harm_pixels=int(switch_harm.sum().item()),
                    help_positive_candidate_gain_pixels=int(
                        (candidate_help & positive_gain).sum().item()),
                    help_positive_margin_gain_pixels=int(
                        (candidate_help & positive_margin_gain).sum().item()),
                    help_candidate_dominant_pixels=int(
                        (candidate_help & candidate_dominant).sum().item()),
                    help_competitor_only_pixels=int(
                        (candidate_help & competitor_only).sum().item()),
                    help_crossing_pixels=int(
                        (candidate_help & crossing).sum().item()),
                    harm_positive_margin_gain_pixels=int(
                        (switch_harm & positive_margin_gain).sum().item()),
                ))

                base_pair_code = (
                    pred_idx * self.num_cls + candidate_class_map)
                for regime_name, regime_mask in active_regimes:
                    for margin_lower, margin_upper in margin_ranges:
                        stratum_mask = (
                            rank_valid
                            & regime_mask
                            & (final_margin >= float(margin_lower))
                            & (final_margin < float(margin_upper))
                        )
                        if int(stratum_mask.sum().item()) < min_pair_pixels:
                            continue
                        pair_codes = torch.unique(
                            base_pair_code[stratum_mask])
                        for pair_code_tensor in pair_codes:
                            pair_code = int(pair_code_tensor.item())
                            base_class = pair_code // self.num_cls
                            candidate_class = pair_code % self.num_cls
                            if base_class == candidate_class:
                                continue
                            pair_mask = (
                                stratum_mask
                                & (base_pair_code == pair_code)
                            )
                            pair_pixels = int(pair_mask.sum().item())
                            if pair_pixels < min_pair_pixels:
                                continue
                            pair_help = (
                                pair_mask
                                & base_wrong
                                & (gt_idx == candidate_class)
                            )
                            pair_harm = pair_mask & base_correct
                            pair_neutral = (
                                pair_mask
                                & (~pair_help)
                                & (~pair_harm)
                            )
                            task_masks = {
                                'candidate_correct': (
                                    pair_help,
                                    pair_mask & (~pair_help),
                                ),
                                'baseline_risk': (
                                    pair_mask & base_wrong,
                                    pair_harm,
                                ),
                                'switch_utility': (
                                    pair_help,
                                    pair_harm,
                                ),
                            }
                            for score_name, score_stack in (
                                    trajectory['score_maps'].items()):
                                score = score_stack[rank_idx]
                                task_metrics = {
                                    task_name:
                                    self._candidate_internal_metric_from_hist(
                                        score,
                                        positive_mask,
                                        negative_mask,
                                        auc_bins,
                                        auc_min,
                                        auc_max,
                                    )
                                    for task_name, (
                                        positive_mask,
                                        negative_mask,
                                    ) in task_masks.items()
                                }
                                margin_name = (
                                    self._candidate_internal_margin_name(
                                        margin_lower,
                                        margin_upper))
                                summary_key = (
                                    score_name,
                                    rank_idx + 1,
                                    regime_name,
                                    margin_name,
                                    float(margin_lower),
                                    (
                                        'inf'
                                        if math.isinf(margin_upper)
                                        else float(margin_upper)
                                    ),
                                )
                                for task_name, metric in (
                                        task_metrics.items()):
                                    add_metric_bucket(
                                        dataset_metric_buckets[
                                            summary_key][task_name],
                                        metric,
                                    )
                                if control_name == 'observed':
                                    matched_rows.append(dict(
                                        row_scope='pair',
                                        layer_index=(
                                            sequence['layer_index']),
                                        variant_name=(
                                            sequence['variant_name']),
                                        control_name=control_name,
                                        trial=int(trial),
                                        score_name=score_name,
                                        candidate_rank=rank_idx + 1,
                                        regime_name=regime_name,
                                        margin_name=margin_name,
                                        margin_lower=float(
                                            margin_lower),
                                        margin_upper=(
                                            'inf'
                                            if math.isinf(margin_upper)
                                            else float(margin_upper)),
                                        base_class_index=base_class,
                                        base_class_name=(
                                            self.class_names[
                                                base_class]),
                                        candidate_class_index=(
                                            candidate_class),
                                        candidate_class_name=(
                                            self.class_names[
                                                candidate_class]),
                                        pair_pixels=pair_pixels,
                                        help_pixels=int(
                                            pair_help.sum().item()),
                                        harm_pixels=int(
                                            pair_harm.sum().item()),
                                        neutral_pixels=int(
                                            pair_neutral.sum().item()),
                                        task_metrics=task_metrics,
                                    ))
            for summary_key, task_buckets in (
                    dataset_metric_buckets.items()):
                (
                    score_name, rank, regime_name, margin_name,
                    margin_lower, margin_upper,
                ) = summary_key
                matched_rows.append(dict(
                    row_scope='dataset',
                    layer_index=sequence['layer_index'],
                    variant_name=sequence['variant_name'],
                    control_name=control_name,
                    trial=int(trial),
                    score_name=score_name,
                    candidate_rank=rank,
                    regime_name=regime_name,
                    margin_name=margin_name,
                    margin_lower=margin_lower,
                    margin_upper=margin_upper,
                    task_metrics={
                        task_name: metric_bucket_values(bucket)
                        for task_name, bucket in (
                            task_buckets.items())
                    },
                ))

        null_trials = max(
            0, int(self.candidate_residual_null_trials))
        for sequence in sequences:
            evaluate_sequence(
                sequence,
                sequence['control_name'],
                sequence['trial'],
                sequence['maps'],
            )
            if sequence['control_name'] != 'observed':
                continue
            for trial in range(null_trials):
                class_shift = 1 + (
                    trial % max(1, self.num_cls - 1))
                shift_y = max(
                    1,
                    int(round(
                        diag_shape[0] * (trial + 1)
                        / float(null_trials + 1))),
                )
                shift_x = max(
                    1,
                    int(round(
                        diag_shape[1] * (null_trials - trial)
                        / float(null_trials + 1))),
                )
                evaluate_sequence(
                    sequence,
                    'class_permutation',
                    trial + 1,
                    [
                        torch.roll(
                            item, shifts=class_shift, dims=0)
                        for item in sequence['maps']
                    ],
                )
                evaluate_sequence(
                    sequence,
                    'spatial_shift',
                    trial + 1,
                    [
                        torch.roll(
                            item,
                            shifts=(shift_y, shift_x),
                            dims=(-2, -1),
                        )
                        for item in sequence['maps']
                    ],
                )

        head_support = []
        for name, score_map in (
                ('semantic', semantic_map),
                ('instance', instance_map)):
            if score_map is None:
                continue
            normalized = self._candidate_internal_normalize_map(
                score_map)
            pred_score = _gather_class_map(normalized, pred_idx)
            candidate_score = torch.gather(
                normalized, 0, candidate_idx)
            head_support.append(dict(
                head_name=name,
                positive_candidate_delta_pixels=int(
                    (
                        candidate_valid
                        & (candidate_score > pred_score.unsqueeze(0))
                    ).sum().item()),
            ))

        return dict(
            dataset_name=self.seed_dataset_name,
            diagnostic_shape=list(diag_shape),
            valid_pixels=int(valid_mask.sum().item()),
            baseline_correct_pixels=int(base_correct.sum().item()),
            baseline_wrong_pixels=int(base_wrong.sum().item()),
            strengths=self._candidate_residual_strengths(),
            matched_stats=matched_rows,
            mechanism_stats=mechanism_rows,
            source_inventory=source_inventory,
            head_support_stats=head_support,
            feature_stats=[
                row for row in (
                    components.get(
                        'scene_common_feature_stats', []) or [])
                if row.get('stat_type') == 'layer_descriptor'
            ],
        )

    def _write_candidate_residual_trajectory_stats(self, record):
        if not self.dump_candidate_residual_trajectory_stats:
            return
        if self._candidate_residual_trajectory_stats_file is None:
            path = (
                self.candidate_residual_trajectory_stats_path
                or './work_dirs/evidence_stats/'
                   'candidate_residual_trajectory_stats.jsonl'
            )
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._candidate_residual_trajectory_stats_file = open(
                path, 'a', buffering=1)
        self._candidate_residual_trajectory_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    def _candidate_residual_miou_confusion(
            self, gt, pred, valid_mask):
        encoded = (
            gt[valid_mask].long() * self.num_cls
            + pred[valid_mask].long()
        )
        matrix = torch.bincount(
            encoded,
            minlength=self.num_cls * self.num_cls,
        ).reshape(self.num_cls, self.num_cls)
        return matrix.detach().cpu().tolist()

    def _candidate_residual_miou_region_reducers(self):
        reducers = self._parse_name_list(
            self.candidate_residual_miou_region_reducer)
        if not reducers:
            reducers = ['mean']
        unknown = set(reducers) - {'mean', 'p25'}
        if unknown:
            raise ValueError(
                'candidate_residual_miou_region_reducer supports '
                f"'mean' and 'p25', got {sorted(unknown)}")
        return reducers

    def _candidate_residual_miou_region_score_map(
            self, score, eligible, pred_idx, candidate_idx, reducer):
        reducer = str(reducer).lower()
        if reducer not in ('mean', 'p25'):
            raise ValueError(
                'candidate_residual_miou_region_reducer must be '
                f"'mean' or 'p25', got {reducer!r}")
        output = torch.full_like(score, float('nan'))
        pair_code = pred_idx * self.num_cls + candidate_idx
        min_pixels = max(
            1, int(self.candidate_residual_miou_region_min_pixels))
        region_count = 0
        retained_regions = 0
        for code_tensor in torch.unique(pair_code[eligible]):
            code = int(code_tensor.item())
            pair_mask = eligible & (pair_code == code)
            labels, count = _connected_component_labels(pair_mask)
            region_count += count
            if count <= 0:
                continue
            labels = labels.to(score.device)
            for label_idx in range(1, count + 1):
                region_mask = labels == label_idx
                pixels = int(region_mask.sum().item())
                if pixels < min_pixels:
                    continue
                values = score[region_mask]
                values = values[torch.isfinite(values)]
                if values.numel() == 0:
                    continue
                if reducer == 'p25':
                    region_score = torch.quantile(values, 0.25)
                else:
                    region_score = values.mean()
                output[region_mask] = region_score
                retained_regions += 1
        return output, dict(
            proposed_regions=region_count,
            retained_regions=retained_regions,
        )

    def _candidate_residual_miou_raw_score_map(
            self, score, eligible, candidate_idx, components, reducer):
        output = torch.full_like(score, float('nan'))
        raw_candidates = (
            None if components is None
            else components.get('raw_mask_candidates'))
        if not raw_candidates:
            return output, dict(
                proposed_regions=0,
                retained_regions=0,
                raw_mask_available=False,
            )
        reducer = str(reducer).lower()
        min_pixels = max(
            1, int(self.candidate_residual_miou_region_min_pixels))
        retained = 0
        proposed = 0
        for prompt_record in raw_candidates:
            class_idx = int(prompt_record.get('class_index', -1))
            raw_masks = prompt_record.get('raw_masks_lowres')
            if (
                    class_idx < 0
                    or class_idx >= self.num_cls
                    or not isinstance(raw_masks, torch.Tensor)):
                continue
            for raw_mask in raw_masks:
                proposed += 1
                mask_prob = torch.sigmoid(self._interpolate_float32(
                    raw_mask.view(1, 1, *raw_mask.shape).to(self.device),
                    score.shape[-2:],
                ).squeeze())
                region_mask = (
                    eligible
                    & (candidate_idx == class_idx)
                    & (
                        mask_prob
                        >= float(
                            self.candidate_residual_miou_raw_bin_thd)
                    )
                )
                pixels = int(region_mask.sum().item())
                if pixels < min_pixels:
                    continue
                values = score[region_mask]
                values = values[torch.isfinite(values)]
                if values.numel() == 0:
                    continue
                if reducer == 'p25':
                    region_score = torch.quantile(values, 0.25)
                else:
                    region_score = values.mean()
                old_values = output[region_mask]
                replacement = torch.full_like(
                    old_values, region_score)
                output[region_mask] = torch.where(
                    torch.isfinite(old_values),
                    torch.maximum(old_values, replacement),
                    replacement,
                )
                retained += 1
        return output, dict(
            proposed_regions=proposed,
            retained_regions=retained,
            raw_mask_available=True,
        )

    def _candidate_residual_miou_transition_table(
            self, gt_data, base_pred, valid_mask, candidate_idx):
        diag_h, diag_w = candidate_idx.shape[-2:]
        diag_pixels = int(diag_h * diag_w)
        class_pairs = int(self.num_cls * self.num_cls)
        diag_id = torch.arange(
            diag_pixels,
            device=self.device,
            dtype=torch.float32,
        ).view(1, 1, diag_h, diag_w)
        diag_id_full = F.interpolate(
            diag_id,
            size=gt_data.shape[-2:],
            mode='nearest',
        ).squeeze().long()
        candidate_full = F.interpolate(
            candidate_idx.float().view(
                1, 1, diag_h, diag_w),
            size=gt_data.shape[-2:],
            mode='nearest',
        ).squeeze().long()
        potential = (
            valid_mask
            & (candidate_full != base_pred)
        )

        delta = torch.zeros(
            (diag_pixels, self.num_cls, self.num_cls),
            device=self.device,
            dtype=torch.long,
        )
        changed_by_diag = torch.zeros(
            diag_pixels, device=self.device, dtype=torch.long)
        improved_by_diag = torch.zeros_like(changed_by_diag)
        harmed_by_diag = torch.zeros_like(changed_by_diag)
        neutral_by_diag = torch.zeros_like(changed_by_diag)
        if potential.any():
            source_id = diag_id_full[potential]
            gt_value = gt_data[potential].long()
            base_value = base_pred[potential].long()
            candidate_value = candidate_full[potential].long()
            old_code = gt_value * self.num_cls + base_value
            new_code = gt_value * self.num_cls + candidate_value
            old_joint = source_id * class_pairs + old_code
            new_joint = source_id * class_pairs + new_code
            old_count = torch.bincount(
                old_joint,
                minlength=diag_pixels * class_pairs,
            )
            new_count = torch.bincount(
                new_joint,
                minlength=diag_pixels * class_pairs,
            )
            delta = (
                new_count - old_count
            ).view(diag_pixels, self.num_cls, self.num_cls)
            changed_by_diag = torch.bincount(
                source_id, minlength=diag_pixels)
            base_correct = base_value == gt_value
            candidate_correct = candidate_value == gt_value
            improved_by_diag = torch.bincount(
                source_id[(~base_correct) & candidate_correct],
                minlength=diag_pixels,
            )
            harmed_by_diag = torch.bincount(
                source_id[base_correct & (~candidate_correct)],
                minlength=diag_pixels,
            )
            neutral_by_diag = torch.bincount(
                source_id[(~base_correct) & (~candidate_correct)],
                minlength=diag_pixels,
            )
        return dict(
            confusion_delta_by_diag=delta.cpu(),
            changed_by_diag=changed_by_diag.cpu(),
            improved_by_diag=improved_by_diag.cpu(),
            harmed_by_diag=harmed_by_diag.cpu(),
            neutral_by_diag=neutral_by_diag.cpu(),
            potential_changed_pixels=int(potential.sum().item()),
        )

    @staticmethod
    def _candidate_residual_sparse_confusion(delta):
        nonzero = torch.nonzero(delta, as_tuple=False)
        return [
            [
                int(index[0].item()),
                int(index[1].item()),
                int(delta[index[0], index[1]].item()),
            ]
            for index in nonzero
        ]

    def _candidate_residual_miou_gate_rows(
            self, score, candidate_idx, eligible, unit_name,
            unit_meta, transition_table, common, region_reducer):
        finite_eligible = eligible & torch.isfinite(score)
        eligible_pixels = int(finite_eligible.sum().item())
        if eligible_pixels == 0:
            return []
        score_values = score[finite_eligible]
        gate_defs = []
        absolute_thresholds = sorted(set(
            float(value)
            for value in self._parse_float_list(
                self.candidate_residual_miou_absolute_thresholds,
                '0.00,0.02,0.05,0.10,0.20',
            )
        ))
        for threshold in absolute_thresholds:
            gate_defs.append((
                'absolute',
                float(threshold),
                float(threshold),
            ))
        coverages = sorted(set(
            min(max(float(value), 0.0), 1.0)
            for value in self._parse_float_list(
                self.candidate_residual_miou_coverages,
                '0.001,0.0025,0.005,0.01,0.02,0.05',
            )
            if float(value) > 0.0
        ))
        for coverage in coverages:
            threshold = float(torch.quantile(
                score_values,
                max(0.0, 1.0 - coverage),
            ).item())
            gate_defs.append((
                'coverage',
                float(coverage),
                threshold,
            ))

        rows = []
        delta_by_diag = transition_table[
            'confusion_delta_by_diag']
        changed_by_diag = transition_table['changed_by_diag']
        improved_by_diag = transition_table['improved_by_diag']
        harmed_by_diag = transition_table['harmed_by_diag']
        neutral_by_diag = transition_table['neutral_by_diag']
        for threshold_type, threshold_value, actual_threshold in gate_defs:
            gate = finite_eligible & (score >= actual_threshold)
            selected_diag = torch.nonzero(
                gate.flatten(), as_tuple=False).flatten().cpu()
            if selected_diag.numel() == 0:
                changed_pixels = 0
                improved_pixels = 0
                harmed_pixels = 0
                neutral_pixels = 0
                delta = torch.zeros(
                    (self.num_cls, self.num_cls), dtype=torch.long)
            else:
                delta = delta_by_diag[selected_diag].sum(dim=0)
                changed_pixels = int(
                    changed_by_diag[selected_diag].sum().item())
                improved_pixels = int(
                    improved_by_diag[selected_diag].sum().item())
                harmed_pixels = int(
                    harmed_by_diag[selected_diag].sum().item())
                neutral_pixels = int(
                    neutral_by_diag[selected_diag].sum().item())
            rows.append(dict(
                **common,
                unit_name=unit_name,
                region_reducer=(
                    'none' if unit_name == 'pixel'
                    else str(region_reducer)
                ),
                threshold_type=threshold_type,
                threshold_value=threshold_value,
                actual_score_threshold=actual_threshold,
                eligible_pixels=eligible_pixels,
                selected_diag_pixels=int(selected_diag.numel()),
                changed_pixels=changed_pixels,
                improved_pixels=improved_pixels,
                harmed_pixels=harmed_pixels,
                neutral_pixels=neutral_pixels,
                proposed_regions=int(
                    unit_meta.get('proposed_regions', 0)),
                retained_regions=int(
                    unit_meta.get('retained_regions', 0)),
                raw_mask_available=bool(
                    unit_meta.get('raw_mask_available', False)),
                potential_changed_pixels=int(
                    transition_table.get(
                        'potential_changed_pixels', 0)),
                confusion_delta_sparse=(
                    self._candidate_residual_sparse_confusion(delta)),
            ))
        return rows

    def _build_candidate_residual_miou_stats(
            self, base_logits, base_pred, data_sample, components):
        if (
                components is None
                or self.num_cls <= 1
                or not self.dump_candidate_residual_miou_stats):
            return None
        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)
        if gt_data is None:
            return None
        valid_mask = gt_data != 255
        if not valid_mask.any():
            return None
        base_pred = base_pred.long()
        baseline_confusion = self._candidate_residual_miou_confusion(
            gt_data, base_pred, valid_mask)
        diag_shape = self._internal_selection_diag_shape(
            int(base_logits.shape[-2]), int(base_logits.shape[-1]))
        pred_diag = F.interpolate(
            base_pred.float().view(1, 1, *base_pred.shape[-2:]),
            size=diag_shape,
            mode='nearest',
        ).squeeze().long()
        valid_diag = F.interpolate(
            valid_mask.float().view(1, 1, *valid_mask.shape[-2:]),
            size=diag_shape,
            mode='nearest',
        ).squeeze().bool()
        final_map = self._resize_internal_source_map(
            base_logits, diag_shape)
        if final_map is None:
            return None
        requested_ranks = sorted({
            int(round(value))
            for value in self._parse_float_list(
                self.candidate_residual_miou_ranks, '2')
            if 2 <= int(round(value)) <= self.num_cls
        })
        if not requested_ranks:
            return None
        topk = max(requested_ranks)
        final_topk = torch.topk(
            torch.nan_to_num(final_map, nan=-1e6),
            k=topk,
            dim=0,
        )
        candidate_stack = final_topk.indices
        final_top1 = candidate_stack[0]
        pred_idx = pred_diag.clamp(0, self.num_cls - 1)
        threshold_reject = (
            pred_idx != final_top1) & valid_diag
        units = set(self._parse_name_list(
            self.candidate_residual_miou_units))
        unknown_units = units - {'pixel', 'component', 'raw_mask'}
        if unknown_units:
            raise ValueError(
                'candidate_residual_miou_units supports pixel, '
                f'component, and raw_mask, got {sorted(unknown_units)}')
        requested_scores = set(self._parse_name_list(
            self.candidate_residual_miou_scores))
        rows = []
        for sequence in self._candidate_residual_source_sequences(
                components):
            if sequence['control_name'] != 'observed':
                continue
            trajectory = self._candidate_residual_trajectory_scores(
                sequence['maps'],
                sequence['alphas'],
                candidate_stack,
                pred_idx,
                (
                    (candidate_stack != pred_idx.unsqueeze(0))
                    & valid_diag.unsqueeze(0)
                ),
            )
            if trajectory is None:
                continue
            for rank in requested_ranks:
                rank_idx = rank - 1
                candidate_idx = candidate_stack[rank_idx]
                eligible = (
                    trajectory['valid'][rank_idx]
                    & (~threshold_reject)
                    & valid_diag
                )
                transition_table = (
                    self._candidate_residual_miou_transition_table(
                        gt_data,
                        base_pred,
                        valid_mask,
                        candidate_idx,
                    )
                )
                for score_name, score_stack in (
                        trajectory['score_maps'].items()):
                    if (
                            requested_scores
                            and score_name not in requested_scores):
                        continue
                    score = score_stack[rank_idx]
                    common = dict(
                        layer_index=int(sequence['layer_index']),
                        variant_name=str(sequence['variant_name']),
                        score_name=str(score_name),
                        candidate_rank=int(rank),
                        regime_name='argmax_competition',
                    )
                    if 'pixel' in units:
                        rows.extend(
                            self._candidate_residual_miou_gate_rows(
                                score,
                                candidate_idx,
                                eligible,
                                'pixel',
                                {},
                                transition_table,
                                common,
                                'none',
                            )
                        )
                    if 'component' in units:
                        for reducer in (
                                self._candidate_residual_miou_region_reducers()):
                            region_score, region_meta = (
                                self._candidate_residual_miou_region_score_map(
                                    score,
                                    eligible,
                                    pred_idx,
                                    candidate_idx,
                                    reducer,
                                )
                            )
                            rows.extend(
                                self._candidate_residual_miou_gate_rows(
                                    region_score,
                                    candidate_idx,
                                    eligible,
                                    'component',
                                    region_meta,
                                    transition_table,
                                    common,
                                    reducer,
                                )
                            )
                    if 'raw_mask' in units:
                        for reducer in (
                                self._candidate_residual_miou_region_reducers()):
                            raw_score, raw_meta = (
                                self._candidate_residual_miou_raw_score_map(
                                    score,
                                    eligible,
                                    candidate_idx,
                                    components,
                                    reducer,
                                )
                            )
                            rows.extend(
                                self._candidate_residual_miou_gate_rows(
                                    raw_score,
                                    candidate_idx,
                                    eligible,
                                    'raw_mask',
                                    raw_meta,
                                    transition_table,
                                    common,
                                    reducer,
                                )
                            )
        return dict(
            dataset_name=self.seed_dataset_name,
            diagnostic_shape=list(diag_shape),
            valid_pixels=int(valid_mask.sum().item()),
            baseline_confusion=baseline_confusion,
            class_names=list(self.class_names),
            feature_stats=[
                row for row in (
                    components.get(
                        'scene_common_feature_stats', []) or [])
                if row.get('stat_type') == 'layer_descriptor'
            ],
            counterfactual_stats=rows,
        )

    def _write_candidate_residual_miou_stats(self, record):
        if not self.dump_candidate_residual_miou_stats:
            return
        if self._candidate_residual_miou_stats_file is None:
            path = (
                self.candidate_residual_miou_stats_path
                or './work_dirs/evidence_stats/'
                   'candidate_residual_miou_stats.jsonl'
            )
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._candidate_residual_miou_stats_file = open(
                path, 'a', buffering=1)
        self._candidate_residual_miou_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    def _write_candidate_internal_verifier_stats(self, record):
        if not self.dump_candidate_internal_verifier_stats:
            return
        if self._candidate_internal_verifier_stats_file is None:
            path = (
                self.candidate_internal_verifier_stats_path
                or './work_dirs/evidence_stats/'
                   'candidate_internal_verifier_stats.jsonl'
            )
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._candidate_internal_verifier_stats_file = open(
                path, 'a', buffering=1)
        self._candidate_internal_verifier_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    def _write_position_bias_stats(self, record):
        if not self.dump_position_bias_stats:
            return
        if self._position_bias_stats_file is None:
            path = (
                self.position_bias_stats_path
                or './work_dirs/evidence_stats/position_bias_stats.jsonl'
            )
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._position_bias_stats_file = open(
                path, 'a', buffering=1)
        self._position_bias_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    def _write_scene_common_bias_stats(self, record):
        if not self.dump_scene_common_bias_stats:
            return
        if self._scene_common_bias_stats_file is None:
            path = (
                self.scene_common_bias_stats_path
                or './work_dirs/evidence_stats/'
                   'scene_common_bias_stats.jsonl'
            )
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._scene_common_bias_stats_file = open(
                path, 'a', buffering=1)
        self._scene_common_bias_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    def _parse_name_list(self, value):
        if value is None:
            return []
        if isinstance(value, str):
            items = [item.strip().lower() for item in value.split(',')]
        elif isinstance(value, (list, tuple)):
            items = [str(item).strip().lower() for item in value]
        else:
            items = [str(value).strip().lower()]
        return [item for item in items if item]

    def _geoer_reject_recovery_allowed(self):
        dataset = str(self.seed_dataset_name or '').lower()
        allowed_datasets = self._parse_name_list(
            self.geoer_reject_recovery_datasets)
        dataset_allowed = (
            not allowed_datasets
            or 'all' in allowed_datasets
            or dataset in allowed_datasets
        )
        if not dataset_allowed:
            return False
        if not (0 <= int(self.bg_idx) < len(self.class_names)):
            return False
        bg_tokens = _class_name_tokens(self.class_names[int(self.bg_idx)])
        allowed_bg_names = self._parse_name_list(
            self.geoer_reject_recovery_bg_names)
        if not allowed_bg_names or 'all' in allowed_bg_names:
            return True
        return bool(bg_tokens & set(allowed_bg_names))

    def _apply_geoer_reject_recovery(self, logits, pred, components):
        context = dict(
            enabled=bool(self.geoer_use_reject_recovery),
            allowed=bool(self._geoer_reject_recovery_allowed()),
            gate_mask=None,
            gate_stats=None,
        )
        if (not bool(self.geoer_use_reject_recovery)
                or not context['allowed']
                or components is None):
            return pred, context
        semantic_logits = components.get('semantic_logits')
        instance_logits = components.get('instance_logits')
        if semantic_logits is None or instance_logits is None:
            return pred, context

        top2_vals, top2_idx = torch.topk(logits, k=min(2, self.num_cls), dim=0)
        top1_score = top2_vals[0]
        top1_idx = top2_idx[0].clamp(min=0, max=self.num_cls - 1)
        top2_score = top2_vals[1] if self.num_cls > 1 else torch.zeros_like(top1_score)
        final_margin = top1_score - top2_score
        bg_idx = int(self.bg_idx)
        bg_idx_map = torch.full_like(top1_idx, bg_idx)

        top1_semantic = _gather_class_map(semantic_logits, top1_idx)
        top1_instance = _gather_class_map(instance_logits, top1_idx)
        bg_semantic = _gather_class_map(semantic_logits, bg_idx_map)
        bg_instance = _gather_class_map(instance_logits, bg_idx_map)
        semantic_bg_margin = top1_semantic - bg_semantic
        instance_bg_margin = top1_instance - bg_instance
        local_consistency = self._same_class_local_consistency(
            top1_idx,
            self.seed_local_kernel,
        )

        threshold_reject = (
            (pred == bg_idx)
            & (top1_score < float(self.prob_thd))
            & (top1_idx != bg_idx)
        )
        evidence_gate = (
            (top1_score >= float(self.geoer_reject_recovery_score_thd))
            & ((top1_semantic >= float(self.geoer_reject_recovery_semantic_thd))
               | (top1_instance >= float(self.geoer_reject_recovery_instance_thd)))
            & (final_margin >= float(self.geoer_reject_recovery_margin_thd))
            & (local_consistency >= float(self.geoer_reject_recovery_local_thd))
        )
        if bool(self.geoer_reject_recovery_require_bg_margin):
            evidence_gate = evidence_gate & (
                (semantic_bg_margin > 0) | (instance_bg_margin > 0))
        gate_mask = threshold_reject & evidence_gate

        recovered = pred.clone()
        if gate_mask.any():
            recovered[gate_mask] = top1_idx[gate_mask]
        context.update(dict(
            gate_mask=gate_mask,
            gate_stats=dict(
                threshold_reject_pixels=int(threshold_reject.sum().item()),
                gate_pixels=int(gate_mask.sum().item()),
                mean_top1_score=_masked_mean(top1_score, gate_mask),
                mean_final_margin=_masked_mean(final_margin, gate_mask),
                mean_top1_semantic=_masked_mean(top1_semantic, gate_mask),
                mean_top1_instance=_masked_mean(top1_instance, gate_mask),
                mean_semantic_bg_margin=_masked_mean(
                    semantic_bg_margin, gate_mask),
                mean_instance_bg_margin=_masked_mean(
                    instance_bg_margin, gate_mask),
                mean_local_consistency=_masked_mean(
                    local_consistency, gate_mask),
            ),
        ))
        return recovered, context

    def _build_geoer_router_prediction(self, base_logits, base_pred, image,
                                       components):
        routed_logits = base_logits
        context = dict(
            dataset_name=self.seed_dataset_name,
            use_scale_stability=bool(self.geoer_use_scale_stability),
            use_reject_recovery=bool(self.geoer_use_reject_recovery),
            scale_context=None,
            reject_context=None,
        )
        if bool(self.geoer_use_scale_stability):
            routed_logits, scale_context = self._build_scale_stability_rerank_logits(
                base_logits, image)
            context['scale_context'] = scale_context
        routed_pred = self._threshold_with_reject_recovery(routed_logits, components)
        if bool(self.geoer_use_reject_recovery):
            routed_pred, reject_context = self._apply_geoer_reject_recovery(
                routed_logits, routed_pred, components)
            context['reject_context'] = reject_context
        return routed_logits, routed_pred, context

    def _geoer_subset_row(self, route_name, mask, base_pred, routed_pred,
                          gt_data, valid_mask, extra=None):
        mask = mask & valid_mask
        pixels = int(mask.sum().item())
        if pixels == 0:
            row = dict(route_name=route_name, pixels=0)
            if extra:
                row.update(extra)
            return row
        base_correct = (base_pred == gt_data) & valid_mask
        routed_correct = (routed_pred == gt_data) & valid_mask
        changed = (base_pred != routed_pred) & valid_mask
        improved = (~base_correct) & routed_correct
        harmed = base_correct & (~routed_correct)
        row = dict(
            route_name=route_name,
            pixels=pixels,
            changed_pixels=int((changed & mask).sum().item()),
            improved_pixels=int((improved & mask).sum().item()),
            harmed_pixels=int((harmed & mask).sum().item()),
            net_improved_pixels=(
                int((improved & mask).sum().item())
                - int((harmed & mask).sum().item())),
            base_correct_pixels=int((base_correct & mask).sum().item()),
            routed_correct_pixels=int((routed_correct & mask).sum().item()),
        )
        row['changed_ratio'] = _safe_div(row['changed_pixels'], pixels)
        row['improved_ratio'] = _safe_div(row['improved_pixels'], pixels)
        row['harmed_ratio'] = _safe_div(row['harmed_pixels'], pixels)
        row['net_improved_ratio'] = _safe_div(row['net_improved_pixels'], pixels)
        if extra:
            row.update(extra)
        return row

    def _build_geoer_router_stats(self, base_logits, routed_logits, base_pred,
                                  routed_pred, data_sample, context):
        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)
        if gt_data is None or context is None:
            return None

        if base_pred.shape[-2:] != gt_data.shape[-2:]:
            base_pred = F.interpolate(
                base_pred.float().unsqueeze(0).unsqueeze(0),
                size=gt_data.shape[-2:],
                mode='nearest').squeeze(0).squeeze(0).long()
        if routed_pred.shape[-2:] != gt_data.shape[-2:]:
            routed_pred = F.interpolate(
                routed_pred.float().unsqueeze(0).unsqueeze(0),
                size=gt_data.shape[-2:],
                mode='nearest').squeeze(0).squeeze(0).long()

        valid_mask = gt_data != 255
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return dict(valid_pixels=0, overall=None, route_stats=[], pair_stats=[])

        all_row = self._geoer_subset_row(
            'all_valid',
            valid_mask,
            base_pred,
            routed_pred,
            gt_data,
            valid_mask,
        )
        all_row['valid_pixels'] = valid_count

        route_stats = []
        scale_context = context.get('scale_context') or {}
        for row in scale_context.get('pair_stats') or []:
            gate_mask = row.get('gate_mask')
            if gate_mask is None:
                continue
            target_class = int(row.get('target_class_index'))
            competitor_class = int(row.get('competitor_class_index'))
            route_stats.append(self._geoer_subset_row(
                f"scale:{row.get('target_class_name')}->{row.get('competitor_class_name')}",
                gate_mask,
                base_pred,
                routed_pred,
                gt_data,
                valid_mask,
                extra=dict(
                    route_type='scale_stability',
                    target_class_index=target_class,
                    target_class_name=row.get('target_class_name'),
                    competitor_class_index=competitor_class,
                    competitor_class_name=row.get('competitor_class_name'),
                    candidate_pixels=row.get('candidate_pixels'),
                    gate_base_margin_sum=row.get('gate_base_margin_sum'),
                    gate_scaled_margin_sum=row.get('gate_scaled_margin_sum'),
                    gate_margin_gain_sum=row.get('gate_margin_gain_sum'),
                )))
        reject_context = context.get('reject_context') or {}
        reject_gate = reject_context.get('gate_mask')
        if reject_gate is not None:
            extra = dict(route_type='reject_recovery')
            extra.update(reject_context.get('gate_stats') or {})
            route_stats.append(self._geoer_subset_row(
                'reject_recovery',
                reject_gate,
                base_pred,
                routed_pred,
                gt_data,
                valid_mask,
                extra=extra))

        pair_stats = []
        base_wrong = (base_pred != gt_data) & valid_mask
        for gt_class in range(self.num_cls):
            gt_mask = (gt_data == gt_class) & valid_mask
            if not gt_mask.any():
                continue
            for pred_class in range(self.num_cls):
                if pred_class == gt_class:
                    continue
                pair_mask = base_wrong & (gt_data == gt_class) & (base_pred == pred_class)
                if not pair_mask.any():
                    continue
                pair_stats.append(self._geoer_subset_row(
                    f'{self.class_names[gt_class]}->{self.class_names[pred_class]}',
                    pair_mask,
                    base_pred,
                    routed_pred,
                    gt_data,
                    valid_mask,
                    extra=dict(
                        gt_class_index=gt_class,
                        gt_class_name=self.class_names[gt_class],
                        base_pred_class_index=pred_class,
                        base_pred_class_name=self.class_names[pred_class],
                    )))
        pair_stats.sort(key=lambda item: item.get('pixels', 0), reverse=True)
        return dict(
            dataset_name=self.seed_dataset_name,
            valid_pixels=valid_count,
            use_scale_stability=bool(context.get('use_scale_stability')),
            use_reject_recovery=bool(context.get('use_reject_recovery')),
            overall=all_row,
            route_stats=route_stats,
            pair_stats=pair_stats,
        )

    def _write_geoer_router_stats(self, record):
        if not self.dump_geoer_router_stats:
            return
        if self._geoer_router_stats_file is None:
            path = (
                self.geoer_router_stats_path
                or './work_dirs/evidence_stats/geoer_router_stats.jsonl'
            )
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._geoer_router_stats_file = open(path, 'a', buffering=1)
        self._geoer_router_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    def _build_geometry_context_stats(self, base_logits, base_pred, data_sample,
                                      components):
        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)
        if gt_data is None or components is None:
            return None
        semantic_logits = components.get('semantic_logits')
        instance_logits = components.get('instance_logits')
        if semantic_logits is None or instance_logits is None:
            return None

        valid_mask = gt_data != 255
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return dict(valid_pixels=0, class_stats=[], pair_stats=[])

        pred_idx = base_pred.clamp(min=0, max=self.num_cls - 1)
        base_correct = (base_pred == gt_data) & valid_mask
        base_wrong = (~base_correct) & valid_mask
        wrong_count = int(base_wrong.sum().item())

        seed_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        seed_class = pred_idx
        seed_context = self._build_seed_rule_context(base_logits, base_pred, components)
        if seed_context is not None:
            seed_mask = seed_context['rule_defs'][-1][1] & valid_mask
            seed_class = seed_context['final_top1_idx'].clamp(min=0, max=self.num_cls - 1)

        support_masks = []
        pred_masks = []
        seed_masks = []
        class_geom = []
        class_stats = []
        score_thd = float(self.geometry_context_score_thd)

        for class_idx in range(self.num_cls):
            gt_mask = (gt_data == class_idx) & valid_mask
            support_mask = (base_logits[class_idx] >= score_thd) & valid_mask
            pred_mask = (pred_idx == class_idx) & valid_mask
            class_seed = seed_mask & (seed_class == class_idx)
            support_masks.append(support_mask)
            pred_masks.append(pred_mask)
            seed_masks.append(class_seed)

            support_geom = self._geometry_context_mask_stats(
                support_mask,
                valid_count,
                score_map=base_logits[class_idx],
                semantic_map=semantic_logits[class_idx],
                instance_map=instance_logits[class_idx])
            pred_geom = self._geometry_context_mask_stats(
                pred_mask,
                valid_count,
                score_map=base_logits[class_idx],
                semantic_map=semantic_logits[class_idx],
                instance_map=instance_logits[class_idx])
            seed_geom = self._geometry_context_mask_stats(
                class_seed,
                valid_count,
                score_map=base_logits[class_idx],
                semantic_map=semantic_logits[class_idx],
                instance_map=instance_logits[class_idx])
            class_geom.append(dict(
                support=support_geom,
                pred=pred_geom,
                seed=seed_geom,
            ))

            support_pixels = int(support_mask.sum().item())
            pred_pixels = int(pred_mask.sum().item())
            seed_pixels = int(class_seed.sum().item())
            gt_pixels = int(gt_mask.sum().item())
            support_correct = int((support_mask & gt_mask).sum().item())
            pred_correct = int((pred_mask & gt_mask).sum().item())
            seed_correct = int((class_seed & gt_mask).sum().item())
            row = dict(
                class_index=class_idx,
                class_name=self.class_names[class_idx],
                gt_pixels=gt_pixels,
                support_pixels=support_pixels,
                pred_pixels=pred_pixels,
                seed_pixels=seed_pixels,
                support_precision=_safe_div(support_correct, support_pixels),
                support_recall=_safe_div(support_correct, gt_pixels),
                pred_precision=_safe_div(pred_correct, pred_pixels),
                pred_recall=_safe_div(pred_correct, gt_pixels),
                seed_purity=_safe_div(seed_correct, seed_pixels),
                seed_recall=_safe_div(seed_correct, gt_pixels),
            )
            row.update(_prefix_geometry_stats('support', support_geom))
            row.update(_prefix_geometry_stats('pred', pred_geom))
            row.update(_prefix_geometry_stats('seed', seed_geom))
            class_stats.append(row)

        pair_stats = []
        for gt_class in range(self.num_cls):
            gt_wrong_mask = base_wrong & (gt_data == gt_class)
            gt_wrong_pixels = int(gt_wrong_mask.sum().item())
            if gt_wrong_pixels == 0:
                continue
            for pred_class in range(self.num_cls):
                if pred_class == gt_class:
                    continue
                pair_mask = gt_wrong_mask & (pred_idx == pred_class)
                pixels = int(pair_mask.sum().item())
                if pixels == 0:
                    continue

                gt_support = support_masks[gt_class]
                pred_support = support_masks[pred_class]
                gt_seed = seed_masks[gt_class]
                pred_seed = seed_masks[pred_class]
                gt_support_geom = class_geom[gt_class]['support']
                pred_support_geom = class_geom[pred_class]['support']
                gt_pred_geom = class_geom[gt_class]['pred']
                pred_pred_geom = class_geom[pred_class]['pred']
                gt_seed_geom = class_geom[gt_class]['seed']
                pred_seed_geom = class_geom[pred_class]['seed']

                row = dict(
                    gt_class_index=gt_class,
                    gt_class_name=self.class_names[gt_class],
                    base_pred_class_index=pred_class,
                    base_pred_class_name=self.class_names[pred_class],
                    pixels=pixels,
                    pixel_ratio_in_error=_safe_div(pixels, wrong_count),
                    pixel_ratio_in_gt_wrong=_safe_div(pixels, gt_wrong_pixels),
                    gt_support_on_pair_pixels=int((pair_mask & gt_support).sum().item()),
                    pred_support_on_pair_pixels=int((pair_mask & pred_support).sum().item()),
                    gt_seed_on_pair_pixels=int((pair_mask & gt_seed).sum().item()),
                    pred_seed_on_pair_pixels=int((pair_mask & pred_seed).sum().item()),
                    gt_score_on_pair_sum=_masked_sum(base_logits[gt_class], pair_mask),
                    pred_score_on_pair_sum=_masked_sum(base_logits[pred_class], pair_mask),
                    gt_semantic_on_pair_sum=_masked_sum(semantic_logits[gt_class], pair_mask),
                    pred_semantic_on_pair_sum=_masked_sum(semantic_logits[pred_class], pair_mask),
                    gt_instance_on_pair_sum=_masked_sum(instance_logits[gt_class], pair_mask),
                    pred_instance_on_pair_sum=_masked_sum(instance_logits[pred_class], pair_mask),
                )
                row.update(_prefix_geometry_stats('gt_support', gt_support_geom))
                row.update(_prefix_geometry_stats('pred_support', pred_support_geom))
                row.update(_prefix_geometry_stats('gt_pred_region', gt_pred_geom))
                row.update(_prefix_geometry_stats('pred_pred_region', pred_pred_geom))
                row.update(_prefix_geometry_stats('gt_seed', gt_seed_geom))
                row.update(_prefix_geometry_stats('pred_seed', pred_seed_geom))
                row.update(self._geometry_context_pair_margins(
                    gt_support_geom, pred_support_geom))
                pair_stats.append(row)

        pair_stats.sort(key=lambda item: item['pixels'], reverse=True)
        return dict(
            dataset_name=self.seed_dataset_name,
            valid_pixels=valid_count,
            baseline_correct_pixels=int(base_correct.sum().item()),
            baseline_wrong_pixels=wrong_count,
            geometry_context_score_thd=float(self.geometry_context_score_thd),
            geometry_context_min_pixels=int(self.geometry_context_min_pixels),
            geometry_context_core_kernel=int(self.geometry_context_core_kernel),
            seed_rule_for_geometry='final_score_margin_sem_inst_final_agree_local_core',
            class_stats=class_stats,
            pair_stats=pair_stats,
        )

    def _geometry_context_mask_stats(self, mask, valid_count, score_map=None,
                                     semantic_map=None, instance_map=None):
        mask = mask.bool()
        mask_pixels = int(mask.sum().item())
        if mask_pixels == 0:
            return dict(mask_pixels=0)
        core_mask = _binary_erode(mask, int(self.geometry_context_core_kernel))
        boundary_mask = mask & (~core_mask)
        component_stats = _connected_component_summary(
            mask,
            mask,
            min_pixels=int(self.geometry_context_min_pixels),
            purity_threshold=1.0,
        )
        region_count = component_stats.get('region_count', 0)
        return dict(
            mask_pixels=mask_pixels,
            mask_ratio=_safe_div(mask_pixels, valid_count),
            bbox_fill_ratio=_bbox_fill_ratio(mask, mask_pixels),
            core_pixels=int(core_mask.sum().item()),
            core_ratio=_safe_div(int(core_mask.sum().item()), mask_pixels),
            boundary_pixels=int(boundary_mask.sum().item()),
            boundary_ratio=_safe_div(int(boundary_mask.sum().item()), mask_pixels),
            component_count=region_count,
            component_density=_safe_div(float(region_count) * 1000.0, mask_pixels),
            mean_component_area=component_stats.get('mean_region_area'),
            max_component_area=component_stats.get('max_region_area'),
            min_component_area=component_stats.get('min_region_area'),
            score_mean=_masked_mean(score_map, mask),
            semantic_mean=_masked_mean(semantic_map, mask),
            instance_mean=_masked_mean(instance_map, mask),
        )

    def _geometry_context_pair_margins(self, gt_geom, pred_geom):
        fields = [
            'mask_ratio',
            'bbox_fill_ratio',
            'core_ratio',
            'boundary_ratio',
            'component_density',
            'mean_component_area',
            'max_component_area',
            'score_mean',
            'semantic_mean',
            'instance_mean',
        ]
        row = {}
        for field in fields:
            gt_value = gt_geom.get(field)
            pred_value = pred_geom.get(field)
            row[f'{field}_gt_minus_pred'] = (
                None if gt_value is None or pred_value is None
                else float(gt_value) - float(pred_value))
        return row

    def _write_geometry_context_stats(self, record):
        if not self.dump_geometry_context_stats:
            return
        if self._geometry_context_stats_file is None:
            path = (self.geometry_context_stats_path
                    or './work_dirs/evidence_stats/geometry_context_stats.jsonl')
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._geometry_context_stats_file = open(path, 'a', buffering=1)
        self._geometry_context_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    def _get_weak_support_thresholds(self):
        if isinstance(self.weak_support_thresholds, str):
            values = [
                item.strip()
                for item in self.weak_support_thresholds.split(',')
                if item.strip()
            ]
        elif isinstance(self.weak_support_thresholds, (list, tuple)):
            values = self.weak_support_thresholds
        else:
            values = [self.weak_support_thresholds]
        thresholds = []
        for value in values:
            thresholds.append(float(value))
        thresholds = sorted(set(thresholds))
        if not thresholds:
            thresholds = [0.5]
        return thresholds

    def _get_weak_support_sources(self):
        if isinstance(self.weak_support_sources, str):
            values = [
                item.strip().lower()
                for item in self.weak_support_sources.split(',')
                if item.strip()
            ]
        elif isinstance(self.weak_support_sources, (list, tuple)):
            values = [str(item).strip().lower() for item in self.weak_support_sources]
        else:
            values = ['final', 'semantic', 'instance', 'any_head']
        allowed = {'final', 'semantic', 'instance', 'any_head'}
        sources = []
        for value in values:
            if value not in allowed:
                raise ValueError(
                    "weak_support_sources must contain only "
                    "'final', 'semantic', 'instance', or 'any_head', "
                    f"but got {value!r}")
            if value not in sources:
                sources.append(value)
        return sources or ['final']

    def _build_weak_support_stats(self, base_logits, base_pred, data_sample,
                                  components):
        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)
        if gt_data is None:
            return None

        valid_mask = gt_data != 255
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return dict(valid_pixels=0, class_stats=[], pair_stats=[])

        semantic_logits = None if components is None else components.get('semantic_logits')
        instance_logits = None if components is None else components.get('instance_logits')
        support_maps = {'final': base_logits}
        if semantic_logits is not None:
            support_maps['semantic'] = semantic_logits
        if instance_logits is not None:
            support_maps['instance'] = instance_logits
        if semantic_logits is not None and instance_logits is not None:
            support_maps['any_head'] = torch.maximum(
                base_logits,
                torch.maximum(semantic_logits, instance_logits))
        elif semantic_logits is not None:
            support_maps['any_head'] = torch.maximum(base_logits, semantic_logits)
        elif instance_logits is not None:
            support_maps['any_head'] = torch.maximum(base_logits, instance_logits)
        else:
            support_maps['any_head'] = base_logits

        thresholds = self._get_weak_support_thresholds()
        sources = [
            source for source in self._get_weak_support_sources()
            if source in support_maps
        ]
        pred_idx = base_pred.clamp(min=0, max=self.num_cls - 1)
        base_correct = (base_pred == gt_data) & valid_mask
        base_wrong = (~base_correct) & valid_mask
        wrong_count = int(base_wrong.sum().item())

        pair_masks = []
        for gt_class in range(self.num_cls):
            gt_wrong_mask = base_wrong & (gt_data == gt_class)
            gt_wrong_pixels = int(gt_wrong_mask.sum().item())
            if gt_wrong_pixels == 0:
                continue
            for pred_class in range(self.num_cls):
                if pred_class == gt_class:
                    continue
                pair_mask = gt_wrong_mask & (pred_idx == pred_class)
                pixels = int(pair_mask.sum().item())
                if pixels < int(self.weak_support_min_pair_pixels):
                    continue
                pair_masks.append(dict(
                    gt_class=gt_class,
                    pred_class=pred_class,
                    mask=pair_mask,
                    pixels=pixels,
                    gt_wrong_pixels=gt_wrong_pixels,
                ))

        class_stats = []
        pair_stats = []
        for source in sources:
            score_map = support_maps[source]
            for thd in thresholds:
                support = score_map >= float(thd)
                for class_idx in range(self.num_cls):
                    gt_mask = (gt_data == class_idx) & valid_mask
                    support_mask = support[class_idx] & valid_mask
                    gt_pixels = int(gt_mask.sum().item())
                    support_pixels = int(support_mask.sum().item())
                    correct_pixels = int((support_mask & gt_mask).sum().item())
                    class_stats.append(dict(
                        source=source,
                        threshold=float(thd),
                        class_index=class_idx,
                        class_name=self.class_names[class_idx],
                        gt_pixels=gt_pixels,
                        support_pixels=support_pixels,
                        support_correct_pixels=correct_pixels,
                        support_precision=_safe_div(correct_pixels, support_pixels),
                        support_recall=_safe_div(correct_pixels, gt_pixels),
                        support_ratio=_safe_div(support_pixels, valid_count),
                        mean_gt_score=_masked_mean(score_map[class_idx], gt_mask),
                    ))

                for item in pair_masks:
                    gt_class = item['gt_class']
                    pred_class = item['pred_class']
                    pair_mask = item['mask']
                    pixels = item['pixels']
                    gt_support = support[gt_class]
                    pred_support = support[pred_class]
                    gt_on_pair = pair_mask & gt_support
                    pred_on_pair = pair_mask & pred_support
                    both_on_pair = gt_on_pair & pred_support
                    gt_only_on_pair = gt_on_pair & (~pred_support)
                    pred_only_on_pair = pred_on_pair & (~gt_support)
                    neither_on_pair = pair_mask & (~gt_support) & (~pred_support)
                    gt_score = score_map[gt_class]
                    pred_score = score_map[pred_class]
                    margin = gt_score - pred_score
                    pair_stats.append(dict(
                        source=source,
                        threshold=float(thd),
                        gt_class_index=gt_class,
                        gt_class_name=self.class_names[gt_class],
                        base_pred_class_index=pred_class,
                        base_pred_class_name=self.class_names[pred_class],
                        pixels=pixels,
                        pixel_ratio_in_error=_safe_div(pixels, wrong_count),
                        pixel_ratio_in_gt_wrong=_safe_div(pixels, item['gt_wrong_pixels']),
                        gt_support_pixels=int(gt_on_pair.sum().item()),
                        pred_support_pixels=int(pred_on_pair.sum().item()),
                        both_support_pixels=int(both_on_pair.sum().item()),
                        gt_only_support_pixels=int(gt_only_on_pair.sum().item()),
                        pred_only_support_pixels=int(pred_only_on_pair.sum().item()),
                        neither_support_pixels=int(neither_on_pair.sum().item()),
                        gt_score_sum=_masked_sum(gt_score, pair_mask),
                        pred_score_sum=_masked_sum(pred_score, pair_mask),
                        margin_sum=_masked_sum(margin, pair_mask),
                        gt_beats_pred_pixels=int(((margin > 0) & pair_mask).sum().item()),
                    ))

        pair_stats.sort(key=lambda item: item['pixels'], reverse=True)
        return dict(
            dataset_name=self.seed_dataset_name,
            valid_pixels=valid_count,
            baseline_correct_pixels=int(base_correct.sum().item()),
            baseline_wrong_pixels=wrong_count,
            weak_support_thresholds=thresholds,
            weak_support_sources=sources,
            weak_support_min_pair_pixels=int(self.weak_support_min_pair_pixels),
            class_stats=class_stats,
            pair_stats=pair_stats,
        )

    def _write_weak_support_stats(self, record):
        if not self.dump_weak_support_stats:
            return
        if self._weak_support_stats_file is None:
            path = (self.weak_support_stats_path
                    or './work_dirs/evidence_stats/weak_support_stats.jsonl')
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._weak_support_stats_file = open(path, 'a', buffering=1)
        self._weak_support_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    def _parse_float_list(self, value, default):
        if value is None:
            value = default
        if isinstance(value, str):
            items = [item.strip() for item in value.split(',') if item.strip()]
        elif isinstance(value, (list, tuple)):
            items = value
        else:
            items = [value]
        parsed = []
        for item in items:
            parsed.append(float(item))
        return sorted(set(parsed))

    def _get_reject_recovery_precision_pe_spaces(self):
        spaces = self.reject_recovery_precision_pe_spaces
        if spaces is None:
            return []
        if isinstance(spaces, str):
            spaces = [item.strip() for item in spaces.split(',') if item.strip()]
        elif isinstance(spaces, (list, tuple)):
            spaces = [str(item).strip() for item in spaces if str(item).strip()]
        else:
            spaces = [str(spaces).strip()]
        return [
            space for space in spaces
            if space.startswith('pe_layer_') or space == 'vision'
        ]

    def _build_reject_recovery_precision_stats(self, base_logits, base_pred,
                                               data_sample, components):
        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)
        if gt_data is None or components is None:
            return None
        semantic_logits = components.get('semantic_logits')
        instance_logits = components.get('instance_logits')
        if semantic_logits is None or instance_logits is None:
            return None

        valid_mask = gt_data != 255
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return dict(valid_pixels=0, gate_stats=[], band_stats=[],
                        pred_class_stats=[], gt_class_stats=[])

        top2_vals, top2_idx = torch.topk(base_logits, k=min(2, self.num_cls), dim=0)
        top1_score = top2_vals[0]
        top1_idx = top2_idx[0].clamp(min=0, max=self.num_cls - 1)
        if self.num_cls > 1:
            top2_score = top2_vals[1]
        else:
            top2_score = torch.zeros_like(top1_score)
        final_margin = top1_score - top2_score

        threshold_reject = (
            valid_mask
            & (base_pred == int(self.bg_idx))
            & (top1_score < float(self.prob_thd))
            & (top1_idx != int(self.bg_idx))
        )
        candidate_pixels = int(threshold_reject.sum().item())
        if candidate_pixels == 0:
            return dict(
                dataset_name=self.seed_dataset_name,
                valid_pixels=valid_count,
                threshold_reject_candidate_pixels=0,
                gate_stats=[],
                band_stats=[],
                pred_class_stats=[],
                gt_class_stats=[],
            )

        gt_idx = gt_data.clamp(min=0, max=self.num_cls - 1)
        recovery_correct = threshold_reject & (top1_idx == gt_idx)
        recovery_true_bg = threshold_reject & (gt_idx == int(self.bg_idx))
        recovery_true_fg = threshold_reject & (gt_idx != int(self.bg_idx))

        top1_semantic = _gather_class_map(semantic_logits, top1_idx)
        top1_instance = _gather_class_map(instance_logits, top1_idx)
        bg_idx_map = torch.full_like(top1_idx, int(self.bg_idx))
        bg_semantic = _gather_class_map(semantic_logits, bg_idx_map)
        bg_instance = _gather_class_map(instance_logits, bg_idx_map)
        semantic_bg_margin = top1_semantic - bg_semantic
        instance_bg_margin = top1_instance - bg_instance
        head_agreement = torch.minimum(top1_semantic, top1_instance)
        local_consistency = self._same_class_local_consistency(
            top1_idx,
            self.seed_local_kernel,
        )

        pe_margins = {}
        if bool(self.reject_recovery_precision_use_pe):
            seed_context = self._build_seed_rule_context(
                base_logits, base_pred, components)
            if seed_context is not None:
                seed_mask = seed_context['rule_defs'][-1][1] & valid_mask
                seed_class = seed_context['final_top1_idx'].clamp(
                    min=0, max=self.num_cls - 1)
                for space in self._get_reject_recovery_precision_pe_spaces():
                    sim_maps, has_seed = self._build_seed_similarity_maps_by_space(
                        space, base_logits, components, seed_mask, seed_class)
                    if sim_maps is None or has_seed is None:
                        continue
                    if not bool(has_seed[int(self.bg_idx)].item()):
                        continue
                    top1_has_seed = has_seed[top1_idx] & threshold_reject
                    if not top1_has_seed.any():
                        continue
                    top1_sim = _gather_class_map(sim_maps, top1_idx)
                    bg_sim = _gather_class_map(sim_maps, bg_idx_map)
                    pe_margins[space] = dict(
                        margin=top1_sim - bg_sim,
                        available=top1_has_seed,
                    )

        feature_maps = dict(
            top1_score=top1_score,
            final_margin=final_margin,
            top1_semantic=top1_semantic,
            top1_instance=top1_instance,
            semantic_bg_margin=semantic_bg_margin,
            instance_bg_margin=instance_bg_margin,
            head_agreement=head_agreement,
            local_consistency=local_consistency,
        )

        def row_for_mask(name, mask, group_type='gate', extra=None):
            mask = mask & threshold_reject
            pixels = int(mask.sum().item())
            row = dict(
                name=name,
                group_type=group_type,
                pixels=pixels,
                pixel_ratio_in_candidates=_safe_div(pixels, candidate_pixels),
                correct_pixels=int((mask & recovery_correct).sum().item()),
                precision=_safe_div(int((mask & recovery_correct).sum().item()), pixels),
                true_foreground_pixels=int((mask & recovery_true_fg).sum().item()),
                true_foreground_ratio=_safe_div(
                    int((mask & recovery_true_fg).sum().item()), pixels),
                true_background_pixels=int((mask & recovery_true_bg).sum().item()),
                true_background_ratio=_safe_div(
                    int((mask & recovery_true_bg).sum().item()), pixels),
                mean_top1_score=_masked_mean(top1_score, mask),
                mean_final_margin=_masked_mean(final_margin, mask),
                mean_top1_semantic=_masked_mean(top1_semantic, mask),
                mean_top1_instance=_masked_mean(top1_instance, mask),
                mean_semantic_bg_margin=_masked_mean(semantic_bg_margin, mask),
                mean_instance_bg_margin=_masked_mean(instance_bg_margin, mask),
                mean_head_agreement=_masked_mean(head_agreement, mask),
                mean_local_consistency=_masked_mean(local_consistency, mask),
            )
            for space, pe_info in pe_margins.items():
                available = pe_info['available'] & mask
                row[f'{space}_available_pixels'] = int(available.sum().item())
                row[f'{space}_available_ratio'] = _safe_div(
                    int(available.sum().item()), pixels)
                row[f'{space}_mean_margin'] = _masked_mean(
                    pe_info['margin'], available)
                row[f'{space}_positive_pixels'] = int(
                    ((pe_info['margin'] > 0) & available).sum().item())
                row[f'{space}_positive_ratio'] = _safe_div(
                    row[f'{space}_positive_pixels'],
                    int(available.sum().item()))
            if extra:
                row.update(extra)
            return row

        gate_stats = [row_for_mask('all_threshold_reject', threshold_reject)]
        score_bins = self._parse_float_list(
            self.reject_recovery_precision_score_bins,
            '0.00,0.03,0.05,0.07,0.10,0.15,0.20,0.30')
        semantic_thds = self._parse_float_list(
            self.reject_recovery_precision_semantic_thds,
            '0.03,0.05,0.10')
        instance_thds = self._parse_float_list(
            self.reject_recovery_precision_instance_thds,
            '0.01,0.03,0.05')
        margin_thds = self._parse_float_list(
            self.reject_recovery_precision_margin_thds,
            '0.00,0.02,0.05')
        local_thds = self._parse_float_list(
            self.reject_recovery_precision_local_thds,
            '0.50,0.70,0.85')

        for thd in score_bins:
            gate_stats.append(row_for_mask(
                f'score_ge_{thd:g}',
                top1_score >= float(thd),
                extra=dict(top1_score_thd=float(thd))))
        for thd in semantic_thds:
            gate_stats.append(row_for_mask(
                f'semantic_ge_{thd:g}',
                top1_semantic >= float(thd),
                extra=dict(semantic_thd=float(thd))))
        for thd in instance_thds:
            gate_stats.append(row_for_mask(
                f'instance_ge_{thd:g}',
                top1_instance >= float(thd),
                extra=dict(instance_thd=float(thd))))
        for thd in margin_thds:
            gate_stats.append(row_for_mask(
                f'margin_ge_{thd:g}',
                final_margin >= float(thd),
                extra=dict(final_margin_thd=float(thd))))
        for thd in local_thds:
            gate_stats.append(row_for_mask(
                f'local_ge_{thd:g}',
                local_consistency >= float(thd),
                extra=dict(local_thd=float(thd))))

        for sem_thd in semantic_thds:
            for inst_thd in instance_thds:
                sem = top1_semantic >= float(sem_thd)
                inst = top1_instance >= float(inst_thd)
                gate_stats.append(row_for_mask(
                    f'sem_or_inst_ge_{sem_thd:g}_{inst_thd:g}',
                    sem | inst,
                    extra=dict(semantic_thd=float(sem_thd),
                               instance_thd=float(inst_thd))))
                gate_stats.append(row_for_mask(
                    f'sem_and_inst_ge_{sem_thd:g}_{inst_thd:g}',
                    sem & inst,
                    extra=dict(semantic_thd=float(sem_thd),
                               instance_thd=float(inst_thd))))

        combined_base = (
            (top1_score >= 0.03)
            & ((top1_semantic >= 0.03) | (top1_instance >= 0.01))
        )
        gate_stats.append(row_for_mask(
            'score_ge_0p03_sem_or_inst',
            combined_base))
        gate_stats.append(row_for_mask(
            'score_ge_0p03_sem_or_inst_local_ge_0p70',
            combined_base & (local_consistency >= 0.70)))
        gate_stats.append(row_for_mask(
            'score_ge_0p03_sem_or_inst_sem_bg_pos',
            combined_base & (semantic_bg_margin > 0)))
        gate_stats.append(row_for_mask(
            'score_ge_0p03_sem_or_inst_inst_bg_pos',
            combined_base & (instance_bg_margin > 0)))
        gate_stats.append(row_for_mask(
            'score_ge_0p03_sem_or_inst_no_bg_dominance',
            combined_base
            & ((semantic_bg_margin > 0) | (instance_bg_margin > 0))))
        for space, pe_info in pe_margins.items():
            pe_positive = pe_info['available'] & (pe_info['margin'] > 0)
            gate_stats.append(row_for_mask(
                f'{space}_pe_margin_pos',
                pe_positive,
                extra=dict(pe_space=space)))
            gate_stats.append(row_for_mask(
                f'combined_{space}_pe_margin_pos',
                combined_base & pe_positive,
                extra=dict(pe_space=space)))

        band_stats = []
        for idx, lo in enumerate(score_bins):
            hi = score_bins[idx + 1] if idx + 1 < len(score_bins) else None
            if hi is None:
                band_mask = top1_score >= lo
                name = f'score_band_ge_{lo:g}'
            else:
                band_mask = (top1_score >= lo) & (top1_score < hi)
                name = f'score_band_{lo:g}_{hi:g}'
            band_stats.append(row_for_mask(
                name,
                band_mask,
                group_type='score_band',
                extra=dict(score_lo=float(lo), score_hi=hi)))

        pred_class_stats = []
        gt_class_stats = []
        for class_idx in range(self.num_cls):
            pred_mask = threshold_reject & (top1_idx == class_idx)
            if pred_mask.any():
                pred_class_stats.append(row_for_mask(
                    f'pred_{self.class_names[class_idx]}',
                    pred_mask,
                    group_type='pred_class',
                    extra=dict(class_index=class_idx,
                               class_name=self.class_names[class_idx])))
            gt_mask = threshold_reject & (gt_idx == class_idx)
            if gt_mask.any():
                gt_class_stats.append(row_for_mask(
                    f'gt_{self.class_names[class_idx]}',
                    gt_mask,
                    group_type='gt_class',
                    extra=dict(class_index=class_idx,
                               class_name=self.class_names[class_idx])))

        return dict(
            dataset_name=self.seed_dataset_name,
            valid_pixels=valid_count,
            threshold_reject_candidate_pixels=candidate_pixels,
            prob_thd=float(self.prob_thd),
            bg_idx=int(self.bg_idx),
            score_bins=score_bins,
            semantic_thds=semantic_thds,
            instance_thds=instance_thds,
            margin_thds=margin_thds,
            local_thds=local_thds,
            pe_spaces=list(pe_margins.keys()),
            gate_stats=gate_stats,
            band_stats=band_stats,
            pred_class_stats=pred_class_stats,
            gt_class_stats=gt_class_stats,
        )

    def _write_reject_recovery_precision_stats(self, record):
        if not self.dump_reject_recovery_precision_stats:
            return
        if self._reject_recovery_precision_stats_file is None:
            path = (
                self.reject_recovery_precision_stats_path
                or './work_dirs/evidence_stats/reject_recovery_precision_stats.jsonl'
            )
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._reject_recovery_precision_stats_file = open(
                path, 'a', buffering=1)
        self._reject_recovery_precision_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    def _get_coco_sec_query_text_features(self):
        if self._coco_sec_query_text_features is not None:
            return self._coco_sec_query_text_features
        with torch.no_grad():
            text_outputs = self.processor.model.backbone.forward_text(
                list(self.query_words),
                device=self.device,
            )
            text_features = text_outputs['language_features'].detach().float()
            text_mask = text_outputs['language_mask'].detach()
            # language_features: [seq, num_queries, dim], language_mask: [num_queries, seq]
            valid = (~text_mask.bool()).float().permute(1, 0).unsqueeze(-1)
            denom = valid.sum(dim=0).clamp(min=1.0)
            pooled = (text_features * valid).sum(dim=0) / denom
            pooled = F.normalize(pooled, dim=-1)
            self._coco_sec_query_text_features = pooled.to(self.device)
        return self._coco_sec_query_text_features

    def _build_coco_sec_prior(self, components, output_size):
        if components is None:
            return None
        cached_prior = components.get('coco_sec_prior')
        if cached_prior is not None:
            prior = cached_prior.detach().float().to(self.device)
            if prior.shape[-2:] != output_size:
                prior = F.interpolate(
                    prior.unsqueeze(0),
                    size=output_size,
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0)
            prior = prior.clamp_min(float(self.coco_sec_eps))
            return prior / prior.sum(
                dim=0, keepdim=True).clamp_min(float(self.coco_sec_eps))
        vision_features = components.get('vision_features')
        if vision_features is None:
            return None
        if vision_features.ndim == 4:
            vision_features = vision_features.squeeze(0)
        if vision_features.ndim != 3:
            return None
        vision_features = vision_features.detach().float().to(self.device)
        text_features = self._get_coco_sec_query_text_features().float().to(self.device)
        if vision_features.shape[0] != text_features.shape[-1]:
            return None
        vision_features = F.normalize(vision_features, dim=0)
        sim = torch.einsum('chw,qc->qhw', vision_features, text_features)
        sim = sim / max(float(self.coco_sec_temperature), 1e-6)
        class_sim = self._aggregate_query_similarity_to_classes(sim)
        prior = torch.softmax(class_sim, dim=0)
        if prior.shape[-2:] != output_size:
            prior = F.interpolate(
                prior.unsqueeze(0),
                size=output_size,
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
            prior = prior.clamp_min(float(self.coco_sec_eps))
            prior = prior / prior.sum(dim=0, keepdim=True).clamp_min(float(self.coco_sec_eps))
        return prior

    def _aggregate_query_similarity_to_classes(self, query_sim):
        if self.num_cls == self.num_queries:
            return query_sim
        rows = []
        reduce = str(self.coco_sec_synonym_reduce).lower()
        for class_idx in range(self.num_cls):
            mask = self.query_idx == class_idx
            values = query_sim[mask]
            if values.numel() == 0:
                rows.append(torch.full_like(query_sim[0], -1e4))
            elif reduce == 'max':
                rows.append(values.max(dim=0)[0])
            elif reduce == 'mean':
                rows.append(values.mean(dim=0))
            elif reduce == 'logsumexp':
                rows.append(torch.logsumexp(values, dim=0))
            else:
                raise ValueError(
                    "coco_sec_synonym_reduce must be one of "
                    "'logsumexp', 'max', or 'mean', "
                    f"but got {self.coco_sec_synonym_reduce!r}")
        return torch.stack(rows, dim=0)

    def _apply_coco_sec_fusion(self, base_logits, components):
        prior = self._build_coco_sec_prior(components, base_logits.shape[-2:])
        if prior is None:
            return base_logits, None
        eps = float(self.coco_sec_eps)
        structural_prob = base_logits.clamp(min=eps, max=1.0 - eps)
        structural_logit = torch.logit(structural_prob)
        log_prior = torch.log(prior.clamp_min(eps))
        if bool(self.coco_sec_center_prior) and self.num_cls > 0:
            log_prior = log_prior + float(np.log(self.num_cls))
        fused_logit = structural_logit + float(self.coco_sec_lambda) * log_prior
        fused_prob = torch.sigmoid(fused_logit)
        context = dict(
            prior=prior,
            log_prior=log_prior,
            fused_logit=fused_logit,
        )
        return fused_prob, context

    def _build_coco_sec_stats(self, base_logits, fused_logits, base_pred,
                              fused_pred, data_sample, context):
        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)
        if gt_data is None or context is None:
            return None
        prior = context.get('prior')
        if prior is None:
            return None

        valid_mask = gt_data != 255
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return dict(valid_pixels=0, class_stats=[], pair_stats=[])

        gt_idx = gt_data.clamp(min=0, max=self.num_cls - 1)
        base_correct = (base_pred == gt_data) & valid_mask
        fused_correct = (fused_pred == gt_data) & valid_mask
        improved = (~base_correct) & fused_correct & valid_mask
        harmed = base_correct & (~fused_correct) & valid_mask
        base_wrong = (~base_correct) & valid_mask
        wrong_count = int(base_wrong.sum().item())

        class_stats = []
        for class_idx in range(self.num_cls):
            gt_mask = (gt_idx == class_idx) & valid_mask
            gt_pixels = int(gt_mask.sum().item())
            if gt_pixels == 0:
                continue
            class_stats.append(dict(
                class_index=class_idx,
                class_name=self.class_names[class_idx],
                gt_pixels=gt_pixels,
                base_correct_pixels=int((base_correct & gt_mask).sum().item()),
                fused_correct_pixels=int((fused_correct & gt_mask).sum().item()),
                improved_pixels=int((improved & gt_mask).sum().item()),
                harmed_pixels=int((harmed & gt_mask).sum().item()),
                base_recall=_safe_div(int((base_correct & gt_mask).sum().item()), gt_pixels),
                fused_recall=_safe_div(int((fused_correct & gt_mask).sum().item()), gt_pixels),
                mean_prior=_masked_mean(prior[class_idx], gt_mask),
                mean_base_score=_masked_mean(base_logits[class_idx], gt_mask),
                mean_fused_score=_masked_mean(fused_logits[class_idx], gt_mask),
            ))

        pair_stats = []
        for gt_class in range(self.num_cls):
            gt_wrong_mask = base_wrong & (gt_idx == gt_class)
            gt_wrong_pixels = int(gt_wrong_mask.sum().item())
            if gt_wrong_pixels == 0:
                continue
            for pred_class in range(self.num_cls):
                if pred_class == gt_class:
                    continue
                pair_mask = gt_wrong_mask & (base_pred == pred_class)
                pixels = int(pair_mask.sum().item())
                if pixels == 0:
                    continue
                prior_margin = prior[gt_class] - prior[pred_class]
                base_margin = base_logits[gt_class] - base_logits[pred_class]
                fused_margin = fused_logits[gt_class] - fused_logits[pred_class]
                pair_stats.append(dict(
                    gt_class_index=gt_class,
                    gt_class_name=self.class_names[gt_class],
                    base_pred_class_index=pred_class,
                    base_pred_class_name=self.class_names[pred_class],
                    pixels=pixels,
                    pixel_ratio_in_error=_safe_div(pixels, wrong_count),
                    pixel_ratio_in_gt_wrong=_safe_div(pixels, gt_wrong_pixels),
                    improved_pixels=int((improved & pair_mask).sum().item()),
                    improved_ratio=_safe_div(int((improved & pair_mask).sum().item()), pixels),
                    mean_prior_gt=_masked_mean(prior[gt_class], pair_mask),
                    mean_prior_pred=_masked_mean(prior[pred_class], pair_mask),
                    mean_prior_margin=_masked_mean(prior_margin, pair_mask),
                    mean_base_margin=_masked_mean(base_margin, pair_mask),
                    mean_fused_margin=_masked_mean(fused_margin, pair_mask),
                    fused_beats_pred_pixels=int((fused_margin > 0).logical_and(pair_mask).sum().item()),
                    fused_beats_pred_ratio=_safe_div(
                        int((fused_margin > 0).logical_and(pair_mask).sum().item()),
                        pixels),
                ))

        pair_stats.sort(key=lambda item: item['pixels'], reverse=True)
        return dict(
            valid_pixels=valid_count,
            base_correct_pixels=int(base_correct.sum().item()),
            fused_correct_pixels=int(fused_correct.sum().item()),
            improved_pixels=int(improved.sum().item()),
            harmed_pixels=int(harmed.sum().item()),
            coco_sec_lambda=float(self.coco_sec_lambda),
            coco_sec_temperature=float(self.coco_sec_temperature),
            coco_sec_center_prior=bool(self.coco_sec_center_prior),
            coco_sec_synonym_reduce=str(self.coco_sec_synonym_reduce),
            class_stats=class_stats,
            pair_stats=pair_stats,
        )

    def _write_coco_sec_stats(self, record):
        if not self.dump_coco_sec_stats:
            return
        if self._coco_sec_stats_file is None:
            path = self.coco_sec_stats_path or './work_dirs/evidence_stats/coco_sec_stats.jsonl'
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._coco_sec_stats_file = open(path, 'a', buffering=1)
        self._coco_sec_stats_file.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _uses_state_action_atlas(self):
        return (
            bool(self.dump_state_action_atlas_stats)
            or str(self.state_action_apply or 'baseline').lower()
            != 'baseline'
        )

    def _get_state_action_names(self):
        supported = {
            'baseline',
            'presence_sqrt',
            'presence_bounded_sqrt',
            'presence_guarded',
            'semantic_presence',
            'instance_presence',
            'fusion_no_presence',
            'semantic_no_presence',
            'instance_no_presence',
            'prompt_mean',
        }
        names = self._parse_name_list(self.state_action_atlas_actions)
        if not names:
            names = ['baseline']
        if 'baseline' not in names:
            names.insert(0, 'baseline')
        apply_name = str(
            self.state_action_apply or 'baseline').strip().lower()
        if apply_name not in names:
            names.append(apply_name)
        unknown = set(names) - supported
        if unknown:
            raise ValueError(
                'state_action_atlas_actions contains unsupported actions: '
                f'{sorted(unknown)}')
        return names

    def _iter_state_action_logits(
            self, base_logits, query_final, query_semantic,
            query_instance, action_names):
        eps = 1e-6
        query_final = query_final.detach().float()
        query_semantic = query_semantic.detach().float()
        query_instance = query_instance.detach().float()
        for action_name in action_names:
            if action_name == 'baseline':
                yield action_name, base_logits
                continue
            if action_name == 'semantic_no_presence':
                yield action_name, self._aggregate_query_logits_to_classes(
                    query_semantic)
                continue
            if action_name == 'instance_no_presence':
                yield action_name, self._aggregate_query_logits_to_classes(
                    query_instance)
                continue
            if action_name == 'prompt_mean':
                yield action_name, self._aggregate_query_logits_to_classes(
                    query_final, reduce='mean')
                continue

            no_presence = torch.maximum(
                query_semantic, query_instance)
            if action_name == 'fusion_no_presence':
                action_query = no_presence
            else:
                effective_presence = torch.where(
                    no_presence > eps,
                    query_final / no_presence.clamp_min(eps),
                    torch.zeros_like(query_final),
                ).clamp(0.0, 1.0)
                if action_name == 'presence_sqrt':
                    gamma = max(
                        1e-3, float(self.state_action_presence_gamma))
                    action_query = (
                        no_presence * effective_presence.pow(gamma))
                elif action_name == 'presence_bounded_sqrt':
                    gamma = max(
                        1e-3, float(self.state_action_presence_gamma))
                    relaxed = (
                        no_presence * effective_presence.pow(gamma))
                    max_boost = (
                        no_presence
                        * max(
                            0.0,
                            float(self.state_action_presence_max_boost),
                        )
                    )
                    boost = (relaxed - query_final).clamp_min(0.0)
                    boost = torch.minimum(boost, max_boost)
                    action_query = (
                        query_final
                        + max(
                            0.0,
                            float(self.state_action_presence_blend),
                        ) * boost
                    )
                    action_query = torch.minimum(
                        action_query, no_presence)
                elif action_name == 'presence_guarded':
                    floor = min(
                        1.0,
                        max(
                            0.0,
                            float(self.state_action_presence_floor),
                        ),
                    )
                    action_query = torch.maximum(
                        query_final, no_presence * floor)
                elif action_name == 'semantic_presence':
                    action_query = (
                        query_semantic * effective_presence)
                elif action_name == 'instance_presence':
                    action_query = (
                        query_instance * effective_presence)
                else:
                    raise ValueError(
                        f'Unsupported state action: {action_name}')
            yield action_name, self._aggregate_query_logits_to_classes(
                action_query)

    def _state_action_feature_shape(self, height, width):
        max_side = max(
            1, int(self.state_action_feature_max_side))
        scale = min(
            1.0, float(max_side) / float(max(height, width)))
        return (
            max(1, int(round(height * scale))),
            max(1, int(round(width * scale))),
        )

    def _build_state_action_class_features(
            self, base_logits, base_pred, query_final,
            query_semantic, query_instance):
        height, width = base_logits.shape[-2:]
        feature_shape = self._state_action_feature_shape(
            height, width)
        final_map = self._interpolate_float32(
            base_logits.detach().unsqueeze(0),
            feature_shape,
        ).squeeze(0)
        semantic_map = self._interpolate_float32(
            self._aggregate_query_logits_to_classes(
                query_semantic.detach().float()).unsqueeze(0),
            feature_shape,
        ).squeeze(0)
        instance_map = self._interpolate_float32(
            self._aggregate_query_logits_to_classes(
                query_instance.detach().float()).unsqueeze(0),
            feature_shape,
        ).squeeze(0)
        no_presence = torch.maximum(semantic_map, instance_map)
        effective_presence = torch.where(
            no_presence > 1e-6,
            final_map / no_presence.clamp_min(1e-6),
            torch.zeros_like(final_map),
        ).clamp(0.0, 1.0)
        pred = F.interpolate(
            base_pred.float().view(1, 1, height, width),
            size=feature_shape,
            mode='nearest',
        ).squeeze().long()
        raw_topk = torch.topk(
            final_map, k=min(2, self.num_cls), dim=0)
        raw_top1 = raw_topk.indices[0]
        if self.num_cls > 1:
            raw_top2 = raw_topk.indices[1]
            margin = raw_topk.values[0] - raw_topk.values[1]
        else:
            raw_top2 = raw_top1
            margin = raw_topk.values[0]
        semantic_top1 = semantic_map.argmax(dim=0)
        instance_top1 = instance_map.argmax(dim=0)
        query_final_diag = self._interpolate_float32(
            query_final.detach().float().unsqueeze(0),
            feature_shape,
        ).squeeze(0)
        prompt_mean = self._aggregate_query_logits_to_classes(
            query_final_diag, reduce='mean')
        prompt_max = self._aggregate_query_logits_to_classes(
            query_final_diag)

        support_thd = float(
            self.state_action_support_threshold)
        low_presence_thd = float(
            self.state_action_low_presence_threshold)
        margin_thd = float(self.state_action_margin_threshold)
        local_kernel = max(
            1, int(self.state_action_local_kernel))
        if local_kernel % 2 == 0:
            local_kernel += 1
        catch_all_tokens = {
            'background', 'other', 'clutter', 'unknown',
            'unlabeled', 'unlabelled',
        }
        catch_all_indices = {
            class_idx
            for class_idx, class_name in enumerate(self.class_names)
            if _class_name_tokens(class_name) & catch_all_tokens
        }
        rows = []
        feature_pixels = int(
            feature_shape[0] * feature_shape[1])
        for class_idx, class_name in enumerate(self.class_names):
            semantic_support = semantic_map[class_idx] >= support_thd
            instance_support = instance_map[class_idx] >= support_thd
            final_support = final_map[class_idx] >= support_thd
            spatial_support = no_presence[class_idx] >= support_thd
            union = semantic_support | instance_support
            intersection = semantic_support & instance_support
            pred_mask = pred == class_idx
            raw_top1_mask = raw_top1 == class_idx
            low_margin_mask = (
                raw_top1_mask & (margin <= margin_thd))
            low_presence_spatial = (
                spatial_support
                & (effective_presence[class_idx]
                   <= low_presence_thd)
            )
            semantic_only = (
                semantic_support & (~instance_support))
            instance_only = (
                instance_support & (~semantic_support))
            if pred_mask.any():
                local = F.avg_pool2d(
                    pred_mask.float().view(
                        1, 1, *feature_shape),
                    kernel_size=local_kernel,
                    stride=1,
                    padding=local_kernel // 2,
                ).squeeze()
                local_consistency_sum = float(
                    local[pred_mask].sum().item())
                core_pixels = int(
                    (pred_mask & (local >= 0.90)).sum().item())
            else:
                local_consistency_sum = 0.0
                core_pixels = 0

            competitor_counts = torch.bincount(
                raw_top2[raw_top1_mask],
                minlength=self.num_cls,
            ) if raw_top1_mask.any() else torch.zeros(
                self.num_cls,
                device=base_logits.device,
                dtype=torch.long,
            )
            competitor_counts[class_idx] = 0
            top_competitor = int(
                competitor_counts.argmax().item())
            top_competitor_pixels = int(
                competitor_counts[top_competitor].item())
            catch_all_competitor_pixels = int(sum(
                int(competitor_counts[idx].item())
                for idx in catch_all_indices
            ))

            query_ids = torch.nonzero(
                self.query_idx == class_idx,
                as_tuple=False,
            ).flatten()
            prompt_winner_entropy = 0.0
            if query_ids.numel() > 1:
                class_queries = query_final_diag[query_ids]
                winners = class_queries.argmax(dim=0)
                winner_counts = torch.bincount(
                    winners.flatten(),
                    minlength=int(query_ids.numel()),
                ).float()
                winner_probs = (
                    winner_counts
                    / winner_counts.sum().clamp_min(1.0)
                )
                nonzero = winner_probs > 0
                entropy = -(
                    winner_probs[nonzero]
                    * winner_probs[nonzero].log()
                ).sum()
                prompt_winner_entropy = float(
                    entropy.item()
                    / math.log(float(query_ids.numel())))

            spatial_pixels = int(spatial_support.sum().item())
            pred_pixels = int(pred_mask.sum().item())
            raw_top1_pixels = int(raw_top1_mask.sum().item())
            semantic_pixels = int(
                semantic_support.sum().item())
            instance_pixels = int(
                instance_support.sum().item())
            final_pixels = int(final_support.sum().item())
            union_pixels = int(union.sum().item())
            intersection_pixels = int(
                intersection.sum().item())
            low_presence_pixels = int(
                low_presence_spatial.sum().item())
            low_margin_pixels = int(
                low_margin_mask.sum().item())
            reject_pixels = int(
                (raw_top1_mask & (pred != class_idx)).sum().item())
            head_agree_pixels = int(
                (
                    raw_top1_mask
                    & (semantic_top1 == class_idx)
                    & (instance_top1 == class_idx)
                ).sum().item())
            rows.append(dict(
                class_index=class_idx,
                class_name=class_name,
                query_count=int(query_ids.numel()),
                is_catch_all=bool(
                    class_idx in catch_all_indices),
                feature_pixels=feature_pixels,
                predicted_pixels=pred_pixels,
                raw_top1_pixels=raw_top1_pixels,
                semantic_support_pixels=semantic_pixels,
                instance_support_pixels=instance_pixels,
                final_support_pixels=final_pixels,
                spatial_support_pixels=spatial_pixels,
                semantic_instance_union_pixels=union_pixels,
                semantic_instance_intersection_pixels=(
                    intersection_pixels),
                semantic_only_pixels=int(
                    semantic_only.sum().item()),
                instance_only_pixels=int(
                    instance_only.sum().item()),
                low_presence_spatial_pixels=low_presence_pixels,
                low_margin_top1_pixels=low_margin_pixels,
                threshold_reject_pixels=reject_pixels,
                head_agree_top1_pixels=head_agree_pixels,
                local_consistency_sum=local_consistency_sum,
                local_core_pixels=core_pixels,
                effective_presence_support_sum=(
                    float(effective_presence[class_idx][
                        spatial_support].sum().item())
                    if spatial_support.any() else 0.0),
                final_score_sum=float(
                    final_map[class_idx].sum().item()),
                semantic_score_sum=float(
                    semantic_map[class_idx].sum().item()),
                instance_score_sum=float(
                    instance_map[class_idx].sum().item()),
                no_presence_score_sum=float(
                    no_presence[class_idx].sum().item()),
                prompt_gap_sum=float(
                    (
                        prompt_max[class_idx]
                        - prompt_mean[class_idx]
                    ).sum().item()),
                prompt_winner_entropy=prompt_winner_entropy,
                top_competitor_class_index=top_competitor,
                top_competitor_class_name=(
                    self.class_names[top_competitor]),
                top_competitor_pixels=top_competitor_pixels,
                catch_all_competitor_pixels=(
                    catch_all_competitor_pixels),
                presence_under_score=(
                    _safe_div(
                        intersection_pixels, union_pixels) or 0.0
                ) * (
                    _safe_div(
                        low_presence_pixels, spatial_pixels) or 0.0
                ),
                catch_all_pressure_score=(
                    _safe_div(
                        reject_pixels, raw_top1_pixels) or 0.0
                ) + (
                    _safe_div(
                        catch_all_competitor_pixels,
                        raw_top1_pixels) or 0.0
                ),
                competition_score=(
                    _safe_div(
                        intersection_pixels, union_pixels) or 0.0
                ) * (
                    _safe_div(
                        low_margin_pixels, raw_top1_pixels) or 0.0
                ),
                weak_dual_score=1.0 - max(
                    _safe_div(semantic_pixels, feature_pixels) or 0.0,
                    _safe_div(instance_pixels, feature_pixels) or 0.0,
                ),
            ))
        return rows, list(feature_shape)

    def _build_state_action_atlas(
            self, base_logits, base_pred, query_final,
            query_semantic, query_instance, class_components,
            data_sample):
        action_names = self._get_state_action_names()
        apply_name = str(
            self.state_action_apply or 'baseline').strip().lower()
        if not self.dump_state_action_atlas_stats:
            action_names = [apply_name]
        gt = None
        if (
                data_sample is not None
                and hasattr(data_sample, 'gt_sem_seg')
                and hasattr(data_sample.gt_sem_seg, 'data')):
            gt = data_sample.gt_sem_seg.data.squeeze().to(
                base_pred.device).long()
        valid = (gt != 255) if gt is not None else None
        base_correct = (
            (base_pred == gt) & valid
            if gt is not None else None)
        wrong_pair_mask = (
            valid & (base_pred != gt)
            if gt is not None else None)
        wrong_pair_codes = (
            (
                gt[wrong_pair_mask] * self.num_cls
                + base_pred[wrong_pair_mask]
            ).long()
            if gt is not None else None)
        wrong_pair_counts = (
            torch.bincount(
                wrong_pair_codes,
                minlength=self.num_cls * self.num_cls,
            )
            if gt is not None else None)
        active_pair_codes = (
            torch.nonzero(
                wrong_pair_counts > 0,
                as_tuple=False,
            ).flatten().tolist()
            if gt is not None else [])
        class_features = []
        feature_shape = None
        if self.dump_state_action_atlas_stats:
            class_features, feature_shape = (
                self._build_state_action_class_features(
                    base_logits,
                    base_pred,
                    query_final,
                    query_semantic,
                    query_instance,
                )
            )

        action_rows = []
        class_rows = []
        pair_rows = []
        pixel_oracle_pred = (
            base_pred.clone() if gt is not None else None)
        applied_logits = None
        applied_pred = None
        for action_name, action_logits in (
                self._iter_state_action_logits(
                    base_logits,
                    query_final,
                    query_semantic,
                    query_instance,
                    action_names,
                )):
            action_pred = self._threshold_with_reject_recovery(
                action_logits, class_components)
            if action_name == apply_name:
                applied_logits = (
                    action_logits
                    if action_name == 'baseline'
                    else action_logits.clone())
                applied_pred = action_pred.clone()
            if gt is None or not self.dump_state_action_atlas_stats:
                continue

            action_correct = (action_pred == gt) & valid
            changed = (action_pred != base_pred) & valid
            improved = (~base_correct) & action_correct
            harmed = base_correct & (~action_correct)
            wrong_to_wrong = (
                (~base_correct) & (~action_correct) & changed)
            pixel_oracle_pred[action_correct] = gt[action_correct]
            action_confusion = (
                self._candidate_residual_miou_confusion(
                    gt, action_pred, valid))
            action_rows.append(dict(
                action_name=action_name,
                confusion=action_confusion,
                changed_pixels=int(changed.sum().item()),
                improved_pixels=int(improved.sum().item()),
                harmed_pixels=int(harmed.sum().item()),
                wrong_to_wrong_pixels=int(
                    wrong_to_wrong.sum().item()),
                net_correct_pixels=(
                    int(improved.sum().item())
                    - int(harmed.sum().item())),
            ))

            changed_by_gt = torch.bincount(
                gt[changed].long(), minlength=self.num_cls)
            improved_by_gt = torch.bincount(
                gt[improved].long(), minlength=self.num_cls)
            harmed_by_gt = torch.bincount(
                gt[harmed].long(), minlength=self.num_cls)
            for class_idx, class_name in enumerate(
                    self.class_names):
                intersection = int(
                    action_confusion[class_idx][class_idx])
                gt_pixels = int(sum(
                    action_confusion[class_idx]))
                pred_pixels = int(sum(
                    action_confusion[row_idx][class_idx]
                    for row_idx in range(self.num_cls)))
                class_rows.append(dict(
                    action_name=action_name,
                    class_index=class_idx,
                    class_name=class_name,
                    gt_pixels=gt_pixels,
                    pred_pixels=pred_pixels,
                    intersection_pixels=intersection,
                    union_pixels=(
                        gt_pixels + pred_pixels - intersection),
                    changed_pixels=int(
                        changed_by_gt[class_idx].item()),
                    improved_pixels=int(
                        improved_by_gt[class_idx].item()),
                    harmed_pixels=int(
                        harmed_by_gt[class_idx].item()),
                ))

            pair_changed_counts = torch.bincount(
                wrong_pair_codes[changed[wrong_pair_mask]],
                minlength=self.num_cls * self.num_cls,
            )
            pair_corrected_counts = torch.bincount(
                wrong_pair_codes[
                    action_correct[wrong_pair_mask]],
                minlength=self.num_cls * self.num_cls,
            )
            pair_other_wrong_counts = torch.bincount(
                wrong_pair_codes[
                    wrong_to_wrong[wrong_pair_mask]],
                minlength=self.num_cls * self.num_cls,
            )
            for pair_code in active_pair_codes:
                gt_idx = int(pair_code // self.num_cls)
                pred_idx = int(pair_code % self.num_cls)
                pair_rows.append(dict(
                    action_name=action_name,
                    gt_class_index=gt_idx,
                    gt_class_name=self.class_names[gt_idx],
                    base_pred_class_index=pred_idx,
                    base_pred_class_name=(
                        self.class_names[pred_idx]),
                    pixels=int(
                        wrong_pair_counts[pair_code].item()),
                    changed_pixels=int(
                        pair_changed_counts[pair_code].item()),
                    corrected_pixels=int(
                        pair_corrected_counts[pair_code].item()),
                    changed_to_other_wrong_pixels=int(
                        pair_other_wrong_counts[
                            pair_code].item()),
                ))

            if action_name != 'baseline':
                del action_pred
                if (
                        applied_logits is not action_logits
                        and action_logits is not base_logits):
                    del action_logits

        stats = None
        if (
                self.dump_state_action_atlas_stats
                and gt is not None):
            stats = dict(
                dataset_name=self.seed_dataset_name,
                valid_pixels=int(valid.sum().item()),
                feature_shape=feature_shape,
                class_names=list(self.class_names),
                baseline_confusion=(
                    self._candidate_residual_miou_confusion(
                        gt, base_pred, valid)),
                pixel_oracle_confusion=(
                    self._candidate_residual_miou_confusion(
                        gt, pixel_oracle_pred, valid)),
                action_rows=action_rows,
                class_feature_rows=class_features,
                class_action_rows=class_rows,
                pair_action_rows=pair_rows,
            )
        return stats, applied_logits, applied_pred

    def _write_state_action_atlas_stats(self, record):
        if not self.dump_state_action_atlas_stats:
            return
        if self._state_action_atlas_stats_file is None:
            path = (
                self.state_action_atlas_stats_path
                or './work_dirs/evidence_stats/'
                   'state_action_atlas_stats.jsonl'
            )
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._state_action_atlas_stats_file = open(
                path, 'a', buffering=1)
        self._state_action_atlas_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    def _aggregate_query_logits_to_classes(
            self, query_logits, reduce='max'):
        if self.num_cls == self.num_queries:
            return query_logits
        reduce = str(reduce).lower()
        if reduce not in ('max', 'mean'):
            raise ValueError(
                "query-logit reduction must be 'max' or 'mean', "
                f'but got {reduce!r}')
        query_logits = query_logits.unsqueeze(0)
        cls_index = nn.functional.one_hot(
            self.query_idx, num_classes=self.num_cls)
        cls_index = cls_index.T.view(self.num_cls, len(self.query_idx), 1, 1)
        if reduce == 'max':
            return (query_logits * cls_index).max(1)[0]
        class_sum = (query_logits * cls_index).sum(1)
        class_count = cls_index.sum(1).clamp_min(1)
        return class_sum / class_count

    def _aggregate_query_scores_to_classes(self, query_scores, reduce='max'):
        if self.num_cls == self.num_queries:
            return query_scores
        reduce = str(reduce).lower()
        if reduce not in ('max', 'mean'):
            raise ValueError(
                "query-score reduction must be 'max' or 'mean', "
                f'but got {reduce!r}')
        scores = query_scores.detach().float().view(-1)
        class_scores = torch.zeros(
            self.num_cls, device=scores.device, dtype=scores.dtype)
        class_counts = torch.zeros(
            self.num_cls, device=scores.device, dtype=scores.dtype)
        if reduce == 'max':
            class_scores.fill_(float('-inf'))
            for query_idx, class_idx in enumerate(self.query_idx.tolist()):
                class_scores[int(class_idx)] = torch.maximum(
                    class_scores[int(class_idx)],
                    scores[query_idx],
                )
            class_scores[~torch.isfinite(class_scores)] = 0.0
            return class_scores
        for query_idx, class_idx in enumerate(self.query_idx.tolist()):
            class_scores[int(class_idx)] += scores[query_idx]
            class_counts[int(class_idx)] += 1
        return class_scores / class_counts.clamp_min(1.0)

    def _uses_active_concept_diagnostic(self):
        return bool(
            self.dump_active_concept_stats
            or self.use_active_concept_pruning)

    def _active_concept_score_threshold(self):
        if self.active_concept_score_thd is not None:
            return float(self.active_concept_score_thd)
        # Dataset prob_thd may be high (e.g. LoveDA), while active-set
        # detection should be a weak support test rather than final rejection.
        return float(min(max(float(self.prob_thd) * 0.5, 0.03), 0.10))

    def _active_concept_roles(self):
        return [
            _remote_sensing_class_role(name)
            for name in self.class_names
        ]

    @staticmethod
    def _active_concept_related_roles(role):
        """Remote-sensing role families used only as conservative safeguards."""
        role = str(role or 'other_landcover')
        families = {
            'built_object': {
                'built_object', 'impervious_surface', 'vehicle',
                'bareland',
            },
            'impervious_surface': {
                'impervious_surface', 'built_object', 'bareland',
                'vehicle',
            },
            'woody_vegetation': {
                'woody_vegetation', 'low_vegetation', 'bareland',
            },
            'low_vegetation': {
                'low_vegetation', 'woody_vegetation', 'bareland',
            },
            'bareland': {
                'bareland', 'impervious_surface', 'low_vegetation',
                'built_object',
            },
            'water': {'water', 'impervious_surface', 'bareland'},
            'vehicle': {'vehicle', 'built_object', 'impervious_surface'},
            'catch_all': {'catch_all'},
            'other_landcover': {'other_landcover', 'catch_all'},
        }
        return families.get(role, {role})

    def _build_active_concept_ontology_mask(
            self, class_features, multi_active, roles, strict=False):
        class_count = int(self.num_cls)
        device = multi_active.device
        active = multi_active.clone()
        context_area = float(self.active_concept_v2_context_area_thd)
        context_topk = float(self.active_concept_v2_context_topk_area_thd)
        context_presence = float(self.active_concept_v2_context_presence_thd)
        min_signals = max(1, int(self.active_concept_v2_min_strong_signals))

        strong_active = torch.zeros(
            class_count, device=device, dtype=torch.bool)
        weak_family_support = torch.zeros_like(strong_active)
        feature_rows = []
        for row in class_features:
            class_idx = int(row.get('class_index', -1))
            if class_idx < 0 or class_idx >= class_count:
                continue
            presence = float(row.get('presence_score', 0.0) or 0.0)
            final_area = float(row.get('final_area', 0.0) or 0.0)
            semantic_area = float(row.get('semantic_area', 0.0) or 0.0)
            instance_area = float(row.get('instance_area', 0.0) or 0.0)
            agreement_area = float(row.get('agreement_area', 0.0) or 0.0)
            pred_area = float(row.get('pred_area', 0.0) or 0.0)
            topk_area = float(row.get('topk_area', 0.0) or 0.0)
            signal_count = int(presence >= context_presence)
            signal_count += int(final_area >= context_area)
            signal_count += int(semantic_area >= context_area)
            signal_count += int(instance_area >= context_area)
            signal_count += int(agreement_area >= context_area * 0.5)
            signal_count += int(pred_area >= float(
                self.active_concept_pred_area_thd))
            signal_count += int(topk_area >= context_topk)
            strong_active[class_idx] = signal_count >= min_signals
            weak_family_support[class_idx] = (
                topk_area >= context_topk
                or pred_area >= float(self.active_concept_pred_area_thd)
                or agreement_area >= context_area * 0.5
                or presence >= context_presence * 0.5
            )
            feature_rows.append((
                class_idx, roles[class_idx], signal_count, topk_area,
                pred_area, agreement_area, presence))

        if strict:
            active = strong_active | (
                multi_active
                & weak_family_support
            )
        else:
            active = active | strong_active

        active_roles = {
            roles[idx]
            for idx, is_active in enumerate(active.cpu().tolist())
            if is_active
        }
        related_roles = set()
        for role in active_roles:
            related_roles.update(self._active_concept_related_roles(role))

        for (
                class_idx, role, signal_count, topk_area, pred_area,
                agreement_area, presence) in feature_rows:
            if bool(active[class_idx].item()):
                continue
            if role == 'catch_all':
                keep_catch_all = (
                    class_idx == int(self.bg_idx)
                    or pred_area >= float(self.active_concept_pred_area_thd)
                    or topk_area >= context_topk
                    or presence >= context_presence)
                if keep_catch_all:
                    active[class_idx] = True
                continue
            if role not in related_roles:
                continue
            if strict:
                keep = (
                    topk_area >= float(self.active_concept_topk_area_thd)
                    or pred_area >= float(self.active_concept_pred_area_thd)
                    or agreement_area >= context_area)
            else:
                keep = (
                    topk_area >= context_topk
                    or pred_area >= float(self.active_concept_pred_area_thd)
                    or agreement_area >= context_area * 0.5
                    or (
                        signal_count >= 1
                        and role in active_roles
                    )
                    or presence >= context_presence * 0.5)
            if keep:
                active[class_idx] = True

        if self.active_concept_keep_bg and 0 <= int(self.bg_idx) < class_count:
            active[int(self.bg_idx)] = True
        return active

    def _build_active_concept_context(
            self, base_logits, base_pred, components, data_sample):
        gt = data_sample.gt_sem_seg.data
        if gt.ndim == 3:
            gt = gt.squeeze(0)
        gt = gt.to(base_logits.device)
        valid = gt != 255
        valid_pixels = int(valid.sum().item())
        class_count = int(self.num_cls)
        score_thd = self._active_concept_score_threshold()
        area_denom = max(1, valid_pixels)

        gt_valid = gt[valid].long().clamp(min=0, max=class_count - 1)
        gt_counts = torch.bincount(gt_valid, minlength=class_count)
        gt_present = gt_counts > 0

        topk = max(1, min(int(self.active_concept_topk), class_count))
        topk_idx = torch.topk(base_logits, k=topk, dim=0).indices
        topk_counts = torch.zeros(
            class_count, device=base_logits.device, dtype=torch.long)
        for rank in range(topk):
            rank_values = topk_idx[rank][valid].long().clamp(
                min=0, max=class_count - 1)
            topk_counts += torch.bincount(
                rank_values, minlength=class_count)

        semantic_logits = (
            components.get('semantic_logits') if components is not None
            else None)
        instance_logits = (
            components.get('instance_logits') if components is not None
            else None)
        presence_scores = (
            components.get('presence_scores') if components is not None
            else None)
        if presence_scores is None:
            presence_scores = torch.zeros(
                class_count, device=base_logits.device,
                dtype=base_logits.dtype)
        else:
            presence_scores = presence_scores.detach().float().to(
                base_logits.device)

        class_features = []
        multi_active = torch.zeros(
            class_count, device=base_logits.device, dtype=torch.bool)
        presence_active = torch.zeros_like(multi_active)
        pred_area_values = []
        topk_area_values = []
        roles = self._active_concept_roles()
        for class_idx in range(class_count):
            cls_mask = valid
            final_map = base_logits[class_idx]
            final_area = _safe_div(
                int(((final_map >= score_thd) & cls_mask).sum().item()),
                area_denom)
            semantic_area = 0.0
            instance_area = 0.0
            agreement_area = 0.0
            if semantic_logits is not None:
                sem_mask = (semantic_logits[class_idx] >= score_thd) & cls_mask
                semantic_area = _safe_div(int(sem_mask.sum().item()), area_denom)
            else:
                sem_mask = None
            if instance_logits is not None:
                inst_mask = (
                    (instance_logits[class_idx] >= score_thd) & cls_mask)
                instance_area = _safe_div(int(inst_mask.sum().item()), area_denom)
            else:
                inst_mask = None
            if sem_mask is not None and inst_mask is not None:
                agreement_area = _safe_div(
                    int((sem_mask & inst_mask).sum().item()), area_denom)
            pred_area = _safe_div(
                int(((base_pred == class_idx) & valid).sum().item()),
                area_denom)
            topk_area = _safe_div(int(topk_counts[class_idx].item()), area_denom)
            pred_area_values.append(pred_area)
            topk_area_values.append(topk_area)
            presence_value = float(presence_scores[class_idx].item())
            presence_active[class_idx] = (
                presence_value >= float(self.active_concept_presence_thd))
            multi_active[class_idx] = (
                presence_active[class_idx]
                or final_area >= float(self.active_concept_area_thd)
                or semantic_area >= float(self.active_concept_area_thd)
                or instance_area >= float(self.active_concept_area_thd)
                or agreement_area >= float(self.active_concept_area_thd)
                or pred_area >= float(self.active_concept_pred_area_thd)
                or topk_area >= float(self.active_concept_topk_area_thd)
            )
            class_features.append(dict(
                class_index=class_idx,
                class_name=self.class_names[class_idx],
                role=roles[class_idx],
                gt_pixels=int(gt_counts[class_idx].item()),
                gt_present=bool(gt_present[class_idx].item()),
                presence_score=presence_value,
                final_area=final_area,
                semantic_area=semantic_area,
                instance_area=instance_area,
                agreement_area=agreement_area,
                pred_area=pred_area,
                topk_area=topk_area,
            ))

        if self.active_concept_keep_bg and 0 <= int(self.bg_idx) < class_count:
            presence_active[int(self.bg_idx)] = True
            multi_active[int(self.bg_idx)] = True

        variants = {}
        requested = self._parse_name_list(self.active_concept_variants)
        if not requested:
            requested = [
                'gt_oracle',
                'presence_only',
                'multi_evidence',
                'conflict_preserving',
            ]
        if 'gt_oracle' in requested:
            gt_active = gt_present.clone()
            if self.active_concept_keep_bg and 0 <= int(self.bg_idx) < class_count:
                gt_active[int(self.bg_idx)] = True
            variants['gt_oracle'] = gt_active
        if 'presence_only' in requested:
            variants['presence_only'] = presence_active
        if 'multi_evidence' in requested:
            variants['multi_evidence'] = multi_active
        if 'conflict_preserving' in requested:
            variants['conflict_preserving'] = (
                self._expand_active_concept_conflicts(
                    multi_active,
                    torch.tensor(
                        topk_area_values,
                        device=base_logits.device,
                        dtype=torch.float32),
                    roles,
                )
            )
        if (
                'ontology_context' in requested
                or 'remote_ontology' in requested):
            variants['ontology_context'] = (
                self._build_active_concept_ontology_mask(
                    class_features,
                    multi_active,
                    roles,
                    strict=False,
                )
            )
        if (
                'ontology_strict' in requested
                or 'remote_ontology_strict' in requested):
            variants['ontology_strict'] = (
                self._build_active_concept_ontology_mask(
                    class_features,
                    multi_active,
                    roles,
                    strict=True,
                )
            )

        baseline_confusion = self._candidate_residual_miou_confusion(
            gt, base_pred, valid)
        variant_rows = []
        for variant_name, active_mask in variants.items():
            pruned_logits = self._apply_active_concept_pruning_from_mask(
                base_logits, active_mask)
            pred = self._threshold_with_reject_recovery(
                pruned_logits, components)
            confusion = self._candidate_residual_miou_confusion(
                gt, pred, valid)
            changed = valid & (pred != base_pred)
            base_correct = base_pred == gt
            new_correct = pred == gt
            gt_active_count = int(gt_present.sum().item())
            active_count = int(active_mask.sum().item())
            active_gt_count = int((active_mask & gt_present).sum().item())
            non_bg = torch.ones_like(gt_present)
            if 0 <= int(self.bg_idx) < class_count:
                non_bg[int(self.bg_idx)] = False
            active_non_bg = active_mask & non_bg
            gt_non_bg = gt_present & non_bg
            variant_rows.append(dict(
                variant=variant_name,
                active_count=active_count,
                active_ratio=_safe_div(active_count, class_count),
                gt_present_count=gt_active_count,
                active_gt_count=active_gt_count,
                active_recall=_safe_div(active_gt_count, gt_active_count),
                active_precision=_safe_div(
                    int((active_mask & gt_present).sum().item()),
                    active_count),
                non_bg_active_count=int(active_non_bg.sum().item()),
                non_bg_gt_present_count=int(gt_non_bg.sum().item()),
                non_bg_active_recall=_safe_div(
                    int((active_mask & gt_non_bg).sum().item()),
                    int(gt_non_bg.sum().item())),
                inactive_gt_classes=[
                    self.class_names[idx]
                    for idx in range(class_count)
                    if bool(gt_present[idx].item())
                    and not bool(active_mask[idx].item())
                ],
                changed_pixels=int(changed.sum().item()),
                improved_pixels=int(
                    (changed & ~base_correct & new_correct).sum().item()),
                harmed_pixels=int(
                    (changed & base_correct & ~new_correct).sum().item()),
                wrong_to_wrong_pixels=int(
                    (changed & ~base_correct & ~new_correct).sum().item()),
                confusion=confusion,
                active_mask=[bool(value) for value in active_mask.cpu().tolist()],
            ))

        return dict(
            score_threshold=score_thd,
            presence_threshold=float(self.active_concept_presence_thd),
            area_threshold=float(self.active_concept_area_thd),
            pred_area_threshold=float(self.active_concept_pred_area_thd),
            topk=int(topk),
            topk_area_threshold=float(self.active_concept_topk_area_thd),
            conflict_mode=str(self.active_concept_conflict_mode),
            ontology_context_area_threshold=float(
                self.active_concept_v2_context_area_thd),
            ontology_context_topk_area_threshold=float(
                self.active_concept_v2_context_topk_area_thd),
            ontology_context_presence_threshold=float(
                self.active_concept_v2_context_presence_thd),
            ontology_min_strong_signals=int(
                self.active_concept_v2_min_strong_signals),
            valid_pixels=valid_pixels,
            class_names=list(self.class_names),
            baseline_confusion=baseline_confusion,
            class_features=class_features,
            variants=variant_rows,
        )

    def _expand_active_concept_conflicts(self, active_mask, topk_area, roles):
        expanded = active_mask.clone()
        mode = str(self.active_concept_conflict_mode or 'none').lower()
        if mode in ('none', 'off', 'false'):
            return expanded
        topk_supported = topk_area >= float(self.active_concept_topk_area_thd)
        active_roles = {
            roles[idx]
            for idx, active in enumerate(active_mask.cpu().tolist())
            if active
        }
        for idx, supported in enumerate(topk_supported.cpu().tolist()):
            if not supported or expanded[idx]:
                continue
            if mode in ('topk', 'role_topk'):
                if mode == 'topk' or roles[idx] in active_roles:
                    expanded[idx] = True
        if self.active_concept_keep_bg and 0 <= int(self.bg_idx) < self.num_cls:
            expanded[int(self.bg_idx)] = True
        return expanded

    def _apply_active_concept_pruning_from_context(
            self, logits, context, variant_name):
        variant_name = str(variant_name or 'conflict_preserving').lower()
        for row in context.get('variants', []):
            if str(row.get('variant')).lower() == variant_name:
                mask = torch.tensor(
                    row.get('active_mask', []),
                    device=logits.device,
                    dtype=torch.bool,
                )
                if mask.numel() == logits.shape[0]:
                    return self._apply_active_concept_pruning_from_mask(
                        logits, mask)
        return logits

    def _apply_active_concept_pruning_from_mask(self, logits, active_mask):
        if active_mask is None or active_mask.numel() != logits.shape[0]:
            return logits
        pruned = logits.clone()
        inactive = ~active_mask.to(device=logits.device, dtype=torch.bool)
        if not inactive.any():
            return pruned
        pruned[inactive] = float(self.active_concept_suppress_value)
        return pruned

    def _write_active_concept_stats(self, record):
        if not self.dump_active_concept_stats:
            return
        if self._active_concept_stats_file is None:
            path = (
                self.active_concept_stats_path
                or './work_dirs/evidence_stats/active_concept_stats.jsonl'
            )
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._active_concept_stats_file = open(path, 'a', buffering=1)
        self._active_concept_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    def _local_active_diag_shape(self, height, width):
        max_side = max(1, int(self.local_active_max_side))
        scale = min(1.0, float(max_side) / float(max(height, width)))
        return (
            max(1, int(round(height * scale))),
            max(1, int(round(width * scale))),
        )

    def _local_active_score_threshold(self):
        if self.local_active_score_thd is not None:
            return float(self.local_active_score_thd)
        return self._active_concept_score_threshold()

    def _local_support_fraction(self, binary_maps, kernel):
        kernel = max(1, int(kernel))
        if kernel % 2 == 0:
            kernel += 1
        radius = kernel // 2
        maps = binary_maps.float().unsqueeze(0)
        maps = F.pad(
            maps,
            (radius, radius, radius, radius),
            mode='replicate',
        )
        return F.avg_pool2d(
            maps,
            kernel_size=kernel,
            stride=1,
            padding=0,
        ).squeeze(0)

    def _build_local_active_set_stats(
            self, base_logits, base_pred, components, data_sample):
        if data_sample is None or not hasattr(data_sample, 'gt_sem_seg'):
            return None
        gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
        if not hasattr(gt_sem_seg, 'data'):
            return None
        gt = gt_sem_seg.data.squeeze().to(base_logits.device)
        valid = gt != 255
        valid_pixels_full = int(valid.sum().item())
        if valid_pixels_full == 0:
            return None

        diag_shape = self._local_active_diag_shape(
            int(base_logits.shape[-2]), int(base_logits.shape[-1]))
        final_map = self._resize_internal_source_map(base_logits, diag_shape)
        if final_map is None:
            return None
        gt_diag = F.interpolate(
            gt.float().view(1, 1, *gt.shape[-2:]),
            size=diag_shape,
            mode='nearest',
        ).squeeze().long()
        valid_diag = F.interpolate(
            valid.float().view(1, 1, *valid.shape[-2:]),
            size=diag_shape,
            mode='nearest',
        ).squeeze().bool()
        pred_diag = F.interpolate(
            base_pred.float().view(1, 1, *base_pred.shape[-2:]),
            size=diag_shape,
            mode='nearest',
        ).squeeze().long().clamp(0, self.num_cls - 1)
        gt_idx = gt_diag.clamp(0, self.num_cls - 1)
        wrong = valid_diag & (pred_diag != gt_idx)
        wrong_pixels = int(wrong.sum().item())

        score_thd = float(self._local_active_score_threshold())
        requested_sources = set(self._parse_name_list(
            self.local_active_sources))
        if not requested_sources:
            requested_sources = {
                'topk', 'final', 'semantic', 'instance', 'agreement'}
        topk = max(1, min(int(self.local_active_topk), self.num_cls))
        topk_idx = torch.topk(
            torch.nan_to_num(final_map, nan=-1e6),
            k=topk,
            dim=0,
        ).indices

        source_binary = {}
        if 'topk' in requested_sources:
            topk_binary = torch.zeros_like(final_map, dtype=torch.bool)
            for rank in range(topk):
                rank_one_hot = F.one_hot(
                    topk_idx[rank].clamp(0, self.num_cls - 1),
                    num_classes=self.num_cls,
                ).permute(2, 0, 1).bool()
                topk_binary |= rank_one_hot
            source_binary['topk'] = topk_binary
        if 'final' in requested_sources:
            source_binary['final'] = final_map >= score_thd

        semantic_map = None
        instance_map = None
        if components is not None:
            if components.get('semantic_logits') is not None:
                semantic_map = self._resize_internal_source_map(
                    components.get('semantic_logits'), diag_shape)
            if components.get('instance_logits') is not None:
                instance_map = self._resize_internal_source_map(
                    components.get('instance_logits'), diag_shape)
        if semantic_map is not None and 'semantic' in requested_sources:
            source_binary['semantic'] = semantic_map >= score_thd
        if instance_map is not None and 'instance' in requested_sources:
            source_binary['instance'] = instance_map >= score_thd
        if (
                semantic_map is not None
                and instance_map is not None
                and 'agreement' in requested_sources):
            source_binary['agreement'] = (
                (semantic_map >= score_thd) & (instance_map >= score_thd)
            )

        windows = sorted({
            max(1, int(round(value)))
            for value in self._parse_float_list(
                self.local_active_windows, '17,33,65')
        })
        area_thresholds = sorted({
            max(0.0, float(value))
            for value in self._parse_float_list(
                self.local_active_area_thresholds,
                '0.001,0.005,0.01,0.02,0.05',
            )
        })
        roles = self._active_concept_roles()
        rows = []
        class_rows = []
        pair_rows = []
        for window in windows:
            support_maps = {
                name: self._local_support_fraction(binary, window)
                for name, binary in source_binary.items()
            }
            for area_thd in area_thresholds:
                active = torch.zeros_like(final_map, dtype=torch.bool)
                for support in support_maps.values():
                    active |= support >= float(area_thd)
                gt_kept = _gather_class_map(active.float(), gt_idx) > 0.5
                pred_kept = (
                    _gather_class_map(active.float(), pred_diag) > 0.5)
                active_count = active.float().sum(dim=0)
                active_count_valid = active_count[valid_diag]
                wrong_gt_kept = wrong & gt_kept
                wrong_pred_removed = wrong & (~pred_kept)
                desired = wrong & gt_kept & (~pred_kept)
                both = wrong & gt_kept & pred_kept
                gt_removed = wrong & (~gt_kept)
                pred_removed_gt_removed = wrong & (~gt_kept) & (~pred_kept)
                rows.append(dict(
                    window=int(window),
                    area_threshold=float(area_thd),
                    sources=','.join(sorted(source_binary.keys())),
                    valid_pixels=int(valid_diag.sum().item()),
                    wrong_pixels=wrong_pixels,
                    mean_active_count=_masked_mean(active_count, valid_diag),
                    wrong_gt_kept_pixels=int(wrong_gt_kept.sum().item()),
                    wrong_gt_kept_ratio=_safe_div(
                        int(wrong_gt_kept.sum().item()), wrong_pixels),
                    wrong_pred_removed_pixels=int(
                        wrong_pred_removed.sum().item()),
                    wrong_pred_removed_ratio=_safe_div(
                        int(wrong_pred_removed.sum().item()), wrong_pixels),
                    desired_keep_gt_remove_pred_pixels=int(
                        desired.sum().item()),
                    desired_keep_gt_remove_pred_ratio=_safe_div(
                        int(desired.sum().item()), wrong_pixels),
                    both_gt_and_pred_kept_pixels=int(both.sum().item()),
                    both_gt_and_pred_kept_ratio=_safe_div(
                        int(both.sum().item()), wrong_pixels),
                    gt_removed_pixels=int(gt_removed.sum().item()),
                    gt_removed_ratio=_safe_div(
                        int(gt_removed.sum().item()), wrong_pixels),
                    pred_removed_but_gt_removed_pixels=int(
                        pred_removed_gt_removed.sum().item()),
                    mean_active_count_on_wrong=_masked_mean(
                        active_count, wrong),
                    mean_active_count_on_valid=(
                        float(active_count_valid.float().mean().item())
                        if active_count_valid.numel() else 0.0),
                ))

                for gt_class in range(self.num_cls):
                    class_mask = wrong & (gt_idx == gt_class)
                    pixels = int(class_mask.sum().item())
                    if pixels <= 0:
                        continue
                    class_rows.append(dict(
                        window=int(window),
                        area_threshold=float(area_thd),
                        class_index=int(gt_class),
                        class_name=self.class_names[gt_class],
                        role=roles[gt_class],
                        wrong_pixels=pixels,
                        gt_kept_pixels=int((class_mask & gt_kept).sum().item()),
                        gt_kept_ratio=_safe_div(
                            int((class_mask & gt_kept).sum().item()), pixels),
                        pred_removed_pixels=int(
                            (class_mask & (~pred_kept)).sum().item()),
                        pred_removed_ratio=_safe_div(
                            int((class_mask & (~pred_kept)).sum().item()), pixels),
                        desired_pixels=int((class_mask & desired).sum().item()),
                        desired_ratio=_safe_div(
                            int((class_mask & desired).sum().item()), pixels),
                        mean_active_count=_masked_mean(
                            active_count, class_mask),
                    ))

                min_pair = int(self.local_active_min_pair_pixels)
                for gt_class in range(self.num_cls):
                    gt_mask = wrong & (gt_idx == gt_class)
                    if not gt_mask.any():
                        continue
                    for pred_class in range(self.num_cls):
                        if pred_class == gt_class:
                            continue
                        pair_mask = gt_mask & (pred_diag == pred_class)
                        pixels = int(pair_mask.sum().item())
                        if pixels < min_pair:
                            continue
                        pair_rows.append(dict(
                            window=int(window),
                            area_threshold=float(area_thd),
                            gt_class_index=int(gt_class),
                            gt_class_name=self.class_names[gt_class],
                            gt_role=roles[gt_class],
                            pred_class_index=int(pred_class),
                            pred_class_name=self.class_names[pred_class],
                            pred_role=roles[pred_class],
                            pair_pixels=pixels,
                            gt_kept_pixels=int(
                                (pair_mask & gt_kept).sum().item()),
                            gt_kept_ratio=_safe_div(
                                int((pair_mask & gt_kept).sum().item()), pixels),
                            pred_removed_pixels=int(
                                (pair_mask & (~pred_kept)).sum().item()),
                            pred_removed_ratio=_safe_div(
                                int((pair_mask & (~pred_kept)).sum().item()), pixels),
                            desired_pixels=int(
                                (pair_mask & desired).sum().item()),
                            desired_ratio=_safe_div(
                                int((pair_mask & desired).sum().item()), pixels),
                            both_kept_pixels=int(
                                (pair_mask & both).sum().item()),
                            both_kept_ratio=_safe_div(
                                int((pair_mask & both).sum().item()), pixels),
                            gt_removed_pixels=int(
                                (pair_mask & gt_removed).sum().item()),
                            gt_removed_ratio=_safe_div(
                                int((pair_mask & gt_removed).sum().item()), pixels),
                            mean_active_count=_masked_mean(
                                active_count, pair_mask),
                        ))

        pair_rows.sort(key=lambda row: row['pair_pixels'], reverse=True)
        return dict(
            dataset_name=self.seed_dataset_name,
            diagnostic_shape=list(diag_shape),
            score_threshold=score_thd,
            topk=topk,
            sources=sorted(source_binary.keys()),
            valid_pixels=int(valid_diag.sum().item()),
            baseline_wrong_pixels=wrong_pixels,
            class_names=list(self.class_names),
            windows=windows,
            area_thresholds=area_thresholds,
            overall=rows,
            class_stats=class_rows,
            pair_stats=pair_rows,
        )

    def _write_local_active_set_stats(self, record):
        if not self.dump_local_active_set_stats:
            return
        if self._local_active_set_stats_file is None:
            path = (
                self.local_active_set_stats_path
                or './work_dirs/evidence_stats/local_active_set_stats.jsonl'
            )
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._local_active_set_stats_file = open(
                path, 'a', buffering=1)
        self._local_active_set_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    def _prompt_winner_diag_shape(self, height, width):
        max_side = max(1, int(self.prompt_winner_max_side))
        scale = min(1.0, float(max_side) / float(max(height, width)))
        return (
            max(1, int(round(height * scale))),
            max(1, int(round(width * scale))),
        )

    def _query_class_max_with_winner(self, query_maps):
        class_scores = torch.full(
            (self.num_cls, *query_maps.shape[-2:]),
            float('-inf'),
            device=query_maps.device,
            dtype=query_maps.dtype,
        )
        winner_queries = torch.full(
            (self.num_cls, *query_maps.shape[-2:]),
            -1,
            device=query_maps.device,
            dtype=torch.long,
        )
        for class_idx in range(self.num_cls):
            query_indices = torch.nonzero(
                self.query_idx == class_idx,
                as_tuple=False,
            ).flatten()
            if query_indices.numel() == 0:
                continue
            values = query_maps[query_indices]
            max_values, local_winner = values.max(dim=0)
            class_scores[class_idx] = max_values
            winner_queries[class_idx] = query_indices[local_winner]
        class_scores[~torch.isfinite(class_scores)] = float('-inf')
        return class_scores, winner_queries

    def _query_class_reduce(self, query_maps, reduce):
        reduce = str(reduce).lower()
        if reduce == 'max':
            return self._query_class_max_with_winner(query_maps)[0]
        class_scores = torch.full(
            (self.num_cls, *query_maps.shape[-2:]),
            float('-inf'),
            device=query_maps.device,
            dtype=query_maps.dtype,
        )
        for class_idx in range(self.num_cls):
            query_indices = torch.nonzero(
                self.query_idx == class_idx,
                as_tuple=False,
            ).flatten()
            if query_indices.numel() == 0:
                continue
            values = query_maps[query_indices]
            if reduce == 'mean':
                class_scores[class_idx] = values.mean(dim=0)
            elif reduce == 'top2_mean':
                topk = min(2, int(values.shape[0]))
                class_scores[class_idx] = torch.topk(
                    values, k=topk, dim=0).values.mean(dim=0)
            else:
                raise ValueError(
                    f'Unknown prompt winner reduction: {reduce}')
        return class_scores

    def _threshold_class_scores(self, class_scores):
        pred = torch.argmax(class_scores, dim=0)
        max_scores = class_scores.max(dim=0)[0]
        pred[max_scores < float(self.prob_thd)] = int(self.bg_idx)
        return pred

    def _top_query_on_mask(self, winner_queries, mask):
        if mask is None or not mask.any():
            return None, None, 0
        selected = winner_queries[mask].detach().long()
        selected = selected[(selected >= 0) & (selected < self.num_queries)]
        if selected.numel() == 0:
            return None, None, 0
        counts = torch.bincount(selected, minlength=self.num_queries)
        query_idx = int(counts.argmax().item())
        count = int(counts[query_idx].item())
        return query_idx, self.query_words[query_idx], count

    def _build_prompt_winner_attribution_stats(
            self, query_final_logits, query_semantic_logits,
            query_instance_logits, base_logits, base_pred, components,
            data_sample):
        if data_sample is None or not hasattr(data_sample, 'gt_sem_seg'):
            return None
        gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
        if not hasattr(gt_sem_seg, 'data'):
            return None
        gt = gt_sem_seg.data.squeeze().to(base_logits.device)
        valid_full = gt != 255
        if int(valid_full.sum().item()) == 0:
            return None

        diag_shape = self._prompt_winner_diag_shape(
            int(base_logits.shape[-2]), int(base_logits.shape[-1]))
        gt_diag = F.interpolate(
            gt.float().view(1, 1, *gt.shape[-2:]),
            size=diag_shape,
            mode='nearest',
        ).squeeze().long()
        valid = F.interpolate(
            valid_full.float().view(1, 1, *valid_full.shape[-2:]),
            size=diag_shape,
            mode='nearest',
        ).squeeze().bool()
        gt_idx = gt_diag.clamp(0, self.num_cls - 1)
        base_pred_diag = F.interpolate(
            base_pred.float().view(1, 1, *base_pred.shape[-2:]),
            size=diag_shape,
            mode='nearest',
        ).squeeze().long().clamp(0, self.num_cls - 1)
        base_wrong = valid & (base_pred_diag != gt_idx)

        source_query_maps = {}
        requested = set(self._parse_name_list(self.prompt_winner_sources))
        if not requested:
            requested = {'final', 'semantic', 'instance'}
        if 'final' in requested and query_final_logits is not None:
            source_query_maps['final'] = self._interpolate_float32(
                query_final_logits.detach().unsqueeze(0),
                diag_shape,
            ).squeeze(0)
        if 'semantic' in requested and query_semantic_logits is not None:
            source_query_maps['semantic'] = self._interpolate_float32(
                query_semantic_logits.detach().unsqueeze(0),
                diag_shape,
            ).squeeze(0)
        if 'instance' in requested and query_instance_logits is not None:
            source_query_maps['instance'] = self._interpolate_float32(
                query_instance_logits.detach().unsqueeze(0),
                diag_shape,
            ).squeeze(0)
        if not source_query_maps:
            return None

        roles = self._active_concept_roles()
        valid_pixels = int(valid.sum().item())
        wrong_pixels = int(base_wrong.sum().item())
        baseline_confusion = self._candidate_residual_miou_confusion(
            gt_idx, base_pred_diag, valid)
        action_rows = [
            dict(
                action_name='baseline_threshold',
                source_name='baseline',
                valid_pixels=valid_pixels,
                confusion=baseline_confusion,
            )
        ]
        prompt_rows = []
        pair_rows = []

        final_query = source_query_maps.get('final')
        if final_query is not None:
            reduction_preds = {}
            for reduce_name in ('max', 'mean', 'top2_mean'):
                reduced = self._query_class_reduce(final_query, reduce_name)
                pred = self._threshold_class_scores(reduced)
                reduction_preds[reduce_name] = pred
                action_rows.append(dict(
                    action_name=f'prompt_{reduce_name}',
                    source_name='final',
                    valid_pixels=valid_pixels,
                    confusion=self._candidate_residual_miou_confusion(
                        gt_idx, pred, valid),
                ))
            oracle_correct = torch.zeros_like(valid, dtype=torch.bool)
            for pred in reduction_preds.values():
                oracle_correct |= pred == gt_idx
            oracle_pred = base_pred_diag.clone()
            oracle_pred[valid & oracle_correct] = gt_idx[valid & oracle_correct]
            action_rows.append(dict(
                action_name='prompt_reduction_oracle',
                source_name='final',
                valid_pixels=valid_pixels,
                oracle_correct_pixels=int(
                    (valid & oracle_correct).sum().item()),
                baseline_correct_pixels=int(
                    (valid & (base_pred_diag == gt_idx)).sum().item()),
                confusion=self._candidate_residual_miou_confusion(
                    gt_idx, oracle_pred, valid),
            ))

        for source_name, query_maps in source_query_maps.items():
            class_scores, winner_queries = (
                self._query_class_max_with_winner(query_maps))
            source_top1 = torch.argmax(class_scores, dim=0)
            for query_idx, query_word in enumerate(self.query_words):
                class_idx = int(self.query_idx[query_idx].item())
                class_winner = valid & (winner_queries[class_idx] == query_idx)
                source_top1_prompt = (
                    class_winner & (source_top1 == class_idx))
                baseline_pred_prompt = (
                    class_winner & (base_pred_diag == class_idx))
                gt_class_prompt = class_winner & (gt_idx == class_idx)
                correct_prompt = baseline_pred_prompt & (gt_idx == class_idx)
                harmful_prompt = baseline_pred_prompt & (gt_idx != class_idx)
                suppressed_prompt = (
                    class_winner & base_wrong & (gt_idx == class_idx))
                prompt_rows.append(dict(
                    source_name=source_name,
                    query_index=int(query_idx),
                    query_word=str(query_word),
                    class_index=class_idx,
                    class_name=self.class_names[class_idx],
                    role=roles[class_idx],
                    class_winner_pixels=int(class_winner.sum().item()),
                    source_top1_prompt_pixels=int(
                        source_top1_prompt.sum().item()),
                    baseline_pred_prompt_pixels=int(
                        baseline_pred_prompt.sum().item()),
                    gt_class_prompt_pixels=int(gt_class_prompt.sum().item()),
                    correct_prompt_pixels=int(correct_prompt.sum().item()),
                    harmful_prompt_pixels=int(harmful_prompt.sum().item()),
                    suppressed_gt_prompt_pixels=int(
                        suppressed_prompt.sum().item()),
                    baseline_prompt_purity=_safe_div(
                        int(correct_prompt.sum().item()),
                        int(baseline_pred_prompt.sum().item())),
                    harmful_ratio=_safe_div(
                        int(harmful_prompt.sum().item()),
                        int(baseline_pred_prompt.sum().item())),
                    gt_capture_ratio=_safe_div(
                        int(gt_class_prompt.sum().item()),
                        int(((gt_idx == class_idx) & valid).sum().item())),
                    suppressed_gt_capture_ratio=_safe_div(
                        int(suppressed_prompt.sum().item()),
                        int((base_wrong & (gt_idx == class_idx)).sum().item())),
                    winner_score_mean=_masked_mean(
                        query_maps[query_idx], class_winner),
                    baseline_pred_score_mean=_masked_mean(
                        query_maps[query_idx], baseline_pred_prompt),
                ))

            min_pair = max(1, int(self.prompt_winner_min_pair_pixels))
            for gt_class in range(self.num_cls):
                gt_wrong = base_wrong & (gt_idx == gt_class)
                if not gt_wrong.any():
                    continue
                for pred_class in range(self.num_cls):
                    if pred_class == gt_class:
                        continue
                    pair_mask = gt_wrong & (base_pred_diag == pred_class)
                    pixels = int(pair_mask.sum().item())
                    if pixels < min_pair:
                        continue
                    gt_query, gt_word, gt_query_pixels = (
                        self._top_query_on_mask(
                            winner_queries[gt_class], pair_mask))
                    pred_query, pred_word, pred_query_pixels = (
                        self._top_query_on_mask(
                            winner_queries[pred_class], pair_mask))
                    gt_score = _gather_class_map(class_scores, gt_idx)
                    pred_score = _gather_class_map(
                        class_scores,
                        torch.full_like(gt_idx, int(pred_class)),
                    )
                    pair_rows.append(dict(
                        source_name=source_name,
                        gt_class_index=int(gt_class),
                        gt_class_name=self.class_names[gt_class],
                        gt_role=roles[gt_class],
                        pred_class_index=int(pred_class),
                        pred_class_name=self.class_names[pred_class],
                        pred_role=roles[pred_class],
                        pixels=pixels,
                        pixel_ratio_in_wrong=_safe_div(pixels, wrong_pixels),
                        gt_winner_query_index=gt_query,
                        gt_winner_query_word=gt_word,
                        gt_winner_query_ratio=_safe_div(
                            gt_query_pixels, pixels),
                        pred_winner_query_index=pred_query,
                        pred_winner_query_word=pred_word,
                        pred_winner_query_ratio=_safe_div(
                            pred_query_pixels, pixels),
                        gt_score_mean=_masked_mean(gt_score, pair_mask),
                        pred_score_mean=_masked_mean(pred_score, pair_mask),
                        gt_minus_pred_score_mean=(
                            None
                            if _masked_mean(gt_score, pair_mask) is None
                            or _masked_mean(pred_score, pair_mask) is None
                            else _masked_mean(gt_score, pair_mask)
                            - _masked_mean(pred_score, pair_mask)),
                    ))

        return dict(
            dataset_name=self.seed_dataset_name,
            diagnostic_shape=list(diag_shape),
            class_names=list(self.class_names),
            query_words=list(self.query_words),
            query_to_class=[int(x) for x in self.query_idx.tolist()],
            prob_thd=float(self.prob_thd),
            bg_idx=int(self.bg_idx),
            valid_pixels=valid_pixels,
            baseline_wrong_pixels=wrong_pixels,
            sources=sorted(source_query_maps.keys()),
            action_stats=action_rows,
            prompt_stats=prompt_rows,
            pair_stats=pair_rows,
        )

    def _write_prompt_winner_attribution_stats(self, record):
        if not self.dump_prompt_winner_attribution_stats:
            return
        if self._prompt_winner_attribution_stats_file is None:
            path = (
                self.prompt_winner_attribution_stats_path
                or './work_dirs/evidence_stats/'
                   'prompt_winner_attribution_stats.jsonl'
            )
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._prompt_winner_attribution_stats_file = open(
                path, 'a', buffering=1)
        self._prompt_winner_attribution_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    def _region_prompt_diag_shape(self, height, width):
        max_side = max(1, int(self.region_prompt_identity_max_side))
        scale = min(1.0, float(max_side) / float(max(height, width)))
        return (
            max(1, int(round(height * scale))),
            max(1, int(round(width * scale))),
        )

    def _place_raw_mask_on_diag(
            self, raw_mask, diag_shape, full_shape, crop_box=None):
        diag_h, diag_w = diag_shape
        full_h, full_w = full_shape
        raw_mask = raw_mask.to(self.device).float()
        if crop_box is None:
            return torch.sigmoid(self._interpolate_float32(
                raw_mask.view(1, 1, *raw_mask.shape),
                diag_shape,
            ).squeeze())
        x1, y1, x2, y2 = [float(value) for value in crop_box]
        gx1 = max(0, min(diag_w - 1, int(round(x1 * diag_w / full_w))))
        gy1 = max(0, min(diag_h - 1, int(round(y1 * diag_h / full_h))))
        gx2 = max(gx1 + 1, min(diag_w, int(round(x2 * diag_w / full_w))))
        gy2 = max(gy1 + 1, min(diag_h, int(round(y2 * diag_h / full_h))))
        placed = torch.zeros(diag_shape, device=self.device)
        resized = torch.sigmoid(self._interpolate_float32(
            raw_mask.view(1, 1, *raw_mask.shape),
            (gy2 - gy1, gx2 - gx1),
        ).squeeze())
        placed[gy1:gy2, gx1:gx2] = resized
        return placed

    @staticmethod
    def _binary_dilate(mask, kernel_size):
        kernel_size = int(kernel_size)
        if kernel_size <= 1:
            return mask
        padding = kernel_size // 2
        value = mask.float().view(1, 1, *mask.shape)
        value = F.pad(value, (padding, padding, padding, padding), value=0.0)
        return F.max_pool2d(
            value,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
        ).squeeze() > 0

    def _region_score_vector(self, score_map, mask, ring=None):
        if score_map is None or mask is None or not mask.any():
            return None
        inside = score_map[:, mask].detach().float().mean(dim=1)
        if ring is None or not ring.any():
            return inside
        outside = score_map[:, ring].detach().float().mean(dim=1)
        return inside - outside

    def _score_rank(self, scores, class_idx):
        if scores is None:
            return None
        finite_scores = torch.nan_to_num(
            scores.detach().float(),
            nan=-1e6,
            neginf=-1e6,
            posinf=1e6,
        )
        order = torch.argsort(finite_scores, descending=True)
        hit = torch.nonzero(order == int(class_idx), as_tuple=False)
        if hit.numel() == 0:
            return None
        return int(hit.flatten()[0].item()) + 1

    def _build_region_prompt_identity_stats(
            self, base_logits, base_pred, components, data_sample):
        if data_sample is None or not hasattr(data_sample, 'gt_sem_seg'):
            return None
        raw_candidates = None if components is None else components.get(
            'raw_mask_candidates')
        if not raw_candidates:
            return None
        gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
        if not hasattr(gt_sem_seg, 'data'):
            return None
        gt = gt_sem_seg.data.squeeze().to(base_logits.device)
        valid_full = gt != 255
        if int(valid_full.sum().item()) == 0:
            return None

        diag_shape = self._region_prompt_diag_shape(
            int(base_logits.shape[-2]), int(base_logits.shape[-1]))
        gt_diag = F.interpolate(
            gt.float().view(1, 1, *gt.shape[-2:]),
            size=diag_shape,
            mode='nearest',
        ).squeeze().long()
        valid = F.interpolate(
            valid_full.float().view(1, 1, *valid_full.shape[-2:]),
            size=diag_shape,
            mode='nearest',
        ).squeeze().bool()
        gt_idx = gt_diag.clamp(0, self.num_cls - 1)
        pred_diag = F.interpolate(
            base_pred.float().view(1, 1, *base_pred.shape[-2:]),
            size=diag_shape,
            mode='nearest',
        ).squeeze().long().clamp(0, self.num_cls - 1)
        base_wrong = valid & (pred_diag != gt_idx)
        final_map = self._resize_internal_source_map(base_logits, diag_shape)
        semantic_map = None
        instance_map = None
        if components is not None:
            semantic_map = self._resize_internal_source_map(
                components.get('semantic_logits'), diag_shape)
            instance_map = self._resize_internal_source_map(
                components.get('instance_logits'), diag_shape)

        candidates = []
        bin_thd = float(self.region_prompt_identity_bin_thd)
        min_pixels = max(1, int(self.region_prompt_identity_min_pixels))
        full_shape = tuple(base_logits.shape[-2:])
        for prompt_record in raw_candidates:
            class_idx = int(prompt_record.get('class_index', -1))
            if class_idx < 0 or class_idx >= self.num_cls:
                continue
            raw_masks = prompt_record.get('raw_masks_lowres')
            if raw_masks is None or int(raw_masks.shape[0]) == 0:
                continue
            raw_scores = prompt_record.get('raw_scores')
            presence_scores = prompt_record.get('raw_presence_scores')
            selected_indices = prompt_record.get('selected_indices')
            crop_box = prompt_record.get('crop_box')
            for local_idx in range(int(raw_masks.shape[0])):
                prob = self._place_raw_mask_on_diag(
                    raw_masks[local_idx],
                    diag_shape,
                    full_shape,
                    crop_box=crop_box,
                )
                mask = (prob >= bin_thd) & valid
                pixels = int(mask.sum().item())
                if pixels < min_pixels:
                    continue
                raw_score = (
                    None if raw_scores is None
                    else float(raw_scores[local_idx].item()))
                presence_score = (
                    None if presence_scores is None
                    else float(presence_scores[local_idx].item()))
                candidates.append(dict(
                    mask=mask,
                    class_index=class_idx,
                    class_name=self.class_names[class_idx],
                    query_index=int(prompt_record.get('query_index', -1)),
                    query_word=str(prompt_record.get('query_word', '')),
                    raw_candidate_index=(
                        None if selected_indices is None
                        else int(selected_indices[local_idx].item())),
                    raw_score=raw_score,
                    raw_presence_score=presence_score,
                    mask_pixels=pixels,
                ))
        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (
                -_zero_if_none(item.get('raw_presence_score')),
                -_zero_if_none(item.get('raw_score')),
                -item['mask_pixels'],
            ))
        max_regions = max(1, int(self.region_prompt_identity_max_regions))
        groups = []
        for candidate in candidates:
            best_idx = None
            best_iou = -1.0
            for group_idx, group in enumerate(groups):
                inter = int((candidate['mask'] & group['mask']).sum().item())
                if inter == 0:
                    continue
                union = int((candidate['mask'] | group['mask']).sum().item())
                iou = _safe_div(inter, union) or 0.0
                containment = _safe_div(
                    inter,
                    min(candidate['mask_pixels'],
                        int(group['mask'].sum().item())),
                ) or 0.0
                if (
                        iou >= float(self.region_prompt_identity_group_iou)
                        or containment >= float(
                            self.region_prompt_identity_group_containment)):
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = group_idx
            if best_idx is None:
                if len(groups) >= max_regions:
                    continue
                groups.append(dict(
                    mask=candidate['mask'].clone(),
                    members=[candidate],
                ))
            else:
                groups[best_idx]['mask'] = (
                    groups[best_idx]['mask'] | candidate['mask'])
                groups[best_idx]['members'].append(candidate)

        roles = self._active_concept_roles()
        scorer_maps = {
            'final_mean': (final_map, False),
            'semantic_mean': (semantic_map, False),
            'instance_mean': (instance_map, False),
            'final_contrast': (final_map, True),
            'semantic_contrast': (semantic_map, True),
            'instance_contrast': (instance_map, True),
        }
        group_rows = []
        pair_rows = []
        pair_oracle_rows = []
        min_pair = max(1, int(self.region_prompt_identity_min_pair_pixels))
        wrong_pixels = int(base_wrong.sum().item())
        for group_id, group in enumerate(groups):
            mask = group['mask'] & valid
            pixels = int(mask.sum().item())
            if pixels < min_pixels:
                continue
            ring = self._binary_dilate(
                mask,
                int(self.region_prompt_identity_ring_kernel),
            ) & (~mask) & valid
            member_classes = sorted({
                int(member['class_index']) for member in group['members']})
            member_queries = sorted({
                int(member['query_index']) for member in group['members']})
            gt_mode = _mode_class(gt_idx, mask)
            pred_mode = _mode_class(pred_diag, mask)
            base_correct_ratio = _masked_mean(
                (gt_idx == pred_diag).float(), mask)
            group_purity = (
                None if gt_mode is None
                else _masked_mean((gt_idx == gt_mode).float(), mask))
            score_vectors = {}
            for scorer_name, (score_map, use_ring) in scorer_maps.items():
                scores = self._region_score_vector(
                    score_map,
                    mask,
                    ring if use_ring else None,
                )
                if scores is None:
                    continue
                score_vectors[scorer_name] = scores
                top1 = int(torch.nan_to_num(
                    scores,
                    nan=-1e6,
                    neginf=-1e6,
                    posinf=1e6,
                ).argmax().item())
                group_rows.append(dict(
                    group_id=int(group_id),
                    scorer_name=scorer_name,
                    mask_pixels=pixels,
                    ring_pixels=int(ring.sum().item()),
                    member_count=len(group['members']),
                    member_class_count=len(member_classes),
                    member_classes='|'.join(
                        self.class_names[idx] for idx in member_classes),
                    member_query_count=len(member_queries),
                    gt_mode_class_index=gt_mode,
                    gt_mode_class_name=(
                        None if gt_mode is None else self.class_names[gt_mode]),
                    pred_mode_class_index=pred_mode,
                    pred_mode_class_name=(
                        None if pred_mode is None else self.class_names[pred_mode]),
                    baseline_correct_ratio=base_correct_ratio,
                    gt_mode_purity=group_purity,
                    top1_class_index=top1,
                    top1_class_name=self.class_names[top1],
                    top1_role=roles[top1],
                    top1_is_gt_mode=(
                        None if gt_mode is None else bool(top1 == gt_mode)),
                    top1_is_pred_mode=(
                        None if pred_mode is None else bool(top1 == pred_mode)),
                    top1_score=float(scores[top1].item()),
                ))

            group_wrong = base_wrong & mask
            if not group_wrong.any():
                continue
            pair_codes = (
                gt_idx[group_wrong] * self.num_cls
                + pred_diag[group_wrong]
            )
            unique_codes, counts = torch.unique(
                pair_codes, sorted=True, return_counts=True)
            for code, count in zip(unique_codes.tolist(), counts.tolist()):
                if int(count) < min_pair:
                    continue
                gt_class = int(code // self.num_cls)
                pred_class = int(code % self.num_cls)
                if gt_class == pred_class:
                    continue
                pair_mask = group_wrong & (gt_idx == gt_class) & (
                    pred_diag == pred_class)
                best_margin = None
                best_scorer = None
                any_favors = False
                best_rank = None
                for scorer_name, scores in score_vectors.items():
                    gt_score = float(scores[gt_class].item())
                    pred_score = float(scores[pred_class].item())
                    margin = gt_score - pred_score
                    gt_rank = self._score_rank(scores, gt_class)
                    pred_rank = self._score_rank(scores, pred_class)
                    top1 = int(torch.nan_to_num(
                        scores,
                        nan=-1e6,
                        neginf=-1e6,
                        posinf=1e6,
                    ).argmax().item())
                    favors = bool(margin > 0.0)
                    any_favors = any_favors or favors
                    if best_margin is None or margin > best_margin:
                        best_margin = margin
                        best_scorer = scorer_name
                        best_rank = gt_rank
                    pair_rows.append(dict(
                        group_id=int(group_id),
                        scorer_name=scorer_name,
                        gt_class_index=gt_class,
                        gt_class_name=self.class_names[gt_class],
                        gt_role=roles[gt_class],
                        pred_class_index=pred_class,
                        pred_class_name=self.class_names[pred_class],
                        pred_role=roles[pred_class],
                        pair_pixels=int(count),
                        pixel_ratio_in_wrong=_safe_div(
                            int(count), wrong_pixels),
                        gt_score=gt_score,
                        pred_score=pred_score,
                        gt_minus_pred=margin,
                        gt_beats_pred=favors,
                        gt_rank=gt_rank,
                        pred_rank=pred_rank,
                        top1_class_index=top1,
                        top1_class_name=self.class_names[top1],
                        top1_is_gt=bool(top1 == gt_class),
                        top1_is_pred=bool(top1 == pred_class),
                    ))
                pair_oracle_rows.append(dict(
                    group_id=int(group_id),
                    gt_class_index=gt_class,
                    gt_class_name=self.class_names[gt_class],
                    gt_role=roles[gt_class],
                    pred_class_index=pred_class,
                    pred_class_name=self.class_names[pred_class],
                    pred_role=roles[pred_class],
                    pair_pixels=int(count),
                    pixel_ratio_in_wrong=_safe_div(int(count), wrong_pixels),
                    any_scorer_favors_gt=bool(any_favors),
                    best_scorer=best_scorer,
                    best_gt_minus_pred=best_margin,
                    best_gt_rank=best_rank,
                ))

        return dict(
            dataset_name=self.seed_dataset_name,
            diagnostic_shape=list(diag_shape),
            class_names=list(self.class_names),
            valid_pixels=int(valid.sum().item()),
            baseline_wrong_pixels=wrong_pixels,
            raw_candidate_count=len(candidates),
            region_group_count=len(groups),
            region_prompt_identity_bin_thd=bin_thd,
            region_prompt_identity_min_pixels=min_pixels,
            region_prompt_identity_group_iou=float(
                self.region_prompt_identity_group_iou),
            region_prompt_identity_group_containment=float(
                self.region_prompt_identity_group_containment),
            region_prompt_identity_max_regions=max_regions,
            group_stats=group_rows,
            pair_scorer_stats=pair_rows,
            pair_oracle_stats=pair_oracle_rows,
        )

    def _write_region_prompt_identity_stats(self, record):
        if not self.dump_region_prompt_identity_stats:
            return
        if self._region_prompt_identity_stats_file is None:
            path = (
                self.region_prompt_identity_stats_path
                or './work_dirs/evidence_stats/'
                   'region_prompt_identity_stats.jsonl'
            )
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._region_prompt_identity_stats_file = open(
                path, 'a', buffering=1)
        self._region_prompt_identity_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    def _apply_reject_aware_calibration(self, seg_logits, components):
        if not self.use_reject_aware_calibration or components is None:
            return seg_logits
        calibrated = seg_logits.clone()
        if self.use_competition_suppression and self.num_cls > 1:
            calibrated = self._apply_competition_suppression(calibrated, components)
        return calibrated

    def _apply_evidence_competition_graph(self, seg_logits, components):
        if (not self.use_evidence_competition_graph
                or components is None
                or self.num_cls <= 1):
            return seg_logits

        semantic_logits = components.get('semantic_logits')
        instance_logits = components.get('instance_logits')
        if semantic_logits is None or instance_logits is None:
            return seg_logits

        top2_vals, top2_idx = torch.topk(seg_logits, k=2, dim=0)
        top1_score = top2_vals[0]
        top2_score = top2_vals[1]
        top1_idx = top2_idx[0]
        top2_class = top2_idx[1]
        margin = top1_score - top2_score

        top1_semantic = _gather_class_map(semantic_logits, top1_idx)
        top1_instance = _gather_class_map(instance_logits, top1_idx)
        top2_semantic = _gather_class_map(semantic_logits, top2_class)
        top2_instance = _gather_class_map(instance_logits, top2_class)

        top1_agreement = torch.minimum(top1_semantic, top1_instance)
        top2_agreement = torch.minimum(top2_semantic, top2_instance)
        top1_sem_only = (top1_semantic - top1_instance).clamp(min=0.0)

        ambiguous = (
            (top2_score >= top1_score * self.ecg_top2_ratio)
            | (margin <= self.ecg_margin_thd)
        )
        top2_supported = (
            (top2_semantic >= self.ecg_semantic_thd)
            & (top2_instance >= self.ecg_instance_thd)
        )
        top1_semantic_overexpands = top1_sem_only >= self.ecg_sem_over_inst_gap
        top2_more_reliable = (
            top2_agreement >= top1_agreement + self.ecg_agreement_gap
        )
        transfer_mask = (
            ambiguous
            & top2_supported
            & (top2_more_reliable | top1_semantic_overexpands)
        )
        if not transfer_mask.any():
            return seg_logits

        agreement_delta = (top2_agreement - top1_agreement).clamp(min=0.0)
        sem_only_delta = torch.minimum(top1_sem_only, top2_agreement).clamp(min=0.0)
        transfer_delta = torch.maximum(
            agreement_delta,
            sem_only_delta * self.ecg_sem_only_transfer,
        )
        transfer_delta = transfer_delta * transfer_mask.float()

        boosted = seg_logits.clone()
        boost = transfer_delta * self.ecg_boost_scale
        boosted.scatter_add_(0, top2_class.unsqueeze(0), boost.unsqueeze(0))
        if self.ecg_suppress_scale > 0:
            suppress = boost * self.ecg_suppress_scale
            boosted.scatter_add_(0, top1_idx.unsqueeze(0), -suppress.unsqueeze(0))
        return boosted

    def _apply_competition_suppression(self, seg_logits, components):
        semantic_logits = components.get('semantic_logits')
        instance_logits = components.get('instance_logits')
        if semantic_logits is None or instance_logits is None:
            return seg_logits

        top2_vals, top2_idx = torch.topk(seg_logits, k=2, dim=0)
        top1_score = top2_vals[0]
        top2_score = top2_vals[1]
        top1_idx = top2_idx[0]

        top1_semantic = _gather_class_map(semantic_logits, top1_idx)
        top1_instance = _gather_class_map(instance_logits, top1_idx)
        semantic_instance_gap = top1_semantic - top1_instance

        suppress_mask = (
            (top1_idx != self.bg_idx)
            & (semantic_instance_gap > self.competition_sem_inst_gap)
            & (top2_score > top1_score * self.competition_top2_ratio)
        )
        if not suppress_mask.any():
            return seg_logits

        penalty = (semantic_instance_gap * self.competition_penalty).clamp(min=0.0)
        penalty = penalty * suppress_mask.float()
        seg_logits = seg_logits.clone()
        seg_logits.scatter_add_(0, top1_idx.unsqueeze(0), -penalty.unsqueeze(0))
        return seg_logits

    def _uses_residual_background_modeling(self):
        return (
            bool(getattr(self, 'use_residual_background_modeling', False))
            or bool(getattr(self, 'dump_residual_background_stats', False))
            or bool(getattr(self, 'dump_background_separability_stats', False))
        )

    def _background_residual_confusion(self, gt, pred, valid_mask):
        class_count = int(self.num_cls)
        if gt is None or pred is None or valid_mask is None:
            return [[0 for _ in range(class_count)] for _ in range(class_count)]
        valid_gt = gt[valid_mask].long()
        valid_pred = pred[valid_mask].long()
        if valid_gt.numel() == 0:
            return [[0 for _ in range(class_count)] for _ in range(class_count)]
        index = valid_gt * class_count + valid_pred.clamp(0, class_count - 1)
        matrix = torch.bincount(
            index,
            minlength=class_count * class_count,
        ).reshape(class_count, class_count)
        return matrix.detach().cpu().tolist()

    def _residual_background_local_support(self, fg_idx):
        kernel = max(1, int(self.residual_background_local_kernel))
        if kernel % 2 == 0:
            kernel += 1
        if kernel <= 1:
            return torch.ones_like(fg_idx, dtype=torch.float32)

        class_count = int(self.num_cls)
        one_hot = F.one_hot(
            fg_idx.clamp(0, class_count - 1).long(),
            num_classes=class_count,
        ).permute(2, 0, 1).float()
        pooled = F.avg_pool2d(
            one_hot.unsqueeze(0),
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        ).squeeze(0)
        return _gather_class_map(pooled, fg_idx.long())

    def _build_residual_background_prediction(
            self, base_logits, base_pred, components):
        if (
                base_logits is None
                or base_pred is None
                or self.num_cls <= 1
                or self.bg_idx < 0
                or self.bg_idx >= self.num_cls):
            return base_pred, None

        bg_idx = int(self.bg_idx)
        class_count = int(self.num_cls)
        raw_top_score, raw_top_idx = base_logits.max(dim=0)
        bg_score = base_logits[bg_idx]

        foreground_logits = base_logits.clone()
        foreground_logits[bg_idx] = -1e6
        fg_score, fg_idx = foreground_logits.max(dim=0)
        fg_bg_margin = fg_score - bg_score

        semantic_support = torch.zeros_like(fg_score)
        instance_support = torch.zeros_like(fg_score)
        semantic_head_support = torch.zeros_like(fg_score, dtype=torch.bool)
        instance_head_support = torch.zeros_like(fg_score, dtype=torch.bool)
        presence_support = torch.zeros_like(fg_score)

        if components is not None:
            semantic_logits = components.get('semantic_logits')
            if semantic_logits is not None:
                semantic_support = _gather_class_map(
                    semantic_logits, fg_idx.long())
                semantic_fg_logits = semantic_logits.clone()
                semantic_fg_logits[bg_idx] = -1e6
                semantic_head_support = (
                    semantic_fg_logits.argmax(dim=0) == fg_idx)

            instance_logits = components.get('instance_logits')
            if instance_logits is not None:
                instance_support = _gather_class_map(
                    instance_logits, fg_idx.long())
                instance_fg_logits = instance_logits.clone()
                instance_fg_logits[bg_idx] = -1e6
                instance_head_support = (
                    instance_fg_logits.argmax(dim=0) == fg_idx)

            presence_scores = components.get('presence_scores')
            if presence_scores is not None:
                presence_support = presence_scores.detach().float()[
                    fg_idx.long()]

        local_support = self._residual_background_local_support(fg_idx)

        score_floor = max(
            float(self.residual_background_min_score),
            float(self.prob_thd) * float(self.residual_background_score_factor),
        )
        evidence_support = (
            (semantic_support >= float(self.residual_background_min_semantic))
            | (instance_support >= float(self.residual_background_min_instance))
        )
        if bool(self.residual_background_require_head_support):
            evidence_support = evidence_support & (
                semantic_head_support | instance_head_support)
        if float(self.residual_background_min_presence) > 0:
            evidence_support = evidence_support & (
                presence_support >= float(
                    self.residual_background_min_presence))

        reliability = (
            fg_score
            + float(self.residual_background_semantic_weight)
            * semantic_support.clamp(min=0.0)
            + float(self.residual_background_instance_weight)
            * instance_support.clamp(min=0.0)
            + float(self.residual_background_presence_weight)
            * presence_support.clamp(min=0.0)
            + float(self.residual_background_local_weight)
            * (local_support - 0.5)
        )
        residual_background = (
            float(self.residual_background_bg_weight)
            * bg_score.clamp(min=0.0)
        )
        reliability_margin = reliability - residual_background

        baseline_bg = base_pred == bg_idx
        threshold_route = (
            baseline_bg
            & (raw_top_idx != bg_idx)
            & (raw_top_score < float(self.prob_thd))
        )
        weak_bg_route = (
            baseline_bg
            & (raw_top_idx == bg_idx)
            & (bg_score <= float(self.prob_thd))
        )

        route = str(self.residual_background_route).lower()
        if route in ('threshold', 'threshold_only'):
            route_mask = threshold_route
        elif route in ('threshold_or_weak_bg', 'weak_bg'):
            route_mask = threshold_route | weak_bg_route
        elif route in ('all_bg', 'all_background'):
            route_mask = baseline_bg
        else:
            raise ValueError(
                "residual_background_route must be one of "
                "'threshold_only', 'threshold_or_weak_bg', or 'all_bg', "
                f"but got {self.residual_background_route!r}")

        recover_mask = (
            route_mask
            & (fg_idx != bg_idx)
            & (fg_score >= score_floor)
            & (fg_bg_margin >= float(
                self.residual_background_min_fg_bg_margin))
            & (local_support >= float(self.residual_background_min_local))
            & evidence_support
            & (reliability_margin >= float(
                self.residual_background_min_reliability_margin))
        )

        residual_pred = base_pred.clone()
        residual_pred[recover_mask] = fg_idx[recover_mask]
        context = dict(
            route=str(self.residual_background_route),
            score_floor=float(score_floor),
            raw_top_score=raw_top_score,
            raw_top_idx=raw_top_idx,
            fg_score=fg_score,
            fg_idx=fg_idx,
            bg_score=bg_score,
            fg_bg_margin=fg_bg_margin,
            semantic_support=semantic_support,
            instance_support=instance_support,
            presence_support=presence_support,
            semantic_head_support=semantic_head_support,
            instance_head_support=instance_head_support,
            local_support=local_support,
            reliability=reliability,
            residual_background=residual_background,
            reliability_margin=reliability_margin,
            threshold_route=threshold_route,
            weak_bg_route=weak_bg_route,
            route_mask=route_mask,
            recover_mask=recover_mask,
        )
        return residual_pred, context

    def _build_residual_background_stats(
            self, base_pred, residual_pred, data_sample, context):
        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)
        if gt_data is None or context is None or residual_pred is None:
            return None

        valid_mask = gt_data != 255
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return dict(valid_pixels=0)

        bg_idx = int(self.bg_idx)
        changed = (base_pred != residual_pred) & valid_mask
        base_correct = (base_pred == gt_data) & valid_mask
        residual_correct = (residual_pred == gt_data) & valid_mask
        improved = changed & (~base_correct) & residual_correct
        harmed = changed & base_correct & (~residual_correct)
        wrong_to_wrong = changed & (~base_correct) & (~residual_correct)

        recover_mask = context['recover_mask'] & valid_mask
        route_mask = context['route_mask'] & valid_mask
        threshold_route = context['threshold_route'] & valid_mask
        weak_bg_route = context['weak_bg_route'] & valid_mask
        gt_is_bg = gt_data == bg_idx
        pred_is_bg = base_pred == bg_idx
        fg_gt_to_bg = valid_mask & pred_is_bg & (gt_data != bg_idx)
        true_bg_pred_bg = valid_mask & pred_is_bg & gt_is_bg

        class_stats = []
        pair_stats = []
        for class_idx in range(int(self.num_cls)):
            class_mask = valid_mask & (gt_data == class_idx)
            class_pixels = int(class_mask.sum().item())
            if class_pixels == 0:
                continue
            class_changed = changed & class_mask
            class_stats.append(dict(
                class_index=class_idx,
                class_name=self.class_names[class_idx],
                role=_remote_sensing_class_role(self.class_names[class_idx]),
                gt_pixels=class_pixels,
                base_bg_pixels=int((class_mask & pred_is_bg).sum().item()),
                recovered_pixels=int((class_mask & recover_mask).sum().item()),
                changed_pixels=int(class_changed.sum().item()),
                improved_pixels=int((improved & class_mask).sum().item()),
                harmed_pixels=int((harmed & class_mask).sum().item()),
                wrong_to_wrong_pixels=int(
                    (wrong_to_wrong & class_mask).sum().item()),
            ))

        for gt_idx in range(int(self.num_cls)):
            gt_mask = valid_mask & (gt_data == gt_idx)
            if not gt_mask.any():
                continue
            for pred_idx in range(int(self.num_cls)):
                pair_mask = gt_mask & (base_pred == pred_idx)
                changed_pair = pair_mask & changed
                pair_pixels = int(pair_mask.sum().item())
                changed_pixels = int(changed_pair.sum().item())
                if pair_pixels == 0 or changed_pixels == 0:
                    continue
                pair_stats.append(dict(
                    gt_class_index=gt_idx,
                    gt_class_name=self.class_names[gt_idx],
                    gt_role=_remote_sensing_class_role(
                        self.class_names[gt_idx]),
                    base_pred_class_index=pred_idx,
                    base_pred_class_name=self.class_names[pred_idx],
                    base_pred_role=_remote_sensing_class_role(
                        self.class_names[pred_idx]),
                    pair_pixels=pair_pixels,
                    changed_pixels=changed_pixels,
                    improved_pixels=int((improved & pair_mask).sum().item()),
                    harmed_pixels=int((harmed & pair_mask).sum().item()),
                    wrong_to_wrong_pixels=int(
                        (wrong_to_wrong & pair_mask).sum().item()),
                ))

        return dict(
            class_names=list(self.class_names),
            class_roles=[
                _remote_sensing_class_role(name)
                for name in self.class_names
            ],
            bg_idx=int(bg_idx),
            valid_pixels=valid_count,
            baseline_confusion=self._background_residual_confusion(
                gt_data, base_pred, valid_mask),
            residual_confusion=self._background_residual_confusion(
                gt_data, residual_pred, valid_mask),
            baseline_bg_pixels=int((valid_mask & pred_is_bg).sum().item()),
            baseline_true_bg_pixels=int(true_bg_pred_bg.sum().item()),
            foreground_gt_to_bg_pixels=int(fg_gt_to_bg.sum().item()),
            route_pixels=int(route_mask.sum().item()),
            threshold_route_pixels=int(threshold_route.sum().item()),
            weak_bg_route_pixels=int(weak_bg_route.sum().item()),
            recovered_pixels=int(recover_mask.sum().item()),
            changed_pixels=int(changed.sum().item()),
            improved_pixels=int(improved.sum().item()),
            harmed_pixels=int(harmed.sum().item()),
            wrong_to_wrong_pixels=int(wrong_to_wrong.sum().item()),
            changed_true_bg_pixels=int((changed & gt_is_bg).sum().item()),
            recovered_foreground_gt_pixels=int(
                (recover_mask & (gt_data != bg_idx)).sum().item()),
            recovered_true_bg_pixels=int(
                (recover_mask & gt_is_bg).sum().item()),
            score_floor=float(context['score_floor']),
            class_stats=class_stats,
            pair_stats=pair_stats,
        )

    def _write_residual_background_stats(self, record):
        if not self.dump_residual_background_stats:
            return
        if self._residual_background_stats_file is None:
            path = (
                self.residual_background_stats_path
                or 'residual_background_stats.jsonl'
            )
            path = _ranked_jsonl_path(path)
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            self._residual_background_stats_file = open(
                path, 'a', buffering=1)
        self._residual_background_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    def _parse_background_separability_thresholds(self, value, default):
        if value is None:
            return list(default)
        if isinstance(value, (list, tuple)):
            raw_values = value
        else:
            raw_values = str(value).split(',')
        parsed = []
        for item in raw_values:
            text = str(item).strip()
            if not text:
                continue
            try:
                parsed.append(float(text))
            except ValueError:
                continue
        return parsed or list(default)

    def _background_feature_bucket(self, values, mask, bins):
        values = values.detach().float()
        mask = mask.detach().bool()
        count = int(mask.sum().item())
        hist_len = len(bins) + 1
        if count == 0:
            return dict(
                pixels=0,
                sum=0.0,
                sumsq=0.0,
                min=None,
                max=None,
                hist=[0 for _ in range(hist_len)],
            )
        selected = values[mask]
        selected = selected[torch.isfinite(selected)]
        count = int(selected.numel())
        if count == 0:
            return dict(
                pixels=0,
                sum=0.0,
                sumsq=0.0,
                min=None,
                max=None,
                hist=[0 for _ in range(hist_len)],
            )
        edges = torch.tensor(
            bins,
            dtype=selected.dtype,
            device=selected.device,
        )
        bucket = torch.bucketize(selected, edges)
        hist = torch.bincount(bucket, minlength=hist_len)
        return dict(
            pixels=count,
            sum=float(selected.sum().item()),
            sumsq=float((selected * selected).sum().item()),
            min=float(selected.min().item()),
            max=float(selected.max().item()),
            hist=[int(v) for v in hist.detach().cpu().tolist()],
        )

    def _background_candidate_matrix(self, gt, candidate, mask):
        class_count = int(self.num_cls)
        matrix = torch.zeros(
            (class_count, class_count),
            dtype=torch.long,
            device=gt.device,
        )
        mask = mask.detach().bool()
        if not mask.any():
            return matrix.detach().cpu().tolist()
        valid_gt = gt[mask].long().clamp(0, class_count - 1)
        valid_candidate = candidate[mask].long().clamp(0, class_count - 1)
        index = valid_gt * class_count + valid_candidate
        matrix = torch.bincount(
            index,
            minlength=class_count * class_count,
        ).reshape(class_count, class_count)
        return matrix.detach().cpu().tolist()

    def _build_background_selector_stats(
            self, selector_name, selected, gt, fg_idx, bg_idx,
            valid_bg_mask):
        selected = selected.detach().bool() & valid_bg_mask
        selected_pixels = int(selected.sum().item())
        gt_is_bg = gt == bg_idx
        improved = selected & (~gt_is_bg) & (fg_idx == gt)
        harmed = selected & gt_is_bg
        wrong_to_wrong = selected & (~gt_is_bg) & (fg_idx != gt)
        return dict(
            name=selector_name,
            selected_pixels=selected_pixels,
            selected_recoverable_pixels=int(
                (selected & (~gt_is_bg)).sum().item()),
            selected_protected_pixels=int((selected & gt_is_bg).sum().item()),
            improved_pixels=int(improved.sum().item()),
            harmed_pixels=int(harmed.sum().item()),
            wrong_to_wrong_pixels=int(wrong_to_wrong.sum().item()),
            selected_candidate_matrix=self._background_candidate_matrix(
                gt, fg_idx, selected),
        )

    def _build_background_separability_stats(
            self, base_pred, data_sample, context):
        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)
        if gt_data is None or context is None or base_pred is None:
            return None

        valid_mask = gt_data != 255
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return dict(valid_pixels=0)

        bg_idx = int(self.bg_idx)
        baseline_bg = valid_mask & (base_pred == bg_idx)
        protected = baseline_bg & (gt_data == bg_idx)
        recoverable = baseline_bg & (gt_data != bg_idx)
        fg_idx = context['fg_idx'].long()
        candidate_correct = recoverable & (fg_idx == gt_data)

        score_bins = self._parse_background_separability_thresholds(
            self.background_separability_score_thresholds,
            [0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50],
        )
        margin_bins = self._parse_background_separability_thresholds(
            self.background_separability_margin_thresholds,
            [-0.10, -0.05, 0.00, 0.02, 0.05, 0.10, 0.20, 0.40],
        )
        reliability_bins = self._parse_background_separability_thresholds(
            self.background_separability_reliability_thresholds,
            [-0.20, -0.10, -0.05, 0.00, 0.02, 0.05, 0.10, 0.20, 0.40],
        )
        local_bins = self._parse_background_separability_thresholds(
            self.background_separability_local_thresholds,
            [0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
        )

        semantic_head = context['semantic_head_support'].float()
        instance_head = context['instance_head_support'].float()
        head_votes = semantic_head + instance_head
        any_head = head_votes >= 1.0
        both_heads = head_votes >= 2.0

        feature_specs = [
            ('fg_score', context['fg_score'], score_bins),
            ('bg_score', context['bg_score'], score_bins),
            ('fg_bg_margin', context['fg_bg_margin'], margin_bins),
            ('semantic_support', context['semantic_support'], score_bins),
            ('instance_support', context['instance_support'], score_bins),
            ('presence_support', context['presence_support'], score_bins),
            ('local_support', context['local_support'], local_bins),
            ('reliability', context['reliability'], reliability_bins),
            ('residual_background', context['residual_background'], score_bins),
            ('reliability_margin', context['reliability_margin'],
             reliability_bins),
            ('head_votes', head_votes, [0.5, 1.5]),
            ('semantic_head_support', semantic_head, [0.5]),
            ('instance_head_support', instance_head, [0.5]),
        ]
        feature_stats = []
        for name, values, bins in feature_specs:
            feature_stats.append(dict(
                feature=name,
                bins=[float(v) for v in bins],
                protected=self._background_feature_bucket(
                    values, protected, bins),
                recoverable=self._background_feature_bucket(
                    values, recoverable, bins),
                candidate_correct=self._background_feature_bucket(
                    values, candidate_correct, bins),
            ))

        selectors = []
        selector_masks = []

        def add_selector(name, mask):
            selectors.append(self._build_background_selector_stats(
                name, mask, gt_data, fg_idx, bg_idx, baseline_bg))

        fg_score = context['fg_score']
        fg_bg_margin = context['fg_bg_margin']
        reliability_margin = context['reliability_margin']
        local_support = context['local_support']
        threshold_route = context['threshold_route']
        weak_bg_route = context['weak_bg_route']

        for threshold in score_bins:
            add_selector(
                f'fg_score_ge_{threshold:g}',
                baseline_bg & (fg_score >= float(threshold)),
            )
        for threshold in margin_bins:
            add_selector(
                f'fg_bg_margin_ge_{threshold:g}',
                baseline_bg & (fg_bg_margin >= float(threshold)),
            )
        for threshold in reliability_bins:
            add_selector(
                f'reliability_margin_ge_{threshold:g}',
                baseline_bg & (reliability_margin >= float(threshold)),
            )
        for threshold in local_bins:
            add_selector(
                f'local_support_ge_{threshold:g}',
                baseline_bg & (local_support >= float(threshold)),
            )

        add_selector('any_head_support', baseline_bg & any_head)
        add_selector('both_head_support', baseline_bg & both_heads)
        add_selector('threshold_route', baseline_bg & threshold_route)
        add_selector('weak_bg_route', baseline_bg & weak_bg_route)

        compact_scores = [0.03, 0.05, 0.10, 0.20]
        compact_reliability = [0.00, 0.05, 0.10]
        compact_local = [0.50, 0.70]
        for score_thd in compact_scores:
            for rel_thd in compact_reliability:
                mask = (
                    baseline_bg
                    & (fg_score >= score_thd)
                    & (reliability_margin >= rel_thd)
                    & any_head
                )
                add_selector(
                    f'score_rel_anyhead_s{score_thd:g}_r{rel_thd:g}',
                    mask,
                )
                add_selector(
                    f'threshold_score_rel_anyhead_s{score_thd:g}_r{rel_thd:g}',
                    mask & threshold_route,
                )
            for local_thd in compact_local:
                add_selector(
                    f'score_local_anyhead_s{score_thd:g}_l{local_thd:g}',
                    baseline_bg
                    & (fg_score >= score_thd)
                    & (local_support >= local_thd)
                    & any_head,
                )

        class_stats = []
        for class_idx in range(int(self.num_cls)):
            class_mask = baseline_bg & (gt_data == class_idx)
            pixels = int(class_mask.sum().item())
            if pixels == 0:
                continue
            class_stats.append(dict(
                class_index=class_idx,
                class_name=self.class_names[class_idx],
                role=_remote_sensing_class_role(self.class_names[class_idx]),
                baseline_bg_pixels=pixels,
                candidate_correct_pixels=int(
                    (class_mask & (fg_idx == gt_data)).sum().item()),
                threshold_route_pixels=int(
                    (class_mask & threshold_route).sum().item()),
                weak_bg_route_pixels=int(
                    (class_mask & weak_bg_route).sum().item()),
                fg_score_sum=float(fg_score[class_mask].sum().item()),
                fg_bg_margin_sum=float(
                    fg_bg_margin[class_mask].sum().item()),
                reliability_margin_sum=float(
                    reliability_margin[class_mask].sum().item()),
                local_support_sum=float(
                    local_support[class_mask].sum().item()),
                head_votes_sum=float(head_votes[class_mask].sum().item()),
            ))

        pair_stats = []
        for gt_idx in range(int(self.num_cls)):
            gt_mask = baseline_bg & (gt_data == gt_idx)
            if not gt_mask.any():
                continue
            for cand_idx in range(int(self.num_cls)):
                pair_mask = gt_mask & (fg_idx == cand_idx)
                pixels = int(pair_mask.sum().item())
                if pixels == 0:
                    continue
                pair_stats.append(dict(
                    gt_class_index=gt_idx,
                    gt_class_name=self.class_names[gt_idx],
                    gt_role=_remote_sensing_class_role(
                        self.class_names[gt_idx]),
                    candidate_class_index=cand_idx,
                    candidate_class_name=self.class_names[cand_idx],
                    candidate_role=_remote_sensing_class_role(
                        self.class_names[cand_idx]),
                    pixels=pixels,
                    correct_candidate_pixels=int(
                        (pair_mask & (gt_data == fg_idx)).sum().item()),
                    threshold_route_pixels=int(
                        (pair_mask & threshold_route).sum().item()),
                    weak_bg_route_pixels=int(
                        (pair_mask & weak_bg_route).sum().item()),
                    fg_score_sum=float(fg_score[pair_mask].sum().item()),
                    fg_bg_margin_sum=float(
                        fg_bg_margin[pair_mask].sum().item()),
                    reliability_margin_sum=float(
                        reliability_margin[pair_mask].sum().item()),
                    local_support_sum=float(
                        local_support[pair_mask].sum().item()),
                    head_votes_sum=float(head_votes[pair_mask].sum().item()),
                ))

        return dict(
            class_names=list(self.class_names),
            class_roles=[
                _remote_sensing_class_role(name)
                for name in self.class_names
            ],
            valid_pixels=valid_count,
            baseline_confusion=self._background_residual_confusion(
                gt_data, base_pred, valid_mask),
            baseline_bg_pixels=int(baseline_bg.sum().item()),
            protected_background_pixels=int(protected.sum().item()),
            recoverable_foreground_pixels=int(recoverable.sum().item()),
            candidate_correct_pixels=int(candidate_correct.sum().item()),
            threshold_route_pixels=int(
                (baseline_bg & threshold_route).sum().item()),
            weak_bg_route_pixels=int(
                (baseline_bg & weak_bg_route).sum().item()),
            candidate_oracle_matrix=self._background_candidate_matrix(
                gt_data, fg_idx, candidate_correct),
            foreground_candidate_matrix=self._background_candidate_matrix(
                gt_data, fg_idx, baseline_bg),
            feature_stats=feature_stats,
            selector_stats=selectors,
            class_stats=class_stats,
            pair_stats=pair_stats,
        )

    def _write_background_separability_stats(self, record):
        if not self.dump_background_separability_stats:
            return
        if self._background_separability_stats_file is None:
            path = (
                self.background_separability_stats_path
                or 'background_separability_stats.jsonl'
            )
            path = _ranked_jsonl_path(path)
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            self._background_separability_stats_file = open(
                path, 'a', buffering=1)
        self._background_separability_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    def _threshold_with_reject_recovery(self, seg_logits, components):
        seg_pred = torch.argmax(seg_logits, dim=0)
        max_vals = seg_logits.max(0)[0]
        if (not self.use_reject_aware_calibration
                or not self.use_reject_recovery
                or components is None
                or not self._allow_reject_recovery_for_bg()):
            seg_pred[max_vals < self.prob_thd] = self.bg_idx
            return seg_pred

        top2_vals, top2_idx = torch.topk(seg_logits, k=min(2, self.num_cls), dim=0)
        top1_score = top2_vals[0]
        top1_idx = top2_idx[0]
        if self.num_cls > 1:
            top2_score = top2_vals[1]
        else:
            top2_score = torch.zeros_like(top1_score)
        margin = top1_score - top2_score

        semantic_support = _gather_class_map(components['semantic_logits'], top1_idx)
        instance_support = _gather_class_map(components['instance_logits'], top1_idx)
        recovery_threshold = self.prob_thd * self.reject_recovery_factor
        recovery_support = (
            (semantic_support >= self.reject_recovery_semantic_thd)
            | (instance_support >= self.reject_recovery_instance_thd)
            | (margin >= self.reject_recovery_margin_thd)
        )
        recover_mask = (
            (top1_idx != self.bg_idx)
            & (top1_score < self.prob_thd)
            & (top1_score >= recovery_threshold)
            & recovery_support
        )

        threshold_mask = max_vals < self.prob_thd
        seg_pred[threshold_mask & ~recover_mask] = self.bg_idx
        seg_pred[recover_mask] = top1_idx[recover_mask]
        return seg_pred

    def _allow_reject_recovery_for_bg(self):
        mode = str(self.reject_recovery_bg_role).lower()
        if mode in ('none', 'false', 'off'):
            return False
        if mode in ('any', 'all', 'true', 'on'):
            return True
        bg_name = self.class_names[self.bg_idx].lower()
        if mode == 'clutter':
            return 'clutter' in bg_name
        if mode == 'background':
            return any(token in bg_name for token in ('background', 'other', 'clutter'))
        if mode == 'auto':
            # Prior diagnostics showed recovery helps clutter fallback but hurts
            # true background/other classes by converting correct background
            # pixels into foreground false positives.
            return 'clutter' in bg_name
        raise ValueError(
            "reject_recovery_bg_role must be one of "
            "'auto', 'clutter', 'background', 'any', or 'none', "
            f"but got {self.reject_recovery_bg_role!r}")

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            # Fallback for meta info construction
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        for i, meta in enumerate(batch_img_metas):
            # Load original image to preserve details for SAM3
            image_path = meta.get('img_path')
            image = Image.open(image_path).convert('RGB')
            ori_shape = meta['ori_shape']

            # Determine inference mode
            evidence_stats = None
            components = None
            return_stats = self.dump_evidence_stats
            return_components = (
                self.dump_competition_stats
                or self.use_reject_aware_calibration
                or self._uses_residual_background_modeling()
                or self.use_evidence_competition_graph
                or self.dump_oracle_stats
                or self.use_topk_candidate_verifier
                or self.dump_topk_verifier_stats
                or self.dump_error_rank_stats
                or self.use_top2_risk_arbitration
                or self.dump_top2_risk_stats
                or self.dump_multiview_oracle_stats
                or self.dump_seed_separability_stats
                or self.dump_raw_mask_oracle_stats
                or self.dump_candidate_quality_stats
                or self.dump_geometry_context_stats
                or self.dump_weak_support_stats
                or self.dump_reject_recovery_precision_stats
                or self.dump_expert_reliability_stats
                or self.use_coco_sec_fusion
                or self.dump_coco_sec_stats
                or self._uses_active_concept_diagnostic()
                or self.dump_local_active_set_stats
                or self.dump_prompt_winner_attribution_stats
                or self.dump_region_prompt_identity_stats
                or self._uses_concept_specificity_diagnostic()
                or self._uses_cross_image_bank_diagnostic()
                or self._uses_cross_image_bank_recomposition()
                or self.dump_topk_conflict_bias_stats
                or self.dump_internal_selection_gap_stats
                or self.dump_candidate_internal_verifier_stats
                or self.dump_position_bias_stats
                or self.dump_scene_common_bias_stats
                or self.dump_candidate_residual_trajectory_stats
                or self.dump_candidate_residual_miou_stats
                or self._uses_ontology_readout_oracle()
                or self._uses_state_action_atlas()
                or self._uses_region_contrastive_readout()
                or self._uses_region_hypothesis_v2()
                or self._uses_candidate_region_quality_diagnostic()
                or self._uses_query_topology_diagnostic()
                or self._uses_rethinking_reviewer()
                or self._uses_ontology_self_verification()
                or self._uses_evidence_enhancement()
                or self._uses_structure_aware_recalibration()
                or self._uses_self_prompted_concept_verification()
                or self._uses_sam3_geometry_requery_diagnostic()
                or self.use_geoer_router
                or self.dump_geoer_router_stats
            )
            if self.slide_crop > 0 and (self.slide_crop < image.size[0] or self.slide_crop < image.size[1]):
                if return_stats and return_components:
                    seg_logits, evidence_stats, components = self.slide_inference(
                        image, self.slide_stride, self.slide_crop,
                        return_stats=True, return_components=True)
                elif return_stats:
                    seg_logits, evidence_stats = self.slide_inference(
                        image, self.slide_stride, self.slide_crop, return_stats=True)
                elif return_components:
                    seg_logits, components = self.slide_inference(
                        image, self.slide_stride, self.slide_crop, return_components=True)
                else:
                    seg_logits = self.slide_inference(image, self.slide_stride, self.slide_crop)
            else:
                if return_stats and return_components:
                    seg_logits, evidence_stats, components = self._inference_single_view(
                        image, return_stats=True, return_components=True, view_id='full_image')
                elif return_stats:
                    seg_logits, evidence_stats = self._inference_single_view(
                        image, return_stats=True, view_id='full_image')
                elif return_components:
                    seg_logits, components = self._inference_single_view(
                        image, return_components=True, view_id='full_image')
                else:
                    seg_logits = self._inference_single_view(image)

            # Resize to original shape if necessary (e.g. padding effects)
            if seg_logits.shape[-2:] != ori_shape:
                seg_logits = F.interpolate(
                    seg_logits.unsqueeze(0),
                    size=ori_shape,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                if components is not None:
                    for key in ['semantic_logits', 'instance_logits']:
                        components[key] = F.interpolate(
                            components[key].unsqueeze(0),
                            size=ori_shape,
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)

            # Post-processing
            query_seg_logits = seg_logits
            query_semantic_logits = (
                components.get('semantic_logits')
                if components is not None else None)
            query_instance_logits = (
                components.get('instance_logits')
                if components is not None else None)
            seg_logits = self._aggregate_query_logits_to_classes(seg_logits)
            if components is not None:
                aggregated_components = {}
                for key, value in components.items():
                    if key in ('semantic_logits', 'instance_logits'):
                        aggregated_components[key] = self._aggregate_query_logits_to_classes(value)
                    elif key == 'presence_query_scores':
                        aggregated_components['presence_scores'] = (
                            self._aggregate_query_scores_to_classes(value))
                    else:
                        aggregated_components[key] = value
                components = aggregated_components
            seg_logits = self._apply_evidence_competition_graph(seg_logits, components)
            seg_logits = self._apply_reject_aware_calibration(seg_logits, components)

            base_seg_logits = seg_logits
            coco_sec_context = None
            coco_sec_logits = None
            coco_sec_pred = None
            if self.use_coco_sec_fusion or self.dump_coco_sec_stats:
                coco_sec_logits, coco_sec_context = self._apply_coco_sec_fusion(
                    base_seg_logits, components)
                coco_sec_pred = self._threshold_with_reject_recovery(
                    coco_sec_logits, components)
                if self.use_coco_sec_fusion:
                    seg_logits = coco_sec_logits

            base_seg_pred = self._threshold_with_reject_recovery(base_seg_logits, components)
            evidence_enhancement_context = None
            evidence_enhancement_logits = None
            evidence_enhancement_pred = None
            if self._uses_evidence_enhancement():
                (
                    evidence_enhancement_logits,
                    evidence_enhancement_context,
                ) = self._build_evidence_enhancement_logits(
                    base_seg_logits,
                    base_seg_pred,
                    image,
                    data_samples[i],
                    components,
                )
                evidence_enhancement_pred = (
                    self._threshold_with_reject_recovery(
                        evidence_enhancement_logits, components)
                )
                if self.use_evidence_enhancement:
                    seg_logits = evidence_enhancement_logits
            structure_recalibration_context = None
            structure_recalibration_logits = None
            structure_recalibration_pred = None
            if self._uses_structure_aware_recalibration():
                (
                    structure_recalibration_logits,
                    structure_recalibration_context,
                ) = self._build_structure_aware_recalibration_logits(
                    base_seg_logits,
                    base_seg_pred,
                    components,
                )
                structure_recalibration_pred = (
                    self._threshold_with_reject_recovery(
                        structure_recalibration_logits, components)
                )
                if self.use_structure_aware_recalibration:
                    seg_logits = structure_recalibration_logits
            active_concept_context = None
            active_concept_logits = None
            active_concept_pred = None
            if self._uses_active_concept_diagnostic():
                active_concept_context = self._build_active_concept_context(
                    base_seg_logits,
                    base_seg_pred,
                    components,
                    data_samples[i],
                )
                if self.use_active_concept_pruning:
                    active_concept_logits = (
                        self._apply_active_concept_pruning_from_context(
                            base_seg_logits,
                            active_concept_context,
                            self.active_concept_apply_variant,
                        )
                    )
                    active_concept_pred = self._threshold_with_reject_recovery(
                        active_concept_logits,
                        components,
                    )
                    seg_logits = active_concept_logits

            concept_specificity_context = None
            concept_specificity_logits = None
            concept_specificity_pred = None
            if self._uses_concept_specificity_diagnostic():
                if query_semantic_logits is None:
                    raise RuntimeError(
                        'Concept-specificity diagnostic requires '
                        'prompt-level semantic maps.')
                concept_specificity_context = (
                    self._build_concept_specificity_context(
                        base_seg_logits,
                        base_seg_pred,
                        query_seg_logits,
                        query_semantic_logits,
                        components,
                        data_samples[i],
                    )
                )
                if self.use_concept_specificity_pruning:
                    concept_specificity_logits = (
                        self._apply_concept_specificity_pruning_from_context(
                            base_seg_logits,
                            concept_specificity_context,
                            self.concept_specificity_apply_ranker,
                        )
                    )
                    concept_specificity_pred = (
                        self._threshold_with_reject_recovery(
                            concept_specificity_logits,
                            components,
                        )
                    )
                    seg_logits = concept_specificity_logits

            cross_image_bank_stats = None
            if self._uses_cross_image_bank_diagnostic():
                cross_image_bank_stats = self._build_cross_image_bank_stats(
                    base_seg_logits,
                    base_seg_pred,
                    data_samples[i],
                    components,
                    image=image,
                )
            cross_image_bank_recomposition_context = None
            cross_image_bank_recomposition_logits = None
            if self._uses_cross_image_bank_recomposition():
                (
                    cross_image_bank_recomposition_logits,
                    cross_image_bank_recomposition_context,
                ) = self._build_cross_image_bank_recomposition_logits(
                    base_seg_logits,
                    base_seg_pred,
                    components,
                    image=image,
                    image_id=image_path,
                    data_sample=data_samples[i],
                )
                if self.use_cross_image_bank_recomposition:
                    seg_logits = cross_image_bank_recomposition_logits
            ontology_self_verification_context = None
            ontology_self_verification_logits = None
            if self._uses_ontology_self_verification():
                (
                    ontology_self_verification_logits,
                    ontology_self_verification_context,
                ) = self._build_ontology_self_verification_logits(
                    base_seg_logits,
                    base_seg_pred,
                    components,
                    data_samples[i],
                )
                if self.use_ontology_self_verification:
                    seg_logits = ontology_self_verification_logits
            sam3_geometry_requery_context = None
            if self._uses_sam3_geometry_requery_diagnostic():
                sam3_geometry_requery_context = (
                    self._build_sam3_geometry_requery_stats(
                        base_seg_logits,
                        base_seg_pred,
                        components,
                        data_samples[i],
                        image,
                    )
                )
            self_prompted_concept_verification_context = None
            if self._uses_self_prompted_concept_verification():
                self_prompted_concept_verification_context = (
                    self._build_self_prompted_concept_verification_stats(
                        base_seg_logits,
                        base_seg_pred,
                        data_samples[i],
                        image,
                    )
                )
            state_action_stats = None
            state_action_logits = None
            state_action_pred = None
            if self._uses_state_action_atlas():
                if (
                        query_semantic_logits is None
                        or query_instance_logits is None):
                    raise RuntimeError(
                        'State-action atlas requires semantic and instance '
                        'query maps.')
                (
                    state_action_stats,
                    state_action_logits,
                    state_action_pred,
                ) = self._build_state_action_atlas(
                    base_seg_logits,
                    base_seg_pred,
                    query_seg_logits,
                    query_semantic_logits,
                    query_instance_logits,
                    components,
                    data_samples[i],
                )
            region_readout_context = None
            region_readout_logits = None
            region_readout_pred = None
            if self._uses_region_contrastive_readout():
                if query_semantic_logits is None:
                    raise RuntimeError(
                        'Region contrastive readout requires prompt-level '
                        'semantic maps.')
                (
                    region_readout_logits,
                    region_readout_pred,
                    region_readout_context,
                ) = self._build_region_contrastive_readout(
                    base_seg_logits,
                    base_seg_pred,
                    query_semantic_logits,
                    components,
                    data_samples[i],
                )
                if self.use_region_contrastive_readout:
                    seg_logits = region_readout_logits

            region_hypothesis_v2_context = None
            if self._uses_region_hypothesis_v2():
                if query_semantic_logits is None:
                    raise RuntimeError(
                        'Region hypothesis v2 diagnostic requires '
                        'prompt-level semantic maps.')
                region_hypothesis_v2_context = (
                    self._build_region_hypothesis_v2_diagnostic(
                        base_seg_logits,
                        base_seg_pred,
                        query_semantic_logits,
                        components,
                        data_samples[i],
                    )
                )

            candidate_region_quality_context = None
            if self._uses_candidate_region_quality_diagnostic():
                candidate_region_quality_context = (
                    self._build_candidate_region_quality_diagnostic(
                        base_seg_logits,
                        base_seg_pred,
                        components,
                        data_samples[i],
                    )
                )

            query_topology_context = None
            if self._uses_query_topology_diagnostic():
                if (
                        query_semantic_logits is None
                        or query_instance_logits is None):
                    raise RuntimeError(
                        'Query-topology diagnostic requires prompt-level '
                        'semantic and instance maps.')
                query_topology_context = (
                    self._build_query_topology_diagnostic(
                        base_seg_logits,
                        base_seg_pred,
                        query_seg_logits,
                        query_semantic_logits,
                        query_instance_logits,
                        components,
                        data_samples[i],
                        image_path,
                    )
                )

            top2_risk_context = None
            top2_risk_logits = None
            top2_risk_pred = None
            if self.use_top2_risk_arbitration or self.dump_top2_risk_stats:
                top2_risk_logits, top2_risk_context = self._build_top2_risk_arbitration_logits(
                    base_seg_logits, components)
                top2_risk_pred = self._threshold_with_reject_recovery(
                    top2_risk_logits, components)
                if self.use_top2_risk_arbitration:
                    seg_logits = top2_risk_logits

            topk_verifier_context = None
            topk_verifier_logits = None
            topk_verifier_pred = None
            if self.use_topk_candidate_verifier or self.dump_topk_verifier_stats:
                topk_verifier_logits, topk_verifier_context = self._build_topk_candidate_verifier_logits(
                    base_seg_logits, components)
                topk_verifier_pred = self._threshold_with_reject_recovery(
                    topk_verifier_logits, components)
                if self.use_topk_candidate_verifier:
                    seg_logits = topk_verifier_logits

            scale_stability_rerank_context = None
            scale_stability_rerank_logits = None
            scale_stability_rerank_pred = None
            if (self.use_scale_stability_rerank
                    or self.dump_scale_stability_rerank_stats
                    or self.dump_expert_reliability_stats):
                scale_stability_rerank_logits, scale_stability_rerank_context = (
                    self._build_scale_stability_rerank_logits(
                        base_seg_logits, image))
                scale_stability_rerank_pred = self._threshold_with_reject_recovery(
                    scale_stability_rerank_logits, components)
                if self.use_scale_stability_rerank:
                    seg_logits = scale_stability_rerank_logits

            geoer_router_context = None
            geoer_router_logits = None
            geoer_router_pred = None
            if self.use_geoer_router or self.dump_geoer_router_stats:
                geoer_router_logits, geoer_router_pred, geoer_router_context = (
                    self._build_geoer_router_prediction(
                        base_seg_logits,
                        base_seg_pred,
                        image,
                        components))
                if self.use_geoer_router:
                    seg_logits = geoer_router_logits

            if self.dump_reviewer_cache:
                self._dump_reviewer_cache_record(
                    image_path,
                    data_samples[i],
                    base_seg_logits,
                    base_seg_pred,
                    components,
                )

            learned_reviewer_context = None
            learned_reviewer_logits = None
            if self.use_learned_reviewer:
                (
                    learned_reviewer_logits,
                    learned_reviewer_context,
                ) = self._apply_learned_reviewer(
                    base_seg_logits,
                    base_seg_pred,
                    components,
                )
                seg_logits = learned_reviewer_logits

            residual_background_context = None
            residual_background_pred = None
            if self._uses_residual_background_modeling():
                (
                    residual_background_pred,
                    residual_background_context,
                ) = self._build_residual_background_prediction(
                    base_seg_logits,
                    base_seg_pred,
                    components,
                )

            # Apply probability threshold, optionally with reject recovery.
            seg_pred = self._threshold_with_reject_recovery(seg_logits, components)
            if (
                    self.use_residual_background_modeling
                    and residual_background_pred is not None):
                seg_pred = residual_background_pred
            if self.use_geoer_router and geoer_router_pred is not None:
                seg_pred = geoer_router_pred
            if (
                    str(
                        self.state_action_apply
                        or 'baseline').strip().lower()
                    != 'baseline'):
                if (
                        state_action_logits is None
                        or state_action_pred is None):
                    raise RuntimeError(
                        'Requested state_action_apply was not built.')
                seg_logits = state_action_logits
                seg_pred = state_action_pred
            if (
                    self.use_region_contrastive_readout
                    and region_readout_logits is not None):
                seg_logits = region_readout_logits
                seg_pred = (
                    region_readout_pred
                    if region_readout_pred is not None
                    else self._threshold_with_reject_recovery(
                        seg_logits, components)
                )

            if self.dump_structure_aware_recalibration_stats:
                if structure_recalibration_logits is None:
                    structure_recalibration_logits = base_seg_logits
                if structure_recalibration_pred is None:
                    structure_recalibration_pred = base_seg_pred
                structure_recalibration_stats = (
                    self._build_structure_aware_recalibration_stats(
                        base_seg_logits,
                        base_seg_pred,
                        structure_recalibration_logits,
                        structure_recalibration_pred,
                        components,
                        data_samples[i],
                        structure_recalibration_context,
                    )
                )
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    use_structure_aware_recalibration=bool(
                        self.use_structure_aware_recalibration),
                    structure_recalibration_variant=str(
                        self.structure_recalibration_variant),
                    structure_aware_recalibration_stats=(
                        structure_recalibration_stats),
                )
                self._write_structure_aware_recalibration_stats(record)

            if self.dump_learned_reviewer_stats:
                gt = data_samples[i].gt_sem_seg.data
                if gt.ndim == 3:
                    gt = gt.squeeze(0)
                gt = gt.to(seg_pred.device)
                valid = gt != 255
                base_correct = base_seg_pred == gt
                reviewed_correct = seg_pred == gt
                changed = valid & (base_seg_pred != seg_pred)
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    dataset_name=self.reviewer_dataset_name,
                    reviewer_variant=self.reviewer_variant,
                    reviewer_checkpoint=self.reviewer_checkpoint,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(
                        self.confidence_threshold),
                    valid_pixels=int(valid.sum().item()),
                    changed_pixels=int(changed.sum().item()),
                    improved_pixels=int(
                        (changed & ~base_correct & reviewed_correct).sum(
                        ).item()),
                    harmed_pixels=int(
                        (changed & base_correct & ~reviewed_correct).sum(
                        ).item()),
                    wrong_to_wrong_pixels=int(
                        (changed & ~base_correct & ~reviewed_correct).sum(
                        ).item()),
                    reviewer_context=learned_reviewer_context,
                )
                self._write_learned_reviewer_stats(record)

            if self.dump_residual_background_stats:
                residual_background_stats = (
                    self._build_residual_background_stats(
                        base_seg_pred,
                        residual_background_pred,
                        data_samples[i],
                        residual_background_context,
                    )
                )
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    use_residual_background_modeling=bool(
                        self.use_residual_background_modeling),
                    residual_background_route=str(
                        self.residual_background_route),
                    residual_background_score_factor=float(
                        self.residual_background_score_factor),
                    residual_background_min_score=float(
                        self.residual_background_min_score),
                    residual_background_min_semantic=float(
                        self.residual_background_min_semantic),
                    residual_background_min_instance=float(
                        self.residual_background_min_instance),
                    residual_background_min_presence=float(
                        self.residual_background_min_presence),
                    residual_background_min_local=float(
                        self.residual_background_min_local),
                    residual_background_min_fg_bg_margin=float(
                        self.residual_background_min_fg_bg_margin),
                    residual_background_min_reliability_margin=float(
                        self.residual_background_min_reliability_margin),
                    residual_background_semantic_weight=float(
                        self.residual_background_semantic_weight),
                    residual_background_instance_weight=float(
                        self.residual_background_instance_weight),
                    residual_background_presence_weight=float(
                        self.residual_background_presence_weight),
                    residual_background_local_weight=float(
                        self.residual_background_local_weight),
                    residual_background_bg_weight=float(
                        self.residual_background_bg_weight),
                    residual_background_local_kernel=int(
                        self.residual_background_local_kernel),
                    residual_background_require_head_support=bool(
                        self.residual_background_require_head_support),
                    residual_background_stats=residual_background_stats,
                )
                self._write_residual_background_stats(record)

            if self.dump_background_separability_stats:
                background_separability_stats = (
                    self._build_background_separability_stats(
                        base_seg_pred,
                        data_samples[i],
                        residual_background_context,
                    )
                )
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    background_separability_stats=(
                        background_separability_stats),
                )
                self._write_background_separability_stats(record)

            if self.dump_active_concept_stats:
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    active_concept_apply_variant=str(
                        self.active_concept_apply_variant),
                    use_active_concept_pruning=bool(
                        self.use_active_concept_pruning),
                    active_concept_stats=active_concept_context,
                )
                self._write_active_concept_stats(record)

            if self.dump_local_active_set_stats:
                local_active_set_stats = self._build_local_active_set_stats(
                    base_seg_logits,
                    base_seg_pred,
                    components,
                    data_samples[i],
                )
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    local_active_max_side=int(self.local_active_max_side),
                    local_active_windows=str(self.local_active_windows),
                    local_active_topk=int(self.local_active_topk),
                    local_active_score_thd=(
                        None
                        if self.local_active_score_thd is None
                        else float(self.local_active_score_thd)),
                    local_active_area_thresholds=str(
                        self.local_active_area_thresholds),
                    local_active_sources=str(self.local_active_sources),
                    local_active_min_pair_pixels=int(
                        self.local_active_min_pair_pixels),
                    local_active_set_stats=local_active_set_stats,
                )
                self._write_local_active_set_stats(record)

            if self.dump_prompt_winner_attribution_stats:
                prompt_winner_stats = (
                    self._build_prompt_winner_attribution_stats(
                        query_seg_logits,
                        query_semantic_logits,
                        query_instance_logits,
                        base_seg_logits,
                        base_seg_pred,
                        components,
                        data_samples[i],
                    ))
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    prompt_winner_max_side=int(self.prompt_winner_max_side),
                    prompt_winner_sources=str(self.prompt_winner_sources),
                    prompt_winner_min_pair_pixels=int(
                        self.prompt_winner_min_pair_pixels),
                    prompt_winner_attribution_stats=prompt_winner_stats,
                )
                self._write_prompt_winner_attribution_stats(record)

            if self.dump_region_prompt_identity_stats:
                region_prompt_identity_stats = (
                    self._build_region_prompt_identity_stats(
                        base_seg_logits,
                        base_seg_pred,
                        components,
                        data_samples[i],
                    ))
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    raw_mask_oracle_topk=int(self.raw_mask_oracle_topk),
                    region_prompt_identity_max_side=int(
                        self.region_prompt_identity_max_side),
                    region_prompt_identity_bin_thd=float(
                        self.region_prompt_identity_bin_thd),
                    region_prompt_identity_min_pixels=int(
                        self.region_prompt_identity_min_pixels),
                    region_prompt_identity_group_iou=float(
                        self.region_prompt_identity_group_iou),
                    region_prompt_identity_group_containment=float(
                        self.region_prompt_identity_group_containment),
                    region_prompt_identity_max_regions=int(
                        self.region_prompt_identity_max_regions),
                    region_prompt_identity_ring_kernel=int(
                        self.region_prompt_identity_ring_kernel),
                    region_prompt_identity_min_pair_pixels=int(
                        self.region_prompt_identity_min_pair_pixels),
                    region_prompt_identity_stats=(
                        region_prompt_identity_stats),
                )
                self._write_region_prompt_identity_stats(record)

            if self.dump_concept_specificity_stats:
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    use_concept_specificity_pruning=bool(
                        self.use_concept_specificity_pruning),
                    concept_specificity_apply_ranker=str(
                        self.concept_specificity_apply_ranker),
                    concept_specificity_max_side=int(
                        self.concept_specificity_max_side),
                    concept_specificity_prune_active_count=(
                        self.concept_specificity_prune_active_count),
                    concept_specificity_stats=(
                        concept_specificity_context),
                )
                self._write_concept_specificity_stats(record)

            if self.dump_cross_image_bank_stats:
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    cross_image_bank_spaces=self._get_cross_image_bank_spaces(),
                    cross_image_bank_seed_rule=str(
                        self.cross_image_bank_seed_rule),
                    cross_image_bank_feature_max_side=int(
                        self.cross_image_bank_feature_max_side),
                    cross_image_bank_stats=cross_image_bank_stats,
                )
                self._write_cross_image_bank_stats(record)

            if self.dump_cross_image_bank_recomposition_stats:
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    use_cross_image_bank_recomposition=bool(
                        self.use_cross_image_bank_recomposition),
                    cross_image_bank_file=str(self.cross_image_bank_file),
                    cross_image_bank_apply_space=str(
                        self.cross_image_bank_apply_space),
                    cross_image_bank_apply_variant=str(
                        self.cross_image_bank_apply_variant),
                    cross_image_bank_apply_topk=int(
                        self.cross_image_bank_apply_topk),
                    cross_image_bank_apply_min_bank_margin=float(
                        self.cross_image_bank_apply_min_bank_margin),
                    cross_image_bank_apply_max_base_gap=float(
                        self.cross_image_bank_apply_max_base_gap),
                    cross_image_bank_recomposition=(
                        cross_image_bank_recomposition_context),
                )
                self._write_cross_image_bank_recomposition_stats(record)

            if self.dump_ontology_self_verification_stats:
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    use_ontology_self_verification=bool(
                        self.use_ontology_self_verification),
                    ontology_self_verification=(
                        ontology_self_verification_context),
                )
                self._write_ontology_self_verification_stats(record)

            if self.dump_sam3_geometry_requery_stats:
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    sam3_geometry_requery_topk=int(
                        self.sam3_geometry_requery_topk),
                    sam3_geometry_requery_max_pairs=int(
                        self.sam3_geometry_requery_max_pairs),
                    sam3_geometry_requery_box_modes=str(
                        self.sam3_geometry_requery_box_modes),
                    sam3_geometry_requery_prompt_mode=str(
                        self.sam3_geometry_requery_prompt_mode),
                    sam3_geometry_requery_max_side=int(
                        self.sam3_geometry_requery_max_side),
                    sam3_geometry_requery=(
                        sam3_geometry_requery_context),
                )
                self._write_sam3_geometry_requery_stats(record)

            if self.dump_self_prompted_concept_verification_stats:
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    self_prompted_concept_verification_topk=int(
                        self.self_prompted_concept_verification_topk),
                    self_prompted_concept_verification_max_pairs=int(
                        self.self_prompted_concept_verification_max_pairs),
                    self_prompted_concept_verification_max_base_gap=float(
                        self.self_prompted_concept_verification_max_base_gap),
                    self_prompted_concept_verification_max_side=int(
                        self.self_prompted_concept_verification_max_side),
                    self_prompted_concept_verification_prompt_mode=str(
                        self.self_prompted_concept_verification_prompt_mode),
                    self_prompted_concept_verification=(
                        self_prompted_concept_verification_context),
                )
                self._write_self_prompted_concept_verification_stats(record)

            if self.dump_evidence_enhancement_stats:
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    use_evidence_enhancement=bool(
                        self.use_evidence_enhancement),
                    evidence_enhancement_classes=str(
                        self.evidence_enhancement_classes),
                    evidence_enhancement_source=str(
                        self.evidence_enhancement_source),
                    evidence_enhancement_presence_mode=str(
                        self.evidence_enhancement_presence_mode),
                    evidence_enhancement_reduce=str(
                        self.evidence_enhancement_reduce),
                    evidence_enhancement_combine=str(
                        self.evidence_enhancement_combine),
                    evidence_enhancement_max_prompts=int(
                        self.evidence_enhancement_max_prompts),
                    evidence_enhancement=(
                        evidence_enhancement_context),
                )
                self._write_evidence_enhancement_stats(record)

            if self.dump_evidence_stats:
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    use_sem_seg=bool(self.use_sem_seg),
                    use_transformer_decoder=bool(self.use_transformer_decoder),
                    use_presence_score=bool(self.use_presence_score),
                    instance_score_type=self.instance_score_type,
                    evidence=evidence_stats,
                    class_stats=self._build_class_evidence_stats(seg_logits, seg_pred, data_samples[i]),
                )
                self._write_evidence_stats(record)

            if self.dump_competition_stats:
                competition_stats = self._build_competition_stats(
                    seg_logits, seg_pred, data_samples[i], components)
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    use_sem_seg=bool(self.use_sem_seg),
                    use_transformer_decoder=bool(self.use_transformer_decoder),
                    use_presence_score=bool(self.use_presence_score),
                    instance_score_type=self.instance_score_type,
                    competition_stats=competition_stats,
                )
                self._write_competition_stats(record)

            if self.dump_oracle_stats:
                oracle_stats = self._build_oracle_stats(
                    seg_logits, seg_pred, data_samples[i], components)
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    use_sem_seg=bool(self.use_sem_seg),
                    use_transformer_decoder=bool(self.use_transformer_decoder),
                    use_presence_score=bool(self.use_presence_score),
                    instance_score_type=self.instance_score_type,
                    oracle_topk=int(self.oracle_topk),
                    use_evidence_competition_graph=bool(self.use_evidence_competition_graph),
                    use_reject_aware_calibration=bool(self.use_reject_aware_calibration),
                    oracle_stats=oracle_stats,
                )
                self._write_oracle_stats(record)

            if self.dump_topk_verifier_stats:
                verifier_stats = self._build_topk_verifier_stats(
                    base_seg_logits,
                    topk_verifier_logits,
                    base_seg_pred,
                    topk_verifier_pred,
                    data_samples[i],
                    components,
                    topk_verifier_context)
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    instance_score_type=self.instance_score_type,
                    use_topk_candidate_verifier=bool(self.use_topk_candidate_verifier),
                    topk_verifier_k=int(self.topk_verifier_k),
                    topk_verifier_apply_mode=self.topk_verifier_apply_mode,
                    topk_verifier_require_final_candidate=bool(self.topk_verifier_require_final_candidate),
                    topk_verifier_min_aux_rank=float(self.topk_verifier_min_aux_rank),
                    topk_verifier_stats=verifier_stats,
                )
                self._write_topk_verifier_stats(record)

            if self.dump_error_rank_stats:
                error_rank_stats = self._build_error_rank_stats(
                    base_seg_logits,
                    base_seg_pred,
                    data_samples[i],
                    components)
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    instance_score_type=self.instance_score_type,
                    error_rank_topk=int(self.error_rank_topk),
                    use_evidence_competition_graph=bool(self.use_evidence_competition_graph),
                    use_reject_aware_calibration=bool(self.use_reject_aware_calibration),
                    error_rank_stats=error_rank_stats,
                )
                self._write_error_rank_stats(record)

            if self.dump_top2_risk_stats:
                top2_risk_stats = self._build_top2_risk_stats(
                    base_seg_logits,
                    top2_risk_logits,
                    base_seg_pred,
                    top2_risk_pred,
                    data_samples[i],
                    top2_risk_context)
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    instance_score_type=self.instance_score_type,
                    use_top2_risk_arbitration=bool(self.use_top2_risk_arbitration),
                    top2_risk_topk=int(self.top2_risk_topk),
                    top2_risk_mode=self.top2_risk_mode,
                    top2_risk_bg_policy=self.top2_risk_bg_policy,
                    top2_risk_min_margin=float(self.top2_risk_min_margin),
                    top2_risk_max_margin=float(self.top2_risk_max_margin),
                    top2_risk_min_sem_adv=float(self.top2_risk_min_sem_adv),
                    top2_risk_min_inst_adv=float(self.top2_risk_min_inst_adv),
                    top2_risk_min_agreement_adv=float(self.top2_risk_min_agreement_adv),
                    top2_risk_min_top1_sem_only=float(self.top2_risk_min_top1_sem_only),
                    top2_risk_require_head_disagree=bool(self.top2_risk_require_head_disagree),
                    top2_risk_stats=top2_risk_stats,
                )
                self._write_top2_risk_stats(record)

            if self.dump_multiview_oracle_stats:
                multiview_oracle_stats = self._build_multiview_oracle_stats(
                    base_seg_logits,
                    base_seg_pred,
                    data_samples[i],
                    components,
                    image,
                    ori_shape)
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    instance_score_type=self.instance_score_type,
                    multiview_oracle_topk=int(self.multiview_oracle_topk),
                    multiview_oracle_views=self.multiview_oracle_views,
                    multiview_oracle_slide_views=self.multiview_oracle_slide_views,
                    multiview_oracle_include_base=bool(self.multiview_oracle_include_base),
                    multiview_oracle_stats=multiview_oracle_stats,
                )
                self._write_multiview_oracle_stats(record)

            if self.dump_seed_separability_stats:
                seed_separability_stats = self._build_seed_separability_stats(
                    base_seg_logits,
                    base_seg_pred,
                    data_samples[i],
                    components)
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    instance_score_type=self.instance_score_type,
                    seed_final_score_thd=float(self.seed_final_score_thd),
                    seed_margin_thd=float(self.seed_margin_thd),
                    seed_local_kernel=int(self.seed_local_kernel),
                    seed_local_consistency_thd=float(self.seed_local_consistency_thd),
                    seed_core_kernel=int(self.seed_core_kernel),
                    seed_core_consistency_thd=float(self.seed_core_consistency_thd),
                    seed_region_min_pixels=int(self.seed_region_min_pixels),
                    seed_region_purity_thd=float(self.seed_region_purity_thd),
                    seed_similarity_spaces=self._get_seed_similarity_spaces(),
                    use_evidence_competition_graph=bool(self.use_evidence_competition_graph),
                    use_reject_aware_calibration=bool(self.use_reject_aware_calibration),
                    seed_separability_stats=seed_separability_stats,
                )
                self._write_seed_separability_stats(record)

            if self.dump_raw_mask_oracle_stats:
                raw_mask_oracle_stats = self._build_raw_mask_oracle_stats(
                    base_seg_logits,
                    base_seg_pred,
                    data_samples[i],
                    components)
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    instance_score_type=self.instance_score_type,
                    raw_mask_oracle_topk=int(self.raw_mask_oracle_topk),
                    raw_mask_oracle_bin_thd=float(self.raw_mask_oracle_bin_thd),
                    raw_mask_oracle_min_pixels=int(self.raw_mask_oracle_min_pixels),
                    use_evidence_competition_graph=bool(self.use_evidence_competition_graph),
                    use_reject_aware_calibration=bool(self.use_reject_aware_calibration),
                    raw_mask_oracle_stats=raw_mask_oracle_stats,
                )
                self._write_raw_mask_oracle_stats(record)

            if self.dump_candidate_quality_stats:
                candidate_quality_stats = self._build_candidate_quality_stats(
                    base_seg_logits,
                    base_seg_pred,
                    data_samples[i],
                    components)
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    instance_score_type=self.instance_score_type,
                    raw_mask_oracle_topk=int(self.raw_mask_oracle_topk),
                    raw_mask_oracle_bin_thd=float(self.raw_mask_oracle_bin_thd),
                    raw_mask_oracle_min_pixels=int(self.raw_mask_oracle_min_pixels),
                    candidate_quality_clean_purity_thd=float(
                        self.candidate_quality_clean_purity_thd),
                    use_evidence_competition_graph=bool(self.use_evidence_competition_graph),
                    use_reject_aware_calibration=bool(self.use_reject_aware_calibration),
                    candidate_quality_stats=candidate_quality_stats,
                )
                self._write_candidate_quality_stats(record)

            if self.dump_prompt_competition_stats:
                prompt_competition_stats = self._build_prompt_competition_stats(
                    base_seg_logits,
                    base_seg_pred,
                    data_samples[i],
                    image,
                    ori_shape)
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    instance_score_type=self.instance_score_type,
                    prompt_competition_class_names=self.prompt_competition_class_names,
                    prompt_competition_templates=self._get_prompt_competition_templates(),
                    prompt_competition_max_variants=int(self.prompt_competition_max_variants),
                    prompt_competition_use_builtin_variants=bool(
                        self.prompt_competition_use_builtin_variants),
                    prompt_competition_stats=prompt_competition_stats,
                )
                self._write_prompt_competition_stats(record)

            if self.dump_pair_prompt_competition_stats:
                pair_prompt_competition_stats = self._build_pair_prompt_competition_stats(
                    base_seg_logits,
                    base_seg_pred,
                    data_samples[i],
                    image,
                    ori_shape)
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    instance_score_type=self.instance_score_type,
                    pair_prompt_competition_pairs=self.pair_prompt_competition_pairs,
                    pair_prompt_competition_templates=self._get_pair_prompt_competition_templates(),
                    pair_prompt_competition_max_variants=int(
                        self.pair_prompt_competition_max_variants),
                    pair_prompt_competition_use_builtin_variants=bool(
                        self.pair_prompt_competition_use_builtin_variants),
                    pair_prompt_competition_stats=pair_prompt_competition_stats,
                )
                self._write_pair_prompt_competition_stats(record)

            if self.dump_geometry_context_stats:
                geometry_context_stats = self._build_geometry_context_stats(
                    base_seg_logits,
                    base_seg_pred,
                    data_samples[i],
                    components)
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    instance_score_type=self.instance_score_type,
                    geometry_context_score_thd=float(self.geometry_context_score_thd),
                    geometry_context_min_pixels=int(self.geometry_context_min_pixels),
                    geometry_context_core_kernel=int(self.geometry_context_core_kernel),
                    geometry_context_stats=geometry_context_stats,
                )
                self._write_geometry_context_stats(record)

            if self.dump_weak_support_stats:
                weak_support_stats = self._build_weak_support_stats(
                    base_seg_logits,
                    base_seg_pred,
                    data_samples[i],
                    components)
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    instance_score_type=self.instance_score_type,
                    weak_support_thresholds=self._get_weak_support_thresholds(),
                    weak_support_sources=self._get_weak_support_sources(),
                    weak_support_min_pair_pixels=int(self.weak_support_min_pair_pixels),
                    weak_support_stats=weak_support_stats,
                )
                self._write_weak_support_stats(record)

            if self.dump_reject_recovery_precision_stats:
                reject_recovery_precision_stats = self._build_reject_recovery_precision_stats(
                    base_seg_logits,
                    base_seg_pred,
                    data_samples[i],
                    components)
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    instance_score_type=self.instance_score_type,
                    reject_recovery_precision_score_bins=self._parse_float_list(
                        self.reject_recovery_precision_score_bins,
                        '0.00,0.03,0.05,0.07,0.10,0.15,0.20,0.30'),
                    reject_recovery_precision_semantic_thds=self._parse_float_list(
                        self.reject_recovery_precision_semantic_thds,
                        '0.03,0.05,0.10'),
                    reject_recovery_precision_instance_thds=self._parse_float_list(
                        self.reject_recovery_precision_instance_thds,
                        '0.01,0.03,0.05'),
                    reject_recovery_precision_margin_thds=self._parse_float_list(
                        self.reject_recovery_precision_margin_thds,
                        '0.00,0.02,0.05'),
                    reject_recovery_precision_local_thds=self._parse_float_list(
                        self.reject_recovery_precision_local_thds,
                        '0.50,0.70,0.85'),
                    reject_recovery_precision_use_pe=bool(
                        self.reject_recovery_precision_use_pe),
                    reject_recovery_precision_pe_spaces=self._get_reject_recovery_precision_pe_spaces(),
                    reject_recovery_precision_stats=reject_recovery_precision_stats,
                )
                self._write_reject_recovery_precision_stats(record)

            if self.dump_context_requery_stats:
                context_requery_stats = self._build_context_requery_stats(
                    base_seg_logits,
                    base_seg_pred,
                    data_samples[i],
                    image,
                    ori_shape)
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    instance_score_type=self.instance_score_type,
                    context_requery_pairs=self.context_requery_pairs,
                    context_requery_modes=self._get_context_requery_modes(),
                    context_requery_thresholds=self._parse_float_list(
                        self.context_requery_thresholds,
                        '0.05,0.10,0.20,0.50'),
                    context_requery_min_pair_pixels=int(
                        self.context_requery_min_pair_pixels),
                    context_requery_min_crop_size=int(
                        self.context_requery_min_crop_size),
                    context_requery_max_crop_size=int(
                        self.context_requery_max_crop_size),
                    context_requery_stats=context_requery_stats,
                )
                self._write_context_requery_stats(record)

            if self.dump_scale_requery_stats:
                scale_requery_stats = self._build_scale_requery_stats(
                    base_seg_logits,
                    base_seg_pred,
                    data_samples[i],
                    image)
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    instance_score_type=self.instance_score_type,
                    scale_requery_pairs=self.scale_requery_pairs,
                    scale_requery_scales=self._get_scale_requery_scales(),
                    scale_requery_thresholds=self._parse_float_list(
                        self.scale_requery_thresholds,
                        '0.05,0.10,0.20,0.50'),
                    scale_requery_min_pair_pixels=int(
                        self.scale_requery_min_pair_pixels),
                    scale_requery_max_side=int(self.scale_requery_max_side),
                    scale_requery_stats=scale_requery_stats,
                )
                self._write_scale_requery_stats(record)

            if self.dump_scale_stability_stats:
                scale_stability_stats = self._build_scale_stability_stats(
                    base_seg_logits,
                    base_seg_pred,
                    data_samples[i],
                    image)
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    instance_score_type=self.instance_score_type,
                    scale_stability_pairs=self.scale_stability_pairs,
                    scale_stability_scale=float(self.scale_stability_scale),
                    scale_stability_topk=int(self.scale_stability_topk),
                    scale_stability_drop_thresholds=self._parse_float_list(
                        self.scale_stability_drop_thresholds,
                        '0.02,0.05,0.10'),
                    scale_stability_min_pair_pixels=int(
                        self.scale_stability_min_pair_pixels),
                    scale_stability_max_side=int(self.scale_stability_max_side),
                    scale_stability_stats=scale_stability_stats,
                )
                self._write_scale_stability_stats(record)

            if self.dump_scale_stability_rerank_stats:
                scale_stability_rerank_stats = self._build_scale_stability_rerank_stats(
                    base_seg_logits,
                    scale_stability_rerank_logits,
                    base_seg_pred,
                    scale_stability_rerank_pred,
                    data_samples[i],
                    scale_stability_rerank_context)
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    instance_score_type=self.instance_score_type,
                    use_scale_stability_rerank=bool(
                        self.use_scale_stability_rerank),
                    scale_stability_rerank_pairs=(
                        self.scale_stability_rerank_pairs),
                    scale_stability_rerank_pair_file=(
                        self.scale_stability_rerank_pair_file),
                    scale_stability_rerank_pair_mode=str(
                        self.scale_stability_rerank_pair_mode),
                    scale_stability_rerank_query_mode=str(
                        self.scale_stability_rerank_query_mode),
                    scale_stability_rerank_auto_min_pixels=int(
                        self.scale_stability_rerank_auto_min_pixels),
                    scale_stability_rerank_auto_exclude_bg=bool(
                        self._scale_stability_rerank_exclude_bg()),
                    scale_stability_rerank_bg_names=str(
                        self.scale_stability_rerank_bg_names),
                    scale_stability_rerank_max_risk_classes=int(
                        self.scale_stability_rerank_max_risk_classes),
                    scale_stability_rerank_scale=float(
                        self.scale_stability_rerank_scale),
                    scale_stability_rerank_topk=int(
                        self.scale_stability_rerank_topk),
                    scale_stability_rerank_gate=str(
                        self.scale_stability_rerank_gate),
                    scale_stability_rerank_drop_threshold=float(
                        self.scale_stability_rerank_drop_threshold),
                    scale_stability_rerank_min_scaled_margin=float(
                        self.scale_stability_rerank_min_scaled_margin),
                    scale_stability_rerank_min_base_margin=float(
                        self.scale_stability_rerank_min_base_margin),
                    scale_stability_rerank_max_base_margin=float(
                        self.scale_stability_rerank_max_base_margin),
                    scale_stability_rerank_max_target_rank=int(
                        self.scale_stability_rerank_max_target_rank),
                    scale_stability_rerank_max_side=int(
                        self.scale_stability_rerank_max_side),
                    scale_stability_rerank_stats=(
                        scale_stability_rerank_stats),
                )
                self._write_scale_stability_rerank_stats(record)

            if self.dump_expert_reliability_stats:
                expert_reliability_stats = self._build_expert_reliability_stats(
                    base_seg_logits,
                    base_seg_pred,
                    data_samples[i],
                    components,
                    scale_stability_rerank_context)
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    instance_score_type=self.instance_score_type,
                    expert_reliability_min_gate_pixels=int(
                        self.expert_reliability_min_gate_pixels),
                    expert_reliability_local_kernel=int(
                        self.expert_reliability_local_kernel),
                    scale_stability_rerank_pair_mode=str(
                        self.scale_stability_rerank_pair_mode),
                    scale_stability_rerank_query_mode=str(
                        self.scale_stability_rerank_query_mode),
                    scale_stability_rerank_pairs=(
                        self.scale_stability_rerank_pairs),
                    scale_stability_rerank_pair_file=(
                        self.scale_stability_rerank_pair_file),
                    scale_stability_rerank_scale=float(
                        self.scale_stability_rerank_scale),
                    scale_stability_rerank_topk=int(
                        self.scale_stability_rerank_topk),
                    scale_stability_rerank_gate=str(
                        self.scale_stability_rerank_gate),
                    scale_stability_rerank_drop_threshold=float(
                        self.scale_stability_rerank_drop_threshold),
                    expert_reliability_stats=expert_reliability_stats,
                )
                self._write_expert_reliability_stats(record)

            if self.dump_evidence_bias_stats:
                evidence_bias_stats = self._build_evidence_bias_stats(
                    base_seg_logits,
                    base_seg_pred,
                    data_samples[i],
                    image)
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    instance_score_type=self.instance_score_type,
                    evidence_bias_pairs=self.evidence_bias_pairs,
                    evidence_bias_pair_mode=str(self.evidence_bias_pair_mode),
                    evidence_bias_probes=self._get_evidence_bias_probes(),
                    evidence_bias_heads=self._get_evidence_bias_heads(),
                    evidence_bias_support_thresholds=self._parse_float_list(
                        self.evidence_bias_support_thresholds,
                        '0.05,0.10,0.20'),
                    evidence_bias_min_pair_pixels=int(
                        self.evidence_bias_min_pair_pixels),
                    evidence_bias_max_gt_pairs=int(
                        self.evidence_bias_max_gt_pairs),
                    evidence_bias_max_side=int(self.evidence_bias_max_side),
                    evidence_bias_stats=evidence_bias_stats,
                )
                self._write_evidence_bias_stats(record)

            if self.dump_topk_conflict_bias_stats:
                topk_conflict_bias_stats = self._build_topk_conflict_bias_stats(
                    base_seg_logits,
                    base_seg_pred,
                    data_samples[i],
                    components,
                    image)
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    instance_score_type=self.instance_score_type,
                    topk_conflict_topk=int(self.topk_conflict_topk),
                    topk_conflict_probes=self._get_topk_conflict_probes(),
                    topk_conflict_heads=self._get_topk_conflict_heads(),
                    topk_conflict_support_thresholds=self._parse_float_list(
                        self.topk_conflict_support_thresholds,
                        '0.05,0.10,0.20'),
                    topk_conflict_min_pair_pixels=int(
                        self.topk_conflict_min_pair_pixels),
                    topk_conflict_max_pairs=int(
                        self.topk_conflict_max_pairs),
                    topk_conflict_max_side=int(
                        self.topk_conflict_max_side),
                    topk_conflict_local_kernel=int(
                        self.topk_conflict_local_kernel),
                    topk_conflict_bias_stats=topk_conflict_bias_stats,
                )
                self._write_topk_conflict_bias_stats(record)

            if self.dump_internal_selection_gap_stats:
                internal_selection_gap_stats = (
                    self._build_internal_selection_gap_stats(
                        base_seg_logits,
                        base_seg_pred,
                        data_samples[i],
                        components,
                    )
                )
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    instance_score_type=self.instance_score_type,
                    internal_selection_sources=(
                        self._get_internal_selection_sources()),
                    internal_selection_topk=int(
                        self.internal_selection_topk),
                    internal_selection_max_side=int(
                        self.internal_selection_max_side),
                    internal_selection_raw_topk=int(
                        self.internal_selection_raw_topk),
                    internal_selection_min_pair_pixels=int(
                        self.internal_selection_min_pair_pixels),
                    internal_selection_seed_rule=str(
                        self.internal_selection_seed_rule),
                    internal_selection_conservative_min_consensus=int(
                        self.internal_selection_conservative_min_consensus),
                    internal_selection_conservative_min_reliability=float(
                        self.internal_selection_conservative_min_reliability),
                    internal_selection_gap_stats=(
                        internal_selection_gap_stats),
                )
                self._write_internal_selection_gap_stats(record)

            if self.dump_candidate_internal_verifier_stats:
                candidate_internal_verifier_stats = (
                    self._build_candidate_internal_verifier_stats(
                        base_seg_logits,
                        base_seg_pred,
                        data_samples[i],
                        components,
                    )
                )
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    instance_score_type=self.instance_score_type,
                    internal_selection_sources=(
                        self._get_internal_selection_sources()),
                    candidate_internal_topk=int(
                        self.candidate_internal_topk),
                    candidate_internal_max_side=int(
                        self.candidate_internal_max_side),
                    candidate_internal_min_pair_pixels=int(
                        self.candidate_internal_min_pair_pixels),
                    candidate_internal_delta_thresholds=(
                        self._parse_float_list(
                            self.candidate_internal_delta_thresholds,
                            '-0.50,-0.25,0.00,0.10,0.25,0.50,1.00',
                        )),
                    candidate_internal_vote_thresholds=(
                        self._parse_float_list(
                            self.candidate_internal_vote_thresholds,
                            '1,2,3',
                        )),
                    candidate_internal_null_trials=int(
                        self.candidate_internal_null_trials),
                    candidate_internal_local_kernel=int(
                        self.candidate_internal_local_kernel),
                    candidate_internal_dump_matched_stats=bool(
                        self.candidate_internal_dump_matched_stats),
                    candidate_internal_matched_families=str(
                        self.candidate_internal_matched_families),
                    candidate_internal_matched_ranks=str(
                        self.candidate_internal_matched_ranks),
                    candidate_internal_matched_regimes=str(
                        self.candidate_internal_matched_regimes),
                    candidate_internal_matched_margin_bins=str(
                        self.candidate_internal_matched_margin_bins),
                    candidate_internal_matched_score_thresholds=str(
                        self.candidate_internal_matched_score_thresholds),
                    candidate_internal_matched_null_trials=int(
                        self.candidate_internal_matched_null_trials),
                    candidate_internal_matched_auc_bins=int(
                        self.candidate_internal_matched_auc_bins),
                    candidate_internal_matched_auc_min=float(
                        self.candidate_internal_matched_auc_min),
                    candidate_internal_matched_auc_max=float(
                        self.candidate_internal_matched_auc_max),
                    candidate_internal_verifier_stats=(
                        candidate_internal_verifier_stats),
                )
                self._write_candidate_internal_verifier_stats(record)

            if self.dump_position_bias_stats:
                position_bias_candidate_stats = (
                    self._build_candidate_internal_verifier_stats(
                        base_seg_logits,
                        base_seg_pred,
                        data_samples[i],
                        components,
                    )
                )
                if position_bias_candidate_stats is not None:
                    position_bias_candidate_stats = {
                        key: position_bias_candidate_stats.get(key)
                        for key in (
                            'dataset_name',
                            'diagnostic_shape',
                            'topk',
                            'valid_pixels',
                            'baseline_correct_pixels',
                            'baseline_wrong_pixels',
                            'family_sources',
                            'matched_stats',
                        )
                    }
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    position_bias_layers=self._get_position_bias_layers(
                        len(components.get('pe_layers', []))
                        if components is not None else 0),
                    position_bias_ranks=str(self.position_bias_ranks),
                    position_bias_controls=str(
                        self.position_bias_controls),
                    position_bias_random_trials=int(
                        self.position_bias_random_trials),
                    position_bias_max_side=int(
                        self.position_bias_max_side),
                    position_bias_seed_rule=str(
                        self.position_bias_seed_rule),
                    candidate_internal_topk=int(
                        self.candidate_internal_topk),
                    candidate_internal_dump_matched_stats=bool(
                        self.candidate_internal_dump_matched_stats),
                    position_bias_feature_stats=(
                        []
                        if components is None
                        else components.get(
                            'position_bias_feature_stats', [])
                    ),
                    position_bias_candidate_stats=(
                        position_bias_candidate_stats),
                )
                self._write_position_bias_stats(record)

            if self.dump_scene_common_bias_stats:
                scene_common_candidate_stats = (
                    self._build_candidate_internal_verifier_stats(
                        base_seg_logits,
                        base_seg_pred,
                        data_samples[i],
                        components,
                    )
                )
                if scene_common_candidate_stats is not None:
                    scene_common_candidate_stats = {
                        key: scene_common_candidate_stats.get(key)
                        for key in (
                            'dataset_name',
                            'diagnostic_shape',
                            'topk',
                            'valid_pixels',
                            'baseline_correct_pixels',
                            'baseline_wrong_pixels',
                            'family_sources',
                            'matched_stats',
                        )
                    }
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(
                        self.confidence_threshold),
                    scene_common_layers=self._get_scene_common_layers(
                        len(components.get('pe_layers', []))
                        if components is not None else 0),
                    scene_common_variants=str(
                        self.scene_common_variants),
                    scene_common_strengths=str(
                        self.scene_common_strengths),
                    scene_common_local_kernels=str(
                        self.scene_common_local_kernels),
                    scene_common_trim_quantile=float(
                        self.scene_common_trim_quantile),
                    scene_common_random_trials=int(
                        self.scene_common_random_trials),
                    scene_common_compute_spectrum=bool(
                        self.scene_common_compute_spectrum),
                    scene_common_max_side=int(
                        self.scene_common_max_side),
                    scene_common_seed_rule=str(
                        self.scene_common_seed_rule),
                    candidate_internal_topk=int(
                        self.candidate_internal_topk),
                    candidate_internal_dump_matched_stats=bool(
                        self.candidate_internal_dump_matched_stats),
                    scene_common_feature_stats=(
                        []
                        if components is None
                        else components.get(
                            'scene_common_feature_stats', [])
                    ),
                    scene_common_candidate_stats=(
                        scene_common_candidate_stats),
                )
                self._write_scene_common_bias_stats(record)

            if self.dump_candidate_residual_trajectory_stats:
                candidate_residual_trajectory_stats = (
                    self._build_candidate_residual_trajectory_stats(
                        base_seg_logits,
                        base_seg_pred,
                        data_samples[i],
                        components,
                    )
                )
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(
                        self.confidence_threshold),
                    candidate_residual_layers=str(
                        self.candidate_residual_layers),
                    candidate_residual_variants=str(
                        self.candidate_residual_variants),
                    candidate_residual_strengths=str(
                        self.candidate_residual_strengths),
                    candidate_residual_scores=str(
                        self.candidate_residual_scores),
                    candidate_residual_ranks=str(
                        self.candidate_residual_ranks),
                    candidate_residual_regimes=str(
                        self.candidate_residual_regimes),
                    candidate_residual_margin_bins=str(
                        self.candidate_residual_margin_bins),
                    candidate_residual_min_pair_pixels=int(
                        self.candidate_residual_min_pair_pixels),
                    candidate_residual_null_trials=int(
                        self.candidate_residual_null_trials),
                    candidate_residual_max_side=int(
                        self.candidate_residual_max_side),
                    candidate_residual_seed_rule=str(
                        self.candidate_residual_seed_rule),
                    candidate_residual_trajectory_stats=(
                        candidate_residual_trajectory_stats),
                )
                self._write_candidate_residual_trajectory_stats(
                    record)

            if self.dump_candidate_residual_miou_stats:
                candidate_residual_miou_stats = (
                    self._build_candidate_residual_miou_stats(
                        base_seg_logits,
                        base_seg_pred,
                        data_samples[i],
                        components,
                    )
                )
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(
                        self.confidence_threshold),
                    candidate_residual_layers=str(
                        self.candidate_residual_layers),
                    candidate_residual_variants=str(
                        self.candidate_residual_variants),
                    candidate_residual_strengths=str(
                        self.candidate_residual_strengths),
                    candidate_residual_miou_scores=str(
                        self.candidate_residual_miou_scores),
                    candidate_residual_miou_ranks=str(
                        self.candidate_residual_miou_ranks),
                    candidate_residual_miou_units=str(
                        self.candidate_residual_miou_units),
                    candidate_residual_miou_absolute_thresholds=str(
                        self.candidate_residual_miou_absolute_thresholds),
                    candidate_residual_miou_coverages=str(
                        self.candidate_residual_miou_coverages),
                    candidate_residual_miou_region_reducer=str(
                        self.candidate_residual_miou_region_reducer),
                    candidate_residual_miou_region_min_pixels=int(
                        self.candidate_residual_miou_region_min_pixels),
                    candidate_residual_miou_raw_bin_thd=float(
                        self.candidate_residual_miou_raw_bin_thd),
                    candidate_residual_miou_max_side=int(
                        self.candidate_residual_miou_max_side),
                    candidate_residual_seed_rule=str(
                        self.candidate_residual_seed_rule),
                    raw_mask_oracle_topk=int(
                        self.raw_mask_oracle_topk),
                    candidate_residual_miou_stats=(
                        candidate_residual_miou_stats),
                )
                self._write_candidate_residual_miou_stats(record)

            if self.dump_ontology_readout_oracle_stats:
                ontology_readout_oracle_stats = (
                    self._build_ontology_readout_oracle_stats(
                        base_seg_logits,
                        base_seg_pred,
                        data_samples[i],
                        components,
                    )
                )
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    instance_score_type=self.instance_score_type,
                    ontology_readout_oracle_sources=(
                        self._ontology_readout_source_names()),
                    ontology_readout_oracle_max_side=int(
                        self.ontology_readout_oracle_max_side),
                    ontology_readout_oracle_topk=int(
                        self.ontology_readout_oracle_topk),
                    ontology_readout_oracle_min_pair_pixels=int(
                        self.ontology_readout_oracle_min_pair_pixels),
                    internal_selection_raw_topk=int(
                        self.internal_selection_raw_topk),
                    internal_selection_seed_rule=str(
                        self.internal_selection_seed_rule),
                    ontology_readout_oracle_stats=(
                        ontology_readout_oracle_stats),
                )
                self._write_ontology_readout_oracle_stats(record)

            if self.dump_region_contrastive_readout_stats:
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(
                        self.confidence_threshold),
                    use_region_contrastive_readout=bool(
                        self.use_region_contrastive_readout),
                    region_readout_max_side=int(
                        self.region_readout_max_side),
                    region_readout_max_regions=int(
                        self.region_readout_max_regions),
                    region_readout_masks_per_prompt=int(
                        self.region_readout_masks_per_prompt),
                    region_readout_mask_threshold=float(
                        self.region_readout_mask_threshold),
                    region_readout_ring_kernel=int(
                        self.region_readout_ring_kernel),
                    region_readout_top_fraction=float(
                        self.region_readout_top_fraction),
                    region_readout_presence_weight=float(
                        self.region_readout_presence_weight),
                    region_readout_blends=str(
                        self.region_readout_blends),
                    region_readout_mix_sources=str(
                        self.region_readout_mix_sources),
                    region_readout_apply_source=str(
                        self.region_readout_apply_source),
                    region_readout_apply_blend=float(
                        self.region_readout_apply_blend),
                    region_readout_apply_topk=int(
                        self.region_readout_apply_topk),
                    region_readout_min_margin=float(
                        self.region_readout_min_margin),
                    region_readout_projection=str(
                        self.region_readout_projection),
                    region_readout_route_variant=str(
                        self.region_readout_route_variant),
                    region_readout_foreground_blend=float(
                        self.region_readout_foreground_blend),
                    region_readout_reject_blend=float(
                        self.region_readout_reject_blend),
                    region_readout_background_blend=float(
                        self.region_readout_background_blend),
                    region_readout_formula_agreement=int(
                        self.region_readout_formula_agreement),
                    region_readout_agreement_sources=str(
                        self.region_readout_agreement_sources),
                    region_contrastive_readout_stats=(
                        region_readout_context),
                )
                self._write_region_contrastive_readout_stats(record)

            if self.dump_region_hypothesis_v2_stats:
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(
                        self.confidence_threshold),
                    region_hypothesis_v2_formulas=str(
                        self.region_hypothesis_v2_formulas),
                    region_hypothesis_v2_formula_selectors=str(
                        self.region_hypothesis_v2_formula_selectors),
                    region_hypothesis_v2_projections=str(
                        self.region_hypothesis_v2_projections),
                    region_hypothesis_v2_selector_thresholds=str(
                        self.region_hypothesis_v2_selector_thresholds),
                    region_hypothesis_v2_selector_projection=str(
                        self.region_hypothesis_v2_selector_projection),
                    region_hypothesis_v2_blend=float(
                        self.region_hypothesis_v2_blend),
                    region_hypothesis_v2_min_margin=float(
                        self.region_hypothesis_v2_min_margin),
                    region_hypothesis_v2_topk=int(
                        self.region_hypothesis_v2_topk),
                    region_hypothesis_v2_overlap_max=int(
                        self.region_hypothesis_v2_overlap_max),
                    region_hypothesis_v2_stats=(
                        region_hypothesis_v2_context),
                )
                self._write_region_hypothesis_v2_stats(record)

            if self.dump_candidate_region_quality_stats:
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    candidate_region_quality_sources=str(
                        self.candidate_region_quality_sources),
                    candidate_region_quality_max_side=int(
                        self.candidate_region_quality_max_side),
                    candidate_region_quality_high_purity=float(
                        self.candidate_region_quality_high_purity),
                    candidate_region_quality_stats=(
                        candidate_region_quality_context),
                )
                self._write_candidate_region_quality_stats(record)

            if self.dump_query_topology_stats:
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(
                        self.confidence_threshold),
                    instance_score_type=str(self.instance_score_type),
                    use_presence_score=bool(self.use_presence_score),
                    query_topology_max_side=int(
                        self.query_topology_max_side),
                    query_topology_support_threshold=float(
                        self.query_topology_support_threshold),
                    query_topology_mask_threshold=float(
                        self.query_topology_mask_threshold),
                    query_topology_stats=query_topology_context,
                )
                self._write_query_topology_stats(record)

            if self.dump_state_action_atlas_stats:
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(
                        os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(
                        self.confidence_threshold),
                    instance_score_type=self.instance_score_type,
                    state_action_atlas_actions=','.join(
                        self._get_state_action_names()),
                    state_action_apply=str(
                        self.state_action_apply),
                    state_action_presence_gamma=float(
                        self.state_action_presence_gamma),
                    state_action_presence_blend=float(
                        self.state_action_presence_blend),
                    state_action_presence_max_boost=float(
                        self.state_action_presence_max_boost),
                    state_action_presence_floor=float(
                        self.state_action_presence_floor),
                    state_action_support_threshold=float(
                        self.state_action_support_threshold),
                    state_action_low_presence_threshold=float(
                        self.state_action_low_presence_threshold),
                    state_action_margin_threshold=float(
                        self.state_action_margin_threshold),
                    state_action_local_kernel=int(
                        self.state_action_local_kernel),
                    state_action_feature_max_side=int(
                        self.state_action_feature_max_side),
                    state_action_atlas_stats=(
                        state_action_stats),
                )
                self._write_state_action_atlas_stats(record)

            if self.dump_geoer_router_stats:
                geoer_router_stats = self._build_geoer_router_stats(
                    base_seg_logits,
                    geoer_router_logits,
                    base_seg_pred,
                    geoer_router_pred,
                    data_samples[i],
                    geoer_router_context)
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    instance_score_type=self.instance_score_type,
                    use_geoer_router=bool(self.use_geoer_router),
                    geoer_use_scale_stability=bool(
                        self.geoer_use_scale_stability),
                    geoer_use_reject_recovery=bool(
                        self.geoer_use_reject_recovery),
                    geoer_reject_recovery_datasets=(
                        self.geoer_reject_recovery_datasets),
                    geoer_reject_recovery_bg_names=(
                        self.geoer_reject_recovery_bg_names),
                    geoer_reject_recovery_score_thd=float(
                        self.geoer_reject_recovery_score_thd),
                    geoer_reject_recovery_semantic_thd=float(
                        self.geoer_reject_recovery_semantic_thd),
                    geoer_reject_recovery_instance_thd=float(
                        self.geoer_reject_recovery_instance_thd),
                    geoer_reject_recovery_margin_thd=float(
                        self.geoer_reject_recovery_margin_thd),
                    geoer_reject_recovery_local_thd=float(
                        self.geoer_reject_recovery_local_thd),
                    geoer_reject_recovery_require_bg_margin=bool(
                        self.geoer_reject_recovery_require_bg_margin),
                    scale_stability_rerank_pairs=(
                        self.scale_stability_rerank_pairs),
                    scale_stability_rerank_pair_file=(
                        self.scale_stability_rerank_pair_file),
                    scale_stability_rerank_pair_mode=str(
                        self.scale_stability_rerank_pair_mode),
                    scale_stability_rerank_query_mode=str(
                        self.scale_stability_rerank_query_mode),
                    scale_stability_rerank_auto_min_pixels=int(
                        self.scale_stability_rerank_auto_min_pixels),
                    scale_stability_rerank_auto_exclude_bg=bool(
                        self._scale_stability_rerank_exclude_bg()),
                    scale_stability_rerank_bg_names=str(
                        self.scale_stability_rerank_bg_names),
                    scale_stability_rerank_max_risk_classes=int(
                        self.scale_stability_rerank_max_risk_classes),
                    scale_stability_rerank_scale=float(
                        self.scale_stability_rerank_scale),
                    scale_stability_rerank_topk=int(
                        self.scale_stability_rerank_topk),
                    scale_stability_rerank_gate=str(
                        self.scale_stability_rerank_gate),
                    scale_stability_rerank_drop_threshold=float(
                        self.scale_stability_rerank_drop_threshold),
                    geoer_router_stats=geoer_router_stats,
                )
                self._write_geoer_router_stats(record)

            if self.dump_coco_sec_stats:
                coco_sec_stats = self._build_coco_sec_stats(
                    base_seg_logits,
                    coco_sec_logits,
                    base_seg_pred,
                    coco_sec_pred,
                    data_samples[i],
                    coco_sec_context)
                record = dict(
                    rank=int(os.environ.get('RANK', 0)),
                    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                    img_path=image_path,
                    ori_shape=list(ori_shape),
                    image_size=list(image.size),
                    dataset_name=self.seed_dataset_name,
                    prob_thd=float(self.prob_thd),
                    confidence_threshold=float(self.confidence_threshold),
                    instance_score_type=self.instance_score_type,
                    use_coco_sec_fusion=bool(self.use_coco_sec_fusion),
                    coco_sec_lambda=float(self.coco_sec_lambda),
                    coco_sec_temperature=float(self.coco_sec_temperature),
                    coco_sec_center_prior=bool(self.coco_sec_center_prior),
                    coco_sec_synonym_reduce=str(self.coco_sec_synonym_reduce),
                    coco_sec_stats=coco_sec_stats,
                )
                self._write_coco_sec_stats(record)

            data_samples[i].set_data({
                'seg_logits': PixelData(**{'data': seg_logits}),
                'pred_sem_seg': PixelData(**{'data': seg_pred.unsqueeze(0)})
            })

        return data_samples

    def _forward(data_samples):
            """
        """

    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    def extract_feat(self, inputs):
        """
        """

    def loss(self, inputs, data_samples):
        """
        """


def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(',')
        names_i = [i.strip() for i in names_i]
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices


def _build_class_names(query_words, query_idx, num_cls):
    class_names = [f'class_{idx}' for idx in range(num_cls)]
    filled = [False for _ in range(num_cls)]
    for word, idx in zip(query_words, query_idx):
        idx = int(idx.item()) if hasattr(idx, 'item') else int(idx)
        if not filled[idx]:
            class_names[idx] = word
            filled[idx] = True
    return class_names


def _tensor_scalar(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().item())
    return float(value)


def _tensor_reduce(value, reduce):
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        return float(value)
    if value.numel() == 0:
        return None
    value = value.detach().float()
    if reduce == 'max':
        return float(value.max().item())
    if reduce == 'mean':
        return float(value.mean().item())
    raise ValueError(f'Unsupported reduce: {reduce}')


def _mask_score_stats(mask, area_threshold=0.5):
    if mask is None:
        return None
    mask = mask.detach().float()
    if mask.numel() == 0:
        return dict(max=None, mean=None, area_ratio=None)
    return dict(
        max=float(mask.max().item()),
        mean=float(mask.mean().item()),
        area_ratio=float((mask >= area_threshold).float().mean().item()),
    )


def _binary_iou(mask_a, mask_b):
    inter = (mask_a & mask_b).sum().item()
    union = (mask_a | mask_b).sum().item()
    return _safe_div(inter, union)


def _masked_mean(value, mask):
    if value is None or mask is None or not mask.any():
        return None
    return float(value[mask].detach().float().mean().item())


def _masked_sum(value, mask):
    if value is None or mask is None or not mask.any():
        return 0.0
    return float(value[mask].detach().float().sum().item())


def _mode_class(value, mask):
    if value is None or mask is None or not mask.any():
        return None
    selected = value[mask].detach().long()
    if selected.numel() == 0:
        return None
    return int(torch.bincount(selected).argmax().item())


def _gather_class_map(score_maps, class_idx):
    return torch.gather(score_maps, 0, class_idx.unsqueeze(0)).squeeze(0)


def _best_candidate_row(candidates, key):
    best = None
    best_value = None
    for candidate in candidates:
        value = candidate['row'].get(key)
        if value is None:
            continue
        if best is None or value > best_value:
            best = candidate
            best_value = value
    return best


def _safe_div(num, den):
    if den is None or den == 0:
        return None
    return float(num) / float(den)


def _zero_if_none(value):
    return 0.0 if value is None else float(value)


def _mean_existing(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _prefix_geometry_stats(prefix, stats):
    row = {}
    for key, value in (stats or {}).items():
        row[f'{prefix}_{key}'] = value
    return row


def _binary_erode(mask, kernel_size):
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return mask
    padding = kernel_size // 2
    # Avoid cuDNN conv2d for very large remote-sensing masks. Erosion is
    # equivalent to asking whether the local window contains no background.
    inv_mask = (~mask).float().view(1, 1, *mask.shape)
    inv_mask = F.pad(inv_mask, (padding, padding, padding, padding), value=1.0)
    has_background = F.max_pool2d(
        inv_mask,
        kernel_size=kernel_size,
        stride=1,
        padding=0,
    ).squeeze()
    return has_background <= 0


def _bbox_fill_ratio(mask, mask_pixels=None):
    if mask is None or not mask.any():
        return None
    if mask_pixels is None:
        mask_pixels = int(mask.sum().item())
    coords = torch.nonzero(mask, as_tuple=False)
    if coords.numel() == 0:
        return None
    y_min = int(coords[:, 0].min().item())
    y_max = int(coords[:, 0].max().item())
    x_min = int(coords[:, 1].min().item())
    x_max = int(coords[:, 1].max().item())
    bbox_area = max(1, (y_max - y_min + 1) * (x_max - x_min + 1))
    return _safe_div(mask_pixels, bbox_area)


def _ranked_jsonl_path(path):
    rank = os.environ.get('RANK')
    if rank is None:
        return path
    base, ext = os.path.splitext(path)
    if ext == '':
        ext = '.jsonl'
    return f'{base}_rank{rank}{ext}'


def _infer_dataset_name(classname_path):
    if classname_path is None:
        return None
    stem = os.path.splitext(os.path.basename(str(classname_path)))[0]
    if stem.startswith('cls_'):
        stem = stem[4:]
    return stem


def _class_name_tokens(name):
    tokens = set()
    normalized = str(name or '').lower().replace('/', ',').replace('-', ' ')
    for item in normalized.split(','):
        item = item.strip()
        if item:
            tokens.add(item)
            for part in item.split():
                if part:
                    tokens.add(part)
    return tokens


def _remote_sensing_class_role(name):
    tokens = _class_name_tokens(name)
    if tokens & {'background', 'other', 'clutter', 'void', 'unknown'}:
        return 'catch_all'
    if tokens & {'tree', 'forest', 'wood', 'canopy'}:
        return 'woody_vegetation'
    if tokens & {
            'grass', 'vegetation', 'low', 'crop', 'cropland',
            'agricultural', 'agriculture', 'farmland', 'field'}:
        return 'low_vegetation'
    if tokens & {'water', 'river', 'lake', 'sea', 'pond'}:
        return 'water'
    if tokens & {
            'road', 'pavement', 'impervious', 'surface',
            'sidewalk', 'parking'}:
        return 'impervious_surface'
    if tokens & {
            'building', 'roof', 'house', 'facade', 'wall',
            'construction'}:
        return 'built_object'
    if tokens & {'car', 'vehicle', 'truck', 'ship', 'airplane', 'plane'}:
        return 'vehicle'
    if tokens & {'bareland', 'barren', 'soil', 'sand', 'bare'}:
        return 'bareland'
    return 'other_landcover'


def _builtin_prompt_variants(class_name):
    name = str(class_name).strip()
    tokens = _class_name_tokens(name)
    variants = [name]
    if 'roof' in tokens:
        variants.extend([
            'building roof',
            'rooftop',
            'roof surface',
            'top of building',
            'flat roof in aerial image',
        ])
    if 'facade' in tokens:
        variants.extend([
            'building facade',
            'building wall',
            'vertical wall',
            'side of building',
            'facade surface',
        ])
    if 'tree' in tokens:
        variants.extend([
            'tree crown',
            'tree canopy',
            'woody vegetation',
            'individual tree',
            'trees in aerial image',
        ])
    if 'grass' in tokens or 'low vegetation' in ' '.join(tokens):
        variants.extend([
            'grass',
            'grassland',
            'low vegetation',
            'lawn',
            'herbaceous vegetation',
        ])
    if 'road' in tokens:
        variants.extend([
            'road',
            'paved road',
            'road surface',
            'street',
            'asphalt road',
        ])
    if 'pavement' in tokens:
        variants.extend([
            'pavement',
            'paved surface',
            'sidewalk',
            'impervious pavement',
            'pavement area',
        ])
    if 'building' in tokens:
        variants.extend([
            'building',
            'building footprint',
            'building rooftop',
            'man-made building',
            'building area',
        ])
    if 'background' in tokens or 'clutter' in tokens:
        variants.extend([
            name,
            'background',
            'unlabeled background',
            'other land cover',
            'clutter region',
        ])
    if 'forest' in tokens:
        variants.extend([
            'forest',
            'forest canopy',
            'dense trees',
            'woodland',
            'forested area',
        ])
    if 'agricultural' in tokens or 'agriculture' in tokens:
        variants.extend([
            'agricultural field',
            'farmland',
            'cropland',
            'cultivated land',
            'agriculture area',
        ])
    return variants


def _builtin_pair_prompt_variants(class_name, competitor_name):
    class_name = str(class_name).strip()
    competitor_name = str(competitor_name).strip()
    class_tokens = _class_name_tokens(class_name)
    competitor_tokens = _class_name_tokens(competitor_name)
    variants = [
        f'{class_name}, not {competitor_name}',
        f'{class_name} and not {competitor_name}',
        f'{class_name} region excluding {competitor_name}',
    ]

    if 'roof' in class_tokens and 'facade' in competitor_tokens:
        variants.extend([
            'horizontal building roof, not vertical facade',
            'rooftop surface, not building wall',
            'top of building, not facade',
            'flat roof in aerial image, not facade',
        ])
    if 'facade' in class_tokens and 'roof' in competitor_tokens:
        variants.extend([
            'vertical building facade, not roof',
            'building wall, not rooftop',
            'side of building, not top of building',
        ])
    if 'tree' in class_tokens and 'grass' in competitor_tokens:
        variants.extend([
            'tree canopy, not grass',
            'tree crown, not grassland',
            'woody vegetation, not low vegetation',
            'individual tree, not lawn',
        ])
    if 'grass' in class_tokens and 'tree' in competitor_tokens:
        variants.extend([
            'grassland, not tree canopy',
            'low vegetation, not tree',
            'lawn, not woody vegetation',
            'herbaceous vegetation, not tree crown',
        ])
    if 'road' in class_tokens and ('background' in competitor_tokens or 'clutter' in competitor_tokens):
        variants.extend([
            'paved road, not background',
            'road surface, not clutter',
            'asphalt road, not other land cover',
        ])
    if 'vegetation' in class_tokens and ('background' in competitor_tokens or 'clutter' in competitor_tokens):
        variants.extend([
            'vegetation region, not background',
            'green vegetation, not clutter',
            'plant-covered land, not other land cover',
        ])
    if 'building' in class_tokens and ('background' in competitor_tokens or 'clutter' in competitor_tokens):
        variants.extend([
            'building area, not background',
            'man-made building, not clutter',
            'building footprint, not other land cover',
        ])
    if 'background' in class_tokens:
        variants.extend([
            f'background, not {competitor_name}',
            f'other land cover, not {competitor_name}',
        ])
    if 'pavement' in class_tokens and 'building' in competitor_tokens:
        variants.extend([
            'paved surface, not building',
            'pavement area, not building footprint',
            'sidewalk, not building',
        ])
    if 'forest' in class_tokens and 'background' in competitor_tokens:
        variants.extend([
            'forest canopy, not background',
            'dense trees, not other land cover',
            'woodland, not background',
        ])
    if ('agricultural' in class_tokens or 'agriculture' in class_tokens
            or 'farmland' in class_tokens):
        variants.extend([
            f'agricultural field, not {competitor_name}',
            f'cropland, not {competitor_name}',
            f'cultivated land, not {competitor_name}',
        ])
    return variants


def _connected_component_summary(mask, correct_mask, min_pixels=1, purity_threshold=0.95):
    mask_np = mask.detach().cpu().numpy().astype(np.bool_)
    correct_np = correct_mask.detach().cpu().numpy().astype(np.bool_)
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_np.astype(np.uint8),
            connectivity=4,
        )
        region_count = 0
        region_pixels = 0
        region_correct_pixels = 0
        region_area_sum = 0
        region_purity_sum = 0.0
        pure_region_count = 0
        max_region_area = 0
        min_region_area = None
        for label_idx in range(1, num_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area < min_pixels:
                continue
            component_mask = labels == label_idx
            correct = int(correct_np[component_mask].sum())
            purity = _safe_div(correct, area)
            region_count += 1
            region_pixels += area
            region_correct_pixels += correct
            region_area_sum += area
            region_purity_sum += purity if purity is not None else 0.0
            pure_region_count += int(purity is not None and purity >= purity_threshold)
            max_region_area = max(max_region_area, area)
            min_region_area = area if min_region_area is None else min(min_region_area, area)
        return dict(
            region_count=region_count,
            region_pixels=region_pixels,
            region_correct_pixels=region_correct_pixels,
            region_purity=_safe_div(region_correct_pixels, region_pixels),
            pure_region_count=pure_region_count,
            pure_region_ratio=_safe_div(pure_region_count, region_count),
            region_area_sum=region_area_sum,
            mean_region_area=_safe_div(region_area_sum, region_count),
            max_region_area=max_region_area if region_count > 0 else None,
            min_region_area=min_region_area,
            region_purity_sum=region_purity_sum,
            mean_region_purity=_safe_div(region_purity_sum, region_count),
        )

    visited = np.zeros_like(mask_np, dtype=np.bool_)
    h, w = mask_np.shape

    region_count = 0
    region_pixels = 0
    region_correct_pixels = 0
    region_area_sum = 0
    region_purity_sum = 0.0
    pure_region_count = 0
    max_region_area = 0
    min_region_area = None

    coords = np.argwhere(mask_np)
    for start_y, start_x in coords:
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        area = 0
        correct = 0
        while stack:
            y, x = stack.pop()
            area += 1
            if correct_np[y, x]:
                correct += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if visited[ny, nx] or not mask_np[ny, nx]:
                    continue
                visited[ny, nx] = True
                stack.append((ny, nx))

        if area < min_pixels:
            continue
        purity = _safe_div(correct, area)
        region_count += 1
        region_pixels += area
        region_correct_pixels += correct
        region_area_sum += area
        region_purity_sum += purity if purity is not None else 0.0
        pure_region_count += int(purity is not None and purity >= purity_threshold)
        max_region_area = max(max_region_area, area)
        min_region_area = area if min_region_area is None else min(min_region_area, area)

    return dict(
        region_count=region_count,
        region_pixels=region_pixels,
        region_correct_pixels=region_correct_pixels,
        region_purity=_safe_div(region_correct_pixels, region_pixels),
        pure_region_count=pure_region_count,
        pure_region_ratio=_safe_div(pure_region_count, region_count),
        region_area_sum=region_area_sum,
        mean_region_area=_safe_div(region_area_sum, region_count),
        max_region_area=max_region_area if region_count > 0 else None,
        min_region_area=min_region_area,
        region_purity_sum=region_purity_sum,
        mean_region_purity=_safe_div(region_purity_sum, region_count),
    )


def _connected_component_labels(mask):
    mask_np = mask.detach().cpu().numpy().astype(np.uint8)
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        count, labels = cv2.connectedComponents(
            mask_np,
            connectivity=4,
        )
        return torch.from_numpy(labels.astype(np.int64)), max(0, count - 1)

    labels = np.zeros_like(mask_np, dtype=np.int64)
    height, width = mask_np.shape
    component_count = 0
    for start_y, start_x in np.argwhere(mask_np):
        if labels[start_y, start_x] != 0:
            continue
        component_count += 1
        labels[start_y, start_x] = component_count
        stack = [(int(start_y), int(start_x))]
        while stack:
            y, x = stack.pop()
            for ny, nx in (
                    (y - 1, x), (y + 1, x),
                    (y, x - 1), (y, x + 1)):
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                if (
                        mask_np[ny, nx] == 0
                        or labels[ny, nx] != 0):
                    continue
                labels[ny, nx] = component_count
                stack.append((ny, nx))
    return torch.from_numpy(labels), component_count


def _resolve_inference_device(device):
    if isinstance(device, str):
        device = torch.device(device)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type != 'cuda':
        return device

    local_rank = os.environ.get('LOCAL_RANK')
    if local_rank is not None:
        device = torch.device(f'cuda:{int(local_rank)}')
    elif device.index is None:
        device = torch.device('cuda:0')

    torch.cuda.set_device(device)
    return device
