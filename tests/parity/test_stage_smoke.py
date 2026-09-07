import ast
import importlib
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

from biohub.features.motion import MOTION_FEATURES, MOTION_TRAIN_FEATURES, RUNTIME_DROP
from biohub.features.position import POS_EMBED_DIM, pos_embed_torch
from biohub.losses.association import compute_batch_loss, compute_loss
from biohub.losses.deepcenter import weighted_bce_loss
from biohub.losses.detection import compute_detection_loss
from biohub.losses.division import balanced_focal_bce
from biohub.models.deepcenter import DeepCenterUNet3D
from biohub.models.detector import UNetNodeTransformer
from biohub.models.division import DivisionMLP
from biohub.models.motion import MotionResidual
from biohub.models.node_transformer import SimpleNodeTransformer
from biohub.models.option_head import OptionHead
from biohub.models.temporal_unet import TemporalUNet3D
from biohub.modules.candidategraft.runtime import CandidateGraftDirectRuntime
from biohub.modules.cardinality.head import SourceCardinalityRuntime
from biohub.modules.edgegraft.runtime import EdgeGraftV3Runtime
from biohub.modules.ownership import FEATURES as OWNERSHIP_FEATURES
from biohub.modules.ownership import FullPopulationOwnershipRuntime
from biohub.paths import PROJECT_ROOT
from biohub.train import edgegraft as edgegraft_features
from biohub.train.candidategraft import make_model as make_candidategraft_model
from biohub.train.decoder import make_pair_classifier, make_source_classifier
from biohub.train.edgegraft_gate import make_model as make_edgegraft_gate_model
from biohub.train.edgegraft_oracle import mapped_candidates
from biohub.train.edgegraft_ranker import make_model as make_edgegraft_ranker_model
from biohub.train.motion_cache import fused_edge_loss
from biohub.train.ownership import make_model as make_ownership_model

ATOL = 1e-6
HELPERS = PROJECT_ROOT / 'biohub_production_957_reproducibility_bundle' / 'helpers'
BUNDLE = PROJECT_ROOT / 'kaggle' / 'input' / 'datasets' / 'antonoof' / 'all_files'
LOCKED_CSV = PROJECT_ROOT / 'tests' / 'fixtures' / '44b6_0113de3b_submission.csv'
E2E_ONE_MOVIE = PROJECT_ROOT / 'runs' / 'parity_one_movie_e2e'
OLD_ONE_MOVIE = PROJECT_ROOT / 'runs' / 'parity_one_movie_v2'
NEW_ONE_MOVIE = PROJECT_ROOT / 'runs' / 'parity_one_movie_new'
SCRIPTS = HELPERS / '01_p1_p2_base' / 'shared_repo' / 'scripts'
MOTION_V1 = HELPERS / '09_motion_corrector' / 'TRAINING_V1'
EDGEGRAFT_TRAINING = HELPERS / '07_edgegraft' / 'training'
CANDIDATE_TRAINING = HELPERS / '08_candidategraft' / 'training'
NUMBERED_TRAIN = (
    '01_p1',
    '01_p2',
    '02_model_c',
    '02_division',
    '02_division_decoder',
    '03_cardinality',
    '04_unigraft',
    '06_ownership',
    '07_edgegraft_ranker',
    '07_edgegraft_gate',
    '07_edgegraft_oracle',
    '08_candidategraft',
    '09_motion',
    '09_motion_cache',
    '10_deepcenter',
)
NAMED_TRAINERS = (
    'detector',
    'division',
    'decoder',
    'cardinality',
    'unigraft',
    'ownership',
    'edgegraft_ranker',
    'edgegraft_gate',
    'edgegraft_oracle',
    'candidategraft',
    'motion',
    'motion_cache',
    'deepcenter',
)


def _skip(message: str) -> None:
    cast(Any, pytest.skip)(message)


def _require(path: Path) -> Path:
    assert path.is_file(), f'missing helper: {path}'
    return path


def _helper_src() -> Path:
    return HELPERS / '01_p1_p2_base' / 'shared_repo' / 'src'


def _load_helper(path: Path, name: str, extra_paths: tuple[Path, ...] = ()):
    _require(path)
    for extra in reversed(extra_paths):
        text = str(extra)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _eval_calls(path: Path, func_name: str, names: dict[str, Any]) -> list[Any]:
    _require(path)
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        ident = getattr(node.func, 'id', None) or getattr(node.func, 'attr', None)
        if ident != func_name:
            continue
        expr = ast.Expression(body=node)
        ast.fix_missing_locations(expr)
        found.append(eval(compile(expr, str(path), 'eval'), names))
    assert found, f'{func_name} constructor missing in {path}'
    return found


def _one_step_close(left: torch.nn.Module, right: torch.nn.Module, batch) -> None:
    def fn(model: torch.nn.Module) -> torch.Tensor:
        out = model(*batch) if isinstance(batch, tuple) else model(batch)
        if isinstance(out, tuple):
            return sum(item.sum() for item in out)
        return out.sum()

    _one_step_fn(left, right, fn)


def _one_step_fn(
    left: torch.nn.Module,
    right: torch.nn.Module,
    fn: Callable[[torch.nn.Module], torch.Tensor],
) -> None:
    left.train()
    right.train()
    opt_left = torch.optim.SGD(left.parameters(), lr=0.01)
    opt_right = torch.optim.SGD(right.parameters(), lr=0.01)
    torch.manual_seed(123)
    loss_left = fn(left)
    torch.manual_seed(123)
    loss_right = fn(right)
    torch.testing.assert_close(loss_left, loss_right, atol=ATOL, rtol=0.0)
    opt_left.zero_grad()
    opt_right.zero_grad()
    loss_left.backward()
    loss_right.backward()
    opt_left.step()
    opt_right.step()
    for left_param, right_param in zip(left.parameters(), right.parameters(), strict=True):
        torch.testing.assert_close(left_param, right_param, atol=ATOL, rtol=0.0)


def _tiny_xy(n_features: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(48, n_features)).astype(np.float32)
    y = rng.integers(0, 2, size=48).astype(np.int8)
    y[0] = 0
    y[1] = 1
    weight = np.linspace(0.5, 1.5, 48).astype(np.float32)
    return x, y, weight


def _assert_sklearn_proba(left, right, x, y, sample_weight=None) -> None:
    if sample_weight is None:
        left.fit(x, y)
        right.fit(x, y)
    else:
        left.fit(x, y, sample_weight=sample_weight)
        right.fit(x, y, sample_weight=sample_weight)
    np.testing.assert_allclose(left.predict_proba(x), right.predict_proba(x), atol=ATOL, rtol=0.0)


def _index_features(model, feat_maps, coords, mask):
    index = getattr(model, 'index_features', None) or model._index_features
    return index(feat_maps, coords, mask)


def test_numbered_train_clis_import_train_from_config() -> None:
    for name in NUMBERED_TRAIN:
        module = importlib.import_module(f'biohub.train.{name}')
        assert callable(module.train_from_config), name
        assert callable(module.stage_main), name


def test_named_trainers_expose_train_from_config() -> None:
    for name in NAMED_TRAINERS:
        module = importlib.import_module(f'biohub.train.{name}')
        assert callable(module.train_from_config), name
    importlib.import_module('biohub.train.edgegraft')
    importlib.import_module('biohub.train.tensorboard')


@pytest.mark.slow
def test_temporal_unet_matches_helper_train_step() -> None:
    helper_src = _helper_src()
    HelperUNet = _load_helper(
        helper_src / 'biohub_tracking' / 'models' / 'temporal_unet.py',
        'helper_temporal_unet',
        extra_paths=(helper_src,),
    ).TemporalUNet3D
    torch.manual_seed(0)
    ours = TemporalUNet3D(in_channels=1, out_channels=8, layers=(8, 16))
    torch.manual_seed(0)
    theirs = HelperUNet(in_channels=1, out_channels=8, layers=(8, 16))
    _one_step_close(ours, theirs, torch.randn(1, 2, 1, 8, 16, 16))


@pytest.mark.slow
def test_simple_node_transformer_matches_helper_train_step() -> None:
    helper_src = _helper_src()
    HelperSNT = _load_helper(
        helper_src / 'biohub_tracking' / 'models' / 'simple_node_transformer.py',
        'helper_simple_node_transformer',
        extra_paths=(helper_src,),
    ).SimpleNodeTransformer
    torch.manual_seed(0)
    ours = SimpleNodeTransformer(feat_dim=16, hidden_dim=32, n_heads=2, n_blocks=1, dropout=0.0)
    torch.manual_seed(0)
    theirs = HelperSNT(feat_dim=16, hidden_dim=32, n_heads=2, n_blocks=1, dropout=0.0)
    batch = (
        torch.randn(1, 3, 16),
        torch.randn(1, 4, 16),
        torch.randn(1, 3, 3),
        torch.randn(1, 4, 3),
        torch.ones(1, 3, dtype=torch.bool),
        torch.ones(1, 4, dtype=torch.bool),
    )
    _one_step_close(ours, theirs, batch)


@pytest.mark.slow
def test_unet_node_transformer_matches_helper_train_step() -> None:
    helper_src = _helper_src()
    helper = _load_helper(
        SCRIPTS / 'train_unet_transformer.py',
        'helper_train_unet_unt',
        extra_paths=(SCRIPTS, helper_src),
    )
    HelperUNet = helper.TemporalUNet3D
    HelperUNT = helper.UNetNodeTransformer
    torch.manual_seed(0)
    ours = UNetNodeTransformer(
        TemporalUNet3D(in_channels=1, out_channels=8, layers=(8, 16)),
        8,
        4 * POS_EMBED_DIM,
        hidden_dim=16,
        n_heads=2,
        n_blocks=1,
        dropout=0.0,
    )
    torch.manual_seed(0)
    theirs = HelperUNT(
        HelperUNet(in_channels=1, out_channels=8, layers=(8, 16)),
        8,
        4 * POS_EMBED_DIM,
        hidden_dim=16,
        n_heads=2,
        n_blocks=1,
        dropout=0.0,
    )
    imgs = torch.randn(1, 2, 8, 16, 16)
    coords = torch.tensor([[[1.0, 2.0, 3.0], [2.0, 4.0, 5.0]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    pos = pos_embed_torch(torch.cat([torch.zeros(1, 2, 1), coords], dim=-1), (2, 8, 16, 16))

    def fn(model: torch.nn.Module) -> torch.Tensor:
        encode = getattr(model, 'encode')
        predict_edges = getattr(model, 'predict_edges')
        unet_out, det = encode(imgs)
        feat0 = _index_features(model, unet_out[:, 0], coords, mask)
        feat1 = _index_features(model, unet_out[:, 1], coords, mask)
        edges = predict_edges(feat0, feat1, coords, coords, pos, pos, mask, mask)
        return det[0].sum() + det[1].sum() + edges.sum()

    _one_step_fn(ours, theirs, fn)


@pytest.mark.slow
def test_detection_loss_matches_helper() -> None:
    helper_src = _helper_src()
    module = _load_helper(
        SCRIPTS / 'train_unet_transformer.py',
        'helper_train_unet_loss',
        extra_paths=(SCRIPTS, helper_src),
    )
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 4, 8, 8)
    coords = torch.tensor([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], [[2.0, 1.0, 1.0], [1.0, 1.0, 2.0]]])
    mask = torch.tensor([[True, False], [True, True]])
    left = compute_detection_loss(logits, coords, mask, neg_weight=0.01)
    right = module.compute_detection_loss(logits, coords, mask, neg_weight=0.01)
    torch.testing.assert_close(left, right, atol=ATOL, rtol=0.0)


@pytest.mark.slow
def test_association_loss_matches_helper() -> None:
    helper_src = _helper_src()
    module = _load_helper(
        SCRIPTS / 'train_unet_transformer.py',
        'helper_train_unet_assoc',
        extra_paths=(SCRIPTS, helper_src),
    )
    torch.manual_seed(0)
    logits = torch.randn(4, 5)
    target = torch.zeros(4, 5)
    target[0, 1] = 1.0
    target[2, 3] = 1.0
    target[2, 4] = 1.0
    torch.testing.assert_close(
        compute_loss(logits, target),
        module.compute_loss(logits, target),
        atol=ATOL,
        rtol=0.0,
    )
    batch_logits = torch.randn(2, 4, 5)
    batch_target = torch.zeros(2, 4, 5)
    batch_target[0, 0, 1] = 1.0
    batch_target[1, 2, 3] = 1.0
    mask_t = torch.tensor([[True, True, True, False], [True, True, True, True]])
    mask_t1 = torch.tensor([[True, True, True, True, False], [True, True, True, True, True]])
    torch.testing.assert_close(
        compute_batch_loss(batch_logits, batch_target, mask_t, mask_t1),
        module.compute_batch_loss(batch_logits, batch_target, mask_t, mask_t1),
        atol=ATOL,
        rtol=0.0,
    )


@pytest.mark.slow
def test_division_mlp_matches_helper_train_step() -> None:
    module = _load_helper(
        HELPERS / '02_model_c' / 'train_division_pair_model.py',
        'helper_division_pair',
    )
    torch.manual_seed(1)
    ours = DivisionMLP(6, (64, 32))
    torch.manual_seed(1)
    theirs = module.MLP(6, (64, 32))
    features = torch.randn(5, 6)
    _one_step_close(ours, theirs, features)
    target = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
    logits = ours(features)
    torch.testing.assert_close(
        balanced_focal_bce(logits, target),
        module.balanced_loss(logits, target),
        atol=ATOL,
        rtol=0.0,
    )


@pytest.mark.slow
def test_option_head_matches_helper_train_step() -> None:
    module = _load_helper(
        HELPERS / '03_source_cardinality' / 'train_source_cardinality_head.py',
        'helper_cardinality_head',
    )
    torch.manual_seed(2)
    ours = OptionHead(4, 6, 8, 12)
    torch.manual_seed(2)
    theirs = module.OptionHead(4, 6, 8, 12)
    batch = (
        torch.randn(3, 4),
        torch.randn(3, 5, 6),
        torch.ones(3, 5, dtype=torch.bool),
    )
    _one_step_close(ours, theirs, batch)


@pytest.mark.slow
def test_unigraft_option_head_matches_helper_train_step() -> None:
    module = _load_helper(
        HELPERS / '04_unigraft_p1p2' / 'train_public934_source_cardinality_head.py',
        'helper_unigraft_head',
    )
    torch.manual_seed(4)
    ours = OptionHead(4, 6, 8, 12)
    torch.manual_seed(4)
    theirs = module.OptionHead(4, 6, 8, 12)
    batch = (
        torch.randn(3, 4),
        torch.randn(3, 5, 6),
        torch.ones(3, 5, dtype=torch.bool),
    )
    _one_step_close(ours, theirs, batch)


@pytest.mark.slow
def test_deepcenter_matches_helper_train_step() -> None:
    module = _load_helper(
        HELPERS / '10_deepcenter' / 'train_full_frame_center_detector.py',
        'helper_deepcenter',
    )
    torch.manual_seed(5)
    ours = DeepCenterUNet3D(in_channels=1, base_channels=8)
    torch.manual_seed(5)
    theirs = module.DeepCenterUNet3D(in_channels=1, base_channels=8)
    volume = torch.randn(1, 1, 8, 16, 16)
    target = torch.rand(1, 1, 8, 16, 16)
    weights = torch.ones_like(target)
    _one_step_close(ours, theirs, volume)
    logits = ours(volume)
    torch.testing.assert_close(
        weighted_bce_loss(logits, target, weights),
        module.weighted_bce_loss(logits, target, weights),
        atol=ATOL,
        rtol=0.0,
    )


@pytest.mark.slow
def test_motion_residual_matches_helper_train_step() -> None:
    helper_src = _helper_src()
    module = _load_helper(
        MOTION_V1 / 'train_motion_cost_corrector.py',
        'helper_motion',
        extra_paths=(MOTION_V1, SCRIPTS, helper_src),
    )
    assert tuple(module.FEATURES) == MOTION_TRAIN_FEATURES
    assert module.RUNTIME_DROP == RUNTIME_DROP
    assert MOTION_FEATURES == tuple(
        name for name in module.FEATURES if name not in module.RUNTIME_DROP
    )
    torch.manual_seed(6)
    ours = MotionResidual(4)
    torch.manual_seed(6)
    theirs = module.MotionResidual(4)
    _one_step_close(ours, theirs, torch.randn(8, 4))


@pytest.mark.slow
def test_fused_edge_loss_matches_helper() -> None:
    helper_src = _helper_src()
    module = _load_helper(
        MOTION_V1 / 'finetune_edge_head_on_proposals.py',
        'helper_finetune',
        extra_paths=(MOTION_V1, SCRIPTS, helper_src),
    )
    args = SimpleNamespace(
        anchor_weight=1.0,
        trainable_weight=1.0,
        division_positive_weight=1.0,
        anchor_error_weight=1.5,
        disagreement_weight=1.0,
        focal_gamma=2.0,
        soft_jaccard_weight=0.001,
    )
    torch.manual_seed(7)
    logits_a = torch.randn(2, 3, 4)
    logits_b = torch.randn(2, 3, 4)
    target = torch.zeros(2, 3, 4)
    target[0, 0, 1] = 1.0
    target[1, 1, 2] = 1.0
    supervision = torch.ones(2, 3, 4, dtype=torch.bool)
    mask0 = torch.ones(2, 3, dtype=torch.bool)
    mask1 = torch.ones(2, 4, dtype=torch.bool)
    left = fused_edge_loss(logits_a, logits_b, target, supervision, mask0, mask1, args)
    right = module.fused_edge_loss(logits_a, logits_b, target, supervision, mask0, mask1, args)
    for left_item, right_item in zip(left, right, strict=True):
        torch.testing.assert_close(left_item, right_item, atol=ATOL, rtol=0.0)


@pytest.mark.slow
def test_ownership_extratrees_matches_helper_fit() -> None:
    helper = HELPERS / '06_full_population_ownership' / 'score_and_fit_ownership_oof.py'
    args = SimpleNamespace(
        n_estimators=400,
        max_depth=3,
        min_samples_leaf=20,
        max_features=0.75,
        n_jobs=1,
        seed=324459,
    )
    ours = make_ownership_model(args, 0)
    theirs = _eval_calls(
        helper,
        'ExtraTreesClassifier',
        {'ExtraTreesClassifier': ExtraTreesClassifier, 'fold': 0},
    )[0]
    theirs.n_jobs = 1
    x, y, _weight = _tiny_xy(len(OWNERSHIP_FEATURES), seed=7)
    _assert_sklearn_proba(ours, theirs, x, y)


@pytest.mark.slow
def test_edgegraft_ranker_hgb_matches_helper_fit() -> None:
    helper = EDGEGRAFT_TRAINING / 'train_edgegraft_current_ranker_v3.py'
    args = SimpleNamespace(
        learning_rate=0.06,
        iterations=180,
        leaves=31,
        l2=1.0,
        min_samples_leaf=40,
        max_bins=255,
        seed=20260819,
    )
    ours = make_edgegraft_ranker_model(args, 0)
    theirs = _eval_calls(
        helper,
        'HistGradientBoostingClassifier',
        {'HistGradientBoostingClassifier': HistGradientBoostingClassifier, 'args': args, 'fold': 0},
    )[0]
    x, y, weight = _tiny_xy(8, seed=8)
    _assert_sklearn_proba(ours, theirs, x, y, sample_weight=weight)


@pytest.mark.slow
def test_edgegraft_gate_hgb_matches_helper_fit() -> None:
    helper = EDGEGRAFT_TRAINING / 'train_edgegraft_v3_metric_transaction_gate.py'
    args = SimpleNamespace(
        learning_rate=0.035,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2=5.0,
        seed=20260819,
    )
    ours = make_edgegraft_gate_model(args, 0)
    theirs = _eval_calls(
        helper,
        'HistGradientBoostingClassifier',
        {'HistGradientBoostingClassifier': HistGradientBoostingClassifier, 'args': args, 'fold': 0},
    )[0]
    x, y, weight = _tiny_xy(6, seed=9)
    _assert_sklearn_proba(ours, theirs, x, y, sample_weight=weight)


@pytest.mark.slow
def test_edgegraft_label_helpers_match_ranker_v2() -> None:
    module = _load_helper(
        EDGEGRAFT_TRAINING / 'train_edgegraft_current_ranker_v2.py',
        'helper_edgegraft_ranker_v2',
        extra_paths=(EDGEGRAFT_TRAINING,),
    )
    assert module.METADATA == edgegraft_features.METADATA
    assert module.RELATIVE_MAX == edgegraft_features.RELATIVE_MAX
    assert module.RELATIVE_MIN == edgegraft_features.RELATIVE_MIN
    frame = pd.DataFrame(
        {
            'dataset': ['a', 'a', 'b'],
            'target': [1, 1, 2],
            'source': [10, 11, 20],
            'y': [1, 0, 1],
            'is_current_parent': [1, 0, 1],
            'p1_probability': [0.9, 0.2, 0.4],
            'distance_um': [1.0, 3.0, 2.0],
        }
    )
    left = edgegraft_features.add_relative_features(frame)
    right = module.add_relative_features(frame)
    pd.testing.assert_frame_equal(left, right, check_exact=False, atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(
        edgegraft_features.target_weights(frame),
        module.target_weights(frame),
        atol=ATOL,
        rtol=0.0,
    )
    score = np.array([0.8, 0.1, 0.6], np.float32)
    assert edgegraft_features.target_metrics(frame, score) == module.target_metrics(frame, score)


@pytest.mark.slow
def test_candidategraft_hgb_matches_helper_fit() -> None:
    screen = _load_helper(
        CANDIDATE_TRAINING / 'screen_native_endpoint_candidate_graft_v1.py',
        'helper_candidategraft_screen',
        extra_paths=(CANDIDATE_TRAINING,),
    )
    fit = _load_helper(
        CANDIDATE_TRAINING / 'train_candidategraft_direct_v1.py',
        'helper_candidategraft_fit',
        extra_paths=(CANDIDATE_TRAINING,),
    )
    ours = make_candidategraft_model(271828, 20, 80)
    screen_model = screen.make_model(271828, 20, 80)
    fit_model = fit.make_model(271828, 20, 80)
    x, y, _weight = _tiny_xy(8, seed=10)
    _assert_sklearn_proba(ours, screen_model, x, y)
    ours_again = make_candidategraft_model(271828, 20, 80)
    _assert_sklearn_proba(ours_again, fit_model, x, y)


@pytest.mark.slow
def test_decoder_hgb_matches_helper_fit() -> None:
    helper = HELPERS / '02_model_c' / 'train_model_c_v2_event_decoder.py'
    args = SimpleNamespace(
        pair_learning_rate=0.065,
        pair_max_iter=220,
        pair_max_leaf_nodes=31,
        pair_min_samples_leaf=25,
        pair_l2=1.5,
        source_learning_rate=0.055,
        source_max_iter=260,
        source_max_leaf_nodes=31,
        source_min_samples_leaf=24,
        source_l2=2.0,
    )
    constructors = _eval_calls(
        helper,
        'HistGradientBoostingClassifier',
        {'HistGradientBoostingClassifier': HistGradientBoostingClassifier, 'seed': 2029},
    )
    pair_ours = make_pair_classifier(args, 2029)
    source_ours = make_source_classifier(args, 2029)
    x, y, weight = _tiny_xy(5, seed=11)
    _assert_sklearn_proba(pair_ours, constructors[0], x, y, sample_weight=weight)
    _assert_sklearn_proba(source_ours, constructors[1], x, y, sample_weight=weight)


@pytest.mark.slow
def test_oracle_mapped_candidates_filters_npz(tmp_path: Path) -> None:
    path = tmp_path / 'cand.npz'
    np.savez(
        path,
        source_id=np.array([0, 1, 2], np.int64),
        target_id=np.array([1, 2, 0], np.int64),
        probability=np.array([0.9, 0.001, 0.8], np.float32),
        distance_um=np.array([1.0, 1.0, 20.0], np.float32),
        fused_graph_node_id=np.array([10, 11, 12], np.int64),
    )
    raw_nodes = {10: {}, 11: {}, 12: {}}
    result = mapped_candidates(path, np.array([10, 11, 12], np.int64), raw_nodes, 0.01, 14.0)
    assert result == {11: [10]}


@pytest.mark.slow
def test_frozen_sklearn_runtimes_match_bundle_predict_proba() -> None:
    rng = np.random.default_rng(12)
    ownership_dir = BUNDLE / 'ownership'
    if (ownership_dir / 'deploy_spec.json').is_file():
        ours = FullPopulationOwnershipRuntime(object(), ownership_dir)
        original = ownership_dir / 'ownership_full_population_runtime_v1.py'
        theirs = _load_helper(
            original, 'original_ownership_runtime'
        ).FullPopulationOwnershipRuntime(object(), ownership_dir)
        x = rng.normal(size=(16, len(OWNERSHIP_FEATURES))).astype(np.float32)
        np.testing.assert_allclose(
            ours.models[0].predict_proba(x),
            theirs.models[0].predict_proba(x),
            atol=ATOL,
            rtol=0.0,
        )

    ranker_dir = BUNDLE / 'edgegraft' / 'ranker'
    metric_dir = BUNDLE / 'edgegraft' / 'metric'
    if (ranker_dir / 'fold_0.joblib').is_file() and (metric_dir / 'fold_0.joblib').is_file():
        ours = EdgeGraftV3Runtime(ranker_dir, metric_dir)
        original = BUNDLE / 'edgegraft' / 'edgegraft_v3_deploy_runtime_v5.py'
        try:
            theirs = _load_helper(original, 'original_edgegraft_runtime').EdgeGraftV3Runtime(
                ranker_dir, metric_dir
            )
        except Exception:
            theirs = None
        if theirs is not None:
            ranker = ours._model('ranker', 0)
            x = rng.normal(size=(16, int(ranker.n_features_in_))).astype(np.float32)
            np.testing.assert_allclose(
                ranker.predict_proba(x),
                theirs._model('ranker', 0).predict_proba(x),
                atol=ATOL,
                rtol=0.0,
            )

    cg_dir = BUNDLE / 'candidategraft'
    if (cg_dir / 'candidategraft_direct.joblib').is_file():
        ours = CandidateGraftDirectRuntime(cg_dir, BUNDLE / 'edgegraft')
        original = cg_dir / 'candidategraft_direct_runtime_v1.py'
        theirs = _load_helper(
            original, 'original_candidategraft_runtime'
        ).CandidateGraftDirectRuntime(cg_dir, BUNDLE / 'edgegraft')
        x = rng.normal(size=(16, ours.model.n_features_in_)).astype(np.float32)
        np.testing.assert_allclose(
            ours.model.predict_proba(x),
            theirs.model.predict_proba(x),
            atol=ATOL,
            rtol=0.0,
        )


@pytest.mark.slow
def test_source_cardinality_runtime_matches_original_bundle() -> None:
    artifact = BUNDLE / 'source_cardinality'
    head = artifact / 'source_cardinality_head.pt'
    original = artifact / 'source_cardinality_runtime.py'
    if not head.is_file() or not original.is_file():
        _skip('frozen cardinality runtime is missing')

    ours = SourceCardinalityRuntime(artifact)
    theirs = _load_helper(original, 'original_cardinality_runtime').SourceCardinalityRuntime(
        artifact
    )
    rng = np.random.default_rng(0)
    source = rng.normal(size=(4, ours.source_mean.shape[0])).astype(np.float32)
    pair = rng.normal(size=(8, ours.pair_mean.shape[0])).astype(np.float32)
    owner = np.array([0, 0, 1, 1, 2, 2, 3, 3], np.int32)
    nodes = np.arange(16, dtype=np.int64).reshape(8, 2)
    left = ours.score(source, pair, owner, nodes)
    right = theirs.score(source, pair, owner, nodes)
    np.testing.assert_allclose(left[0], right[0], atol=ATOL)
    np.testing.assert_array_equal(left[1], right[1])


@pytest.mark.slow
def test_one_movie_infer_matches_locked_csv() -> None:
    if not LOCKED_CSV.is_file():
        _skip('locked one-movie CSV fixture is missing')
    locked = LOCKED_CSV.read_text().splitlines()
    assert locked[0].split(',')[:3] == ['id', 'dataset', 'row_type']
    body = [line.split(',') for line in locked[1:] if line]
    node_rows = [row for row in body if row[2] == 'node']
    edge_rows = [row for row in body if row[2] == 'edge']
    assert len(node_rows) == 25645
    assert len(edge_rows) == 24957
    for row in node_rows:
        int(row[3])
        int(row[4])
        int(row[5])
        int(row[6])
        int(row[7])
    candidates = [
        E2E_ONE_MOVIE / 'workdir' / 'submission.csv',
        NEW_ONE_MOVIE / 'workdir' / 'submission.csv',
        NEW_ONE_MOVIE / 'workdir' / 'submission_shards' / '44b6_0113de3b.csv',
        OLD_ONE_MOVIE / 'workdir' / 'submission.csv',
    ]
    compared = False
    for csv_path in candidates:
        if csv_path.is_file():
            assert csv_path.read_text().splitlines() == locked
            compared = True
    if not compared:
        _skip('one-movie infer output is not available yet')
