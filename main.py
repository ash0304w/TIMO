import argparse
import os
import random

import torch
import yaml

import clip
from models import APE, GDA_CLIP, TIMO, run_tip_adapter
from unifsl_rl.config import RLConfig
from unifsl_rl.infer import eval_joint_exact, eval_pure_rl, eval_safe_or_strict, probe_pure_rl, probe_safe
from unifsl_rl.methods import build_adapter
from unifsl_rl.train import train_pure_rl
from utils import image_guide_text, image_guide_text_search, load_few_shot_feature, loda_val_test_feature, save_log


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dbg', type=float, default=0)
    parser.add_argument('--config', required=True)

    parser.add_argument('--method', type=str, default='timo', choices=['timo', 'gda_clip'])
    parser.add_argument('--backend', type=str, default='safe_rl_legacy', choices=['paper', 'joint_exact', 'safe_rl_legacy', 'pure_rl'])
    parser.add_argument('--mode', type=str, default='eval', choices=['train', 'eval', 'probe'])
    parser.add_argument('--ckpt', type=str, default='')

    parser.add_argument('--timo_mode', type=str, default='safe_rl', choices=['paper', 'joint_exact', 'safe_rl', 'strict_rl'])
    parser.add_argument('--rl_mode', type=str, default='eval', choices=['train', 'eval', 'probe'])
    parser.add_argument('--rl_ckpt', type=str, default='')

    parser.add_argument('--safe_floor_mode', type=str, default='safe', choices=['paper', 'joint_exact', 'safe'])
    parser.add_argument('--require_significant_gain', type=int, default=1)
    parser.add_argument('--significance_margin', type=float, default=0.0)
    parser.add_argument('--verify_repeats', type=int, default=3)
    parser.add_argument('--verify_use_ci', type=int, default=1)

    parser.add_argument('--subset_mode', type=str, default='residual_prefix_swap', choices=['prefix', 'residual_prefix_swap'])
    parser.add_argument('--swap_window', type=int, default=8)
    parser.add_argument('--swap_budget', type=int, default=2)
    parser.add_argument('--beta_domain_mode', type=str, default='repo_compat', choices=['paper_strict', 'repo_compat'])

    parser.add_argument('--rl_train_tasks', type=str, default='auto')
    parser.add_argument('--rl_train_epochs', type=int, default=20)
    parser.add_argument('--rl_batch_size', type=int, default=8)
    parser.add_argument('--rl_lr', type=float, default=1e-3)
    parser.add_argument('--rl_entropy_coef', type=float, default=1e-3)
    parser.add_argument('--rl_value_coef', type=float, default=0.5)
    parser.add_argument('--rl_cost_lambda', type=float, default=1e-4)
    parser.add_argument('--rl_violation_lambda', type=float, default=1e-4)
    parser.add_argument('--rl_seed', type=int, default=1)
    parser.add_argument('--rl_device', type=str, default='cuda')
    parser.add_argument('--warm_start_ckpt', type=str, default='')

    parser.add_argument('--rl_trials', type=int, default=3)
    parser.add_argument('--rl_budget', type=int, default=2)
    parser.add_argument('--probe_metric', type=str, default='support_loo', choices=['support_loo'])
    parser.add_argument('--probe_refine_neighbors', type=int, default=1)

    parser.add_argument('--dump_jsonl', type=int, default=1)
    parser.add_argument('--dump_csv', type=int, default=1)
    parser.add_argument('--dump_candidate_pool', type=int, default=1)
    parser.add_argument('--save_rl_outputs', type=int, default=1)
    return parser.parse_args()


def resolve_backend_mode(args):
    if args.backend == 'safe_rl_legacy' and args.timo_mode in {'paper', 'joint_exact', 'safe_rl', 'strict_rl'}:
        mode_map = {'paper': 'paper', 'joint_exact': 'joint_exact', 'safe_rl': 'safe_rl_legacy', 'strict_rl': 'safe_rl_legacy'}
        backend = mode_map[args.timo_mode]
        mode = args.rl_mode if backend == 'safe_rl_legacy' else args.mode
        return backend, mode, (args.timo_mode == 'strict_rl')
    return args.backend, args.mode, False


def run_paper_mode(cfg):
    clip_weights_cupl_all = torch.load(cfg['cache_dir'] + "/text_weights_cupl_t_all.pt", weights_only=False)
    cate_num, _, _ = clip_weights_cupl_all.shape
    clip_weights_cupl = clip_weights_cupl_all.mean(dim=1).t()
    clip_weights_cupl = clip_weights_cupl / clip_weights_cupl.norm(dim=0, keepdim=True)

    cache_keys, cache_values = load_few_shot_feature(cfg)
    val_features, val_labels = loda_val_test_feature(cfg, "val")
    if cfg['dataset'] == 'imagenet':
        test_features, test_labels = loda_val_test_feature(cfg, "val")
    else:
        test_features, test_labels = loda_val_test_feature(cfg, "test")

    image_weights_all = torch.stack([cache_keys.t()[torch.argmax(cache_values, dim=1) == i] for i in range(cate_num)])
    image_weights = image_weights_all.mean(dim=1)
    image_weights = image_weights / image_weights.norm(dim=1, keepdim=True)

    clip_weights_IGT, matching_score = image_guide_text(cfg, clip_weights_cupl_all, image_weights, return_matching=True)
    clip_weights_IGT = clip_weights_IGT.t()
    metric = {}
    metric['Tip_Adapter'] = run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights_cupl)
    metric['APE'] = APE(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights_cupl)
    metric['GDA_CLIP'] = GDA_CLIP(cfg, val_features, val_labels, test_features, test_labels, clip_weights_cupl)
    metric['TIMO'] = TIMO(cfg, val_features, val_labels, test_features, test_labels, clip_weights_IGT, clip_weights_cupl_all, matching_score, grid_search=False, is_print=True)
    clip_weights_IGT, matching_score = image_guide_text_search(cfg, clip_weights_cupl_all, val_features, val_labels, image_weights)
    metric['TIMO_S'] = TIMO(cfg, val_features, val_labels, test_features, test_labels, clip_weights_IGT, clip_weights_cupl_all, matching_score, grid_search=True, n_quick_search=10, is_print=True)
    save_log(cfg, metric)


def build_rl_cfg(args):
    return RLConfig(
        timo_mode=args.backend,
        rl_mode=args.mode,
        ckpt=args.ckpt or args.rl_ckpt,
        save_rl_outputs=args.save_rl_outputs,
        safe_floor_mode=args.safe_floor_mode,
        require_significant_gain=args.require_significant_gain,
        significance_margin=args.significance_margin,
        verify_repeats=args.verify_repeats,
        verify_use_ci=args.verify_use_ci,
        subset_mode=args.subset_mode,
        swap_window=args.swap_window,
        swap_budget=args.swap_budget,
        beta_domain_mode=args.beta_domain_mode,
        rl_train_tasks=args.rl_train_tasks,
        rl_train_epochs=args.rl_train_epochs,
        rl_batch_size=args.rl_batch_size,
        rl_lr=args.rl_lr,
        rl_entropy_coef=args.rl_entropy_coef,
        rl_value_coef=args.rl_value_coef,
        rl_cost_lambda=args.rl_cost_lambda,
        rl_violation_lambda=args.rl_violation_lambda,
        rl_seed=args.rl_seed,
        rl_device=args.rl_device,
        warm_start_ckpt=args.warm_start_ckpt,
        rl_trials=args.rl_trials,
        rl_budget=args.rl_budget,
        probe_metric=args.probe_metric,
        probe_refine_neighbors=args.probe_refine_neighbors,
        dump_jsonl=args.dump_jsonl,
        dump_csv=args.dump_csv,
        dump_candidate_pool=args.dump_candidate_pool,
    )


def print_pure_summary(step, protocol, backend, method):
    print(f"method={method} backend={backend} protocol={protocol}")
    print(f"chosen alpha idx/value={step.actions.get('alpha_idx')}/{step.actions.get('alpha')}")
    print(f"chosen beta={step.actions.get('beta')}")
    print(f"chosen subset={step.actions.get('subset_indices')}")
    print(f"chosen gamma idx/value={step.actions.get('gamma_idx')}/{step.actions.get('gamma')}")
    print(f"reward={step.reward}")
    print(f"cost={step.cost}")
    print(f"violation={step.violation}")
    print(f"repair flag={step.report['stage0'].repaired or step.report['stage1'].repaired}")


def main():
    args = get_arguments()
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfg['shots'] = args.shot
    cfg['seed'] = args.seed
    cfg['dbg'] = args.dbg
    os.makedirs('outputs', exist_ok=True)
    cache_dir = os.path.join(f'./caches/{cfg["backbone"]}/{cfg["seed"]}/{cfg["dataset"]}')
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    clip_model, _ = clip.load(cfg['backbone'])
    clip_model.eval()
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])

    backend, mode, strict = resolve_backend_mode(args)
    rl_cfg = build_rl_cfg(args)

    if backend == 'paper':
        run_paper_mode(cfg)
        return

    adapter = build_adapter(args.method, cfg, device=rl_cfg.rl_device)

    if backend == 'joint_exact':
        res = eval_joint_exact(adapter, rl_cfg)
        print(f"[JOINT_EXACT] alpha={res.alpha_value} beta={res.beta} gamma={res.gamma_value} score={res.selection_score}")
        return

    if backend == 'pure_rl':
        if mode == 'train':
            ckpt, stats = train_pure_rl(adapter, rl_cfg)
            print(f"[PURE_RL TRAIN] ckpt={ckpt} stats={stats}")
        elif mode == 'eval':
            step = eval_pure_rl(adapter, rl_cfg)
            print_pure_summary(step, 'offline_eval', backend, args.method)
        else:
            step = probe_pure_rl(adapter, rl_cfg)
            print_pure_summary(step, 'test_time_probe', backend, args.method)
        return

    if backend == 'safe_rl_legacy':
        if mode == 'train':
            ckpt, stats = train_pure_rl(adapter, rl_cfg)
            print(f"[LEGACY ABLATION TRAIN] ckpt={ckpt} stats={stats}")
        elif mode == 'eval':
            res = eval_safe_or_strict(adapter, rl_cfg, strict=strict)
            print(f"[LEGACY {'STRICT' if strict else 'SAFE'}] chosen={res['chosen'].selection_score}")
        else:
            res = probe_safe(adapter, rl_cfg, strict=strict)
            print(f"[LEGACY PROBE] chosen={res['chosen'].selection_score}")


if __name__ == '__main__':
    main()
