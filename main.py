import os
import random
import argparse
import yaml

import torch

import clip
from utils import *
from models import *
from unifsl_rl.config import RLConfig
from unifsl_rl.methods.timo.adapter import TIMOAdapter
from unifsl_rl.train import train_rl
from unifsl_rl.infer import eval_rl, probe_rl

from unifsl_rl.config import RLConfig
from unifsl_rl.methods.timo.adapter import TIMOAdapter
from unifsl_rl.train import train_rl
from unifsl_rl.infer import eval_joint_exact, eval_safe_or_strict, probe_safe

from unifsl_rl.config import RLConfig
from unifsl_rl.methods.timo.adapter import TIMOAdapter
from unifsl_rl.train import train_rl
from unifsl_rl.infer import eval_joint_exact, eval_safe_or_strict, probe_safe

from unifsl_rl.config import RLConfig
from unifsl_rl.methods.timo.adapter import TIMOAdapter
from unifsl_rl.train import train_rl
from unifsl_rl.infer import eval_joint_exact, eval_safe_or_strict, probe_safe

from unifsl_rl.config import RLConfig
from unifsl_rl.methods.timo.adapter import TIMOAdapter
from unifsl_rl.train import train_rl
from unifsl_rl.infer import eval_joint_exact, eval_safe_or_strict, probe_safe


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shot', dest='shot', type=int, default=1, help='shots number')
    parser.add_argument('--seed', dest='seed', type=int, default=1, help='seed')
    parser.add_argument('--dbg', dest='dbg', type=float, default=0, help='debug mode')
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')

    parser.add_argument('--timo_mode', type=str, default='safe_rl', choices=['paper', 'joint_exact', 'safe_rl', 'strict_rl'])
    parser.add_argument('--rl_mode', type=str, default='eval', choices=['train', 'eval', 'probe'])
    parser.add_argument('--rl_ckpt', type=str, default='')
    parser.add_argument('--save_rl_outputs', type=int, default=1)

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
    args = parser.parse_args()
    return args


def run_paper_mode(cfg):
    clip_weights_cupl_all = torch.load(cfg['cache_dir'] + "/text_weights_cupl_t_all.pt", weights_only=False)
    cate_num, prompt_cupl_num, dim = clip_weights_cupl_all.shape
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
        timo_mode=args.timo_mode,
        rl_mode=args.rl_mode,
        ckpt=args.rl_ckpt,
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


def print_safe_summary(mode_name, protocol, incumbent, chosen, ckpt_path=''):
    print(f"mode_name={mode_name}")
    print(f"protocol={protocol}")
    print(f"incumbent_mode={incumbent.mode_name}")
    print(f"incumbent alpha_idx/value={incumbent.alpha_idx}/{incumbent.alpha_value}")
    print(f"incumbent beta={incumbent.beta}")
    print(f"incumbent gamma_idx/value={incumbent.gamma_idx}/{incumbent.gamma_value}")
    print(f"incumbent subset summary=len:{len(incumbent.subset_indices)}")
    print(f"chosen alpha_idx/value={chosen.alpha_idx}/{chosen.alpha_value}")
    print(f"chosen beta={chosen.beta}")
    print(f"chosen gamma_idx/value={chosen.gamma_idx}/{chosen.gamma_value}")
    print(f"chosen subset indices={chosen.subset_indices}")
    print(f"chosen source_tag={chosen.source_tag}")
    print(f"incumbent selection_score={incumbent.selection_score}")
    print(f"chosen selection_score={chosen.selection_score}")
    print(f"delta vs incumbent={chosen.selection_score - incumbent.selection_score}")
    print(f"raw accuracy={chosen.raw_accuracy}")
    print(f"reward={(chosen.selection_score - incumbent.selection_score)}")
    print(f"cost={chosen.cost}")
    print(f"violation={chosen.violation}")
    print(f"repair flag={chosen.repair_flag}")
    print(f"ckpt path={ckpt_path}")


def main():
    args = get_arguments()
    assert os.path.exists(args.config)

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfg['shots'] = args.shot
    cfg['seed'] = args.seed
    cfg['dbg'] = args.dbg

    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    cache_dir = os.path.join(f'./caches/{cfg["backbone"]}/{cfg["seed"]}/{cfg["dataset"]}')
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])

    if args.timo_mode == 'paper':
        run_paper_mode(cfg)
        return

    rl_cfg = build_rl_cfg(args)
    adapter = TIMOAdapter(cfg, device=rl_cfg.rl_device)

    if args.timo_mode == 'joint_exact':
        res = eval_joint_exact(adapter, rl_cfg)
        print(f"[JOINT_EXACT] alpha={res.alpha_value} beta={res.beta} gamma={res.gamma_value} score={res.selection_score}")
        return

    if args.timo_mode == 'safe_rl':
        if args.rl_mode == 'train':
            ckpt, stats = train_rl(adapter, rl_cfg)
            print(f"[SAFE_RL TRAIN] ckpt={ckpt} stats={stats}")
        elif args.rl_mode == 'eval':
            res = eval_safe_or_strict(adapter, rl_cfg, strict=False)
            print_safe_summary('safe_rl', 'offline_eval', res['safe_inc'], res['chosen'], args.rl_ckpt)
        else:
            res = probe_safe(adapter, rl_cfg, strict=False)
            print_safe_summary('safe_rl', 'test_time_probe', res['safe_inc'], res['chosen'], args.rl_ckpt)
        return

    if args.timo_mode == 'strict_rl':
        if args.rl_mode == 'train':
            ckpt, stats = train_rl(adapter, rl_cfg)
            print(f"[STRICT_RL TRAIN] ckpt={ckpt} stats={stats}")
        elif args.rl_mode == 'eval':
            res = eval_safe_or_strict(adapter, rl_cfg, strict=True)
            print_safe_summary('strict_rl', 'strict_rl', res['safe_inc'], res['chosen'], args.rl_ckpt)
        else:
            res = probe_safe(adapter, rl_cfg, strict=True)
            print_safe_summary('strict_rl', 'test_time_probe', res['safe_inc'], res['chosen'], args.rl_ckpt)


if __name__ == '__main__':
    main()
