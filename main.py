import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *
from models import *
from unifsl_rl.config import RLConfig
from unifsl_rl.methods.timo.adapter import TIMOAdapter
from unifsl_rl.train import train_rl
from unifsl_rl.infer import eval_rl, probe_rl


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--shot', dest='shot', type=int, default=1, help='shots number')
    parser.add_argument('--seed', dest='seed', type=int, default=1, help='seed')
    parser.add_argument('--dbg', dest='dbg', type=float, default=0, help='debug mode')
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    parser.add_argument('--use_rl', type=int, default=0)
    parser.add_argument('--rl_mode', type=str, default='none', choices=['none', 'train', 'eval', 'probe'])
    parser.add_argument('--rl_ckpt', type=str, default='')
    parser.add_argument('--rl_train_episodes', type=int, default=50)
    parser.add_argument('--rl_lr', type=float, default=1e-3)
    parser.add_argument('--rl_budget', type=int, default=2)
    parser.add_argument('--rl_trials', type=int, default=1)
    parser.add_argument('--rl_cost_lambda', type=float, default=0.01)
    parser.add_argument('--rl_violation_lambda', type=float, default=0.1)
    parser.add_argument('--rl_align_lambda', type=float, default=0.0)
    parser.add_argument('--rl_anom_lambda', type=float, default=0.0)
    parser.add_argument('--rl_seed', type=int, default=1)
    parser.add_argument('--rl_device', type=str, default='cuda')
    parser.add_argument('--save_rl_outputs', type=int, default=1)
    args = parser.parse_args()
    return args


def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfg['shots'] = args.shot
    cfg['seed'] = args.seed
    cfg['dbg'] = args.dbg
    cfg['use_rl'] = args.use_rl
    print("shots", cfg['shots'])
    print("seed", cfg['seed'])
    print("dbg", cfg['dbg'])
    
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    cache_dir = os.path.join(f'./caches/{cfg["backbone"]}/{cfg["seed"]}/{cfg["dataset"]}')
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir
    print(cfg)

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # Prepare dataset
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights_cupl_all = torch.load(cfg['cache_dir'] + "/text_weights_cupl_t_all.pt", weights_only=False)
    cate_num, prompt_cupl_num, dim = clip_weights_cupl_all.shape
    clip_weights_cupl = clip_weights_cupl_all.mean(dim=1).t()
    clip_weights_cupl = clip_weights_cupl / clip_weights_cupl.norm(dim=0, keepdim=True)
    
    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = load_few_shot_feature(cfg)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = loda_val_test_feature(cfg, "val")
    
    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    if cfg['dataset'] == 'imagenet':
        test_features, test_labels = loda_val_test_feature(cfg, "val")
    else:
        test_features, test_labels = loda_val_test_feature(cfg, "test")


    # ------------------------------------------ RL path ------------------------------------------
    if args.use_rl == 1:
        rl_cfg = RLConfig(
            mode=args.rl_mode,
            train_episodes=args.rl_train_episodes,
            lr=args.rl_lr,
            budget=args.rl_budget,
            trials=args.rl_trials,
            cost_lambda=args.rl_cost_lambda,
            violation_lambda=args.rl_violation_lambda,
            align_lambda=args.rl_align_lambda,
            anom_lambda=args.rl_anom_lambda,
            seed=args.rl_seed,
            device=args.rl_device,
            ckpt=args.rl_ckpt,
        )
        adapter = TIMOAdapter(cfg, device=rl_cfg.device)
        if rl_cfg.mode == 'train':
            ckpt_path, _ = train_rl(adapter, rl_cfg)
            print(f"[RL] checkpoint: {ckpt_path}")
            if args.save_rl_outputs:
                with open('outputs/UniFSL_RL_train.txt', 'a') as f:
                    f.write(f"{cfg['dataset']}_{cfg['shots']}_{cfg['seed']}: ckpt={ckpt_path}\n")
        elif rl_cfg.mode == 'eval':
            test_acc, step = eval_rl(adapter, rl_cfg)
            print(f"[RL-EVAL] test_acc={test_acc:.2f}")
            print(f"[RL-EVAL] chosen alpha={step.actions['alpha']} beta={step.actions['beta']} gamma={step.actions['gamma']} subset={step.actions.get('subset').tolist() if step.actions.get('subset') is not None else None}")
            print(f"[RL-EVAL] reward={step.reward:.4f} cost={step.actions['beta']} violation={step.report['stage0'].violation + step.report['stage1'].violation} repair={step.report['stage0'].repaired or step.report['stage1'].repaired} protocol=eval ckpt={rl_cfg.ckpt}")
        elif rl_cfg.mode == 'probe':
            test_acc, step = probe_rl(adapter, rl_cfg)
            print(f"[RL-PROBE] test_acc={test_acc:.2f}")
            print(f"[RL-PROBE] chosen alpha={step.actions['alpha']} beta={step.actions['beta']} gamma={step.actions['gamma']} subset={step.actions.get('subset').tolist() if step.actions.get('subset') is not None else None}")
            print(f"[RL-PROBE] reward={step.reward:.4f} cost={step.actions['beta']} violation={step.report['stage0'].violation + step.report['stage1'].violation} repair={step.report['stage0'].repaired or step.report['stage1'].repaired} protocol=probe ckpt={rl_cfg.ckpt}")
        else:
            raise ValueError("--use_rl=1 时必须指定 --rl_mode train/eval/probe")
        return

    # ------------------------------------------ Fusion ------------------------------------------
    
    image_weights_all = torch.stack([cache_keys.t()[torch.argmax(cache_values, dim=1)==i] for i in range(cate_num)])
    image_weights = image_weights_all.mean(dim=1)
    image_weights = image_weights / image_weights.norm(dim=1, keepdim=True) 
    
    clip_weights_IGT, matching_score = image_guide_text(cfg, 
        clip_weights_cupl_all, image_weights, return_matching=True)
    clip_weights_IGT = clip_weights_IGT.t()
    metric = {}
    
    # ------------------------------------------ Baseline ------------------------------------------
    # Tip-Adapter
    acc_free = run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, 
        test_features, test_labels, clip_weights_cupl)
    metric['Tip_Adapter'] = acc_free
    
    # APE
    acc_free = APE(cfg, cache_keys, cache_values, val_features, val_labels,  
        test_features, test_labels, clip_weights_cupl)
    metric['APE'] = acc_free
    
    # GDA-CLIP
    acc_free = GDA_CLIP(cfg, val_features, val_labels, test_features, test_labels, clip_weights_cupl)
    metric['GDA_CLIP'] = acc_free
    
    # ------------------------------------------ Ours ------------------------------------------
    # TIMO   
    acc_free = TIMO(cfg, val_features, val_labels, test_features, test_labels, 
        clip_weights_IGT, clip_weights_cupl_all, matching_score,
        grid_search=False, is_print=True)
    metric['TIMO'] = acc_free

    # TIMO-S
    clip_weights_IGT, matching_score = image_guide_text_search(cfg, 
        clip_weights_cupl_all, val_features, val_labels, image_weights)
    acc_free = TIMO(cfg, val_features, val_labels, test_features, test_labels, 
        clip_weights_IGT, clip_weights_cupl_all, matching_score, 
        grid_search=True, n_quick_search=10, is_print=True)
    metric['TIMO_S'] = acc_free

    save_log(cfg, metric)
    
    
    
if __name__ == '__main__':
    main()
