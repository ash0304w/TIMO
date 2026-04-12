# UniFSL-RL Phase-2 Implementation Report

## Modified files and purpose

- `main.py`: cleaned duplicate imports, added unified CLI (`--method`, `--backend`, `--mode`, `--ckpt`), wired `pure_rl` train/eval/probe to unified core pipeline, preserved legacy paper/joint/safe paths as legacy ablations.
- `unifsl_rl/methods/__init__.py`: added method registry and factory (`register_method`, `build_adapter`) with `timo` and `gda_clip` registrations.
- `unifsl_rl/core/protocol.py`: promoted protocol into executable policy (`TRAIN_PROTOCOL`, `EVAL_PROTOCOL`, `PROBE_PROTOCOL`) with explicit signal permissions and selection split.
- `unifsl_rl/core/access_guard.py`: added `GuardedMapping` and strict probe checks; fixed probe reward guard to `test_time_probe`.
- `unifsl_rl/core/slot.py`: replaced dataclass shell with abstract `Slot` contract.
- `unifsl_rl/core/action_spec.py`: kept typed specs and canonical projection/index-value behavior.
- `unifsl_rl/core/state_builder.py`: stage-aware state construction using real `slot.observe`, guard checks, fixed-size tensor output.
- `unifsl_rl/core/policy_factory.py`: multi-head policy from `CompositeActionSpec`, mixed discrete/subset/box heads, returning actions/log_prob/entropy/value.
- `unifsl_rl/core/controller.py`: implemented coordinator main loop (`decide_stage0/materialize/decide_stage1/apply_constraints/evaluate`).
- `unifsl_rl/core/constraints.py`: implemented repair and projection (`beta`, subset top-k repair, alpha/gamma projection, cost/penalty hooks).
- `unifsl_rl/core/method_wrapper.py`: upgraded wrapper interface for pure-rl lifecycle.
- `unifsl_rl/train.py`: rewritten to pure online actor-critic style loop with optimizer updates each epoch, output artifacts (`best_pure_rl.pt`, `train_log.csv`, `final_decision.json`).
- `unifsl_rl/infer.py`: added `eval_pure_rl` and `probe_pure_rl` on unified coordinator path; retained legacy safe functions.
- `unifsl_rl/methods/timo/slots.py`: implemented real four Slot subclasses with stage split (stage0: gamma/beta/subset; stage1: alpha).
- `unifsl_rl/methods/timo/ops.py`: added pure-rl shared functional aliases (`run_igt`, `fit_gda_classifier`, `run_timo_config`, `support_loo_score`, `build_state_stats`) and unified subset index space handling.
- `unifsl_rl/methods/timo/adapter.py`: converted TIMO into full `MethodWrapper` with protocol views, materialize/run/evaluate/cost.
- `unifsl_rl/methods/gda_clip/adapter.py`, `unifsl_rl/methods/gda_clip/slots.py`, `unifsl_rl/methods/gda_clip/__init__.py`: added real second method adapter and slot wiring.
- `unifsl_rl/methods/gda_clip_skeleton/README.md`: marked skeleton as deprecated alias.
- `tests/*.py`: added protocol, slot conflict, registry, subset-space, and pure-rl boundary tests.

## Layer mapping

### MethodWrapper Layer
- `unifsl_rl/core/method_wrapper.py`
- `unifsl_rl/methods/timo/adapter.py`
- `unifsl_rl/methods/gda_clip/adapter.py`

### Protocol Layer
- `unifsl_rl/core/protocol.py`
- `unifsl_rl/core/access_guard.py`

### RL Template Layer
- `unifsl_rl/core/action_spec.py`
- `unifsl_rl/core/slot.py`
- `unifsl_rl/core/state_builder.py`
- `unifsl_rl/core/policy_factory.py`
- `unifsl_rl/train.py`
- `unifsl_rl/infer.py`

### Coordinator Layer
- `unifsl_rl/core/controller.py`
- `unifsl_rl/core/constraints.py`

## Legacy safe_rl vs pure_rl boundary

- `pure_rl` route: only through wrapper + state builder + policy factory + coordinator, no incumbent residual floor in the pure path.
- `safe_rl_legacy` route: keeps old `infer_safe` based ablation behavior for comparison and backward compatibility.

## Why second method proves transferability

- `gda_clip` is registered through method factory and runs with the same core coordinator/policy/state machinery.
- Adding `gda_clip` required method-specific adapter/slots only; no changes were made to controller/policy_factory/constraints logic to enable it.

## 论文原始复现路径与当前多后端路径说明

### 1) 论文原始复现路径（legacy/paper）
该路径仍然走原项目 `models.py` 中的 `Tip-Adapter/APE/GDA_CLIP/TIMO/TIMO_S` 组合流程，不经过 pure RL coordinator：

```bash
python main.py --config configs/imagenet.yaml --shot 1 --seed 1 --backend paper --method timo
```

兼容旧参数也可触发 paper：

```bash
python main.py --config configs/imagenet.yaml --shot 1 --seed 1 --timo_mode paper
```

### 2) joint_exact 上界/基线路径（legacy deterministic baseline）
用于确定性上界搜索对比：

```bash
python main.py --config configs/imagenet.yaml --shot 1 --seed 1 --backend joint_exact --method timo
```

兼容旧参数：

```bash
python main.py --config configs/imagenet.yaml --shot 1 --seed 1 --timo_mode joint_exact
```

### 3) safe_rl / strict_rl 遗留消融路径（legacy ablation）
该路径保留供对照，不作为默认主实现：

```bash
# safe_rl legacy
python main.py --config configs/imagenet.yaml --shot 1 --seed 1 --backend safe_rl_legacy --mode eval --method timo --ckpt <path>

# strict_rl legacy（通过旧参数兼容）
python main.py --config configs/imagenet.yaml --shot 1 --seed 1 --backend safe_rl_legacy --mode eval --method timo --timo_mode strict_rl --ckpt <path>
```

### 4) pure_rl 主路径（Phase-2 hardened）
该路径走统一核心：
`MethodWrapper -> JointStateBuilder -> PolicyFactory -> RLCoordinator -> ConstraintEngine -> Wrapper.evaluate`

```bash
# train
python main.py --config configs/imagenet.yaml --shot 1 --seed 1 --backend pure_rl --mode train --method timo

# eval
python main.py --config configs/imagenet.yaml --shot 1 --seed 1 --backend pure_rl --mode eval --method timo --ckpt <path>

# probe
python main.py --config configs/imagenet.yaml --shot 1 --seed 1 --backend pure_rl --mode probe --method timo --ckpt <path> --rl_budget 2
```

### 5) 第二方法迁移路径（gda_clip）
证明 method-agnostic：不改 core coordinator/policy/constraints 即可接入：

```bash
python main.py --config configs/imagenet.yaml --shot 1 --seed 1 --backend pure_rl --mode eval --method gda_clip --ckpt <path>
```

### 6) 旧参数到新参数映射
- `timo_mode=paper` -> `backend=paper`
- `timo_mode=joint_exact` -> `backend=joint_exact`
- `timo_mode=safe_rl` -> `backend=safe_rl_legacy`
- `timo_mode=strict_rl` -> `backend=safe_rl_legacy` + strict 标志

因此历史脚本可继续跑，建议新脚本统一切换到 `--backend/--mode/--method`。
