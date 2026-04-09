# TIMO 仓库 RL 接入映射（UniFSL-RL-Pure）

## 1) 当前 TIMO / TIMO-S 入口与调用链
- 入口：`main.py`。
- 加载缓存：
  - `text_weights_cupl_t_all.pt`（CuPL prompts 全量文本特征）
  - `keys_{shot}shots.pt` / `values_{shot}shots.pt`（few-shot cache）
  - `{shot}_vecs_f.pt` / `{shot}_labels_f.pt`（few-shot support 特征与标签）
  - `val_f.pt` / `val_l.pt`，`test_f.pt` / `test_l.pt`（或 ImageNet 特例重用 val）
- 调用链（paper 路径）
  1. `main.py -> image_guide_text(...)` 得到 IGT 权重与 matching score
  2. `main.py -> TIMO(..., grid_search=False)` 跑 TIMO
  3. `main.py -> image_guide_text_search(...)` 搜 gamma
  4. `main.py -> TIMO(..., grid_search=True)` 跑 TIMO-S

## 2) alpha / beta / gamma 搜索位置
- `gamma` 搜索：`utils.py::image_guide_text_search`，候选 `range(5, 101, 5)`。
- `beta` 搜索：`models.py::TIMO`，`grid_search=True` 时在 prompt 数范围扫描（快速模式用 linspace 采样）。
- `alpha` 搜索：`models.py::GDA`，固定候选 `{1e-4,...,1e4}` 网格。

## 3) 可复用缓存与评估函数
- 缓存读取：`utils.py::load_few_shot_feature`、`loda_val_test_feature`。
- 核心算子可复用：
  - `utils.py::image_guide_text`（IGT 生成）
  - `utils.py::vec_sort`（prompt bank 排序）
  - `utils.py::cls_acc`（top1 准确率）
- GDA 参数估计逻辑可拆：`models.py::GDA` 中 `W,b` 求解部分。

## 4) prompt 数 P 的运行时来源
- 来源于 `text_weights_cupl_t_all.pt` 的形状 `[C, P, D]`，即第二维。
- 在 `main.py` 中由 `cate_num, prompt_cupl_num, dim = clip_weights_cupl_all.shape` 获取。

## 5) 可复用 vs 必须拆分
- 可复用：`image_guide_text`、`vec_sort`、accuracy 计算、缓存读取。
- 必须拆分：
  - `models.py::TIMO` 内部混合了 prompt 排序、beta 扫描、transfer set 构建、GDA 调用。
  - `models.py::GDA` 将参数估计与 alpha 搜索耦合。
- 需要拆出纯函数：`run_igt`、`select_tgi_prompts`、`fit_gda_classifier`、`run_timo_config`、`support_loo_score`、`build_state_stats`。

## 6) RL path 必须绕开的旧入口
RL `train/eval/probe` 模式必须绕开以下旧 selector：
- `utils.py::image_guide_text_search`（gamma 搜索）
- `models.py::GDA` 中 alpha 网格
- `models.py::TIMO` 中 beta 扫描 + top-beta 固定选择

新的 RL path 必须由策略动作直接决定 `{gamma,beta,subset,alpha}`，并仅在缓存张量上推理，不重复 CLIP 编码。
