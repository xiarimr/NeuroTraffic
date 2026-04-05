# Window-Level Oracle 项目方案

## 1. 项目目标

把当前仓库从“`rule/llm` 直接选择 `reward_mode`”推进到“基于 `window-level oracle` 监督出来的 selector 在线选择 `reward_mode`”。

目标闭环：

1. 运行实验并沉淀 `train_round` 日志
2. 针对 `train:*` window 生成 `oracle_mode`
3. 把 `oracle_mode` 贴回 selector 原始数据集
4. 训练 selector baseline
5. 用 oracle 数据对 LLM mode selector 做微调
6. 在线回接并做端到端评估

## 2. 当前仓库的真实训练链

当前代码的关键因果链如下：

1. `utils/cityflow_env.py` 在环境交互时把 `reward_mode` 写入 `inter_*.pkl`
2. `utils/construct_sample.py` 离线读取日志，并依据日志里的 `reward_mode` 重算 reward
3. `utils/updater.py` 在 round 末使用这些样本更新模型
4. `utils/model_test.py` 加载该 round 训练后的模型做 test

因此，当前仓库中真正合理的 `window oracle` 定义是：

- 固定同一个 round 的起点模型
- 只改一个 `train:*` window 的 `reward_mode`
- 重建该 round 的 sample 和 updater 结果
- 比较最终 round 模型的 test 指标
- 最优 mode 记作该 window 的 `oracle_mode`

这不是全局最优 oracle，而是“单窗口边际 oracle”。

## 3. 统一术语

- `window sample`：一个 selector 样本，主键是 `sample_id = episode_id:time_step`
- `base model`：某个 round 开始前的模型，即 `round_{r-1}_inter_*.pt`
- `oracle_mode`：在固定 `base model` 下，仅修改当前 window 的 `reward_mode` 后，能让 round `r` 训练后 test 最优的 mode
- `oracle_confidence`：oracle 第一名相对于第二名的优势强弱

## 4. 目录规划

以下目录是推荐保留或新增的：

- `docs/`
  - 存放项目设计文档和任务清单
- `scripts/`
  - 存放数据生成和批处理脚本
- `selector/`
  - 存放 selector baseline、训练代码、推理代码
- `llm/`
  - 保留现有 LLM selector，并新增微调后 selector 入口

以下目录建议作为生成产物，不纳入版本控制：

- `window_oracle_labels/`
- `window_oracle_selector_datasets/`
- `selector_training_subsets/`
- `selector_models/`
- `selector_finetune_data/`
- `tmp_window_oracle/`

## 5. 数据流方案

### 阶段 A：Window Oracle Teacher 生成

输入：

- `records/<memo>/<experiment>/train_round/...`
- `model/<memo>/<experiment>/round_*.pt`
- `selector_dataset_exports/<memo>/<experiment>/dataset.json`

输出：

- `window_oracle_labels/<memo>/<experiment>/window_oracle_labels.json`
- `window_oracle_labels/<memo>/<experiment>/candidate_metrics.csv`

### 阶段 B：Oracle 数据集构建

输入：

- `selector_dataset_exports/...`
- `window_oracle_labels/...`

输出：

- `window_oracle_selector_datasets/.../dataset.json`
- `window_oracle_selector_datasets/.../train.jsonl`
- `window_oracle_selector_datasets/.../val.jsonl`

### 阶段 C：Selector 训练

输入：

- `window_oracle_selector_datasets/...`

输出：

- `selector_models/baseline/...`
- `selector_models/llm_ft/...`

## 6. 模型路线

推荐按两条路线推进：

### 路线 1：结构化 selector baseline

先训练一个简单的多分类器，验证 oracle 标签本身是否有信息量。

输入：

- `features`
- `context`
- 可选的 `stage_feature`

输出：

- 四分类 mode

价值：

- 成本低
- 可以快速验证 oracle 是否有效
- 为 LLM 微调提供先验

### 路线 2：LLM 微调

在 baseline 验证通过后，再用高置信 oracle 样本微调 LLM mode selector。

输入：

- `system + user`

监督：

- `assistant = oracle_mode`

## 7. 样本筛选原则

为避免标签噪声，window oracle 样本建议满足：

1. 来自 `train:*`
2. 优先来自中后期 round，例如 `round >= 20`
3. 优先选择这些 window：
   - `mode_changed = true`
   - `rule_mode != llm_mode`
   - `outcomes` 变化幅度大
4. 对 oracle 结果计算置信度
5. 低置信样本可降权或丢弃

## 8. 评估指标

### 离线标签与分类评估

- macro-F1
- per-class recall
- high-confidence subset accuracy
- 与原始 `llm` 标签的分歧率

### 在线控制评估

- `average_travel_time`
- `average_waiting_time`
- `average_queue_length`
- `throughput`
- `mode_switch_count`
- 多随机种子稳定性

## 9. 推荐开发顺序

1. 先实现单样本 `window oracle`
2. 再实现小批量 `window oracle`
3. 再构建 oracle-attached 数据集
4. 再训练结构化 selector baseline
5. 最后做 LLM 微调和在线接入

## 10. 当前清理结论

本次仓库清理采用“非破坏性清理”原则：

- 不直接删除已有 `records/`、`model/`、`selector_dataset_exports/`
- 先通过文档和 `.gitignore` 把“代码”与“产物”边界划清
- 把后续真正要维护的代码入口收敛到 `scripts/`、`selector/`、`llm/`
