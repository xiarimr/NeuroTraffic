# Window-Level Oracle 任务清单

## P0：仓库整理

- [ ] 创建 `scripts/` 目录下的 window oracle 脚本
- [ ] 创建 `selector/` 目录下的 baseline 训练代码
- [ ] 确认 `.gitignore` 只忽略产物目录，不忽略源码目录
- [ ] 明确所有新产物的输出目录约定

## P1：Window Oracle Teacher MVP

### 目标

跑通“单个 `train:*` window -> 四个 mode 反事实 -> 选出 `oracle_mode`”。

### 任务

- [ ] 新建 `scripts/build_window_oracle_labels.py`
- [ ] 实现 `parse_train_sample_id(sample_id)`
- [ ] 实现 `resolve_experiment_from_sample(dataset_meta, sample)`
- [ ] 实现 `rewrite_reward_mode_for_window(inter_pkl_path, start_time, end_time, mode)`
- [ ] 实现 `copy_required_checkpoints(...)`
- [ ] 实现 `rebuild_history_samples_until_round(...)`
- [ ] 实现 `train_candidate_for_window(...)`
- [ ] 实现 `evaluate_candidate_model(...)`
- [ ] 实现 `rank_candidate_modes(...)`
- [ ] 输出 `window_oracle_labels.json`
- [ ] 输出 `candidate_metrics.csv`

### 验收标准

- [ ] 单个样本四个 mode 都能成功跑完
- [ ] 四个候选 mode 的指标不完全相同
- [ ] 能稳定输出 `oracle_mode`

## P2：Window Oracle 小批量生成

### 目标

对 20 到 50 个高价值 window 批量生成 oracle。

### 任务

- [ ] 新建 `scripts/build_window_oracle_batch.py`
- [ ] 增加样本筛选策略
- [ ] 增加失败重试与日志落盘
- [ ] 增加批处理 summary 输出

### 推荐样本筛选

- [ ] `train:*`
- [ ] `round >= 20`
- [ ] `mode_changed = true`
- [ ] `rule_mode != llm_mode`
- [ ] `outcomes` 绝对值较大

### 验收标准

- [ ] 至少生成 20 个高质量 oracle 样本
- [ ] 成功率可统计
- [ ] 有 per-sample 结果和 batch summary

## P3：Oracle 数据集构建

### 目标

把 `oracle_mode` 贴回 selector 原始数据集。

### 任务

- [ ] 新建 `scripts/attach_window_oracle_labels.py`
- [ ] 建立 `sample_id -> oracle_info` 的索引
- [ ] 将 `target_mode` 替换为 `oracle_mode`
- [ ] 保留 `llm_mode`、`rule_mode`、`label_source_before_oracle`
- [ ] 新增 `oracle_confidence`
- [ ] 生成新的 `dataset.json/train.jsonl/val.jsonl`

### 验收标准

- [ ] 任一 `sample_id` 可查回 oracle 来源
- [ ] 数据集统计字段完整
- [ ] 旧标签信息未丢失

## P4：Selector Baseline

### 目标

先用轻量分类器验证 oracle 标签是否有信息量。

### 任务

- [ ] 新建 `selector/train_classifier.py`
- [ ] 新建 `selector/eval_classifier.py`
- [ ] 定义特征抽取逻辑
- [ ] 训练四分类 baseline
- [ ] 输出混淆矩阵与 per-class 指标

### 推荐输入

- [ ] `features.queue`
- [ ] `features.queue_delta`
- [ ] `features.trunk`
- [ ] `features.trunk_delta`
- [ ] `features.throughput`
- [ ] `features.throughput_delta`
- [ ] `features.throughput_change_rate`
- [ ] `features.spillback`
- [ ] `features.spillback_delta`
- [ ] `context.current_mode`
- [ ] `context.last_mode`
- [ ] `context.mode_duration`
- [ ] 可选 `round_ratio`

### 验收标准

- [ ] oracle 标签训练效果优于原始 llm 标签训练效果
- [ ] 少数类召回不再严重塌陷

## P5：LLM 微调数据

### 目标

把 oracle 数据转换成可用于 SFT 的 `jsonl`。

### 任务

- [ ] 新建 `selector/export_sft_jsonl.py`
- [ ] 只导出高置信 oracle 样本
- [ ] 支持 train/val 划分
- [ ] 输出微调数据统计

### 建议字段

- [ ] `messages[0]`: system prompt
- [ ] `messages[1]`: window features + context + stage info
- [ ] `messages[2]`: `oracle_mode`

### 验收标准

- [ ] 可直接用于 SFT
- [ ] 样本类别分布基本平衡

## P6：LLM 微调与在线接入

### 目标

把微调后的 selector 接入在线环境。

### 任务

- [ ] 新建 `llm/finetuned_mode_selector.py`
- [ ] 在 `utils/selector_factory.py` 注册新 selector 类型
- [ ] 在 `utils/cityflow_env.py` 保持与现有 selector 接口兼容
- [ ] 保留 `rule` fallback

### 验收标准

- [ ] 在线运行稳定
- [ ] 能完整记录 `mode_selector_log.csv`
- [ ] 端到端指标优于现有 `rule/llm`

## P7：最终实验与报告

### 对比基线

- [ ] rule selector
- [ ] 原始 llm selector
- [ ] 原始标签训练的 baseline
- [ ] oracle 标签训练的 baseline
- [ ] oracle 标签微调的 LLM

### 输出物

- [ ] 实验 summary
- [ ] 在线指标对比图
- [ ] 标签分布对比图
- [ ] 失败案例分析
