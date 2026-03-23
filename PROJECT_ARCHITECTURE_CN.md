# 项目架构说明

## 1. 项目定位

这个项目用于做基于强化学习的交通信号灯控制实验，核心目标是：

- 读取 CityFlow 交通仿真环境
- 让不同的信号灯控制算法与环境交互
- 采样训练数据
- 更新模型参数
- 按轮次保存模型并评估效果

目前主学习型模型已经迁移到原生 PyTorch，主要覆盖：

- `EfficientMPLight`
- `AdvancedMPLight`
- `EfficientPressLight`
- `AdvancedDQN`
- `Attend`
- `EfficientColight`
- `AdvancedColight`
- `PPOColight`

这些模型现在统一保存为 `.pt` checkpoint。

## 2. 顶层目录职责

### `models/`

存放所有 agent 和神经网络定义，是项目的“算法层”。

主要文件：

- `agent.py`
  所有 agent 的基础接口定义。
- `network_agent.py`
  DQN 类模型的公共基类，负责：
  - 模型初始化
  - 经验样本转训练数据
  - epsilon-greedy 选动作
  - 训练循环
  - `.pt` 模型保存与加载
- `mplight_agent.py`
  `EfficientMPLight` 的 PyTorch 实现。
- `advanced_mplight_agent.py`
  `AdvancedMPLight` 的 PyTorch 实现。
- `presslight_one.py`
  `EfficientPressLight` 的 PyTorch 实现。
- `simple_dqn_one.py`
  `AdvancedDQN` 的 PyTorch 实现。
- `attendlight_agent.py`
  `AttendLight` 的 PyTorch 实现。
- `colight_net.py`
  CoLight / PPOCoLight 共用的图注意力编码器。
- `colight_agent.py`
  `EfficientColight` / `AdvancedColight` 的 PyTorch 实现，属于图注意力 + DQN 路线。
- `ppo_colight_agent.py`
  `PPOColight` 的 PyTorch 实现，属于图注意力 + Actor-Critic + PPO 路线。
- `fixedtime_agent.py`
  固定时长控制，不依赖学习。
- `maxpressure_agent.py`
  MaxPressure 基线方法。
- `efficient_maxpressure_agent.py`
  Efficient MaxPressure 版本。
- `advanced_maxpressure_agent.py`
  Advanced MaxPressure 版本。

### `utils/`

存放训练调度、环境交互、采样和测试逻辑，是项目的“工程层”。

主要文件：

- `config.py`
  注册模型名和类路径，保存默认配置。
  现在使用延迟导入，避免没用到的模型提前导入。
- `cityflow_env.py`
  对 CityFlow 环境的封装。
- `generator.py`
  负责运行环境、调用 agent 采样数据。
- `construct_sample.py`
  对采样结果做整理，形成训练样本。
- `updater.py`
  从样本中构造训练数据并更新模型。
- `pipeline.py`
  串起完整训练流程，是多轮训练的主控制器。
- `model_test.py`
  加载训练好的模型做测试。
- `oneline.py`
  简化版训练入口。
- `utils.py`
  一些通用工具函数。

### `run_*.py`

这些是实验入口脚本，用于选择模型、数据集和超参数。

例如：

- `run_mplight.py`
- `run_advanced_mplight.py`
- `run_efficient_colight.py`
- `run_ppo_colight.py`

它们的职责是：

- 组合配置
- 指定数据集
- 指定模型名
- 调用 `pipeline` 启动训练

### `data/`

存放 CityFlow 所需的数据文件：

- 路网文件
- 交通流文件

### `records/`

存放实验运行记录，例如：

- 配置快照
- 日志
- 采样结果
- 测试结果

### `readme.md`

项目说明和使用方式。

## 3. 训练主流程

整体训练链路可以理解成下面这条线：

1. `run_xxx.py` 组织配置并启动训练
2. `utils/pipeline.py` 创建目录、复制配置、控制轮次
3. `utils/generator.py` 调用 agent 与 `CityFlowEnv` 交互，生成样本
4. `utils/construct_sample.py` 整理采样结果
5. `utils/updater.py` 加载样本并调用 agent 训练
6. agent 在 `models/` 中完成前向、loss 计算、参数更新
7. 模型保存到 `model/.../*.pt`
8. `utils/model_test.py` 在测试环境上评估模型

## 4. 模型分层理解

当前项目里的方法大致分成三类：

### 规则类方法

不训练参数，直接根据规则控制信号灯。

- `Fixedtime`
- `MaxPressure`
- `EfficientMaxPressure`
- `AdvancedMaxPressure`

### DQN 类方法

这类模型输出每个动作的 Q 值，再用 epsilon-greedy 选动作。

- `EfficientMPLight`
- `AdvancedMPLight`
- `EfficientPressLight`
- `AdvancedDQN`
- `Attend`
- `EfficientColight`
- `AdvancedColight`

其中：

- `network_agent.py` 主要服务单路口或共享参数的 DQN 模型
- `colight_agent.py` 主要服务多路口图注意力 DQN 模型

### PPO 类方法

这类模型同时输出：

- 策略概率 `actor`
- 状态价值 `critic`

当前对应：

- `PPOColight`

## 5. 现在环境主要需要的库

按当前代码状态，最主要的运行依赖是：

- `torch`
  现在主学习模型已经迁到原生 PyTorch，这是最核心依赖。
- `numpy`
  状态、经验样本、张量前的数据处理都依赖它。
- `pandas`
  主要用于日志、结果统计和分析。
- `cityflow`
  交通仿真环境本体，没有它就无法训练和测试。

推荐最小安装命令：

```bash
pip install torch numpy pandas cityflow
```

## 6. 当前这台环境的依赖状态

我刚检查了当前环境，下面这些关键库目前都是缺失状态：

- `torch: MISSING`
- `numpy: MISSING`
- `pandas: MISSING`
- `cityflow: MISSING`

也就是说，现在这台环境还不能直接跑训练或测试，需要先安装依赖。

## 7. 你现在最该先做什么

建议顺序：

1. 先装 `torch numpy pandas cityflow`
2. 再优先跑 `run_mplight.py` 或 `run_efficient_colight.py`
3. 确认 `.pt` 模型能正常生成
4. 最后再跑 `run_ppo_colight.py`

如果后面你要继续整理项目，我建议下一步再补两份文档：

- “训练流程时序图”
- “各个 agent 的输入输出说明”
