# Experiment Comparison

| Experiment | Model | Selector | Episodes | Total Reward | Avg Wait | Avg Queue | Throughput | Avg Travel | Current Mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pure DQN | AdvancedDQN | off | 160 | -680404.00 | 557.71 | 63.00 | 2903.00 | 725.86 | balanced |
| RuleMode + DQN | AdvancedDQN | on | 160 | -80326.50 | 58.58 | 7.44 | 5015.00 | 295.94 | balanced |

![Total Reward Trend](total_reward_trend.svg)

![Final Average Travel Time](final_average_travel_time.svg)
