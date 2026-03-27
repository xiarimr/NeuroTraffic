# Experiment Comparison

| Experiment | Model | Selector | Episodes | Total Reward | Avg Wait | Avg Queue | Throughput | Avg Travel | Current Mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pure DQN | AdvancedDQN | off | 160 | -389978.25 | 370.32 | 36.11 | 3315.00 | 588.53 | balanced |
| RuleMode + DQN | AdvancedDQN | on | 160 | -64660.55 | 52.10 | 5.25 | 4130.00 | 297.72 | balanced |

![Total Reward Trend](total_reward_trend.svg)

![Final Average Travel Time](final_average_travel_time.svg)
