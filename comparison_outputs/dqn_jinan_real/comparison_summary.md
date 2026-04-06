# Experiment Comparison

| Experiment | Model | Selector | Episodes | Total Reward | Avg Wait | Avg Queue | Throughput | Avg Travel | Current Mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pure DQN | AdvancedDQN | off | 160 | -486597.50 | 334.75 | 45.06 | 4345.00 | 546.34 | balanced |
| RuleMode + DQN | AdvancedDQN | on | 160 | -648832.90 | 146.45 | 21.32 | 5392.00 | 380.57 | congestion_resistance |

![Total Reward Trend](total_reward_trend.svg)

![Final Average Travel Time](final_average_travel_time.svg)
