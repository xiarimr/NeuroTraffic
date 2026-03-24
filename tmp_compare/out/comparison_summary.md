# Experiment Comparison

| Experiment | Model | Selector | Episodes | Total Reward | Avg Wait | Avg Queue | Throughput | Avg Travel | Current Mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pure PPO | PPOColight | off | 2 | 120.00 | 11.00 | 7.00 | 55.00 | 110.00 | balanced |
| RuleMode + PPO | PPOColight | on | 2 | 145.00 | 9.00 | 5.00 | 63.00 | 95.00 | main_road_priority |

![Total Reward Trend](total_reward_trend.svg)

![Final Average Travel Time](final_average_travel_time.svg)
