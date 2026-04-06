# Experiment Comparison

| Experiment | Model | Selector | Episodes | Total Reward | Avg Wait | Avg Queue | Throughput | Avg Travel | Current Mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pure DQN | AdvancedDQN | off | 160 | -102069.00 | 65.00 | 9.45 | 5748.00 | 304.37 | balanced |
| LLMMode + DQN | AdvancedDQN | llm:api | 160 | -408791.65 | 86.75 | 12.62 | 5645.00 | 326.78 | queue_clearance |
| RuleMode + DQN | AdvancedDQN | rule | 160 | -423308.53 | 111.86 | 16.28 | 5623.00 | 354.99 | queue_clearance |

![Total Reward Trend](total_reward_trend.svg)

![Final Average Travel Time](final_average_travel_time.svg)
