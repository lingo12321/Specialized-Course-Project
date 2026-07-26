# Agent Behavior Comparison

This report describes observed associations in the evaluation data. It does not establish that the penalty caused any difference.

Reference model: **Baseline**  
Second model: **Penalty 0.1**

## Observed differences

- Episode reward: the second model was lower (-187.111889 vs. -187.085467).
- Average coverage: the second model was higher (0.052267 vs. 0.041067).
- Final coverage: the second model was higher (0.043333 vs. 0.040000).
- Success rate: the second model was unchanged (0.000000 vs. 0.000000).
- Duplicate step rate: the second model was unchanged (0.000000 vs. 0.000000).
- Collision count: the second model was lower (3.840000 vs. 4.060000).

The plots use mean ± sample standard deviation. Results should be interpreted with the episode count, seed coverage, and occupancy definition used by the evaluator in mind.

## Suggested next step

Duplicate occupancy did not show a clear numerical reduction. Before tuning lambda, inspect the penalty trigger, reward integration, penalized entities, and distance threshold.
