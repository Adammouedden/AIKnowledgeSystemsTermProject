| checkpoint | groups | scheme | coherence (mean pairwise cos) | 4-group batch cosine | noise scale B (groups) |
|---|---|---|---|---|---|
| base model | 16 | uniform | 0.007 [-0.013, 0.027] | 0.026 [-0.048, 0.100] | 146.148 [-409.640, 701.936] |
| base model | 16 | action | 0.011 [-0.009, 0.031] | 0.038 [-0.034, 0.111] | 97.318 [-100.386, 295.021] |
| base model | 16 | probe | 0.021 [-0.000, 0.043] | 0.077 [0.002, 0.153] | 45.889 [-2.938, 94.717] |
| trained COMMIT | 16 | uniform | -0.001 [-0.009, 0.007] | -0.004 [-0.038, 0.030] | inf [nan, nan] |
| trained COMMIT | 16 | action | 0.004 [-0.007, 0.015] | 0.014 [-0.028, 0.056] | 314.394 [nan, nan] |
| trained COMMIT | 16 | probe | 0.013 [-0.006, 0.032] | 0.049 [-0.021, 0.119] | 84.977 [-61.304, 231.258] |

| checkpoint | comparison | coherence diff [95% CI] | batch cosine diff [95% CI] |
|---|---|---|---|
| base model | action - uniform | +0.003 [-0.002, +0.009] | +0.013 [-0.006, +0.031] |
| base model | probe - uniform | +0.014 [-0.004, +0.031] | +0.052 [-0.010, +0.113] |
| trained COMMIT | action - uniform | +0.005 [-0.002, +0.012] | +0.018 [-0.008, +0.044] |
| trained COMMIT | probe - uniform | +0.014 [-0.005, +0.033] | +0.053 [-0.020, +0.125] |
