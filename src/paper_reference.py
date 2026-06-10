"""
Published reference values from Makarov et al., Fuel 406 (2026) 136847.

Transcribed from the paper's Tables 3-6 so the reproduction can be compared
side-by-side. All error metrics in m3/day.
"""

# Table 3 - training subset
TABLE3_TRAIN = {
    "CNN":      {"ARE": 1.4935, "MAE": 4.0037, "RMSE": 4.5737, "R2": 0.9999, "SD": 4.5105},
    "LSTM":     {"ARE": 0.6841, "MAE": 3.2653, "RMSE": 4.0061, "R2": 0.9991, "SD": 3.9985},
    "CNN-COA":  {"ARE": -0.1372, "MAE": 1.6894, "RMSE": 3.0894, "R2": 0.9992, "SD": 3.0879},
    "LSTM-COA": {"ARE": -0.0004, "MAE": 0.4538, "RMSE": 0.9345, "R2": 0.9999, "SD": 0.9347},
    "CNN-PSO":  {"ARE": 0.7021, "MAE": 1.8570, "RMSE": 3.3204, "R2": 0.9990, "SD": 3.3195},
    "LSTM-PSO": {"ARE": -0.0049, "MAE": 1.1956, "RMSE": 1.9105, "R2": 0.9996, "SD": 1.9109},
}

# Table 4 - blind testing subset (Well 15/9-F-11)
TABLE4_TEST = {
    "CNN":      {"ARE": 0.1445, "MAE": 8.1895, "RMSE": 8.6427, "R2": 0.9954, "SD": 2.7638},
    "LSTM":     {"ARE": 0.1203, "MAE": 6.0998, "RMSE": 7.4345, "R2": 0.9815, "SD": 4.2532},
    "CNN-COA":  {"ARE": 0.0212, "MAE": 2.1948, "RMSE": 2.7329, "R2": 0.9948, "SD": 2.6753},
    "LSTM-COA": {"ARE": 0.0189, "MAE": 1.8860, "RMSE": 2.1534, "R2": 0.9978, "SD": 2.0573},
    "CNN-PSO":  {"ARE": 0.0381, "MAE": 2.4476, "RMSE": 3.1470, "R2": 0.9983, "SD": 2.8411},
    "LSTM-PSO": {"ARE": 0.0249, "MAE": 1.9063, "RMSE": 2.4004, "R2": 0.9995, "SD": 2.2878},
}

# Table 5 - bootstrapped test RMSE
TABLE5_BOOT = {
    "CNN":      {"mean_rmse": 9.1819, "ci_low": 8.2419, "ci_high": 10.1219, "ci_width": 1.8800},
    "LSTM":     {"mean_rmse": 7.6329, "ci_low": 7.1852, "ci_high": 8.0806, "ci_width": 0.8954},
    "CNN-COA":  {"mean_rmse": 3.0821, "ci_low": 2.8957, "ci_high": 3.2685, "ci_width": 0.3728},
    "LSTM-COA": {"mean_rmse": 2.1575, "ci_low": 2.0896, "ci_high": 2.2254, "ci_width": 0.1358},
    "CNN-PSO":  {"mean_rmse": 3.1828, "ci_low": 2.9859, "ci_high": 3.3797, "ci_width": 0.3938},
    "LSTM-PSO": {"mean_rmse": 2.4211, "ci_low": 2.2795, "ci_high": 2.5627, "ci_width": 0.2832},
}

# Table 6 - total combined score (train + test rank scores)
TABLE6_TOTALS = {
    "CNN": 19, "LSTM": 19, "CNN-COA": 38, "LSTM-COA": 57, "CNN-PSO": 29, "LSTM-PSO": 49,
}

# Paper's performance ranking (Fig. 11 / abstract)
RANKING = ["LSTM-COA", "LSTM-PSO", "CNN-COA", "CNN-PSO", "LSTM", "CNN"]
