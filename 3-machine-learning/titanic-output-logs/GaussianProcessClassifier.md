kernel = 1**2 * RBF(length_scale=1)
Prediction accuracy (Training Set): 85.53 %
Prediction accuracy (Test Set): 77.65 %

kernel = 1**2 * Matern(length_scale=1, nu=0.5)
Prediction accuracy (Training Set): 85.81 %
Prediction accuracy (Test Set): 78.77 %

kernel = 1**2 * Matern(length_scale=1, nu=1.5)
Prediction accuracy (Training Set): 85.53 %
Prediction accuracy (Test Set): 78.21 %

kernel = 1**2 * Matern(length_scale=1, nu=2.5)
Prediction accuracy (Training Set): 85.53 %
Prediction accuracy (Test Set): 78.21 %

kernel = 1**2 * DotProduct(sigma_0=1)
Prediction accuracy (Training Set): 83.85 %
Prediction accuracy (Test Set): 79.89 %

Best kernel: 1**2 * DotProduct(sigma_0=1)
Prediction accuracy (Training Set): 83.85 %
Prediction accuracy (Test Set): 79.89 %
