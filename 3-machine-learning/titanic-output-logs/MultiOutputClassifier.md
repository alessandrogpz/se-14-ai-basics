estimator = LogisticRegression(C=10, max_iter=50)
Prediction accuracy (Training Set): 83.57 %
Prediction accuracy (Test Set): 80.45 %

estimator = RidgeClassifier(alpha=1)
Prediction accuracy (Training Set): 82.72 %
Prediction accuracy (Test Set): 79.89 %

estimator = SVC(C=1, gamma=0.1)
Prediction accuracy (Training Set): 84.41 %
Prediction accuracy (Test Set): 80.45 %

estimator = NuSVC(gamma=0.1)
Prediction accuracy (Training Set): 84.13 %
Prediction accuracy (Test Set): 81.01 %

estimator = LinearSVC(C=1, random_state=1)
Prediction accuracy (Training Set): 83.43 %
Prediction accuracy (Test Set): 79.89 %

estimator = KNeighborsClassifier(n_neighbors=16)
Prediction accuracy (Training Set): 84.97 %
Prediction accuracy (Test Set): 81.01 %

estimator = RadiusNeighborsClassifier(outlier_label='most_frequent', radius=1)
Prediction accuracy (Training Set): 84.69 %
Prediction accuracy (Test Set): 82.12 %

estimator = SGDClassifier(alpha=0.001, loss='log_loss', random_state=1)
Prediction accuracy (Training Set): 83.29 %
Prediction accuracy (Test Set): 79.89 %

estimator = DecisionTreeClassifier(max_depth=3, random_state=1)
Prediction accuracy (Training Set): 83.99 %
Prediction accuracy (Test Set): 80.45 %

estimator = RandomForestClassifier(max_depth=3, n_estimators=50, random_state=1)
Prediction accuracy (Training Set): 84.13 %
Prediction accuracy (Test Set): 81.56 %

Best estimator: RadiusNeighborsClassifier(outlier_label='most_frequent', radius=1)
Prediction accuracy (Training Set): 84.69 %
Prediction accuracy (Test Set): 82.12 %
