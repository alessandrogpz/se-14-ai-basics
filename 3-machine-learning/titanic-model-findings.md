# Titanic Challenge – Model Findings

Decisions are based on **test accuracy**, not training accuracy.


---

## Comparing models

### Summary

todo

### Experiment log

### Linear Models

#### 1. LogisticRegression

```python
best_C = None
best_test_acc = 0
best_train_acc = 0

for C in [0.001, 0.01, 0.1, 1, 10, 100]:

  model = linear_model.LogisticRegression(C=C, max_iter=50)
  model.fit(X_train, y_train)

  train_acc_log = round(model.score(X_train, y_train) * 100, 2)
  test_acc_log = round(model.score(X_test, y_test) * 100, 2)
  print("iteration =", C)
  print("Prediction accuracy (Training Set):", train_acc_log, "%")
  print("Prediction accuracy (Test Set):", test_acc_log, "%\n")

  # keep this one if it beats the best so far
  if test_acc_log > best_test_acc:
    best_C = C
    best_test_acc = test_acc_log
    best_train_acc = train_acc_log

print("Best iteration:", best_C)
print("Prediction accuracy (Training Set):", best_train_acc, "%")
print("Prediction accuracy (Test Set):", best_test_acc, "%")
```

Best Result:
- Best iteration: 10 
	Prediction accuracy (Training Set): 83.57 %
	Prediction accuracy (Test Set): 80.45 %

#### 1.2 RidgeClassifier

```python
best_k = None
best_test_acc = 0
best_train_acc = 0

for k in range(1, 101):

  model = linear_model.RidgeClassifier(alpha=k)
  model.fit(X_train, y_train)

  train_acc_log = round(model.score(X_train, y_train) * 100, 2)
  test_acc_log = round(model.score(X_test, y_test) * 100, 2)
  print("alpha =", k)
  print("Prediction accuracy (Training Set):", train_acc_log, "%")
  print("Prediction accuracy (Test Set):", test_acc_log, "%\n")

  # keep this one if it beats the best so far
  if test_acc_log > best_test_acc:
    best_k = k
    best_test_acc = test_acc_log
    best_train_acc = train_acc_log

print("Best alpha:", best_k)
print("Prediction accuracy (Training Set):", best_train_acc, "%")
print("Prediction accuracy (Test Set):", best_test_acc, "%")
```

Best Result: 
- Best alpha: 1
	Prediction accuracy (Training Set): 82.72 %
	Prediction accuracy (Test Set): 79.89 %

Observation:
- alpha=0 was removed from the range: it turns regularization off, and because Family_Alone + Family_Small + Family_Large always add up to 1, there is no single best solution, so the result changes between machines (82.72 % train in Colab, 82.3 % locally)

### Discriminant Analysis

#### LinearDiscriminantAnalysis

```python
from sklearn import discriminant_analysis

best_solver = None
best_shrinkage = None
best_test_acc = 0
best_train_acc = 0

for solver in ["lsqr", "eigen"]:
  for shrinkage in [None, "auto", 0.1, 0.5, 0.9]:

    model = discriminant_analysis.LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
    model.fit(X_train, y_train)

    train_acc_log = round(model.score(X_train, y_train) * 100, 2)
    test_acc_log = round(model.score(X_test, y_test) * 100, 2)
    print("solver =", solver, "shrinkage =", shrinkage, "| train", train_acc_log, "% | test", test_acc_log, "%")

    # keep this one if it beats the best so far
    if test_acc_log > best_test_acc:
      best_solver = solver
      best_shrinkage = shrinkage
      best_test_acc = test_acc_log
      best_train_acc = train_acc_log

print("Best solver:", best_solver, "Best shrinkage:", best_shrinkage)
print("Prediction accuracy (Training Set):", best_train_acc, "%")
print("Prediction accuracy (Test Set):", best_test_acc, "%")
```

Best Result:
- Best solver: lsqr Best shrinkage: None
	Prediction accuracy (Training Set): 82.44 %
	Prediction accuracy (Test Set): 79.89 %

Observation:
- the default solver "svd" doesn't support shrinkage, it gives the same result as lsqr with no shrinkage (82.44 % / 79.89 %)
- eigen without shrinkage drops to 70.39 % test, probably because Family_Alone + Family_Small + Family_Large always add up to 1; any shrinkage fixes it

#### QuadraticDiscriminantAnalysis

```python
from sklearn import discriminant_analysis

best_reg_param = None
best_test_acc = 0
best_train_acc = 0

for reg_param in [0.001, 0.01, 0.1, 0.5, 1]:

  model = discriminant_analysis.QuadraticDiscriminantAnalysis(reg_param=reg_param)
  model.fit(X_train, y_train)

  train_acc_log = round(model.score(X_train, y_train) * 100, 2)
  test_acc_log = round(model.score(X_test, y_test) * 100, 2)
  print("reg_param =", reg_param)
  print("Prediction accuracy (Training Set):", train_acc_log, "%")
  print("Prediction accuracy (Test Set):", test_acc_log, "%\n")

  # keep this one if it beats the best so far
  if test_acc_log > best_test_acc:
    best_reg_param = reg_param
    best_test_acc = test_acc_log
    best_train_acc = train_acc_log

print("Best reg_param:", best_reg_param)
print("Prediction accuracy (Training Set):", best_train_acc, "%")
print("Prediction accuracy (Test Set):", best_test_acc, "%")
```

Best Result:
- Best reg_param: 0.001
	Prediction accuracy (Training Set): 83.01 %
	Prediction accuracy (Test Set): 78.77 %

Observation:
- reg_param=0 (the default) errors: "LinAlgError: The covariance matrix of class 0 is not full rank. Increase the value of `reg_param` to reduce the collinearity." (Family_Alone + Family_Small + Family_Large always add up to 1), so the range starts at 0.001

### Support Vector Machines (SVM)

#### SVC

```python
best_C = None
best_gamma = None
best_test_acc = 0
best_train_acc = 0

for C in [0.1, 1, 10, 100]:
  for gamma in [0.001, 0.01, 0.1, 1]:

    model = svm.SVC(C=C, gamma=gamma)
    model.fit(X_train, y_train)

    train_acc_log = round(model.score(X_train, y_train) * 100, 2)
    test_acc_log = round(model.score(X_test, y_test) * 100, 2)
    print("C =", C, "gamma =", gamma, "| train", train_acc_log, "% | test", test_acc_log, "%")

    if test_acc_log > best_test_acc:
      best_C = C
      best_gamma = gamma
      best_test_acc = test_acc_log
      best_train_acc = train_acc_log

print("Best C:", best_C, "Best gamma:", best_gamma)
print("Prediction accuracy (Training Set):", best_train_acc, "%")
print("Prediction accuracy (Test Set):", best_test_acc, "%")
```

Best Restul:
- Best C: 1 Best gamma: 0.1
	Prediction accuracy (Training Set): 84.41 %
	Prediction accuracy (Test Set): 80.45 %

#### NuSVC

```python

best_nu = None
best_gamma = None
best_test_acc = 0
best_train_acc = 0

for nu in [0.3, 0.4, 0.5, 0.6, 0.7]:
  for gamma in [0.001, 0.01, 0.1, 1]:

    model = svm.NuSVC(nu=nu, gamma=gamma)
    model.fit(X_train, y_train)

    train_acc_log = round(model.score(X_train, y_train) * 100, 2)
    test_acc_log = round(model.score(X_test, y_test) * 100, 2)
    print("nu =", nu, "| gamma =", gamma)
    print("Prediction accuracy (Training Set):", train_acc_log, "%")
    print("Prediction accuracy (Test Set):", test_acc_log, "%\n")

    # keep this one if it beats the best so far
    if test_acc_log > best_test_acc:
      best_nu = nu
      best_gamma = gamma
      best_test_acc = test_acc_log
      best_train_acc = train_acc_log

print("Best nu:", best_nu)
print("Best gamma:", best_gamma)
print("Prediction accuracy (Training Set):", best_train_acc, "%")
print("Prediction accuracy (Test Set):", best_test_acc, "%")
```

- Best Result
	Best nu: 0.5 Best gamma: 0.1 
	Prediction accuracy (Training Set): 84.13 % 
	Prediction accuracy (Test Set): 81.01 %

#### LinearSVC

```python
best_C = None
best_test_acc = 0
best_train_acc = 0

for C in [0.001, 0.01, 0.1, 1, 10, 100]:

  model = svm.LinearSVC(C=C, random_state=1)
  model.fit(X_train, y_train)

  train_acc_log = round(model.score(X_train, y_train) * 100, 2)
  test_acc_log = round(model.score(X_test, y_test) * 100, 2)
  print("C =", C)
  print("Prediction accuracy (Training Set):", train_acc_log, "%")
  print("Prediction accuracy (Test Set):", test_acc_log, "%\n")

  # keep this one if it beats the best so far
  if test_acc_log > best_test_acc:
    best_C = C
    best_test_acc = test_acc_log
    best_train_acc = train_acc_log

print("Best C:", best_C)
print("Prediction accuracy (Training Set):", best_train_acc, "%")
print("Prediction accuracy (Test Set):", best_test_acc, "%")
```

Best Result:
- Best C: 1
	Prediction accuracy (Training Set): 83.43 %
	Prediction accuracy (Test Set): 79.89 %

### Nearest Neighbors

#### KNeighboursClassifier

``` python

for k in range (1, 101):

	model = neighbors.KNeighborsClassifier(n_neighbors=k)
	
	model.fit(X_train, y_train)
	
	train_acc_log = round(model.score(X_train, y_train) * 100, 2)
	test_acc_log = round(model.score(X_test, y_test) * 100, 2)  
	
	print(f"k={k}: train {train_acc_log}%, test {test_acc_log}%")
```

Best Result:
- k=16: train 84.97%, test 81.01%

#### RadiusNeighborsClassifier

```python

for k in range (2, 101):
	model = neighbors.RadiusNeighborsClassifier(radius=k)
	
	model.fit(X_train, y_train)
	
	train_acc_log = round(model.score(X_train, y_train) * 100, 2)
	test_acc_log = round(model.score(X_test, y_test) * 100, 2)
	
	print(f"k={k}: train {train_acc_log}%, test {test_acc_log}%")
```

Best Result:
- k=2: train 81.04%, test 74.86%

Observation:
- a k=1 errors on "No neighbors found for test samples array 148"

#### RadiusNeighborsClassifier with outlier_label

```python
best_radius = None
best_weights = None
best_test_acc = 0
best_train_acc = 0

for radius in [0.5, 1, 1.5, 2, 3]:
  for weights in ["uniform", "distance"]:

    model = neighbors.RadiusNeighborsClassifier(radius=radius, weights=weights, outlier_label="most_frequent")
    model.fit(X_train, y_train)

    train_acc_log = round(model.score(X_train, y_train) * 100, 2)
    test_acc_log = round(model.score(X_test, y_test) * 100, 2)
    print("radius =", radius, "weights =", weights, "| train", train_acc_log, "% | test", test_acc_log, "%")

    # keep this one if it beats the best so far
    if test_acc_log > best_test_acc:
      best_radius = radius
      best_weights = weights
      best_test_acc = test_acc_log
      best_train_acc = train_acc_log

print("Best radius:", best_radius, "Best weights:", best_weights)
print("Prediction accuracy (Training Set):", best_train_acc, "%")
print("Prediction accuracy (Test Set):", best_test_acc, "%")
```

Best Result:
- Best radius: 1 Best weights: uniform
	Prediction accuracy (Training Set): 84.69 %
	Prediction accuracy (Test Set): 82.12 %

Observation:
- outlier_label="most_frequent" gives passengers with no neighbours inside the radius the most common class (died) instead of raising the error, so radius=1 works now
- weights="distance" always reaches 86.8 % training accuracy but lower test accuracy: a passenger with exactly the same features (distance 0) decides the vote, so it memorises the training set
- 86.8 % is the highest training accuracy any model can get: the 712 training passengers only have 138 different feature combinations, and some passengers with the same features had different outcomes

### Stochastic Gradient Descent

#### SGDClassifier

```python
from sklearn import linear_model

best_loss = None
best_alpha = None
best_test_acc = 0
best_train_acc = 0

for loss in ["hinge", "log_loss", "modified_huber"]:
  for alpha in [0.0001, 0.001, 0.01, 0.1]:

    model = linear_model.SGDClassifier(loss=loss, alpha=alpha, random_state=1)
    model.fit(X_train, y_train)

    train_acc_log = round(model.score(X_train, y_train) * 100, 2)
    test_acc_log = round(model.score(X_test, y_test) * 100, 2)
    print("loss =", loss, "alpha =", alpha, "| train", train_acc_log, "% | test", test_acc_log, "%")

    # keep this one if it beats the best so far
    if test_acc_log > best_test_acc:
      best_loss = loss
      best_alpha = alpha
      best_test_acc = test_acc_log
      best_train_acc = train_acc_log

print("Best loss:", best_loss, "Best alpha:", best_alpha)
print("Prediction accuracy (Training Set):", best_train_acc, "%")
print("Prediction accuracy (Test Set):", best_test_acc, "%")
```

Best Result:
- Best loss: log_loss Best alpha: 0.001
	Prediction accuracy (Training Set): 83.29 %
	Prediction accuracy (Test Set): 79.89 %

Observation:
- random_state=1 so the random shuffling is the same every run and the log can be reproduced

### Gaussian Processes

#### GaussianProcessClassifier

```python
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct

best_kernel = None
best_test_acc = 0
best_train_acc = 0

for kernel in [1.0 * RBF(1.0), 1.0 * Matern(nu=0.5), 1.0 * Matern(nu=1.5), 1.0 * Matern(nu=2.5), 1.0 * DotProduct()]:

  model = gaussian_process.GaussianProcessClassifier(kernel=kernel, random_state=1)
  model.fit(X_train, y_train)

  train_acc_log = round(model.score(X_train, y_train) * 100, 2)
  test_acc_log = round(model.score(X_test, y_test) * 100, 2)
  print("kernel =", kernel)
  print("Prediction accuracy (Training Set):", train_acc_log, "%")
  print("Prediction accuracy (Test Set):", test_acc_log, "%\n")

  # keep this one if it beats the best so far
  if test_acc_log > best_test_acc:
    best_kernel = kernel
    best_test_acc = test_acc_log
    best_train_acc = train_acc_log

print("Best kernel:", best_kernel)
print("Prediction accuracy (Training Set):", best_train_acc, "%")
print("Prediction accuracy (Test Set):", best_test_acc, "%")
```

Best Result:
- Best kernel: 1**2 * DotProduct(sigma_0=1)
	Prediction accuracy (Training Set): 83.85 %
	Prediction accuracy (Test Set): 79.89 %

Observation:
- slowest model so far: about 1 minute for the 5 kernels
- RationalQuadratic was left out: it took ~30 s and gave a ConvergenceWarning

### Naive Bayes

#### GaussianNB

```python
from sklearn import naive_bayes

best_var_smoothing = None
best_test_acc = 0
best_train_acc = 0

for var_smoothing in [1e-9, 1e-6, 1e-3, 0.01, 0.1, 1]:

  model = naive_bayes.GaussianNB(var_smoothing=var_smoothing)
  model.fit(X_train, y_train)

  train_acc_log = round(model.score(X_train, y_train) * 100, 2)
  test_acc_log = round(model.score(X_test, y_test) * 100, 2)
  print("var_smoothing =", var_smoothing)
  print("Prediction accuracy (Training Set):", train_acc_log, "%")
  print("Prediction accuracy (Test Set):", test_acc_log, "%\n")

  # keep this one if it beats the best so far
  if test_acc_log > best_test_acc:
    best_var_smoothing = var_smoothing
    best_test_acc = test_acc_log
    best_train_acc = train_acc_log

print("Best var_smoothing:", best_var_smoothing)
print("Prediction accuracy (Training Set):", best_train_acc, "%")
print("Prediction accuracy (Test Set):", best_test_acc, "%")
```

Best Result:
- Best var_smoothing: 1e-09
	Prediction accuracy (Training Set): 81.6 %
	Prediction accuracy (Test Set): 77.65 %

Observation:
- var_smoothing up to 0.01 changes nothing, bigger values make it worse

#### MultinomialNB

```python
from sklearn import naive_bayes

best_alpha = None
best_test_acc = 0
best_train_acc = 0

for alpha in [0.001, 0.01, 0.1, 1, 10, 100]:

  model = naive_bayes.MultinomialNB(alpha=alpha)
  model.fit(X_train, y_train)

  train_acc_log = round(model.score(X_train, y_train) * 100, 2)
  test_acc_log = round(model.score(X_test, y_test) * 100, 2)
  print("alpha =", alpha)
  print("Prediction accuracy (Training Set):", train_acc_log, "%")
  print("Prediction accuracy (Test Set):", test_acc_log, "%\n")

  # keep this one if it beats the best so far
  if test_acc_log > best_test_acc:
    best_alpha = alpha
    best_test_acc = test_acc_log
    best_train_acc = train_acc_log

print("Best alpha:", best_alpha)
print("Prediction accuracy (Training Set):", best_train_acc, "%")
print("Prediction accuracy (Test Set):", best_test_acc, "%")
```

Best Result:
- Best alpha: 0.001
	Prediction accuracy (Training Set): 79.35 %
	Prediction accuracy (Test Set): 77.09 %

#### ComplementNB

```python
from sklearn import naive_bayes

best_alpha = None
best_test_acc = 0
best_train_acc = 0

for alpha in [0.001, 0.01, 0.1, 1, 10, 100]:

  model = naive_bayes.ComplementNB(alpha=alpha)
  model.fit(X_train, y_train)

  train_acc_log = round(model.score(X_train, y_train) * 100, 2)
  test_acc_log = round(model.score(X_test, y_test) * 100, 2)
  print("alpha =", alpha)
  print("Prediction accuracy (Training Set):", train_acc_log, "%")
  print("Prediction accuracy (Test Set):", test_acc_log, "%\n")

  # keep this one if it beats the best so far
  if test_acc_log > best_test_acc:
    best_alpha = alpha
    best_test_acc = test_acc_log
    best_train_acc = train_acc_log

print("Best alpha:", best_alpha)
print("Prediction accuracy (Training Set):", best_train_acc, "%")
print("Prediction accuracy (Test Set):", best_test_acc, "%")
```

Best Result:
- Best alpha: 0.001
	Prediction accuracy (Training Set): 79.49 %
	Prediction accuracy (Test Set): 74.86 %

#### BernoulliNB

```python
from sklearn import naive_bayes

best_alpha = None
best_binarize = None
best_test_acc = 0
best_train_acc = 0

for alpha in [0.01, 1, 10]:
  for binarize in [0, 0.5, 1, 2]:

    model = naive_bayes.BernoulliNB(alpha=alpha, binarize=binarize)
    model.fit(X_train, y_train)

    train_acc_log = round(model.score(X_train, y_train) * 100, 2)
    test_acc_log = round(model.score(X_test, y_test) * 100, 2)
    print("alpha =", alpha, "binarize =", binarize, "| train", train_acc_log, "% | test", test_acc_log, "%")

    # keep this one if it beats the best so far
    if test_acc_log > best_test_acc:
      best_alpha = alpha
      best_binarize = binarize
      best_test_acc = test_acc_log
      best_train_acc = train_acc_log

print("Best alpha:", best_alpha, "Best binarize:", best_binarize)
print("Prediction accuracy (Training Set):", best_train_acc, "%")
print("Prediction accuracy (Test Set):", best_test_acc, "%")
```

Best Result:
- Best alpha: 0.01 Best binarize: 0
	Prediction accuracy (Training Set): 80.76 %
	Prediction accuracy (Test Set): 77.65 %

Observation:
- binarize turns every feature into 0 or 1 (1 if above the threshold), so Title, AgeBin and Pclass_ordinal lose most of their information; binarize=1 or 2 makes it much worse

#### CategoricalNB

```python
from sklearn import naive_bayes

best_alpha = None
best_test_acc = 0
best_train_acc = 0

for alpha in [0.001, 0.01, 0.1, 1, 10, 100]:

  model = naive_bayes.CategoricalNB(alpha=alpha)
  model.fit(X_train, y_train)

  train_acc_log = round(model.score(X_train, y_train) * 100, 2)
  test_acc_log = round(model.score(X_test, y_test) * 100, 2)
  print("alpha =", alpha)
  print("Prediction accuracy (Training Set):", train_acc_log, "%")
  print("Prediction accuracy (Test Set):", test_acc_log, "%\n")

  # keep this one if it beats the best so far
  if test_acc_log > best_test_acc:
    best_alpha = alpha
    best_test_acc = test_acc_log
    best_train_acc = train_acc_log

print("Best alpha:", best_alpha)
print("Prediction accuracy (Training Set):", best_train_acc, "%")
print("Prediction accuracy (Test Set):", best_test_acc, "%")
```

Best Result:
- Best alpha: 0.001
	Prediction accuracy (Training Set): 81.6 %
	Prediction accuracy (Test Set): 78.77 %

Observation:
- treats every value of a feature as its own category, which fits this data because all the features are small whole numbers; best Naive Bayes model

### Decision Trees

#### DecisionTreeClassifier

```python
best_max_depth = None
best_min_samples_leaf = None
best_test_acc = 0
best_train_acc = 0

for max_depth in [1, 2, 3, 4, 5, 6, 8, 10, None]:
  for min_samples_leaf in [1, 5, 10, 20]:

    model = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=1)
    model.fit(X_train, y_train)

    train_acc_log = round(model.score(X_train, y_train) * 100, 2)
    test_acc_log = round(model.score(X_test, y_test) * 100, 2)
    print("max_depth =", max_depth, "min_samples_leaf =", min_samples_leaf, "| train", train_acc_log, "% | test", test_acc_log, "%")

    # keep this one if it beats the best so far
    if test_acc_log > best_test_acc:
      best_max_depth = max_depth
      best_min_samples_leaf = min_samples_leaf
      best_test_acc = test_acc_log
      best_train_acc = train_acc_log

print("Best max_depth:", best_max_depth, "Best min_samples_leaf:", best_min_samples_leaf)
print("Prediction accuracy (Training Set):", best_train_acc, "%")
print("Prediction accuracy (Test Set):", best_test_acc, "%")
```

Best Result:
- Best max_depth: 3 Best min_samples_leaf: 1
	Prediction accuracy (Training Set): 83.99 %
	Prediction accuracy (Test Set): 80.45 %

Observation:
- deeper trees keep raising training accuracy (86.8 % with no limit) while test accuracy drops: overfitting

### Ensembles

#### RandomForestClassifier

```python
from sklearn import ensemble

best_n_estimators = None
best_max_depth = None
best_test_acc = 0
best_train_acc = 0

for n_estimators in [50, 100, 200]:
  for max_depth in [3, 5, 7, None]:

    model = ensemble.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=1)
    model.fit(X_train, y_train)

    train_acc_log = round(model.score(X_train, y_train) * 100, 2)
    test_acc_log = round(model.score(X_test, y_test) * 100, 2)
    print("n_estimators =", n_estimators, "max_depth =", max_depth, "| train", train_acc_log, "% | test", test_acc_log, "%")

    # keep this one if it beats the best so far
    if test_acc_log > best_test_acc:
      best_n_estimators = n_estimators
      best_max_depth = max_depth
      best_test_acc = test_acc_log
      best_train_acc = train_acc_log

print("Best n_estimators:", best_n_estimators, "Best max_depth:", best_max_depth)
print("Prediction accuracy (Training Set):", best_train_acc, "%")
print("Prediction accuracy (Test Set):", best_test_acc, "%")
```

Best Result:
- Best n_estimators: 50 Best max_depth: 3
	Prediction accuracy (Training Set): 84.13 %
	Prediction accuracy (Test Set): 81.56 %

Observation:
- shallow trees (max_depth=3) do best, deeper trees raise training accuracy up to 86.8 % but lower test accuracy
- the number of trees changes the result noticeably at the same depth (81.56 % with 50 trees, 79.89 % with 100), so small differences between the top models are partly luck

#### ExtraTreesClassifier

```python
from sklearn import ensemble

best_n_estimators = None
best_max_depth = None
best_test_acc = 0
best_train_acc = 0

for n_estimators in [50, 100, 200]:
  for max_depth in [3, 5, 7, None]:

    model = ensemble.ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=1)
    model.fit(X_train, y_train)

    train_acc_log = round(model.score(X_train, y_train) * 100, 2)
    test_acc_log = round(model.score(X_test, y_test) * 100, 2)
    print("n_estimators =", n_estimators, "max_depth =", max_depth, "| train", train_acc_log, "% | test", test_acc_log, "%")

    # keep this one if it beats the best so far
    if test_acc_log > best_test_acc:
      best_n_estimators = n_estimators
      best_max_depth = max_depth
      best_test_acc = test_acc_log
      best_train_acc = train_acc_log

print("Best n_estimators:", best_n_estimators, "Best max_depth:", best_max_depth)
print("Prediction accuracy (Training Set):", best_train_acc, "%")
print("Prediction accuracy (Test Set):", best_test_acc, "%")
```

Best Result:
- Best n_estimators: 50 Best max_depth: 5
	Prediction accuracy (Training Set): 84.27 %
	Prediction accuracy (Test Set): 81.56 %

#### GradientBoostingClassifier

```python
from sklearn import ensemble

best_n_estimators = None
best_learning_rate = None
best_test_acc = 0
best_train_acc = 0

for n_estimators in [50, 100, 200]:
  for learning_rate in [0.01, 0.1, 0.5]:

    model = ensemble.GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=1)
    model.fit(X_train, y_train)

    train_acc_log = round(model.score(X_train, y_train) * 100, 2)
    test_acc_log = round(model.score(X_test, y_test) * 100, 2)
    print("n_estimators =", n_estimators, "learning_rate =", learning_rate, "| train", train_acc_log, "% | test", test_acc_log, "%")

    # keep this one if it beats the best so far
    if test_acc_log > best_test_acc:
      best_n_estimators = n_estimators
      best_learning_rate = learning_rate
      best_test_acc = test_acc_log
      best_train_acc = train_acc_log

print("Best n_estimators:", best_n_estimators, "Best learning_rate:", best_learning_rate)
print("Prediction accuracy (Training Set):", best_train_acc, "%")
print("Prediction accuracy (Test Set):", best_test_acc, "%")
```

Best Result:
- Best n_estimators: 100 Best learning_rate: 0.5
	Prediction accuracy (Training Set): 86.8 %
	Prediction accuracy (Test Set): 80.45 %

#### HistGradientBoostingClassifier

```python
from sklearn import ensemble

best_learning_rate = None
best_max_depth = None
best_test_acc = 0
best_train_acc = 0

for learning_rate in [0.01, 0.1, 0.5]:
  for max_depth in [3, 5, None]:

    model = ensemble.HistGradientBoostingClassifier(learning_rate=learning_rate, max_depth=max_depth, random_state=1)
    model.fit(X_train, y_train)

    train_acc_log = round(model.score(X_train, y_train) * 100, 2)
    test_acc_log = round(model.score(X_test, y_test) * 100, 2)
    print("learning_rate =", learning_rate, "max_depth =", max_depth, "| train", train_acc_log, "% | test", test_acc_log, "%")

    # keep this one if it beats the best so far
    if test_acc_log > best_test_acc:
      best_learning_rate = learning_rate
      best_max_depth = max_depth
      best_test_acc = test_acc_log
      best_train_acc = train_acc_log

print("Best learning_rate:", best_learning_rate, "Best max_depth:", best_max_depth)
print("Prediction accuracy (Training Set):", best_train_acc, "%")
print("Prediction accuracy (Test Set):", best_test_acc, "%")
```

Best Result:
- Best learning_rate: 0.01 Best max_depth: 3
	Prediction accuracy (Training Set): 82.58 %
	Prediction accuracy (Test Set): 80.45 %

#### AdaBoostClassifier

```python
from sklearn import ensemble

best_n_estimators = None
best_learning_rate = None
best_test_acc = 0
best_train_acc = 0

for n_estimators in [50, 100, 200]:
  for learning_rate in [0.01, 0.1, 0.5, 1]:

    model = ensemble.AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=1)
    model.fit(X_train, y_train)

    train_acc_log = round(model.score(X_train, y_train) * 100, 2)
    test_acc_log = round(model.score(X_test, y_test) * 100, 2)
    print("n_estimators =", n_estimators, "learning_rate =", learning_rate, "| train", train_acc_log, "% | test", test_acc_log, "%")

    # keep this one if it beats the best so far
    if test_acc_log > best_test_acc:
      best_n_estimators = n_estimators
      best_learning_rate = learning_rate
      best_test_acc = test_acc_log
      best_train_acc = train_acc_log

print("Best n_estimators:", best_n_estimators, "Best learning_rate:", best_learning_rate)
print("Prediction accuracy (Training Set):", best_train_acc, "%")
print("Prediction accuracy (Test Set):", best_test_acc, "%")
```

Best Result:
- Best n_estimators: 100 Best learning_rate: 1
	Prediction accuracy (Training Set): 84.27 %
	Prediction accuracy (Test Set): 80.45 %

#### BaggingClassifier

```python
from sklearn import ensemble

best_n_estimators = None
best_max_samples = None
best_test_acc = 0
best_train_acc = 0

for n_estimators in [10, 50, 100]:
  for max_samples in [0.5, 0.8, 1.0]:

    model = ensemble.BaggingClassifier(n_estimators=n_estimators, max_samples=max_samples, random_state=1)
    model.fit(X_train, y_train)

    train_acc_log = round(model.score(X_train, y_train) * 100, 2)
    test_acc_log = round(model.score(X_test, y_test) * 100, 2)
    print("n_estimators =", n_estimators, "max_samples =", max_samples, "| train", train_acc_log, "% | test", test_acc_log, "%")

    # keep this one if it beats the best so far
    if test_acc_log > best_test_acc:
      best_n_estimators = n_estimators
      best_max_samples = max_samples
      best_test_acc = test_acc_log
      best_train_acc = train_acc_log

print("Best n_estimators:", best_n_estimators, "Best max_samples:", best_max_samples)
print("Prediction accuracy (Training Set):", best_train_acc, "%")
print("Prediction accuracy (Test Set):", best_test_acc, "%")
```

Best Result:
- Best n_estimators: 50 Best max_samples: 0.5
	Prediction accuracy (Training Set): 86.66 %
	Prediction accuracy (Test Set): 79.89 %

Observation:
- uses full-depth decision trees by default, so training accuracy is always 86.1 % or more while test accuracy stays below 80 %

#### VotingClassifier

```python
from sklearn import ensemble, linear_model, naive_bayes, neural_network

# one model from different families, with the best settings found for each
# (SVC and NuSVC are left out: they can't give probabilities for soft voting by default)
estimators = [
  ("logistic_regression", linear_model.LogisticRegression(C=10, max_iter=50)),
  ("knn", neighbors.KNeighborsClassifier(n_neighbors=16)),
  ("random_forest", ensemble.RandomForestClassifier(n_estimators=50, max_depth=3, random_state=1)),
  ("categorical_nb", naive_bayes.CategoricalNB(alpha=0.001)),
  ("mlp", neural_network.MLPClassifier(hidden_layer_sizes=(50,), alpha=0.1, max_iter=2000, random_state=1)),
]

best_voting = None
best_test_acc = 0
best_train_acc = 0

for voting in ["hard", "soft"]:

  model = ensemble.VotingClassifier(estimators=estimators, voting=voting)
  model.fit(X_train, y_train)

  train_acc_log = round(model.score(X_train, y_train) * 100, 2)
  test_acc_log = round(model.score(X_test, y_test) * 100, 2)
  print("voting =", voting)
  print("Prediction accuracy (Training Set):", train_acc_log, "%")
  print("Prediction accuracy (Test Set):", test_acc_log, "%\n")

  # keep this one if it beats the best so far
  if test_acc_log > best_test_acc:
    best_voting = voting
    best_test_acc = test_acc_log
    best_train_acc = train_acc_log

print("Best voting:", best_voting)
print("Prediction accuracy (Training Set):", best_train_acc, "%")
print("Prediction accuracy (Test Set):", best_test_acc, "%")
```

Best Result:
- Best voting: hard
	Prediction accuracy (Training Set): 84.27 %
	Prediction accuracy (Test Set): 81.56 %

Observation:
- hard voting (majority vote) beats soft voting (average of the probabilities): 81.56 % vs 79.33 %
- ties RandomForestClassifier, the best model inside it

#### StackingClassifier

```python
from sklearn import ensemble, linear_model, naive_bayes, neural_network

# the same models used in VotingClassifier
estimators = [
  ("logistic_regression", linear_model.LogisticRegression(C=10, max_iter=50)),
  ("knn", neighbors.KNeighborsClassifier(n_neighbors=16)),
  ("random_forest", ensemble.RandomForestClassifier(n_estimators=50, max_depth=3, random_state=1)),
  ("categorical_nb", naive_bayes.CategoricalNB(alpha=0.001)),
  ("mlp", neural_network.MLPClassifier(hidden_layer_sizes=(50,), alpha=0.1, max_iter=2000, random_state=1)),
]

best_final_estimator = None
best_test_acc = 0
best_train_acc = 0

# the model that learns how to combine the predictions of the models above
for final_estimator in [
  linear_model.LogisticRegression(),
  svm.SVC(),
  neighbors.KNeighborsClassifier(n_neighbors=16),
  tree.DecisionTreeClassifier(max_depth=3, random_state=1),
]:

  model = ensemble.StackingClassifier(estimators=estimators, final_estimator=final_estimator)
  model.fit(X_train, y_train)

  train_acc_log = round(model.score(X_train, y_train) * 100, 2)
  test_acc_log = round(model.score(X_test, y_test) * 100, 2)
  print("final_estimator =", final_estimator)
  print("Prediction accuracy (Training Set):", train_acc_log, "%")
  print("Prediction accuracy (Test Set):", test_acc_log, "%\n")

  # keep this one if it beats the best so far
  if test_acc_log > best_test_acc:
    best_final_estimator = final_estimator
    best_test_acc = test_acc_log
    best_train_acc = train_acc_log

print("Best final_estimator:", best_final_estimator)
print("Prediction accuracy (Training Set):", best_train_acc, "%")
print("Prediction accuracy (Test Set):", best_test_acc, "%")
```

Best Result:
- Best final_estimator: LogisticRegression()
	Prediction accuracy (Training Set): 84.69 %
	Prediction accuracy (Test Set): 79.33 %

Observation:
- slowest ensemble (about 17 s): each of the 5 models is trained 6 times (5 cross-validation folds + once on all the training data)
- worse than 4 of the 5 models it combines on their own

### Multiclass and Multioutput

#### MultiOutputClassifier

```python
from sklearn import ensemble, linear_model
from sklearn.multioutput import MultiOutputClassifier

# MultiOutputClassifier needs y as a table (2D), not a single column
y_train_2d = y_train.to_frame()
y_test_2d = y_test.to_frame()

best_estimator = None
best_test_acc = 0
best_train_acc = 0

# some of the best models found above
for estimator in [
  linear_model.LogisticRegression(C=10, max_iter=50),
  linear_model.RidgeClassifier(alpha=1),
  svm.SVC(C=1, gamma=0.1),
  svm.NuSVC(nu=0.5, gamma=0.1),
  svm.LinearSVC(C=1, random_state=1),
  neighbors.KNeighborsClassifier(n_neighbors=16),
  neighbors.RadiusNeighborsClassifier(radius=1, outlier_label="most_frequent"),
  linear_model.SGDClassifier(loss="log_loss", alpha=0.001, random_state=1),
  tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf=1, random_state=1),
  ensemble.RandomForestClassifier(n_estimators=50, max_depth=3, random_state=1),
]:

  model = MultiOutputClassifier(estimator)
  model.fit(X_train, y_train_2d)

  train_acc_log = round(model.score(X_train, y_train_2d) * 100, 2)
  test_acc_log = round(model.score(X_test, y_test_2d) * 100, 2)
  print("estimator =", estimator)
  print("Prediction accuracy (Training Set):", train_acc_log, "%")
  print("Prediction accuracy (Test Set):", test_acc_log, "%\n")

  # keep this one if it beats the best so far
  if test_acc_log > best_test_acc:
    best_estimator = estimator
    best_test_acc = test_acc_log
    best_train_acc = train_acc_log

print("Best estimator:", best_estimator)
print("Prediction accuracy (Training Set):", best_train_acc, "%")
print("Prediction accuracy (Test Set):", best_test_acc, "%")
```

Best Result:
- Best estimator: RadiusNeighborsClassifier(outlier_label='most_frequent', radius=1)
	Prediction accuracy (Training Set): 84.69 %
	Prediction accuracy (Test Set): 82.12 %

Observation:
- Titanic has only one output column (Survived), so MultiOutputClassifier trains a single copy of the model: every result is identical to running that model on its own
- passing y_train directly errors: "ValueError: y must have at least two dimensions for multi-output regression but has only one."

### Neural Networks (supervised)

#### MLPClassifier

```python
from sklearn import neural_network

best_hidden_layer_sizes = None
best_alpha = None
best_test_acc = 0
best_train_acc = 0

for hidden_layer_sizes in [(10,), (50,), (100,), (50, 50)]:
  for alpha in [0.0001, 0.001, 0.01, 0.1, 1]:

    model = neural_network.MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, max_iter=2000, random_state=1)
    model.fit(X_train, y_train)

    train_acc_log = round(model.score(X_train, y_train) * 100, 2)
    test_acc_log = round(model.score(X_test, y_test) * 100, 2)
    print("hidden_layer_sizes =", hidden_layer_sizes, "alpha =", alpha, "| train", train_acc_log, "% | test", test_acc_log, "%")

    # keep this one if it beats the best so far
    if test_acc_log > best_test_acc:
      best_hidden_layer_sizes = hidden_layer_sizes
      best_alpha = alpha
      best_test_acc = test_acc_log
      best_train_acc = train_acc_log

print("Best hidden_layer_sizes:", best_hidden_layer_sizes, "Best alpha:", best_alpha)
print("Prediction accuracy (Training Set):", best_train_acc, "%")
print("Prediction accuracy (Test Set):", best_test_acc, "%")
```

Best Result:
- Best hidden_layer_sizes: (50,) Best alpha: 0.1
	Prediction accuracy (Training Set): 85.25 %
	Prediction accuracy (Test Set): 79.89 %

Observation:
- max_iter=2000 because the default (200) gave a ConvergenceWarning in 16 of the 20 runs
- random_state=1 so the starting weights are the same every run and the log can be reproduced

---

## Part 2 – Tuning the best model

**Model:**

**How it works:**

### Hyperparameters

| Hyperparameter | What it does | Default | Values tried |
|----------------|--------------|---------|--------------|
| | | | |

### Code

```python
for value in [...]:
    model = ...
    model.fit(X_train, y_train)
    train_acc = round(model.score(X_train, y_train) * 100, 2)
    test_acc = round(model.score(X_test, y_test) * 100, 2)
    print(f"{value}: train {train_acc}%, test {test_acc}%")
```

### Results

| Hyperparameters | Train acc | Test acc | Notes |
|-----------------|-----------|----------|-------|
| | | | |

### Conclusion

**Best hyperparameters:**

**Test accuracy:**

**Why these work best:**

---

## Template (copy for a new experiment)

````markdown
#### N. Model name

```python
model = ...
```

- Train: / Test:
- Observations:
````
