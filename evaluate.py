from sklearn.model_selection import KFold, cross_val_score

def cross_validate(nested, features, target, hypothesis, scoring, num_folds, cv_kwargs):
    if nested:
        nested_score = cross_val_score(
            estimator=hypothesis.estimator, X=features, y=target, cv=KFold(num_folds),
            scoring=scoring, error_score='raise'
        )
        score = nested_score.mean()
    else:
        hypothesis.estimator.fit(X=features, y=target)
    return

