from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.base import clone
import numpy as np
import pandas as pd

def _get_best_conduits_by_scoring_method(
        X, y, conduits, rand_search_kwargs, scoring, inner_splits, outer_splits
):
    # LOOP OVER THE CONDUITS (MODELS) TO FIND THE ONE WITH THE BEST ESTIMATED GENERALISATION ERROR (SCORE)
    agg_results = {}
    for conduit_name, conduit in conduits.items():
        conduit_results = {}
        agg_results[conduit_name] = conduit_results

        hyperparam_optimizer = RandomizedSearchCV(
            estimator=conduit.pipeline,
            param_distributions=conduit.hyperparam_dist,
            iid=False,
            cv=inner_splits,
            return_train_score=False,
            refit=False,
            scoring=scoring,
            **rand_search_kwargs
        )

        # CREATE AN OUTER CROSS VAL LOOP TO MORE ACCURATELY ESTIMATE THE SCORES ON UNSEEN DATA
        outer_cv_results = {key: [] for key in hyperparam_optimizer.scoring.keys()}
        for train_index, test_index in outer_splits.split(X):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y[train_index]
            y_test = y[test_index]
            # SAMPLE THE HYPERPARAMETER SPACE FOR THIS TRAINING SPLIT/FOLD USING RANDOM SEARCH
            # AND RECORD THE PERFORMANCE OF TRAINED MACHINES ACROSS MULTIPLE SCORING METRICS
            hyperparam_optimizer.fit(X=X_train, y=y_train)
            # FOR EACH SCORING METRIC GET THE HYPERPARAMETERS OF THE BEST PERFORMING MACHINE AND EVALUATE ON OUTER SPLIT
            # TO GET UNBIASED ESTIMATE OF REAL WORLD GENERALISATION PERFORMANCE
            for scoring_name, scorer in hyperparam_optimizer.scoring.items():
                best_param_index = np.argmax(hyperparam_optimizer.cv_results_['mean_test_{}'.format(scoring_name)])
                best_hyperparams = hyperparam_optimizer.cv_results_['params'][best_param_index]
                estimator_clone = clone(estimator=conduit.pipeline, safe=True)
                estimator_clone.set_params(**best_hyperparams)
                estimator_clone.fit(X=X_train, y=y_train)
                test_score = scorer(estimator=estimator_clone, X=X_test, y_true=y_test)
                outer_cv_results[scoring_name].append(test_score)

        # AGGREGATE THE SCORES ACROSS ALL THE OUTER SPLITS
        for scoring_name in hyperparam_optimizer.scoring.keys():
            agg_results[conduit_name]['est_gen_{}'.format(scoring_name)] = np.mean(outer_cv_results[scoring_name])
            agg_results[conduit_name]['stddev_{}'.format(scoring_name)] = np.std(outer_cv_results[scoring_name], ddof=1)

    df_results = pd.DataFrame.from_dict(agg_results, orient='index')
    print('PEAK PERFORMANCE BY CONDUIT BY SCORING METHOD')
    print(df_results.to_string())

    # RETURN THE CONDUITS (NOT MACHINES) WITH THE HIGHEST UNBIASED ESTIMATES FOR EACH SCORING METHOD
    best_conduits_by_scoring_method = {}
    for scoring_name in hyperparam_optimizer.scoring.keys():
        best_conduits_by_scoring_method[scoring_name] = df_results['est_gen_{}'.format(scoring_name)].idxmax()
    return best_conduits_by_scoring_method


def _train_machine(X, y, conduit, rand_search_kwargs, splits):
    final_machine = RandomizedSearchCV(
            estimator=conduit.pipeline,
            param_distributions=conduit.hyperparam_dist,
            iid=False,
            cv=splits,
            return_train_score=False,
            **rand_search_kwargs
        )
    final_machine.fit(X, y)

    return final_machine.best_estimator_


def _report_on_machine(conduit, machine, run_id):
    folder = './logs/hyperparameter_search/'.format(run_id=run_id)
    conduit.write_training_report(folder=folder, run_id=run_id, pipeline=machine)


def get_best_machines_by_scoring_method(
        X, y, hypothesis, num_folds, inner_rand_search_kwargs, best_rand_search_kwargs, scoring, run_id
):
    best_machines_by_scoring_method = {}
    if len(hypothesis) > 1:
        best_conduits_by_scoring_method = _get_best_conduits_by_scoring_method(
            X=X,
            y=y,
            conduits=hypothesis,
            rand_search_kwargs=inner_rand_search_kwargs,
            scoring=scoring,
            inner_splits=KFold(num_folds),
            outer_splits=KFold(num_folds),
        )

        for scoring_method_name, best_conduit_name in best_conduits_by_scoring_method.items():
            best_machine = _train_machine(
                X=X,
                y=y,
                conduit=hypothesis[best_conduit_name],
                rand_search_kwargs={**best_rand_search_kwargs, 'scoring': scoring[scoring_method_name]},
                splits=KFold(num_folds)
            )
            # _report_on_machine(conduit=conduits[best_conduit_name], machine=best_machine, run_id=run_id)
            best_machines_by_scoring_method[scoring_method_name + '_' + best_conduit_name] = best_machine

    else:
        conduit_name, conduit = next(iter(hypothesis.items()))
        for scoring_method_name, scoring_callable in scoring.items():
            best_machine = _train_machine(
                X=X,
                y=y,
                conduit=conduit,
                rand_search_kwargs={**best_rand_search_kwargs, 'scoring': scoring_callable},
                splits=KFold(num_folds)
            )
            # _report_on_machine(conduit=conduits[best_conduit_name], machine=best_machine, run_id=run_id)
            best_machines_by_scoring_method[scoring_method_name + '_' + conduit_name] = best_machine

    return best_machines_by_scoring_method
