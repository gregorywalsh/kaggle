from time import time
from util import process_cv_results, save_cv_results

REPORTING_KEYS = ('run_id', 'hypothesis_name', 'cv_folds', 'cv_repeats', 'num_hyp_samples',
                  'rank_test_score', 'mean_test_score', 'std_test_score', 'min_score',
                  'max_score', 'mean_fit_time', 'params')


class Experiment:
    """
    Define an experiment and run it to evaluate one or more
    hypotheses, record performance and return the
    best performing machine from each hypothesis
    """

    def __init__(self, datareader, x_train, x_test, y_train, hypotheses):
        self.datareader = datareader
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.hypotheses = hypotheses

    def run(self, num_hyper_samples, cv_folds, cv_repeats, reporting_dir, best_machine_dir,
            cv_reporting_keys=REPORTING_KEYS, report_limit=5):
        run_time = int(time())
        hyper_searcher_kwargs = {
            'n_iter': num_hyper_samples,
            'iid': True,
            'return_train_score': False,
            'refit': True,
            'error_score': 'raise'
        }
        for hypothesis_name, hypothesis in self.hypotheses.items():
            x_train, x_test = self.hypotheses.preprocess_x(x_train=self.x_train, x_test=self.x_test)
            y_train = self.hypotheses.preprocess_y(y_train=self.y_train)
            hyper_searcher = hypothesis.hyper_searcher
            if num_hyper_samples == 0:
                print('Fitting base model without sampling hyperparameters')
                hyper_searcher.set_params(**{'param_distributions': {}, 'n_iter': 1})
            hypothesis.hyper_searcher.fit(X=x_train, y=y_train)
            hypothesis.best_model = hyper_searcher.best_estimator_
            hypothesis.best_hyperparams = hyper_searcher.best_params_
            hypothesis.cv_results = process_cv_results(
                run_id=run_time,
                hypothesis_name=hypothesis_name,
                cv_results=hyper_searcher.cv_results_,
                cv_folds=cv_folds,
                cv_repeats=cv_repeats,
                num_hyp_samples=num_hyper_samples
            )
            save_cv_results(
                path='{d}/results.csv'.format(d=reporting_dir),
                processed_cv_results=hypothesis.cv_results,
                top_n=report_limit,
                reporting_keys=cv_reporting_keys
            )
            hypothesis.save(
                path='{d}/hypotheses/{h}_{r}.dump'.format(d=best_machine_dir, r=run_time, h=hypothesis_name)
            )
            # TODO: PRINT RESULTS FOR TOP ARGS.REPORT_N MODELS
