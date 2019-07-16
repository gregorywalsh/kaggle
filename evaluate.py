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

# # print_title("EVALUATING TRAINING VARIANCE FOR HYPOTHESES WITH NON-DETERMINISTIC TRAINING")
# # for hypothesis_name, hypothesis in hypotheses:
# #     if hypothesis.is_trained_stochastically():
# #         # TODO implement: evaluate_training_variance(hypothesis)
# #         pass
# #
# # print_title('GETTING BEST STACKED MODELS')
# # if len(hypotheses) == 1:
# #     print('Training {m} machines for 1 hypothesis for {k} folds. Total: {t}.'.format(
# #         m=args.num_hyp_samp_by_hypoth, k=args.num_folds, t=args.num_hyp_samp_by_hypoth * args.num_folds)
# #     )
# # else:
# #     print('Training {m} machines for each of {c} hypotheses for {k} inner and {k} outer folds. Total: {t}.'.format(
# #         m=args.num_hyp_samp_by_hypoth, c=len(hypotheses), k=args.num_folds,
# #         t=args.num_hyp_samp_by_hypoth * len(hypotheses) * args.num_folds ** 2
# #     ))
# # best_machine = None
# # best_rand_search_kwargs = {
# #     'n_iter': args.num_hyp_samp_for_best,
# #     'n_jobs': args.num_jobs,
# #     'verbose': 1
# # }
# # print_title('SAVING KERNEL')
# # filepath = 'challenges/{c}/kernels/{r}.joblib'.format(c=args.challenge, r=run_id)
# # joblib.dump(best_machine, filename=filepath)
# # print('newly trained machine saved to "{}"'.format(filepath))
# #
# # print_title()