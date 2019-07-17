import yaml
import pandas as pd


def print_title(title):
    many_stars = '*' * 100
    print(many_stars + '\n' + title + '\n' + many_stars)


def load_yaml_config(fp):
    with open(fp, 'r') as f:
        try:
            raw_config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    return raw_config


def process_cv_results(run_id, hypothesis_name, cv_results, cv_folds, cv_repeats, num_hyp_samples):
    df = pd.DataFrame.from_dict(data=cv_results)
    df['run_id'] = run_id
    df['hypothesis_name'] = hypothesis_name
    df['cv_folds'] = cv_folds
    df['cv_repeats'] = cv_repeats
    df['num_hyp_samples'] = num_hyp_samples
    df['min_score'] = df.filter(regex=r'^split\d+_test_score').min(axis=1)
    df['max_score'] = df.filter(regex=r'^split\d+_test_score').max(axis=1)
    df.sort_values(by='rank_test_score', ascending=True, inplace=True)
    return df


def save_cv_results(path, processed_cv_results, reporting_keys, top_n=None):
    with open(file=path, mode='a') as f:
        processed_cv_results[reporting_keys].iloc[0:top_n].to_csv(path_or_buf=f, index=False, header=False)
    return


def save_model_repr(path, model):
    with open(file=path, mode='a+') as f:
        f.write(model.__repr__(float('inf')))  # Get all chars with float('inf')
