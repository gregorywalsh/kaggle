from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from hypothesis import RandomForrestClassifierHypothesis


def initial_data_munge():
    pass


def get_hypotheses(train_features, test_features):

    # SKLEARN RANDOM FOREST CLASSIFIER

    def cv_transformer():
        get_embedding = RandomForrestClassifierHypothesis.get_categorical_embedding_pipeline
        categorical_embedding_transforms = {
            col: get_embedding(test_feature=train_features[col], train_feature=test_features[col])
            for col in train_features.columns if col.split('|')[1] == 'categorical'
        }
        transformer = Pipeline(
            steps=[
                ('column_selector', FunctionTransformer(func=lambda x: x['sex|categorical'].cat.codes.to_numpy(dtype='f4').reshape(-1,1), validate=False)),
                ('embedding', categorical_embedding_transforms['sex|categorical'])
            ]
        )
        return transformer

    rf = RandomForrestClassifierHypothesis(
        features=train_features,
        cv_transformer=cv_transformer(),
        n_trees=200
    )

    hypotheses = {
        'rf': rf
    }
    return hypotheses


def get_scoring():
    return 'accuracy'
