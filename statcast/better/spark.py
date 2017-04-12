import pandas as pd
from sklearn import model_selection

try:
    import spark_sklearn
    import pyspark
except ImportError:
    sparkRuns = False
else:
    sparkRuns = True


class GridSearchCV(model_selection.GridSearchCV):
    '''Doc String'''

    def fit(self, X, y=None):
        '''Doc String'''

        if isinstance(self.n_jobs, int):
            return super().fit(X, y)
        elif sparkRuns:
            self.sc = self.n_jobs
            if not isinstance(self.sc, pyspark.SparkContext):
                raise RuntimeError('n_jobs parameter was not an int, meaning '
                                   'it should have been a SparkContext, but '
                                   'it was not.')
            return spark_sklearn.GridSearchCV.fit(self, X, y)
        else:
            raise RuntimeError('n_jobs parameter was not an int, meaning it '
                               'should have been a SparkContext, but spark '
                               'was unable to be imported.')


def cross_val_score(estimator, X, y=None, groups=None, scoring=None, cv=None,
                    n_jobs=1, verbose=0, fit_params=None,
                    pre_dispatch='2*n_jobs'):
    '''Doc String'''

    if isinstance(n_jobs, int):
        return model_selection.cross_val_score(estimator, X, y, groups,
                                               scoring, cv, n_jobs, verbose,
                                               fit_params, pre_dispatch)
    elif sparkRuns:
        sc = n_jobs
        if not isinstance(sc, pyspark.SparkContext):
            raise RuntimeError('n_jobs parameter was not an int, meaning '
                               'it should have been a SparkContext, but '
                               'it was not.')
        clf = spark_sklearn.GridSearchCV(sc=sc,
                                         estimator=estimator,
                                         param_grid={},
                                         scoring=scoring,
                                         fit_params=fit_params,
                                         n_jobs=n_jobs,
                                         iid=True,
                                         refit=False,
                                         cv=cv,
                                         verbose=verbose,
                                         pre_dispatch=pre_dispatch,
                                         error_score='raise')
        df = pd.DataFrame(clf.cv_results_)
        score = df.loc[0, ['split{}_test_score'.format(ii)
                           for ii in range(cv)]].astype(float).values
        return score
    else:
        raise RuntimeError('n_jobs parameter was not an int, meaning it '
                           'should have been a SparkContext, but spark '
                           'was unable to be imported.')
