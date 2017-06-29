from collections import Sized, defaultdict
from functools import partial

import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import ParameterGrid, check_cv
from sklearn.model_selection._validation import _fit_and_score
from sklearn.base import is_classifier, clone
from sklearn.metrics.scorer import check_scoring
from sklearn.utils.validation import indexable
from sklearn.utils.fixes import rankdata, MaskedArray

try:
    import pyspark
except ImportError:
    sparkRuns = False
else:
    sparkRuns = True


class GridSearchCV(model_selection.GridSearchCV):
    '''Doc String'''

    def fit(self, X, y=None, groups=None):
        '''Doc String'''

        if isinstance(self.n_jobs, int):
            super().fit(X, y, groups)
            self.cv_results_ = pd.DataFrame(self.cv_results_)
            return self
        elif sparkRuns:
            if not isinstance(self.n_jobs, pyspark.SparkContext):
                raise RuntimeError('n_jobs parameter was not an int, meaning '
                                   'it should have been a SparkContext, but '
                                   'it was not.')
            return self._scFit(X, y, groups, ParameterGrid(self.param_grid))
        else:
            raise RuntimeError('n_jobs parameter was not an int, meaning it '
                               'should have been a SparkContext, but spark '
                               'was unable to be imported.')

    def _scFit(self, X, y, groups, parameter_iterable):
        '''Doc String'''

        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        X, y, groups = indexable(X, y, groups)
        n_splits = cv.get_n_splits(X, y, groups)
        if self.verbose > 0 and isinstance(parameter_iterable, Sized):
            n_candidates = len(parameter_iterable)
            print("Fitting {0} folds for each of {1} candidates, totalling"
                  " {2} fits".format(n_splits, n_candidates,
                                     n_candidates * n_splits))

        base_estimator = clone(self.estimator)

        cv_iter = list(cv.split(X, y, groups))
        param_grid = [(parameters, train, test)
                      for parameters in parameter_iterable
                      for (train, test) in cv_iter]
        # Because the original python code expects a certain order for the
        # elements, we need to respect it.
        indexed_param_grid = list(zip(range(len(param_grid)), param_grid))
        par_param_grid = self.n_jobs.parallelize(indexed_param_grid,
                                                 len(indexed_param_grid))
        X_bc = self.n_jobs.broadcast(X)
        y_bc = self.n_jobs.broadcast(y)

        scorer = self.scorer_
        verbose = self.verbose
        fit_params = self.fit_params
        return_train_score = self.return_train_score
        error_score = self.error_score
        fas = _fit_and_score

        def fun(tup):
            (index, (parameters, train, test)) = tup
            local_estimator = clone(base_estimator)
            local_X = X_bc.value
            local_y = y_bc.value
            res = fas(local_estimator, local_X, local_y, scorer, train, test,
                      verbose, parameters,
                      fit_params=fit_params,
                      return_train_score=return_train_score,
                      return_n_test_samples=True,
                      return_times=True,
                      return_parameters=True,
                      error_score=error_score)
            return (index, res)
        indexed_out0 = dict(par_param_grid.map(fun).collect())
        out = [indexed_out0[idx] for idx in range(len(param_grid))]

        X_bc.unpersist()
        y_bc.unpersist()

        # if one choose to see train score, "out" will contain train score info
        if self.return_train_score:
            (train_scores, test_scores, test_sample_counts,
             fit_time, score_time, parameters) = zip(*out)
        else:
            (test_scores, test_sample_counts,
             fit_time, score_time, parameters) = zip(*out)

        candidate_params = parameters[::n_splits]
        n_candidates = len(candidate_params)

        results = dict()

        def _store(key_name, array, weights=None, splits=False, rank=False):
            """A small helper to store the scores/times to the cv_results_"""
            array = np.array(array, dtype=np.float64).reshape(n_candidates,
                                                              n_splits)
            if splits:
                for split_i in range(n_splits):
                    results["split%d_%s"
                            % (split_i, key_name)] = array[:, split_i]

            array_means = np.average(array, axis=1, weights=weights)
            results['mean_%s' % key_name] = array_means
            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(np.average((array -
                                             array_means[:, np.newaxis]) ** 2,
                                            axis=1, weights=weights))
            results['std_%s' % key_name] = array_stds

            if rank:
                results["rank_%s" % key_name] = np.asarray(
                    rankdata(-array_means, method='min'), dtype=np.int32)

        # Computed the (weighted) mean and std for test scores alone
        # NOTE test_sample counts (weights) remain the same for all candidates
        test_sample_counts = np.array(test_sample_counts[:n_splits],
                                      dtype=np.int)

        _store('test_score', test_scores, splits=True, rank=True,
               weights=test_sample_counts if self.iid else None)
        if self.return_train_score:
            _store('train_score', train_scores, splits=True)
        _store('fit_time', fit_time)
        _store('score_time', score_time)

        best_index = np.flatnonzero(results["rank_test_score"] == 1)[0]
        best_parameters = candidate_params[best_index]

        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(partial(MaskedArray,
                                            np.empty(n_candidates,),
                                            mask=True,
                                            dtype=object))
        for cand_i, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)

        # Store a list of param dicts at the key 'params'
        results['params'] = candidate_params

        self.cv_results_ = pd.DataFrame(results)
        self.best_index_ = best_index
        self.n_splits_ = n_splits

        if self.refit:
            # fit the best estimator using the entire dataset
            # clone first to work around broken estimators
            best_estimator = clone(base_estimator).set_params(
                **best_parameters)
            if y is not None:
                best_estimator.fit(X, y, **self.fit_params)
            else:
                best_estimator.fit(X, **self.fit_params)
            self.best_estimator_ = best_estimator
        return self


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
        gs = GridSearchCV(estimator=estimator,
                          param_grid={},
                          scoring=scoring,
                          fit_params=fit_params,
                          n_jobs=n_jobs,
                          iid=True,
                          refit=False,
                          cv=cv,
                          verbose=verbose,
                          pre_dispatch=pre_dispatch,
                          error_score='raise').fit(X, y, groups)
        df = pd.DataFrame(gs.cv_results_)
        score = df.loc[0, ['split{}_test_score'.format(ii)
                           for ii in range(gs.n_splits_)]].astype(float).values
        return score
    else:
        raise RuntimeError('n_jobs parameter was not an int, meaning it '
                           'should have been a SparkContext, but spark '
                           'was unable to be imported.')
