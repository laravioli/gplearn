"""Testing the Genetic Programming module's underlying datastructure
(gplearn.genetic._Program) as well as the classes that use it,
gplearn.genetic.SymbolicRegressor and gplearn.genetic.SymbolicTransformer."""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import pickle
import pytest
import sys
from io import StringIO

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_raises
from sklearn.utils.validation import check_random_state

from gplearn.genetic import SymbolicClassifier, SymbolicRegressor
from gplearn.genetic import SymbolicTransformer
from gplearn.fitness import weighted_pearson, weighted_spearman
from gplearn._program import _GeneticProgram
from gplearn._graph import _Genotype, _Graph
from gplearn._tree import _Tree
from gplearn.fitness import _fitness_map
from gplearn.functions import (add2, sub2, mul2, div2, sqrt1, log1, abs1, max2,
                               min2)
from gplearn.functions import _Function

cls_test = _Graph

# load the diabetes dataset and randomly permute it
rng = check_random_state(0)
diabetes = load_diabetes()
perm = rng.permutation(diabetes.target.size)
diabetes.data = diabetes.data[perm]
diabetes.target = diabetes.target[perm]

# load the breast cancer dataset and randomly permute it
rng = check_random_state(0)
cancer = load_breast_cancer()
perm = rng.permutation(cancer.target.size)
cancer.data = cancer.data[perm]
cancer.target = cancer.target[perm]

def test_input_validation():
    for Symbolic in (SymbolicRegressor, SymbolicTransformer):
        """Check that guarded input validation raises errors"""
        # Check regressor metrics
        for m in ['mean absolute error', 'mse', 'rmse', 'pearson', 'spearman']:
            est = SymbolicRegressor(population_size=100, generations=1, metric=m, 
                                    representation= 'graph')
            est.fit(diabetes.data, diabetes.target)
        # And check a fake one
        est = SymbolicRegressor(metric='the larch', representation= "graph")
        assert_raises(ValueError, est.fit, diabetes.data, diabetes.target)
        # Check transformer metrics
        for m in ['pearson', 'spearman']:
            est = SymbolicTransformer(population_size=100, generations=1, metric=m, 
                                      representation= 'graph')
            est.fit(diabetes.data, diabetes.target)
        # And check the regressor metrics as well as a fake one
        for m in ['mean absolute error', 'mse', 'rmse', 'the larch']:
            est = SymbolicTransformer(metric=m, representation= 'graph')
            assert_raises(ValueError, est.fit, diabetes.data, diabetes.target)

def test_input_validation_classifier():
    """Check that guarded input validation raises errors"""

    # Check too much proba
    est = SymbolicClassifier(p_point_mutation=1.1, representation = "graph")
    assert_raises(ValueError, est.fit, cancer.data, cancer.target)

    # Check classifier metrics
    for m in ['log loss']:
        est = SymbolicClassifier(population_size=100, generations=1, metric=m, 
                                 representation= 'graph')
        est.fit(cancer.data, cancer.target)
    # And check a fake one
    est = SymbolicClassifier(metric='the larch', representation= 'graph')
    assert_raises(ValueError, est.fit, cancer.data, cancer.target)

    # Check classifier transformers
    for t in ['sigmoid']:
        est = SymbolicClassifier(population_size=100, generations=1,
                                 transformer=t, representation= 'graph')
        est.fit(cancer.data, cancer.target)
    # And check an incompatible one with wrong arity
    est = SymbolicClassifier(transformer=sub2, representation= 'graph')
    assert_raises(ValueError, est.fit, cancer.data, cancer.target)
    # And check a fake one
    est = SymbolicClassifier(transformer='the larch', representation= 'graph')
    assert_raises(ValueError, est.fit, cancer.data, cancer.target)

def test_sample_weight_and_class_weight():
    """Check sample_weight param works"""

    # Check constant sample_weight has no effect
    sample_weight = np.ones(diabetes.target.shape[0])
    est1 = SymbolicRegressor(population_size=100, generations=2,
                             random_state=0, representation= 'graph')
    est1.fit(diabetes.data, diabetes.target)
    est2 = SymbolicRegressor(population_size=100, generations=2,
                             random_state=0, representation= 'graph')
    est2.fit(diabetes.data, diabetes.target, sample_weight=sample_weight)
    # And again with a scaled sample_weight
    est3 = SymbolicRegressor(population_size=100, generations=2,
                             random_state=0, representation= 'graph')
    est3.fit(diabetes.data, diabetes.target, sample_weight=sample_weight * 1.1)

    assert_almost_equal(est1._program.fitness_, est2._program.fitness_)
    assert_almost_equal(est1._program.fitness_, est3._program.fitness_)

    # And again for the classifier
    sample_weight = np.ones(cancer.target.shape[0])
    est1 = SymbolicClassifier(population_size=100, generations=2,
                              random_state=0, representation= 'graph')
    est1.fit(cancer.data, cancer.target)
    est2 = SymbolicClassifier(population_size=100, generations=2,
                              random_state=0, representation= 'graph')
    est2.fit(cancer.data, cancer.target, sample_weight=sample_weight)
    # And again with a scaled sample_weight
    est3 = SymbolicClassifier(population_size=100, generations=2,
                              random_state=0, representation= 'graph')
    est3.fit(cancer.data, cancer.target, sample_weight=sample_weight * 1.1)
    # And then using class weight to do the same thing
    est4 = SymbolicClassifier(class_weight={0: 1, 1: 1}, population_size=100,
                              generations=2, random_state=0, representation= 'graph')
    est4.fit(cancer.data, cancer.target)
    est5 = SymbolicClassifier(class_weight={0: 1.1, 1: 1.1},
                              population_size=100, generations=2,
                              random_state=0, representation= 'graph')
    est5.fit(cancer.data, cancer.target)

    assert_almost_equal(est1._program.fitness_, est2._program.fitness_)
    assert_almost_equal(est1._program.fitness_, est3._program.fitness_)
    assert_almost_equal(est1._program.fitness_, est4._program.fitness_)
    assert_almost_equal(est1._program.fitness_, est5._program.fitness_)

def test_trigonometric():
    """Check that using trig functions work and that results differ"""

    est1 = SymbolicRegressor(population_size=100, generations=2,
                             random_state=0, representation= 'graph')
    est1.fit(diabetes.data[:400, :], diabetes.target[:400])
    est1 = mean_absolute_error(est1.predict(diabetes.data[400:, :]),
                               diabetes.target[400:])

    est2 = SymbolicRegressor(population_size=100, generations=2,
                             function_set=['add', 'sub', 'mul', 'div',
                                           'sin', 'cos', 'tan'],
                             random_state=0, representation= 'graph')
    est2.fit(diabetes.data[:400, :], diabetes.target[:400])
    est2 = mean_absolute_error(est2.predict(diabetes.data[400:, :]),
                               diabetes.target[400:])

    assert(abs(est1 - est2) > 0.01)

def test_subsample():
    """Check that subsample work and that results differ"""

    est1 = SymbolicRegressor(population_size=100, generations=2,
                             max_samples=1.0, random_state=0, representation= 'graph')
    est1.fit(diabetes.data[:400, :], diabetes.target[:400])
    est1 = mean_absolute_error(est1.predict(diabetes.data[400:, :]),
                               diabetes.target[400:])

    est2 = SymbolicRegressor(population_size=100, generations=2,
                             max_samples=0.1, random_state=0, representation= 'graph')
    est2.fit(diabetes.data[:400, :], diabetes.target[:400])
    est2 = mean_absolute_error(est2.predict(diabetes.data[400:, :]),
                               diabetes.target[400:])

    assert(abs(est1 - est2) > 0.01)

def test_parsimony_coefficient():
    """Check that parsimony coefficients work and that results differ"""

    est1 = SymbolicRegressor(population_size=100, generations=2,
                             parsimony_coefficient=0.001, random_state=0, representation='graph')
    est1.fit(diabetes.data[:400, :], diabetes.target[:400])
    est1 = mean_absolute_error(est1.predict(diabetes.data[400:, :]),
                               diabetes.target[400:])

    est2 = SymbolicRegressor(population_size=100, generations=2,
                             parsimony_coefficient='auto', random_state=0, representation='graph')
    est2.fit(diabetes.data[:400, :], diabetes.target[:400])
    est2 = mean_absolute_error(est2.predict(diabetes.data[400:, :]),
                               diabetes.target[400:])

    assert(abs(est1 - est2) > 0.01)

def test_early_stopping():
    """Check that early stopping works"""

    est1 = SymbolicRegressor(population_size=100, generations=2,
                             stopping_criteria=200, random_state=0, representation= 'graph')
    est1.fit(diabetes.data[:400, :], diabetes.target[:400])
    assert(len(est1._programs) == 1)

    est1 = SymbolicTransformer(population_size=100, generations=2,
                               stopping_criteria=100, random_state=0, representation= 'graph')
    est1.fit(cancer.data[:400, :], cancer.target[:400])
    assert(len(est1._programs) == 2)

def test_verbose_output():
    """Check verbose=1 does not cause error"""

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    est = SymbolicRegressor(population_size=100, generations=10,
                            random_state=0, verbose=1, representation= 'graph')
    est.fit(diabetes.data, diabetes.target)
    verbose_output = sys.stdout
    sys.stdout = old_stdout

    # check output
    verbose_output.seek(0)
    header1 = verbose_output.readline().rstrip()
    true_header = '    |{:^25}|{:^42}|'.format('Population Average',
                                               'Best Individual')
    assert(true_header == header1)

    header2 = verbose_output.readline().rstrip()
    true_header = '-' * 4 + ' ' + '-' * 25 + ' ' + '-' * 42 + ' ' + '-' * 10
    assert(true_header == header2)

    header3 = verbose_output.readline().rstrip()

    line_format = '{:>4} {:>8} {:>16} {:>8} {:>16} {:>16} {:>10}'
    true_header = line_format.format('Gen', 'Length', 'Fitness', 'Length',
                                     'Fitness', 'OOB Fitness', 'Time Left')
    assert(true_header == header3)

    n_lines = sum(1 for l in verbose_output.readlines())
    assert(10 == n_lines)

def test_verbose_with_oob():
    """Check oob scoring for subsample does not cause error"""

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    est = SymbolicRegressor(population_size=100, generations=10,
                            max_samples=0.9, random_state=0, verbose=1, representation= 'graph')
    est.fit(diabetes.data, diabetes.target)
    verbose_output = sys.stdout
    sys.stdout = old_stdout

    # check output
    verbose_output.seek(0)
    # Ignore header rows
    _ = verbose_output.readline().rstrip()
    _ = verbose_output.readline().rstrip()
    _ = verbose_output.readline().rstrip()

    n_lines = sum(1 for l in verbose_output.readlines())
    assert(10 == n_lines)


def test_more_verbose_output():
    """Check verbose=2 does not cause error"""

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    est = SymbolicRegressor(population_size=100, generations=10,
                            random_state=0, verbose=2, representation= 'graph')
    est.fit(diabetes.data, diabetes.target)
    verbose_output = sys.stdout
    joblib_output = sys.stderr
    sys.stdout = old_stdout
    sys.stderr = old_stderr

    # check output
    verbose_output.seek(0)
    # Ignore header rows
    _ = verbose_output.readline().rstrip()
    _ = verbose_output.readline().rstrip()
    _ = verbose_output.readline().rstrip()

    n_lines = sum(1 for l in verbose_output.readlines())
    assert(10 == n_lines)

    joblib_output.seek(0)
    n_lines = sum(1 for l in joblib_output.readlines())
    # New version of joblib appears to output sys.stderr
    assert(0 == n_lines % 10)

def test_parallel_train():
    """Check predictions are the same for different n_jobs"""

    # Check the regressor
    ests = [
        SymbolicRegressor(population_size=100, generations=4, n_jobs=n_jobs,
                          random_state=0, representation= 'graph').fit(diabetes.data[:100, :],
                                              diabetes.target[:100])
        for n_jobs in [1, 2, 3, 8, 16]
    ]

    preds = [e.predict(diabetes.data[400:, :]) for e in ests]
    for pred1, pred2 in zip(preds, preds[1:]):
        assert_array_almost_equal(pred1, pred2)
    lengths = np.array([[gp.length_ for gp in e._programs[-1]] for e in ests])
    for len1, len2 in zip(lengths, lengths[1:]):
        assert_array_almost_equal(len1, len2)

    # Check the transformer
    ests = [
        SymbolicTransformer(population_size=100, hall_of_fame=50,
                            generations=4, n_jobs=n_jobs,
                            random_state=0, representation= 'graph').fit(diabetes.data[:100, :],
                                                diabetes.target[:100])
        for n_jobs in [1, 2, 3, 8, 16]
    ]

    preds = [e.transform(diabetes.data[400:, :]) for e in ests]
    for pred1, pred2 in zip(preds, preds[1:]):
        assert_array_almost_equal(pred1, pred2)
    lengths = np.array([[gp.length_ for gp in e._programs[-1]] for e in ests])
    for len1, len2 in zip(lengths, lengths[1:]):
        assert_array_almost_equal(len1, len2)

    # Check the classifier
    ests = [
        SymbolicClassifier(population_size=100, generations=4, n_jobs=n_jobs,
                           random_state=0, representation='graph').fit(cancer.data[:100, :],
                                               cancer.target[:100])
        for n_jobs in [1, 2, 3, 8, 16]
    ]

    preds = [e.predict(cancer.data[400:, :]) for e in ests]
    for pred1, pred2 in zip(preds, preds[1:]):
        assert_array_almost_equal(pred1, pred2)
    lengths = np.array([[gp.length_ for gp in e._programs[-1]] for e in ests])
    for len1, len2 in zip(lengths, lengths[1:]):
        assert_array_almost_equal(len1, len2)

def test_pickle():
    """Check pickability"""

    # Check the regressor
    est = SymbolicRegressor(population_size=100, generations=2,
                            random_state=0, representation= 'graph')
    est.fit(diabetes.data[:100, :], diabetes.target[:100])
    score = est.score(diabetes.data[400:, :], diabetes.target[400:])
    pickle_object = pickle.dumps(est)

    est2 = pickle.loads(pickle_object)
    assert(type(est2) == est.__class__)
    score2 = est2.score(diabetes.data[400:, :], diabetes.target[400:])
    assert(score == score2)

    # Check the transformer
    est = SymbolicTransformer(population_size=100, generations=2,
                              random_state=0, representation= 'graph')
    est.fit(diabetes.data[:100, :], diabetes.target[:100])
    X_new = est.transform(diabetes.data[400:, :])
    pickle_object = pickle.dumps(est)

    est2 = pickle.loads(pickle_object)
    assert(type(est2) == est.__class__)
    X_new2 = est2.transform(diabetes.data[400:, :])
    assert_array_almost_equal(X_new, X_new2)

    # Check the classifier
    est = SymbolicClassifier(population_size=100, generations=2,
                             random_state=0, representation= 'graph')
    est.fit(cancer.data[:100, :], cancer.target[:100])
    score = est.score(cancer.data[500:, :], cancer.target[500:])
    pickle_object = pickle.dumps(est)

    est2 = pickle.loads(pickle_object)
    assert(type(est2) == est.__class__)
    score2 = est2.score(cancer.data[500:, :], cancer.target[500:])
    assert(score == score2)


def test_output_shape():
    """Check output shape is as expected"""

    random_state = check_random_state(415)
    X = np.reshape(random_state.uniform(size=50), (5, 10))
    y = random_state.uniform(size=5)

    # Check the transformer
    est = SymbolicTransformer(population_size=100, generations=2,
                              n_components=5, random_state=0, representation= 'graph')
    est.fit(X, y)
    assert(est.transform(X).shape == (5, 5))

def test_print_overloading_estimator():
    """Check that printing a fitted estimator results in 'pretty' output"""

    random_state = check_random_state(415)
    X = np.reshape(random_state.uniform(size=50), (5, 10))
    y = random_state.uniform(size=5)

    # Check the regressor
    est = SymbolicRegressor(population_size=100, generations=2, random_state=0, 
                            representation='graph')

    # Unfitted
    orig_stdout = sys.stdout
    try:
        out = StringIO()
        sys.stdout = out
        print(est)
        output_unfitted = out.getvalue().strip()
    finally:
        sys.stdout = orig_stdout

    # Fitted
    est.fit(X, y)
    orig_stdout = sys.stdout
    try:
        out = StringIO()
        sys.stdout = out
        print(est)
        output_fitted = out.getvalue().strip()
    finally:
        sys.stdout = orig_stdout

    orig_stdout = sys.stdout
    try:
        out = StringIO()
        sys.stdout = out
        print(est._program)
        output_program = out.getvalue().strip()
    finally:
        sys.stdout = orig_stdout

    assert(output_unfitted != output_fitted)
    assert(output_unfitted == est.__repr__())
    assert(output_fitted == output_program)

    # Check the transformer
    est = SymbolicTransformer(population_size=100, generations=2,
                              random_state=0, representation='graph')

    # Unfitted
    orig_stdout = sys.stdout
    try:
        out = StringIO()
        sys.stdout = out
        print(est)
        output_unfitted = out.getvalue().strip()
    finally:
        sys.stdout = orig_stdout

    # Fitted
    est.fit(X, y)
    orig_stdout = sys.stdout
    try:
        out = StringIO()
        sys.stdout = out
        print(est)
        output_fitted = out.getvalue().strip()
    finally:
        sys.stdout = orig_stdout

    orig_stdout = sys.stdout
    try:
        out = StringIO()
        sys.stdout = out
        output = str([gp.__str__() for gp in est])
        print(output.replace("',", ",\n").replace("'", ""))
        output_program = out.getvalue().strip()
    finally:
        sys.stdout = orig_stdout

    assert(output_unfitted != output_fitted)
    assert(output_unfitted == est.__repr__())
    assert(output_fitted == output_program)

    # Check the classifier
    y = (y > .5).astype(int)
    est = SymbolicClassifier(population_size=100, generations=2, random_state=0, 
                             representation= 'graph')

    # Unfitted
    orig_stdout = sys.stdout
    try:
        out = StringIO()
        sys.stdout = out
        print(est)
        output_unfitted = out.getvalue().strip()
    finally:
        sys.stdout = orig_stdout

    # Fitted
    est.fit(X, y)
    orig_stdout = sys.stdout
    try:
        out = StringIO()
        sys.stdout = out
        print(est)
        output_fitted = out.getvalue().strip()
    finally:
        sys.stdout = orig_stdout

    orig_stdout = sys.stdout
    try:
        out = StringIO()
        sys.stdout = out
        print(est._program)
        output_program = out.getvalue().strip()
    finally:
        sys.stdout = orig_stdout

    assert(output_unfitted != output_fitted)
    assert(output_unfitted == est.__repr__())
    assert(output_fitted == output_program)

def test_validate_functions():
    """Check that valid functions are accepted & invalid ones raise error"""

    for Symbolic in (SymbolicRegressor, SymbolicTransformer):
        # These should be fine
        est = Symbolic(population_size=100, generations=2, random_state=0,
                       function_set=(add2, sub2, mul2, div2), representation='graph')
        est.fit(diabetes.data, diabetes.target)
        est = Symbolic(population_size=100, generations=2, random_state=0,
                       function_set=('add', 'sub', 'mul', div2))
        est.fit(diabetes.data, diabetes.target)

        # These should fail
        est = Symbolic(generations=2, random_state=0,
                       function_set=('ni', 'sub', 'mul', div2), representation= 'graph')
        assert_raises(ValueError, est.fit, diabetes.data, diabetes.target)
        est = Symbolic(generations=2, random_state=0,
                       function_set=(7, 'sub', 'mul', div2), representation= 'graph')
        assert_raises(ValueError, est.fit, diabetes.data, diabetes.target)
        est = Symbolic(generations=2, random_state=0, function_set=())
        assert_raises(ValueError, est.fit, diabetes.data, diabetes.target)

def test_run_details():
    """Check the run_details_ attribute works as expected."""

    est = SymbolicRegressor(population_size=100, generations=5, random_state=0, representation='graph')
    est.fit(diabetes.data, diabetes.target)
    # Check generations are indexed as expected without warm_start
    assert(est.run_details_['generation'] == list(range(5)))
    est.set_params(generations=10, warm_start=True)
    est.fit(diabetes.data, diabetes.target)
    # Check generations are indexed as expected with warm_start
    assert(est.run_details_['generation'] == list(range(10)))
    # Check all details have expected number of elements
    for detail in est.run_details_:
        assert(len(est.run_details_[detail]) == 10)

def test_warm_start():
    """Check the warm_start functionality works as expected."""

    est = SymbolicRegressor(population_size=50, generations=10, random_state=0, 
                            representation='graph')
    est.fit(diabetes.data, diabetes.target)
    cold_fitness = est._program.fitness_
    cold_program = est._program.__str__()

    # Check fitting fewer generations raises error
    est.set_params(generations=5, warm_start=True)
    assert_raises(ValueError, est.fit, diabetes.data, diabetes.target)

    # Check fitting the same number of generations warns
    est.set_params(generations=10, warm_start=True)
    with pytest.warns(UserWarning):
        est.fit(diabetes.data, diabetes.target)

    # Check warm starts get the same result
    est = SymbolicRegressor(population_size=50, generations=5, random_state=0, 
                            representation='graph')
    est.fit(diabetes.data, diabetes.target)
    est.set_params(generations=10, warm_start=True)
    est.fit(diabetes.data, diabetes.target)
    warm_fitness = est._program.fitness_
    warm_program = est._program.__str__()
    assert_almost_equal(cold_fitness, warm_fitness)
    assert(cold_program == warm_program)


def test_low_memory():
    """Check the low_memory functionality works as expected."""

    est = SymbolicRegressor(population_size=50,
                            generations=10,
                            random_state=56,
                            low_memory=True,
                            representation= 'graph')
    # Check there are no parents
    est.fit(diabetes.data, diabetes.target)
    assert(est._programs[-2] is None)


def test_low_memory_warm_start():
    """Check the warm_start functionality works as expected with low_memory."""

    est = SymbolicRegressor(population_size=50,
                            generations=20,
                            random_state=415,
                            low_memory=True,
                            representation='graph')
    est.fit(diabetes.data, diabetes.target)
    cold_fitness = est._program.fitness_
    cold_program = est._program.__str__()

    # Check warm start with low memory gets the same result
    est = SymbolicRegressor(population_size=50,
                            generations=10,
                            random_state=415,
                            low_memory=True, 
                            representation='graph')
    est.fit(diabetes.data, diabetes.target)
    est.set_params(generations=20, warm_start=True)
    est.fit(diabetes.data, diabetes.target)
    warm_fitness = est._program.fitness_
    warm_program = est._program.__str__()
    assert_almost_equal(cold_fitness, warm_fitness)
    assert(cold_program == warm_program)

