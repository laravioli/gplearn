"""Testing the Genetic Programming module's underlying datastructure
(gplearn.genetic._graph) as well as the classes that use it,
gplearn.genetic.SymbolicRegressor and gplearn.genetic.SymbolicTransformer."""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import pickle
import pytest
import sys
import copy
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

from sklearn.utils.estimator_checks import check_estimator
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

# Function used to create quick graph with a lisp program input
def lisp_to_graph(lisp, random_state, **params):

    # sanity check
    if not isinstance(lisp[0], _Function):
        raise TypeError('first element must be a function')
    if params['n_rows'] > 1:
        params['n_cols'] = params['n_cols'] * params['n_rows']
        params['n_rows'] = 1

    # genotype initialisation
    max_arity = max([f.arity for f in params['function_set']])
    inp_genes = [[0]*params['n_cols'] for i in range(max_arity)]
    func_genes = [0]*params['n_cols']
    out_genes = [params['n_cols'] + params['n_features'] -1]
    genotype = _Genotype(inp_genes, func_genes, out_genes)

    # first node initialisation
    node_count = 1
    terminal_stack = [[lisp[0].arity, 0, params['n_cols'] - node_count]]
    genotype.func_genes[terminal_stack[-1][2]] = params['function_set'].index(lisp[0])

    # fill genotype
    for elt in lisp[1:]:
        parent = terminal_stack[-1]
        if isinstance(elt, _Function):
            node_count += 1
            genotype.inp_genes[parent[1]][parent[2]] = params['n_cols'] + params['n_features'] - node_count
            parent[0] += -1
            parent[1] += 1
            if (parent[2] - node_count) < 0:
                raise ValueError('number of nodes must be greater than number of functions in lisp')
            terminal_stack.append([elt.arity, 0, params['n_cols'] - node_count])
            genotype.func_genes[terminal_stack[-1][2]] = params['function_set'].index(elt)

        else:
            genotype.inp_genes[parent[1]][parent[2]] = elt
            parent[0] += -1
            parent[1] += 1
            while terminal_stack[-1][0] == 0:
                terminal_stack.pop()
                if not terminal_stack:
                    return _Graph(random_state=random_state, program=genotype, **params)
    # We should never get here
    return None

#############################################################################################
################################# _GRAPH CLASS TEST #########################################
#############################################################################################
def test_graph_init_method():
    """Check genotype details"""

    params = {'function_set': [add2, sub2, mul2, div2, sqrt1, log1, abs1, max2,
                               min2],
              'n_features': 10,
              'metric': 'mean absolute error',
              'p_point_replace': 0.05,
              'parsimony_coefficient': 0.1,
              'n_cols' : 4,
              'n_rows' : 4,
              'n_outputs' : 1
              }
    random_state = check_random_state(None)
    program = _Graph(random_state=random_state, **params)
    
    #genotype
    assert(isinstance(program._genotype, _Genotype))
    assert(len(program._genotype.inp_genes[0]) == program.n_cols * program.n_rows)
    assert(type(program._genotype.out_genes[0]) == np.int64)
    
    #active_nodes
    assert1 = len(program.active_nodes) == 0
    assert2 = program._genotype.out_genes[0] >= program.n_features
    assert(assert1 or assert2)

def test_validate_genotype():
    """Check that valid programs are accepted & invalid ones raise error"""

    function_set = [add2, sub2, mul2, div2]
    n_features = 10
    metric = 'mean absolute error'
    p_point_replace = 0.05
    parsimony_coefficient = 0.1
    n_cols = 2
    n_rows = 2
    n_outputs = 1
    random_state = check_random_state(415)

    # Test for a small program
    inp_genes = np.array([[8,9,10,0], [1,2,11,0]])
    func_genes = np.array([3,1,2,0])
    out_genes = [12]
    test_genotype = _Genotype(inp_genes= inp_genes, func_genes= func_genes, out_genes= out_genes)

    # This one should be fine
    _ = _Graph(function_set, n_features,
                       metric, p_point_replace, parsimony_coefficient, random_state,
                       n_cols = n_cols, n_rows = n_rows, n_outputs = n_outputs,
                       program = test_genotype)
    # Now try one that shouldn't be
    test_genotype.inp_genes[0][1] = 24
    assert_raises((ValueError,IndexError), _Graph, function_set,
                  n_features, metric, p_point_replace, 
                  parsimony_coefficient, random_state,
                  n_cols = n_cols, n_rows = n_rows, n_outputs = n_outputs,
                  program = test_genotype)

def test_execute():
    """Check executing the program works"""

    params = {'function_set': [add2, sub2, mul2, div2],
              'n_features': 10,
              'metric': 'mean absolute error',
              'p_point_replace': 0.05,
              'parsimony_coefficient': 0.1,
              'n_cols' : 2,
              'n_rows' : 2,
              'n_outputs' : 1
              }
    random_state = check_random_state(415)

    # Test for a small program
    inp_genes = np.array([[8,9,10,0], [1,2,11,0]])
    func_genes = np.array([3,1,2,0])
    out_genes = [12]
    test_genotype = _Genotype(inp_genes= inp_genes, func_genes= func_genes, out_genes= out_genes)

    # test_gp = [mul2, div2, 8, 1, sub2, 9, .5]

    X = np.reshape(random_state.uniform(size=50), (5, 10))
    X[:,2] = [0.5]*5

    gp = _Graph(random_state=random_state, program=test_genotype, **params)
    result = gp.execute(X)
    expected = [-0.19656208, 0.78197782, -1.70123845, -0.60175969, -0.01082618]
    assert_array_almost_equal(result, expected)

def test_execute_lisp_graph():
    """Check executing a lisp program works"""

    params = {'function_set': [add2, sub2, mul2, div2],
              'n_features': 10,
              'metric': 'mean absolute error',
              'p_point_replace': 0.05,
              'parsimony_coefficient': 0.1,
              'n_cols' : 2,
              'n_rows' : 2,
              'n_outputs' : 1
              }
    random_state = check_random_state(415)

    # Test for a small program
    test_gp = [mul2, div2, 8, 1, sub2, 9, .5]
    X = np.reshape(random_state.uniform(size=50), (5, 10))

    gp = lisp_to_graph(lisp = test_gp, random_state=random_state, **params)
    result = gp.execute(X)
    expected = [-0.19656208, 0.78197782, -1.70123845, -0.60175969, -0.01082618]
    assert_array_almost_equal(result, expected)

def test_execute_with_one_arity_functions():
    params = {'function_set': [sqrt1, log1, abs1, add2, mul2, div2],
              'n_features': 10,
              'metric': 'mean absolute error',
              'p_point_replace': 0.05,
              'parsimony_coefficient': 0.1,
              'n_cols' : 2,
              'n_rows' : 2,
              'n_outputs' : 1
              }
    random_state = check_random_state(40)
    gp = _Graph(random_state=random_state, **params)
    X = np.reshape(random_state.uniform(size=50), (5, 10))
    gp.execute(X)
    assert(np.array([params['function_set'][elt].arity == 1 for elt in gp._genotype.func_genes]).any())

def test_validate_active_nodes():
    """Check if active_nodes is immutable when we conserve active graph"""

    params = {'function_set': [add2, sub2, mul2, div2, sqrt1, log1, abs1, max2,
                               min2],
              'n_features': 4,
              'metric': 'mean absolute error',
              'p_point_replace': 0.05,
              'parsimony_coefficient': 0.1,
              'n_cols' : 10,
              'n_rows' : 4,
              'n_outputs' : 2
              }
    random_state = check_random_state(415)
    graph1 = _Graph(random_state=random_state, **params)
    active_nodes1 = graph1.active_nodes
    assert(len(active_nodes1) >= 3)

    genotype1 = graph1._genotype
    genotype2 = copy.deepcopy(genotype1)

    gpos = 0
    for c in range(graph1.n_cols):
        for _ in range(graph1.n_rows):
            if gpos not in [an.idx - graph1.n_features for an in active_nodes1]:
                genotype2.inp_genes[0][gpos] = random_state.randint(\
                    graph1.n_features + c * graph1.n_rows)
                genotype2.inp_genes[1][gpos] = random_state.randint(\
                    graph1.n_features + c * graph1.n_rows)
                genotype2.func_genes[gpos] = random_state.randint(\
                    len(graph1.function_set))
            gpos = gpos + 1

    program2 = _Graph(random_state=random_state, program= genotype2, **params)
    active_nodes2 = program2.active_nodes

    array1 = [an.idx for an in active_nodes1]
    array2 = [an.idx for an in active_nodes2]
    assert_array_equal(array1, array2)

def test_point_mutation():
    '''test if mutations are done correctly'''

    params = {'function_set': [add2, sub2, mul2, div2, sqrt1, log1, abs1, max2,
                            min2],
            'n_features': 4,
            'metric': 'mean absolute error',
            'p_point_replace': 0.05,
            'parsimony_coefficient': 0.1,
            'n_cols' : 10,
            'n_rows' : 4,
            'n_outputs' : 2
            }
    random_state = check_random_state(415)
    graph = _Graph(random_state=random_state, **params)
    genotype = graph._genotype
    nodes_x = genotype.inp_genes[0]
    nodes_y = genotype.inp_genes[1]
    nodes_f = genotype.func_genes
    outputs = genotype.out_genes

    genotype_mutated , mutation = graph.point_mutation(random_state)
    nodes_x_mutated = genotype_mutated.inp_genes[0]
    nodes_y_mutated = genotype_mutated.inp_genes[1]
    nodes_f_mutated = genotype_mutated.func_genes
    outputs_mutated = genotype_mutated.out_genes

    assert_array_equal(np.where(nodes_x != nodes_x_mutated)[0], mutation[0])
    assert_array_equal(np.where(nodes_y != nodes_y_mutated)[0], mutation[1])
    assert_array_equal(np.where(nodes_f != nodes_f_mutated)[0], mutation[2])
    assert_array_equal(np.where(outputs != outputs_mutated)[0], mutation[3])

#############################################################################################
################################# _GRAPH AND GENETIC TEST ###################################
#############################################################################################

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

#############################################################################################
################################# ESTIMATOR CHECK TEST  #####################################
#############################################################################################

def test_sklearn_regressor_checks():
    """Run the sklearn estimator validation checks on SymbolicRegressor"""

    check_estimator(SymbolicRegressor(population_size=1000,
                                    generations=5, representation='graph'))


def test_sklearn_classifier_checks():
    """Run the sklearn estimator validation checks on SymbolicClassifier"""

    check_estimator(SymbolicClassifier(population_size=50,
                                       generations=5, representation='graph'))


def test_sklearn_transformer_checks():
    """Run the sklearn estimator validation checks on SymbolicTransformer"""

    check_estimator(SymbolicTransformer(population_size=50,
                                        hall_of_fame=10,
                                        generations=5, representation='graph'))