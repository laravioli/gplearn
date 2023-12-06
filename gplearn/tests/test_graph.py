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

# convert a lisp program into a graph
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

# _GRAPH CLASS TEST

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