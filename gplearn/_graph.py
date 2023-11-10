"""The :mod:`gplearn._graph` module contains the underlying representation of a graph program.
It is used as a _GeneticProgram subclass for creating and evolving graph.
"""

from copy import copy, deepcopy

import numpy as np
from sklearn.utils.random import sample_without_replacement

from .functions import _Function
from .utils import check_random_state
from ._program import _GeneticProgram

class _Genotype:

    def __init__(self, nodes : dict, outputs : list):
        self.nodes = nodes
        self.outputs = outputs

    def __len__(self):
        return len(self.nodes['x']) * 3 + len(self.outputs)

class _Graph(_GeneticProgram):

    """A program-like representation of the evolved program.

    This is the underlying data-structure used by the public classes in the
    :mod:`gplearn.genetic` module. It should not be used directly by the user.

    Parameters
    ----------
    function_set : list
        A list of valid functions to use in the program.

    arities : dict
        A dictionary of the form `{arity: [functions]}`. The arity is the
        number of arguments that the function takes, the functions must match
        those in the `function_set` parameter.

    init_depth : tuple of two ints
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    n_features : int
        The number of features in `X`.

    const_range : tuple of two floats
        The range of constants to include in the formulas.

    metric : _Fitness object
        The raw fitness metric.

    p_point_replace : float
        The probability that any given node will be mutated during point
        mutation.

    parsimony_coefficient : float
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

    random_state : RandomState instance
        The random number generator. Note that ints, or None are not allowed.
        The reason for this being passed is that during parallel evolution the
        same program object may be accessed by multiple parallel processes.

    transformer : _Function object, optional (default=None)
        The function to transform the output of the program to probabilities,
        only used for the SymbolicClassifier.

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    program : list, optional (default=None)
        The flattened tree representation of the program. If None, a new naive
        random tree will be grown. If provided, it will be validated.

    Attributes
    ----------
    program : list
        The flattened tree representation of the program.

    raw_fitness_ : float
        The raw fitness of the individual program.

    fitness_ : float
        The penalized fitness of the individual program.

    oob_fitness_ : float
        The out-of-bag raw fitness of the individual program for the held-out
        samples. Only present when sub-sampling was used in the estimator by
        specifying `max_samples` < 1.0.

    parents : dict, or None
        If None, this is a naive random program from the initial population.
        Otherwise it includes meta-data about the program's parent(s) as well
        as the genetic operations performed to yield the current program. This
        is set outside this class by the controlling evolution loops.

    depth_ : int
        The maximum depth of the program tree.

    length_ : int
        The number of functions and terminals in the program.

    """

    def __init__(self,
                 function_set,
                 arities,
                 init_depth,
                 init_method,
                 n_features,
                 const_range,
                 metric,
                 p_point_replace,
                 parsimony_coefficient,
                 random_state,
                 n_inputs,
                 n_cols,
                 n_rows,
                 n_outputs,
                 transformer=None,
                 feature_names=None,
                 program=None):

        super(_Graph, self).__init__(
                 function_set,
                 arities,
                 init_depth,
                 init_method,
                 n_features,
                 const_range,
                 metric,
                 p_point_replace,
                 parsimony_coefficient,
                 random_state,
                 transformer=None,
                 feature_names=None,
                 program=None)

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.n_nodes = n_cols * n_rows

        if self._genotype is not None:
            if not self.validate_genotype():
                raise ValueError('The supplied genotype is incomplete.')

        else:
            self._genotype = self.build_genotype(random_state)
            
        self.active_graph = self.build_active_graph()

        self.raw_fitness_ = None
        self.fitness_ = None
        self.parents = None
        self._n_samples = None
        self._max_samples = None
        self._indices_state = None

    @property
    def program(self): return self._genotype

    @program.setter
    def program(self, value):
        self._genotype = value

    def build_genotype(self, random_state):
        """Build a naive random genome | spec : max_arities = 2, l_value = num_columns.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        genotype : _Genotype instance
            The genotype of the program. Nodes are subscribed in Cmajor style

        """
        x_values = np.zeros(self.n_nodes, dtype= int)
        y_values = np.zeros(self.n_nodes, dtype= int)
        f_values = np.zeros(self.n_nodes, dtype= int)

        genotype_nodes = dict(x = x_values, y = y_values, f = f_values)
        genotypes_outputs = np.zeros(self.n_outputs, dtype= int)

        gpos = 0

        for c in range(self.n_cols):
            for _ in range(self.n_rows):
                genotype_nodes['x'][gpos] = random_state.randint(\
                    self.n_inputs + c * self.n_rows)
                genotype_nodes['y'][gpos] = random_state.randint(\
                    self.n_inputs + c * self.n_rows)
                genotype_nodes['f'][gpos] = random_state.randint(\
                    len(self.function_set))
                gpos = gpos + 1

        for o in range(self.n_outputs):
            genotypes_outputs[o] = random_state.randint(self.n_inputs + self.n_nodes)

        return _Genotype(genotype_nodes, genotypes_outputs)
    
    def build_active_graph(self):
        """Build genotype active graph.
        
         Returns
        -------
         active_graph : list
            The active_graph of the program.
        
        """
        active_graph = []
        nodes_to_evaluate = np.zeros(self.n_nodes, dtype=bool)
        p = 0
        while p < self.n_outputs:
            if self._genotype.outputs[p] >= self.n_inputs:
                nodes_to_evaluate[self._genotype.outputs[p] - self.n_inputs] = True
            p = p + 1
        p = self.n_nodes - 1
        while p >= 0:
            if nodes_to_evaluate[p]:
                for var in ['x', 'y']:
                    arg = self._genotype.nodes[var][p]
                    if arg - self.n_inputs >= 0:
                        nodes_to_evaluate[arg - self.n_inputs] = True
                active_graph.append(p)
            p = p - 1

        return np.array(active_graph, dtype= int)

#---validate---------------------------------------------------------------------#
#---graph------------------------------------------------------------------------#
    def validate_genotype(self):
        """Check that the embedded genotype in the object is valid."""
        state = True

        for genotype_output in self._genotype.outputs:
            state = genotype_output in range(self.n_inputs + self.n_nodes)
            if not state:
                break
        if state:
            for node_input, node_values in self._genotype.nodes.items():
                for i, node_value in enumerate(node_values):
                    if node_input == 'f':
                        state = node_value in range(len(self.function_set))
                    else:
                        column = (i+1) % self.n_rows
                        state = node_value in range(self.n_inputs + column * self.n_rows)
                    if not state:
                        break
        return state
        
#---str--------------------------------------------------------------------------#
#---graph------------------------------------------------------------------------#
    def __str__(self):
        """Overloads `print` output of the genotype object graph."""
        nodes = self._genotype.nodes
        outputs = self._genotype.outputs
        output = '['
        for i in range(self.n_nodes):
            for node_input in ['x','y','f']:
                output += '%d ' % nodes[node_input][i]
            output += '|'
        for i in range(len(outputs)):
            output += '%d ' % outputs[i]
        output += ']'

        return output

#---export_graphviz--------------------------------------------------------------#
#---graph------------------------------------------------------------------------#
    def export_graphviz(self, fade_nodes=None):
      """work in progress"""

#---length-----------------------------------------------------------------------#
#---graph----------------------------------------------------------------------#
    def _length(self):
        """Calculates the lenght of the genotype."""
        return len(self._genotype)

#---execute----------------------------------------------------------------------#
#---graph------------------------------------------------------------------------#
    def execute(self, X):
        """Execute the program according to X | spec : max_arities = 2.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.
        """

        p= len(self.active_graph) - 1
        nodes_output = np.zeros((self.n_nodes + self.n_inputs, X.shape[0]))
        output = np.zeros((self.n_outputs, X.shape[0]))

        for i in range(X.shape[1]):
            nodes_output[i] = X[:,i]

        while p >= 0:
            args = np.zeros((2, X.shape[0]))
            for i, var in enumerate(['x','y']):
                args[i] = nodes_output[self._genotype.nodes[var][self.active_graph[p]]]
            f = self.function_set[self._genotype.nodes['f'][self.active_graph[p]]]
            nodes_output[self.active_graph[p] + self.n_inputs] = f(*args)
            p = p - 1

        for i in range(self.n_outputs):
            output[i] = nodes_output[self._genotype.outputs[i]].copy()

        if self.n_outputs == 1:
            output = np.squeeze(output)
        return output
    
    def fitness(self, parsimony_coefficient=None):
        """Evaluate the penalized fitness of the program according to X, y.

        Parameters
        ----------
        parsimony_coefficient : float, optional
            If automatic parsimony is being used, the computed value according
            to the population. Otherwise the initialized value is used.

        Returns
        -------
        fitness : float
            The penalized fitness of the program.

        """
        if parsimony_coefficient is None:
            parsimony_coefficient = self.parsimony_coefficient
        penalty = parsimony_coefficient * len(self.active_graph) * self.metric.sign
        return self.raw_fitness_ - penalty

#---mutation----------------------------------------------------------------#
#---graph-------------------------------------------------------------------#
    def point_mutation(self, random_state):
        """Perform the point mutation operation on the genotype.

        Point mutation selects random genes from the embedded genotype to be
        replaced. The resulting genotype forms an offspring.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        genotype : _Genotype instance
            The new genotype of the graph.
        """
        genotype = deepcopy(self._genotype)

        # Roll a dice for each gene
        node_mut = random_state.uniform(size = (3,self.n_nodes))
        output_mut = random_state.uniform(size = self.n_outputs)

        # Get genes to mutate
        mutate_x = np.where(node_mut[0] < self.p_point_replace)[0]
        mutate_y = np.where(node_mut[1] < self.p_point_replace)[0]
        mutate_f = np.where(node_mut[2] < self.p_point_replace)[0]
        mutate_o = np.where(output_mut < self.p_point_replace)[0]

        # Apply mutation on genotype
        for gene in mutate_x:
            column = (gene + 1) % self.n_rows
            replacement = random_state.randint(self.n_inputs + column * self.n_rows)
            genotype.nodes['x'][gene] = replacement
        
        for gene in mutate_y:
            column = (gene + 1) % self.n_rows
            replacement = random_state.randint(self.n_inputs + column * self.n_rows)
            genotype.nodes['y'][gene] = replacement

        for gene in mutate_f:
            replacement = random_state.randint(len(self.function_set))
            genotype.nodes['f'][gene] = replacement

        for gene in mutate_o:
            replacement = random_state.randint(self.n_outputs)
            genotype.outputs[gene] = replacement

        mutate = [mutate_x, mutate_y, mutate_f, mutate_o]

        return genotype, mutate

    length_ = property(_length)