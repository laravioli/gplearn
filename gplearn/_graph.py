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

    def __init__(self, genes_inp, genes_func, genes_out):
        self.genes_inp = genes_inp
        self.genes_func= genes_func
        self.genes_out = genes_out

    def __str__(self):
        output = '['
        for gene in range(len(self.genes_inp[0])):
            for var in range(len(self.genes_inp)):
                if isinstance(self.genes_inp[var][gene], int):
                    output += '%d ' % self.genes_inp[var][gene]
                else:
                    output += '%.3f ' % self.genes_inp[var][gene]
            output += '%d' % (self.genes_func[gene])
            output += '|'
        for gene in range(len(self.genes_out)):
            output += '%d ' % self.genes_out[gene]
        return output + ']'

class _Node:

    def __init__(self, idx, args, function):
        self.idx = idx
        self.args = args
        self.function = function

    def __str__(self):
        args = ''
        for arg in self.args:
            if isinstance(arg, int):
                args += '%d ' % arg
            else:
                args += '%.3f ' % arg
        return 'idx: %d , args: %s,name: %s' % (self.idx, args, self.function.name)

class _Graph(_GeneticProgram):

    """A program-like representation of the evolved program.

    This is the underlying data-structure used by the public classes in the
    :mod:`gplearn.genetic` module. It should not be used directly by the user.

    """

    def __init__(self,
                 function_set,
                 n_features,
                 metric,
                 p_point_replace,
                 parsimony_coefficient,
                 random_state,
                 n_cols,
                 n_rows,
                 n_outputs,
                 *args, **kwargs):

        super(_Graph, self).__init__(
            function_set,
            n_features,
            metric,
            p_point_replace,
            parsimony_coefficient,
            random_state, 
            **kwargs)

        self.n_cols = n_cols
        self.n_rows = n_rows
        self.n_outputs = n_outputs
        self.n_nodes = n_cols * n_rows
        self.max_arity = max([f.arity for f in self.function_set])

        if self._genotype is not None:
            self.validate_genotype()
        else:
            self._genotype = self.build_genotype(random_state)
            
        self.active_nodes = self.build_active_nodes()

    # CLASS ATTRIBUTE : Define default mutation probability values
    p_crossover = 0.0
    p_subtree_mutation = 0.0
    p_hoist_mutation = 0.0
    p_point_mutation = 0.93

    # descriptor
    # aliases : _genotype is accessed with the name program outside the class
    @property
    def program(self): return self._genotype

    @program.setter
    def program(self, value):
        self._genotype = value

    # METHOD
    def build_genotype(self, random_state):
        """Build a naive random genotype with l_value = num_columns.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        genotype : _Genotype instance
            The genotype of the program. Nodes are subscribed in Cmajor style

        """
        genes_i = np.zeros((self.max_arity, self.n_nodes), dtype= int)
        genes_f = np.zeros(self.n_nodes, dtype= int)
        genes_o = np.zeros(self.n_outputs, dtype= int)

        gpos = 0
        for c in range(self.n_cols):
            for _ in range(self.n_rows):
                for i in range(len(genes_i)):
                    genes_i[i][gpos] = random_state.randint(self.n_features + c * self.n_rows)
                    genes_i[i][gpos] = random_state.randint(self.n_features + c * self.n_rows)
                genes_f[gpos] = random_state.randint(len(self.function_set))
                gpos = gpos + 1

        for o in range(self.n_outputs):
            genes_o[o] = random_state.randint(self.n_features + self.n_nodes)

        return _Genotype(genes_i, genes_f, genes_o)

    def validate_genotype(self):
        """Check that the embedded genotype in the graph is valid."""

        if len(self._genotype.genes_out) != self.n_outputs:
            raise ValueError('lenght of output genes must equal %d' % self.n_outputs)

        if len(self._genotype.genes_func) != self.n_nodes:
            raise ValueError('lenght of function genes must equal %d' % len(self.function_set))

        if len(self._genotype.genes_inp) != self.max_arity:
            raise ValueError('The genotype must provide %d input lists' % self.max_arity)

        for var in range(len(self._genotype.genes_inp)):
            if len(self._genotype.genes_inp[var]) != self.n_nodes:
                raise ValueError('lenght of input genes must equal %d' % self.n_nodes)

        for gene in self._genotype.genes_out:
            if gene not in range(self.n_features + self.n_nodes):
                raise ValueError('output genes must be in range(%d)' % self.n_features + self.n_nodes)

        for gene in self._genotype.genes_func:
            if gene not in range(len(self.function_set)):
                raise ValueError('function genes must be in range(%d)' % len(self.function_set))

        for var in range(len(self._genotype.genes_inp)):
            for i in range(self.n_nodes):
                column = i // self.n_rows
                if self._genotype.genes_inp[var][i] not in range(self.n_features + column * self.n_rows):
                    raise ValueError ('input genes must respect the graph l-value (%d)' % self.n_cols)

    def build_active_nodes(self):
        """Build genotype active nodes.
        
         Returns
        -------
         active_nodes : list[Node]
            The active_nodes of the program.
        
        """
        active_nodes = []
        nodes_to_evaluate = np.zeros(self.n_nodes, dtype=bool)

        for i in range(self.n_outputs):
            if self._genotype.genes_out[i] >= self.n_features:
                nodes_to_evaluate[self._genotype.genes_out[i] - self.n_features] = True

        
        for p in reversed(range(self.n_nodes)):
            if nodes_to_evaluate[p]:
                args = []
                function = self.function_set[self._genotype.genes_func[p]]
                for var in range(function.arity):
                    arg = self._genotype.genes_inp[var][p]
                    args.append(arg)
                    if arg >= self.n_features:
                        nodes_to_evaluate[arg - self.n_features] = True
                active_node = _Node(p + self.n_features, args, function)
                active_nodes.append(active_node)

        return active_nodes
        
    def __str__(self):
        """Overloads `print` output of the graph to display his genotype and active nodes."""

        genotype = self._genotype.__str__() + '\n'
        active_nodes = ''
        for active_node in self.active_nodes:
            active_nodes += active_node.__str__() + '\n'
        return genotype + active_nodes

    def export_graphviz(self):
        """Returns a string, Graphviz script for visualizing the program.

        Returns
        -------
        output : string
            The Graphviz script to plot the graph representation of the program.

        """
        output = 'digraph program {\nnode [style=filled, ordering=out];\n'
        edges = ''

        for feature in range(self.n_features):
            if feature in self._genotype.genes_out:
                if self.feature_names is None:
                    feature_name = 'X%d' % feature
                else:
                    feature_name = self.feature_names[feature] 
                output += ('%d [label="%s", fillcolor="#60a6f6"];\n'
                            % (feature, feature_name))

        for node in self.active_nodes:
            output += ('%d [label="%s", fillcolor="#136ed4"];\n'
                        % (node.idx, node.function.name))
            for arg in node.args:
                if arg >= self.n_features:
                    edges += ('%d -> %d;\n' % (node.idx,
                                               arg))
                else:
                    edges += ('%d -> X%d%d;\n' % (node.idx,
                                                  node.idx,
                                                  arg))
                    if self.feature_names is None:
                        feature_name = 'X%d' % arg
                    else:
                        feature_name = self.feature_names[arg]
                    output += ('X%d%d [label="%s", fillcolor="#60a6f6"];\n'
                                % (node.idx, arg, feature_name))
        return output + edges + '}'

    def pickle_save_graph(self, filename):
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
                    
    def _length(self):
        """Calculates the lenght of the active_nodes."""
        return len(self.active_nodes)

    length_ = property(_length)

    def execute(self, X):
        """Execute the program according to X

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_hats : array-like, shape = (n_samples,) if n_outputs == 1 
                                     else (n_outputs, n_samples)
            The result of executing the program on X.
        """

        nodes_output = np.zeros((self.n_features + self.n_nodes, X.shape[0]))
        output = np.zeros((self.n_outputs, X.shape[0]))

        for i in range(self.n_features):
            nodes_output[i] = X[:,i]

        for active_node in reversed(self.active_nodes):
            args = np.zeros((len(active_node.args), X.shape[0]))
            for i in range(len(args)):
                args[i] = nodes_output[active_node.args[i]]
                nodes_output[active_node.idx] = active_node.function(*args)

        for i in range(self.n_outputs):
            output[i] = nodes_output[self._genotype.genes_out[i]].copy()

        if self.n_outputs == 1:
            return output[0]
        else:
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
        penalty = parsimony_coefficient * len(self.active_nodes) * self.metric.sign
        return self.raw_fitness_ - penalty
    
    def reproduce(self):
        """Return a copy of the embedded _genotype."""
        return deepcopy(self._genotype)

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
            column = gene // self.n_rows
            replacement = random_state.randint(self.n_features + column * self.n_rows)
            genotype.genes_inp[0][gene] = replacement
        
        for gene in mutate_y:
            column = gene // self.n_rows
            replacement = random_state.randint(self.n_features + column * self.n_rows)
            genotype.genes_inp[1][gene] = replacement

        for gene in mutate_f:
            replacement = random_state.randint(len(self.function_set))
            genotype.genes_func[gene] = replacement

        for gene in mutate_o:
            replacement = random_state.randint(self.n_outputs)
            genotype.genes_out[gene] = replacement

        mutate = [mutate_x, mutate_y, mutate_f, mutate_o]

        return genotype, mutate

    # STATIC METHOD
    @staticmethod
    def validate_mutation_probs(p_crossover, p_subtree, p_hoist, p_point):
        if int(p_crossover) != 0 or int(p_subtree) != 0 or int(p_hoist) != 0:
            raise ValueError("Graph doesn't have crossover, subtree or hoist mutations, probabilities should be equals to 0")
    
    @staticmethod
    def lisp_to_genotype(lisp):
        print('a')
        # test_gp = [mul2, div2, 8, 1, sub2, 9, 2]