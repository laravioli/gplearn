"""The :mod:`gplearn._graph` module contains the underlying representation of a graph program.
It is used as a _GeneticProgram subclass for creating and evolving graph.
"""

import numpy as np
from copy import copy, deepcopy
from .functions import _Function
from ._program import _GeneticProgram

class _Genotype:

    """data structure used by the class _Graph int the :mod:`gplearn._graph` module.
    It contains the genetic material needed for graph evolution. Genes are the genetic material of graph nodes.  
    They are stored in a flattened manner. The format and value of the genes expected in a graph are described 
    in the validate_genotype method of the _Graph class.

    Parameters
    ----------
    inp_genes : array-like
        Input genes for graph nodes.

    func_genes : list
        Function genes for graph nodes.

    out_genes : list
        Output genes. Code anticipates more than one output.

    """
    def __init__(self, inp_genes, func_genes, out_genes):
        self.inp_genes = inp_genes
        self.func_genes = func_genes
        self.out_genes = out_genes

    def __str__(self):
        """Used to overlad `print` _Graph method"""
        output = '['
        for gene in range(len(self.inp_genes[0])):
            for entry in range(len(self.inp_genes)):
                if isinstance(self.inp_genes[entry][gene], (int, np.int_)):
                    output += '%d ' % self.inp_genes[entry][gene]
                else:
                    output += '%.3f ' % self.inp_genes[entry][gene]
            output += '%d' % (self.func_genes[gene])
            output += '|'
        for gene in range(len(self.out_genes)):
                output += ' %d' % self.out_genes[gene]
        return output + ']'

class _ActiveNode:

    """Represent the active nodes, those that will be used during graph execution.

    Parameters
    ----------
    idx : int
        Index of node. As data inputs are considered nodes, the node index starts at n_features,
        dimension of input vectors.

    args : list
        Node inputs. They are extracted from the genotype. Argument length depends on node function

    function : _Function
        Node function. It belongs to the graph's function_set

    """
    def __init__(self, idx, args, function):
        self.idx = idx
        self.args = args
        self.function = function

    def __str__(self):
        """Used to overlad `print` _Graph method"""
        args = ''
        for arg in self.args:
            if isinstance(arg, (int,np.int_)):
                args += '%d ' % arg
            else:
                args += '%.3f ' % arg
        return 'idx: %d , args: %s,function: %s' % (self.idx, args, self.function.name)

class _Graph(_GeneticProgram):

    """A cartesian graph representation of the evolved program.

    This is the underlying data-structure used by the public classes in the
    :mod:`gplearn.genetic` module. It should not be used directly by the user.

    Parameters
    ----------
    function_set : list
        A list of valid functions to use in the program.

    n_features : int
        The number of features in `X`.

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

    n_cols : int (default = 40)
        Define the width of the graph node grid

    n_rows : int (default = 1)
        Define the height of the graph node grid

    n_outputs : int (default = 1)
        Define the number of outputs. Gplearn currently supports only one output

    program : _Genotype, optional (default=None)
        The graph's genetic material. If None, a new naive
        random genotype will be grown. If provided, it will be validated.

    Attributes
    ----------
    _max_arity : int
        Define the maximum input a node can have.

    active_nodes : list[_ActiveNode]
        Store all active nodes. Used in graph execution and representation with graphviz.

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
        self._max_arity = max([f.arity for f in self.function_set])

        if self._genotype is not None:
            self.validate_genotype()
        else:
            self._genotype = self.build_genotype(random_state)

        self.active_nodes = self.build_active_nodes()

    # CLASS ATTRIBUTE : Define default mutation probability values
    p_crossover = 0.0
    p_subtree_mutation = 0.0
    p_hoist_mutation = 0.0
    p_point_mutation = 0.8

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
            The genotype of the program.

        """
        inp_genes = np.zeros((self._max_arity, self.n_nodes), dtype= int)
        func_genes = np.zeros(self.n_nodes, dtype= int)
        out_genes = np.zeros(self.n_outputs, dtype= int)

        gpos = 0
        for c in range(self.n_cols):
            for _ in range(self.n_rows):
                for i in range(len(inp_genes)):
                    inp_genes[i][gpos] = random_state.randint(self.n_features + c * self.n_rows)
                    inp_genes[i][gpos] = random_state.randint(self.n_features + c * self.n_rows)
                func_genes[gpos] = random_state.randint(len(self.function_set))
                gpos = gpos + 1

        for o in range(self.n_outputs):
            out_genes[o] = random_state.randint(self.n_features + self.n_nodes)

        return _Genotype(inp_genes, func_genes, out_genes)

    def validate_genotype(self):
        """Check that the embedded genotype in the graph is valid.
        
        """
        # dimension check
        if len(self._genotype.out_genes) != self.n_outputs:
            raise ValueError('lenght of output genes must equal %d' % self.n_outputs)

        if len(self._genotype.func_genes) != self.n_nodes:
            raise ValueError('lenght of function genes must equal %d' % len(self.function_set))

        for var in range(len(self._genotype.inp_genes)):
            if len(self._genotype.inp_genes[var]) != self.n_nodes:
                raise ValueError('lenght of input genes must equal %d' % self.n_nodes)

        if len(self._genotype.inp_genes) != self._max_arity:
            raise ValueError('The genotype must provide %d input lists' % self._max_arity)

        # value ranges check
        for gene in self._genotype.out_genes:
            if gene not in range(self.n_features + self.n_nodes):
                raise ValueError('output genes must be in range(%d)' % self.n_features + self.n_nodes)

        for gene in self._genotype.func_genes:
            if gene not in range(len(self.function_set)):
                raise ValueError('function genes must be in range(%d)' % len(self.function_set))

        for var in range(len(self._genotype.inp_genes)):
            for i in range(self.n_nodes):
                gene = self._genotype.inp_genes[var][i]
                column = i // self.n_rows
                if isinstance(gene,(int, np.int_)) and gene not in range(self.n_features + column * self.n_rows):
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
            if self._genotype.out_genes[i] >= self.n_features:
                nodes_to_evaluate[self._genotype.out_genes[i] - self.n_features] = True

        
        for p in reversed(range(self.n_nodes)):
            if nodes_to_evaluate[p]:
                args = []
                function = self.function_set[self._genotype.func_genes[p]]
                for var in range(function.arity):
                    arg = self._genotype.inp_genes[var][p]
                    args.append(arg)
                    if isinstance(arg, (int, np.int_)) and arg >= self.n_features:
                        nodes_to_evaluate[arg - self.n_features] = True
                active_node = _ActiveNode(p + self.n_features, args, function)
                active_nodes.append(active_node)

        return active_nodes
        
    def __str__(self):
        """Overloads `print` output of the graph to display his genotype and active nodes.
        
        """
        genotype = self._genotype.__str__() + '\n'
        active_nodes = ''
        for active_node in self.active_nodes:
            active_nodes += active_node.__str__() + '\n'
        return 'genotype:\n' + genotype + 'active nodes:\n' + active_nodes

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
            if feature in self._genotype.out_genes:
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
                if isinstance(arg,(int,np.int_)) and arg >= self.n_features:
                    edges += ('%d -> %d;\n' % (node.idx,
                                               arg))
                else:
                    if isinstance(arg,(int,np.int_)):
                        edges += ('%d -> X%d%d;\n' % (node.idx,
                                                        node.idx,
                                                        arg))
                        if self.feature_names is None:
                            feature_name = 'X%d' % arg
                        else:
                            feature_name = self.feature_names[arg]
                        output += ('X%d%d [label="%s", fillcolor="#60a6f6"];\n'
                                    % (node.idx, arg, feature_name))
                    else:
                        id_ = str(arg).replace('.','')
                        edges += ('%d -> F%d%s;\n' % (node.idx,
                                                        node.idx,
                                                        id_))
                        output += ('F%d%s [label="%.3f", fillcolor="#60a6f6"];\n'
                                    % (node.idx, id_, arg))

        return output + edges + '}'

    def pickle_save_graph(self, filename):
        """
        method to store graph object

        """
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
                    
    def _length(self):
        """Calculates the lenght of active nodes.
        
        """
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

        for i in range(X.shape[1]):
            nodes_output[i] = X[:,i]

        for active_node in reversed(self.active_nodes):
            args = np.zeros((len(active_node.args), X.shape[0]))
            for i in range(len(args)):
                argi = active_node.args[i]
                if isinstance(argi,(int,np.int_)):
                    args[i] = nodes_output[argi]
                else:
                    args[i] = np.repeat(argi, X.shape[0])            
            nodes_output[active_node.idx] = active_node.function(*args)

        for i in range(self.n_outputs):
            output[i] = nodes_output[self._genotype.out_genes[i]].copy()

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
        dice_1 = random_state.uniform(size = (self._max_arity + 1,self.n_nodes))
        dice_2 = random_state.uniform(size = self.n_outputs)

        # Get genes to mutate
        mutate_inp = [np.where(dice_1[i] < self.p_point_replace)[0] for i in range(self._max_arity)]
        mutate_func = np.where(dice_1[self._max_arity] < self.p_point_replace)[0]
        mutate_out = np.where(dice_2 < self.p_point_replace)[0]

        # Apply mutation on genotype
        for i in range(self._max_arity):
            for gene in mutate_inp[i]:
                column = gene // self.n_rows
                replacement = random_state.randint(self.n_features + column * self.n_rows)
                genotype.inp_genes[i][gene] = replacement

        for gene in mutate_func:
            replacement = random_state.randint(len(self.function_set))
            genotype.func_genes[gene] = replacement

        for gene in mutate_out:
            replacement = random_state.randint(self.n_outputs)
            genotype.out_genes[gene] = replacement

        mutate = [list(mut) for mut in mutate_inp + [mutate_func] + [mutate_out]]
        return genotype, mutate

    # STATIC METHOD
    @staticmethod
    def validate_mutation_probs(p_crossover, p_subtree, p_hoist, p_point):
        """
        Check if user-defined value of specific _Tree mutation are tuned to 0.
        
        """
        if int(p_crossover) != 0 or int(p_subtree) != 0 or int(p_hoist) != 0:
            raise ValueError("Graph doesn't have crossover, subtree or hoist mutations, probabilities should be equals to 0")