"""The underlying data structure used in gplearn.

The :mod:`gplearn._program` module contains the underlying representation of a
computer program. It is used for creating and evolving programs used in the
:mod:`gplearn.genetic` module.
"""

import numpy as np
from abc import ABCMeta, abstractmethod
from sklearn.utils.random import sample_without_replacement
from .utils import check_random_state

class _GeneticProgram(object, metaclass = ABCMeta):

    @abstractmethod
    def __init__(self,
                 function_set,
                 n_features,
                 metric,
                 p_point_replace,
                 parsimony_coefficient,
                 random_state,
                 transformer = None,
                 feature_names = None,
                 program =  None,
                 **kwargs):

        self.function_set = function_set
        self.n_features = n_features
        self.metric = metric
        self.p_point_replace = p_point_replace
        self.parsimony_coefficient = parsimony_coefficient
        self.transformer = transformer
        self.feature_names = feature_names
        self.program = program

        self.raw_fitness_ = None
        self.fitness_ = None
        self.parents = None
        self._n_samples = None
        self._max_samples = None
        self._indices_state = None

    # abstract class attribute
    p_crossover : float
    p_subtree_mutation : float
    p_hoist_mutation : float
    p_point_mutation : float

    #static method
    @staticmethod
    def validate_mutation_probs(p_crossover, p_subtree, p_hoist, p_point):
        pass

    # abstract method
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def _length(self):
        pass

    @abstractmethod
    def execute(self, X):
        pass

    @abstractmethod
    def fitness(self, parsimony_coefficient=None):
        pass

    @abstractmethod
    def reproduce(self):
        pass

    @abstractmethod
    def point_mutation(self, random_state):
        pass

    # method
    def get_subtree(self, random_state, program=None):
        raise NotImplementedError

    def reproduce(self):
        raise NotImplementedError

    def crossover(self, donor, random_state):
        raise NotImplementedError

    def subtree_mutation(self, random_state):
        raise NotImplementedError

    def hoist_mutation(self, random_state):
        raise NotImplementedError

    def get_all_indices(self, n_samples=None, max_samples=None,
                        random_state=None):
        """Get the indices on which to evaluate the fitness of a program.

        Parameters
        ----------
        n_samples : int
            The number of samples.

        max_samples : int
            The maximum number of samples to use.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        indices : array-like, shape = [n_samples]
            The in-sample indices.

        not_indices : array-like, shape = [n_samples]
            The out-of-sample indices.

        """
        if self._indices_state is None and random_state is None:
            raise ValueError('The program has not been evaluated for fitness '
                             'yet, indices not available.')

        if n_samples is not None and self._n_samples is None:
            self._n_samples = n_samples
        if max_samples is not None and self._max_samples is None:
            self._max_samples = max_samples
        if random_state is not None and self._indices_state is None:
            self._indices_state = random_state.get_state()

        indices_state = check_random_state(None)
        indices_state.set_state(self._indices_state)

        not_indices = sample_without_replacement(
            self._n_samples,
            self._n_samples - self._max_samples,
            random_state=indices_state)
        sample_counts = np.bincount(not_indices, minlength=self._n_samples)
        indices = np.where(sample_counts == 0)[0]

        return indices, not_indices

    def _indices(self):
        """Get the indices used to measure the program's fitness."""
        return self.get_all_indices()[0]

    def raw_fitness(self, X, y, sample_weight):
        """Evaluate the raw fitness of the program according to X, y.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples]
            Weights applied to individual samples.

        Returns
        -------
        raw_fitness : float
            The raw fitness of the program.

        """
        y_pred = self.execute(X)
        if self.transformer:
            y_pred = self.transformer(y_pred)
        raw_fitness = self.metric(y, y_pred, sample_weight)

        return raw_fitness

    indices_ = property(_indices)