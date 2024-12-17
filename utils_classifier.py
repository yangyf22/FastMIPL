import numpy as np
from time import time
from sys import stderr
from inspect import signature
from collections import defaultdict

########################################################################################
### Define the SoftmaxRegression class based on the following references:
# @article{raschkas_2018_mlxtend,
#   author       = {Sebastian Raschka},
#   title        = {MLxtend: Providing machine learning and data science 
#                   utilities and extensions to Python’s  
#                   scientific computing stack},
#   journal      = {The Journal of Open Source Software},
#   volume       = {3},
#   number       = {24},
#   month        = apr,
#   year         = 2018,
#   publisher    = {The Open Journal},
#   doi          = {10.21105/joss.00638},
#   url          = {https://joss.theoj.org/papers/10.21105/joss.00638}
# }
########################################################################################


class _BaseModel(object):
    def __init__(self):
        self._init_time = time()

    def _check_arrays(self, X, y=None):
        if isinstance(X, list):
            raise ValueError("X must be a numpy array")
        if not len(X.shape) == 2:
            raise ValueError("X must be a 2D array. Try X[:, numpy.newaxis]")
        try:
            if y is None:
                return
        except AttributeError:
            if not len(y.shape) == 1:
                raise ValueError("y must be a 1D array.")

        if not len(y) == X.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator

        adapted from
        https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/base.py
        Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
        License: BSD 4 clause



        """
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "scikit-learn estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.'

        adapted from
        https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/base.py
        Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
        License: BSD 3 clause

        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self

        adapted from
        https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/base.py
        Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
        License: BSD 3 clause

        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                raise ValueError(
                    "Invalid parameter %s for estimator %s. "
                    "Check the list of available parameters "
                    "with `estimator.get_params().keys()`." % (key, self)
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self


class _IterativeModel(object):
    def __init__(self):
        pass

    def _shuffle_arrays(self, arrays):
        """Shuffle arrays in unison."""
        r = np.random.permutation(len(arrays[0]))
        return [ary[r] for ary in arrays]

    def _print_progress(self, iteration, n_iter, cost=None, time_interval=10):
        if self.print_progress > 0:
            s = "\rIteration: %d/%d" % (iteration, n_iter)
            if cost:
                s += " | Cost %.2f" % cost
            if self.print_progress > 1:
                if not hasattr(self, "ela_str_"):
                    self.ela_str_ = "00:00:00"
                if not iteration % time_interval:
                    ela_sec = time() - self._init_time
                    self.ela_str_ = self._to_hhmmss(ela_sec)
                s += " | Elapsed: %s" % self.ela_str_
                if self.print_progress > 2:
                    if not hasattr(self, "eta_str_"):
                        self.eta_str_ = "00:00:00"
                    if not iteration % time_interval:
                        eta_sec = (ela_sec / float(iteration)) * n_iter - ela_sec
                        if eta_sec < 0.0:
                            eta_sec = 0.0
                        self.eta_str_ = self._to_hhmmss(eta_sec)
                    s += " | ETA: %s" % self.eta_str_
            stderr.write(s)
            stderr.flush()

    def _to_hhmmss(self, sec):
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)

    def _yield_minibatches_idx(self, rgen, n_batches, data_ary, shuffle=True):
        indices = np.arange(data_ary.shape[0])

        if shuffle:
            indices = rgen.permutation(indices)
        if n_batches > 1:
            remainder = data_ary.shape[0] % n_batches

            if remainder:
                minis = np.array_split(indices[:-remainder], n_batches)
                minis[-1] = np.concatenate((minis[-1], indices[-remainder:]), axis=0)
            else:
                minis = np.array_split(indices, n_batches)

        else:
            minis = (indices,)

        for idx_batch in minis:
            yield idx_batch

    def _init_params(
        self,
        weights_shape,
        bias_shape=(1,),
        random_seed=None,
        dtype="float64",
        scale=0.01,
        bias_const=0.0,
    ):
        """Initialize weight coefficients."""
        rgen = np.random.RandomState(random_seed)
        w = rgen.normal(loc=0.0, scale=scale, size=weights_shape)
        b = np.zeros(shape=bias_shape)
        if bias_const != 0.0:
            b += bias_const
        return b.astype(dtype), w.astype(dtype)


class _Classifier(object):
    def __init__(self):
        pass

    def _check_target_array(self, y, allowed=None):
        if not isinstance(y[0], (int, np.integer)):
            raise AttributeError("y must be an integer array.\nFound %s" % y.dtype)
        found_labels = np.unique(y)
        if (found_labels < 0).any():
            raise AttributeError(
                "y array must not contain negative labels." "\nFound %s" % found_labels
            )
        if allowed is not None:
            found_labels = tuple(found_labels)
            if found_labels not in allowed:
                raise AttributeError(
                    "Labels not in %s.\nFound %s" % (allowed, found_labels)
                )

    def score(self, X, y):
        """Compute the prediction accuracy

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values (true class labels).

        Returns
        ---------
        acc : float
            The prediction accuracy as a float
            between 0.0 and 1.0 (perfect score).

        """
        y_pred = self.predict(X)
        acc = np.sum(y == y_pred, axis=0) / float(X.shape[0])
        return acc

    def fit(self, X, y, init_params=True):
        """Learn model from training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        init_params : bool (default: True)
            Re-initializes model parameters prior to fitting.
            Set False to continue training with weights from
            a previous model fitting.

        Returns
        -------
        self : object

        """
        self._is_fitted = False
        self._check_arrays(X=X, y=y)
        # self._check_target_array(y)
        if hasattr(self, "self.random_seed") or self.random_seed:
            self._rgen = np.random.RandomState(self.random_seed)
        self._init_time = time()
        self._fit(X=X, y=y, init_params=init_params)
        self._is_fitted = True
        return self

    def predict(self, X):
        """Predict targets from X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        target_values : array-like, shape = [n_samples]
          Predicted target values.

        """
        self._check_arrays(X=X)
        if not self._is_fitted:
            raise AttributeError("Model is not fitted, yet.")
        return self._predict(X)


class _MultiClass(object):
    def __init__(self):
        pass

    def _one_hot(self, y, n_labels, dtype):
        """Returns a matrix where each sample in y is represented
           as a row, and each column represents the class label in
           the one-hot encoding scheme.

        Example:

            y = np.array([0, 1, 2, 3, 4, 2])
            mc = _BaseMultiClass()
            mc._one_hot(y=y, n_labels=5, dtype='float')

            np.array([[1., 0., 0., 0., 0.],
                      [0., 1., 0., 0., 0.],
                      [0., 0., 1., 0., 0.],
                      [0., 0., 0., 1., 0.],
                      [0., 0., 0., 0., 1.],
                      [0., 0., 1., 0., 0.]])

        """
        mat = np.zeros((len(y), n_labels))
        for i, val in enumerate(y):
            mat[i, val] = 1
        return mat.astype(dtype)



class SoftmaxRegression(_BaseModel, _IterativeModel, _Classifier, _MultiClass):

    """Softmax regression classifier.

    Parameters
    ------------
    eta : float (default: 0.01)
        Learning rate (between 0.0 and 1.0)

    epochs : int (default: 50)
        Passes over the training dataset.
        Prior to each epoch, the dataset is shuffled
        if `minibatches > 1` to prevent cycles in stochastic gradient descent.

    l2 : float
        Regularization parameter for L2 regularization.
        No regularization if l2=0.0.

    minibatches : int (default: 1)
        The number of minibatches for gradient-based optimization.
        If 1: Gradient Descent learning
        If len(y): Stochastic Gradient Descent (SGD) online learning
        If 1 < minibatches < len(y): SGD Minibatch learning

    n_classes : int (default: None)
        A positive integer to declare the number of class labels
        if not all class labels are present in a partial training set.
        Gets the number of class labels automatically if None.

    random_seed : int (default: None)
        Set random state for shuffling and initializing the weights.

    print_progress : int (default: 0)
        Prints progress in fitting to stderr.
        0: No output
        1: Epochs elapsed and cost
        2: 1 plus time elapsed
        3: 2 plus estimated time until completion

    Attributes
    -----------
    w_ : 2d-array, shape={n_features, 1}
      Model weights after fitting.

    b_ : 1d-array, shape={1,}
      Bias unit after fitting.

    cost_ : list
        List of floats, the average cross_entropy for each epoch.

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/classifier/SoftmaxRegression/

    """

    def __init__(
        self,
        eta=0.01,
        epochs=50,
        l2=0.0,
        minibatches=1,
        n_classes=None,
        random_seed=None,
        print_progress=0,
    ):
        
        self.eta = eta
        self.epochs = epochs
        self.l2 = l2
        self.minibatches = minibatches
        self.n_classes = n_classes
        self.random_seed = random_seed
        self.print_progress = print_progress
        self._is_fitted = False

        _BaseModel.__init__(self)
        _IterativeModel.__init__(self)
        _Classifier.__init__(self)
        _MultiClass.__init__(self)

    def _net_input(self, X):
        return X.dot(self.w_) + self.b_

    def _softmax_activation(self, z):
        e_x = np.exp(z - z.max(axis=1, keepdims=True))
        out = e_x / e_x.sum(axis=1, keepdims=True)
        return out

    def _cross_entropy(self, output, y_target):
        return -np.sum(np.log(output + 1e-8) * (y_target), axis=1)

    def _cost(self, cross_entropy):
        L2_term = self.l2 * np.sum(self.w_**2)
        cross_entropy = cross_entropy + L2_term
        return 0.5 * np.mean(cross_entropy)

    def _to_classlabels(self, z):
        return z.argmax(axis=1)

    def _forward(self, X):
        z = self._net_input(X)
        a = self._softmax_activation(z)
        return a

    def _backward(self, X, y_true, y_probas):
        grad_loss_wrt_out = y_true - y_probas
        # gradient -> n_features x n_classes
        grad_loss_wrt_w = -np.dot(X.T, grad_loss_wrt_out)
        grad_loss_wrt_b = -np.sum(grad_loss_wrt_out, axis=0)
        return grad_loss_wrt_w, grad_loss_wrt_b

    def _fit(self, X, y, init_params=True):
        if init_params:
            if self.n_classes is None:
                self.n_classes = np.max(y) + 1
            self._n_features = X.shape[1]

            self.b_, self.w_ = self._init_params(
                weights_shape=(self._n_features, self.n_classes),
                bias_shape=(self.n_classes,),
                random_seed=self.random_seed,
            )
            self.cost_ = []

        self.init_time_ = time()
        rgen = np.random.RandomState(self.random_seed)
        for i in range(self.epochs):
            for idx in self._yield_minibatches_idx(
                rgen=rgen, n_batches=self.minibatches, data_ary=y, shuffle=True
            ):
                # net_input, softmax and diff -> n_samples x n_classes:
                y_probas = self._forward(X[idx])

                # w_ -> n_feat x n_classes
                # b_  -> n_classes
                grad_loss_wrt_w, grad_loss_wrt_b = self._backward(
                    # X[idx], y_true=y_enc[idx], y_probas=y_probas
                    X[idx], y_true=y[idx], y_probas=y_probas
                )

                # update in opp. direction of the cost gradient
                l2_reg = self.l2 * self.w_
                self.w_ += self.eta * (-grad_loss_wrt_w - l2_reg)
                self.b_ += self.eta * -grad_loss_wrt_b

            # compute cost of the whole epoch
            y_probas = self._forward(X)
            cross_ent = self._cross_entropy(output=y_probas, y_target=y)
            cost = self._cost(cross_ent)
            self.cost_.append(cost)

            if self.print_progress:
                self._print_progress(iteration=i + 1, n_iter=self.epochs, cost=cost)

        return self

    def predict_proba(self, X):
        """Predict class probabilities of X from the net input.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        Class probabilties : array-like, shape= [n_samples, n_classes]

        """
        return self._forward(X)

    def _predict(self, X):
        probas = self.predict_proba(X)
        return self._to_classlabels(probas)
