import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer, normalize
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.estimator_checks import check_estimator


class NBTransformer(BaseEstimator, TransformerMixin):
    """
    A class for transforming features into multinomial naive bayes ratios
    Inspired by https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf
    """
    def __init__(self, alpha=0.0, binarize=False):
        """
        :param alpha: float, optional
            Smoothing parameter for naive bayes ratio (default: 0.0)
        :param binarize: bool, optional
            A flag used to replace all non-zero values with 1 (default: False)
        """
        super(NBTransformer, self).__init__()

        self.alpha = alpha
        self.binarize = binarize

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("NB Transformer requires labels")
        if X.min() < 0:
            raise ValueError("NB Transformer only works with non-negative features")

        self._compute_ratios(X, y)
        return self

    def transform(self, X, y=None):
        result = np.zeros((X.shape[0], X.shape[1], self.ratios_.shape[0]))

        for i in range(self.ratios_.shape[0]):
            result[:, :, i] = X.multiply(self.ratios_[i]).todense()

        return result

    def _compute_ratios(self, X, y):
        if self.binarize:
            X = (X > 0).astype(np.float)

        label_encoder = LabelBinarizer()
        y = label_encoder.fit_transform(y)

        n_classes = label_encoder.classes_.shape[0]

        self.ratios_ = np.full((n_classes, X.shape[1]), self.alpha)
        self.ratios_ += safe_sparse_dot(y.T, X)
        normalize(self.ratios_, norm="l1", copy=False)

        self.ratios_ = np.log(self.ratios_) - np.log(1 - self.ratios_)

        self.ratios_ = sparse.csr_matrix(self.ratios_)

    def _more_tags(self):
        return {
            "requires_positive_X": True,
            "requires_y": True,
        }
