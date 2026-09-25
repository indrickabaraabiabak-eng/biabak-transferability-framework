"""Learners and geostatistical predictors, each fitted inside a training fold.

Every preprocessing step (imputation, rare-level pooling, one-hot encoding,
standardisation) lives inside a scikit-learn Pipeline, so it is re-estimated on
each training fold. Hyperparameters are prespecified (Table in SI) except for
the Elastic Net penalty, tuned by inner spatial folds built on the training
coordinates only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier,
                              GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from . import variography as V

CONTINUOUS = ["elevation_m", "slope_deg", "profile_curvature", "plan_curvature", "lineament_density",
              "rainfall_mm_yr", "aet_mm_yr", "ndvi", "ndbi", "dist_fault_km"]
CATEGORICAL = ["lithology_code", "landcover_code", "soil_dominant"]
RARE_MIN = 5


class RarePooler(BaseEstimator, TransformerMixin):
    """Levels seen fewer than RARE_MIN times in the training fold become 'other'."""
    def __init__(self, min_count=RARE_MIN):
        self.min_count = min_count

    def fit(self, X, y=None):
        X = pd.DataFrame(X).astype(str)
        self.keep_ = {c: set(X[c].value_counts()[lambda s: s >= self.min_count].index) for c in X.columns}
        return self

    def transform(self, X):
        X = pd.DataFrame(X).astype(str).copy()
        for c in X.columns:
            X[c] = np.where(X[c].isin(self.keep_[c]), X[c], "other")
        return X


def preprocessor():
    cont = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("pool", RarePooler()),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("cont", cont, CONTINUOUS), ("cat", cat, CATEGORICAL)])


class SpatialENet(BaseEstimator):
    """Elastic Net whose penalty is chosen by inner spatial folds (k-means on training coordinates)."""
    def __init__(self, xy=None, inner_k=5, seed=0):
        self.xy, self.inner_k, self.seed = xy, inner_k, seed

    def fit(self, X, y, xy=None):
        xy = self.xy if xy is None else xy
        groups = KMeans(self.inner_k, n_init=5, random_state=self.seed).fit(xy).labels_
        cv = list(GroupKFold(self.inner_k).split(X, y, groups))
        self.m_ = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], n_alphas=40, cv=cv, max_iter=20000).fit(X, y)
        return self

    def predict(self, X):
        return self.m_.predict(X)


def regressors(seed):
    return {
        "random_forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.33, random_state=seed),
        "extra_trees": ExtraTreesRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.33, random_state=seed),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=2,
                                                       subsample=0.8, random_state=seed),
        "svr_rbf": SVR(C=1.0, epsilon=0.1, gamma="scale"),
        "mlp": MLPRegressor(hidden_layer_sizes=(16,), alpha=1.0, solver="lbfgs", max_iter=3000, random_state=seed),
    }


def classifiers(seed):
    return {
        "logistic": LogisticRegression(C=0.1, max_iter=5000),
        "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=3, max_features=0.33, random_state=seed),
        "extra_trees": ExtraTreesClassifier(n_estimators=200, min_samples_leaf=3, max_features=0.33, random_state=seed),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2,
                                                        subsample=0.8, random_state=seed),
        "mlp": MLPClassifier(hidden_layer_sizes=(16,), alpha=1.0, solver="lbfgs", max_iter=3000, random_state=seed),
        "knn": KNeighborsClassifier(n_neighbors=15, weights="distance"),
    }


HYPERPARAMETERS = {
    "Elastic Net": "l1_ratio in {0.1, 0.5, 0.9}, 40 penalties, chosen by 5 inner spatial folds (k-means on training coordinates)",
    "Random forest (reg.)": "200 trees, min_samples_leaf 5, max_features 0.33",
    "Extra trees (reg.)": "200 trees, min_samples_leaf 5, max_features 0.33",
    "Gradient boosting (reg.)": "200 trees, learning rate 0.05, depth 2, subsample 0.8",
    "SVR (RBF)": "C 1.0, epsilon 0.1, gamma 'scale'",
    "MLP (reg.)": "one hidden layer of 16 units, L2 penalty 1.0, L-BFGS",
    "Logistic regression": "L2, C 0.1",
    "Random forest (clf.)": "200 trees, min_samples_leaf 3, max_features 0.33",
    "Extra trees (clf.)": "200 trees, min_samples_leaf 3, max_features 0.33",
    "Gradient boosting (clf.)": "150 trees, learning rate 0.05, depth 2, subsample 0.8",
    "MLP (clf.)": "one hidden layer of 16 units, L2 penalty 1.0, L-BFGS",
    "k-nearest neighbours": "15 neighbours in standardised covariate space, distance weights",
    "Ordinary kriging": "spherical model refitted on each training fold (Cressie weights, 2 km lags to 70 km); global neighbourhood",
    "Inverse distance": "power 2, all training points",
    "Preprocessing": "median imputation and z-scores for continuous covariates; mode imputation, pooling of levels seen < 5 times, one-hot encoding for categorical covariates; all fitted on the training fold",
}


# ---------------------------------------------------------------------------
# Geostatistical predictors refitted per fold
# ---------------------------------------------------------------------------

def fit_variogram_fold(xy, z, lag=2.0, hmax=70.0, min_pairs=30):
    i, j, h, _ = V.pairs(xy[:, 0], xy[:, 1])
    edges = np.arange(0, hmax + lag, lag)
    hm, g, n, keep = V.experimental(z, i, j, h, edges, min_pairs=min_pairs)
    var = float(np.var(z, ddof=1))
    if keep.sum() < 3 or var <= 0:
        return dict(model="nugget", c0=max(var, 1e-9), c=0.0, a=1.0)
    return V.fit_model("spherical", hm[keep], g[keep], n[keep], "cressie", (0.5, 150.0), var)


def ordinary_kriging(xy_tr, z_tr, xy_te, fit):
    n = len(z_tr)
    G = V.model_gamma(fit["model"], cdist(xy_tr, xy_tr), fit["c0"], fit["c"], fit["a"])
    A = np.ones((n + 1, n + 1)); A[:n, :n] = G; A[n, n] = 0.0
    g0 = V.model_gamma(fit["model"], cdist(xy_tr, xy_te), fit["c0"], fit["c"], fit["a"])
    B = np.vstack([g0, np.ones((1, len(xy_te)))])
    sol = np.linalg.lstsq(A, B, rcond=None)[0]
    lam = sol[:n]
    return lam.T @ z_tr


def idw(xy_tr, z_tr, xy_te, power=2.0):
    d = np.maximum(cdist(xy_te, xy_tr), 1e-3)
    w = d ** (-power)
    return (w @ z_tr) / w.sum(1)
