"""Fixed-budget, train-only power models for the V2 benchmark."""

from time import perf_counter

RATED_KW = 2050
CONFIG = {"min_bin": 5, "bin_width_ms": 0.5, "spline_knots": 8,
          "ridge_alpha": 10, "boost_iterations": 60, "boost_leaves": 15, "seed": 42}


def bin_index(wind):
    import numpy as np

    return np.floor(2 * wind + 0.5).astype(int)


def fit_models(features, power):
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer, StandardScaler

    start = perf_counter()
    wind = features[:, 0]
    bins = bin_index(wind)
    indices, counts = np.unique(bins, return_counts=True)
    sums = np.bincount(bins, weights=power)
    indices = indices[counts >= CONFIG["min_bin"]]
    means = dict(zip(indices, sums[indices] / np.bincount(bins)[indices], strict=True))

    def support(x):
        return np.isin(bin_index(x[:, 0]), indices)

    def binned(x):
        return np.array([means.get(k, np.nan) for k in bin_index(x[:, 0])])

    def piecewise(x):
        v = x[:, 0]
        k = bin_index(v)
        mean = binned(x)
        fraction = 2 * (v - k / 2)
        neighbors = np.array([means.get(i + np.sign(f), m)
                              for i, f, m in zip(k, fraction, mean, strict=True)])
        return mean + np.abs(fraction) * (neighbors - mean)

    def booster():
        return HistGradientBoostingRegressor(
            max_iter=CONFIG["boost_iterations"], max_leaf_nodes=CONFIG["boost_leaves"],
            early_stopping=False, random_state=CONFIG["seed"],
        )

    median = float(np.median(power))
    spline = make_pipeline(SplineTransformer(n_knots=CONFIG["spline_knots"]),
                           StandardScaler(), Ridge(alpha=CONFIG["ridge_alpha"]))
    spline.fit(features, power)
    tree = booster().fit(features, power)
    shape = IsotonicRegression(y_min=0, y_max=RATED_KW, out_of_bounds="clip").fit(wind, power)

    def reference(x):
        return shape.predict(x[:, 0])

    residual = booster().fit(features, power - reference(features))

    def hybrid(x):
        return reference(x) + residual.predict(x)

    models = {
        "median": lambda x: np.full(len(x), median),
        "binned": binned, "piecewise": piecewise,
        "spline": spline.predict, "boosting": tree.predict, "shape": reference,
        "hybrid": hybrid, "hybrid_bounded": lambda x: np.clip(hybrid(x), 0, RATED_KW),
    }
    return models, support, {
        "fit_seconds": perf_counter() - start, "supported_bins": indices.tolist(),
        "shape_wind_ms": shape.X_thresholds_.tolist(), "shape_power_kw": shape.y_thresholds_.tolist(),
        "boost_iterations": tree.n_iter_, "hybrid_iterations": residual.n_iter_,
    }
