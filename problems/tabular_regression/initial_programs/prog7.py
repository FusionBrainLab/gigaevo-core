import numpy as np
import optuna
import xgboost as xgb


class Model:
    def fit_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_query: np.ndarray,
    ) -> np.ndarray:
        np.random.seed(0)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def make_model(params: dict) -> xgb.XGBRegressor:
            return xgb.XGBRegressor(
                n_estimators=2000,
                tree_method="hist",
                n_jobs=-1,
                random_state=0,
                verbosity=0,
                early_stopping_rounds=50,
                **params,
            )

        def objective(trial: optuna.Trial) -> float:
            params = dict(
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                min_child_weight=trial.suggest_float(
                    "min_child_weight", 1e-2, 50.0, log=True
                ),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            )
            m = make_model(params)
            m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            return float(np.sqrt(np.mean((m.predict(X_val) - y_val) ** 2)))

        sampler = optuna.samplers.TPESampler(seed=0)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=30, show_progress_bar=False)

        m = make_model(study.best_params)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return m.predict(X_query)


def entrypoint() -> type:
    return Model
