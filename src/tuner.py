from itertools import product
from sklearn.metrics import mean_squared_error
from tensorflow.keras.optimizers import Adam
from .architectures import WildfireArchitectures


class ModelCalibrator:
    """Grid search engine for calibrating models to the configurations
    reported in the manuscript (Section 2.5, Table 2).

    Selection follows the manuscript: for each model-target pair the
    configuration with the lowest validation error is retained. The searched
    ranges are learning rate, dropout rate, batch size and number of epochs;
    ReLU hidden activations, the linear output unit, the Adam optimizer and the
    mean-squared-error objective are held fixed across the deep-learning models.
    """

    # Deep-learning models that consume the ordered feature vector as a
    # pseudo-sequence of shape (input_shape, 1).
    SEQUENCE_MODELS = ['BiRNN', 'ED-BiRNN', 'LSTM', 'RNN']

    def __init__(self, model_type, input_shape):
        self.model_type = model_type
        self.input_shape = input_shape
        self.grid = {
            'lr': [0.01, 0.001, 0.0005, 0.0002],
            'dropout': [0.1, 0.2, 0.3, 0.4],
            'batch': [16, 32, 64],
            'epochs': [50, 60, 70, 80, 90, 100],
        }

    def _build_model(self, dropout):
        """Instantiate the requested architecture by name."""
        if self.model_type in self.SEQUENCE_MODELS:
            model_func = getattr(
                WildfireArchitectures,
                f"get_{self.model_type.lower().replace('-', '_')}"
            )
            return model_func(self.input_shape, dropout)
        # Fully connected baseline (DNN)
        return WildfireArchitectures.get_dnn(self.input_shape, dropout)

    def _reshape(self, X):
        """Sequence models expect (n_samples, input_shape, 1); DNN expects 2-D."""
        if self.model_type in self.SEQUENCE_MODELS:
            return X.values.reshape(-1, self.input_shape, 1)
        return X.values

    def run_calibration(self, X_t, y_t, X_v, y_v):
        """Grid search; returns the configuration with the lowest validation MSE."""
        best_cfg, best_err = None, float('inf')
        configs = [dict(zip(self.grid.keys(), v)) for v in product(*self.grid.values())]

        X_train_final = self._reshape(X_t)
        X_val_final = self._reshape(X_v)

        for cfg in configs:
            model = self._build_model(cfg['dropout'])
            model.compile(optimizer=Adam(cfg['lr']), loss='mse')
            model.fit(
                X_train_final, y_t,
                validation_data=(X_val_final, y_v),
                epochs=cfg['epochs'], batch_size=cfg['batch'], verbose=0
            )

            preds = model.predict(X_val_final, verbose=0).reshape(-1)
            val_err = mean_squared_error(y_v, preds)
            if val_err < best_err:
                best_err, best_cfg = val_err, cfg

        return best_cfg