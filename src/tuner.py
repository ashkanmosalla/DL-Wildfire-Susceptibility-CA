from itertools import product
from sklearn.metrics import r2_score
from tensorflow.keras.optimizers import Adam
from .architectures import WildfireArchitectures

class ModelCalibrator:
    """Grid search engine for calibrating models to match manuscript metrics."""
    def __init__(self, model_type, input_shape):
        self.model_type = model_type
        self.input_shape = input_shape
        self.grid = {'lr': [0.01, 0.001, 0.0005, 0.0002], 'dropout': [0.1, 0.2, 0.3, 0,4], 'batch': [16, 32, 64], 'epochs': [50, 60, 70, 80, 90, 100]}

    def run_calibration(self, X_t, y_t, X_v, y_v):
        best_cfg, best_r2 = None, -1
        configs = [dict(zip(self.grid.keys(), v)) for v in product(*self.grid.values())]
        
        for cfg in configs:
            if self.model_type in ['BiRNN', 'ED-BiRNN', 'LSTM']:
                X_train_final = X_t.values.reshape(-1, self.input_shape, 1)
                X_val_final = X_v.values.reshape(-1, self.input_shape, 1)
                model_func = getattr(WildfireArchitectures, f"get_{self.model_type.lower().replace('-','_')}")
                model = model_func(self.input_shape, cfg['dropout'])
            else:
                X_train_final, X_val_final = X_t.values, X_v.values
                model = WildfireArchitectures.get_dnn(self.input_shape, cfg['dropout'])

            model.compile(optimizer=Adam(cfg['lr']), loss='mse')
            model.fit(X_train_final, y_t, validation_data=(X_val_final, y_v), epochs=cfg['epochs'], batch_size=cfg['batch'], verbose=0)
            
            score = r2_score(y_v, model.predict(X_val_final).reshape(-1))
            if score > best_r2:
                best_r2, best_cfg = score, cfg
        return best_cfg