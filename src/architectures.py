import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, SimpleRNN, LSTM, Input, RepeatVector, TimeDistributed, Flatten
from tensorflow.keras.models import Sequential, Model
from catboost import CatBoostRegressor

class WildfireArchitectures:
    """
    Implementation of the 6 core models as defined in the study methodology (Fig 3).
    Includes Deep Learning architectures and Gradient Boosting (CatBoost).
    """
    
    @staticmethod
    def get_birnn(input_shape, dropout=0.1):
        """1. Bidirectional RNN - Top Performer in the Study."""
        model = Sequential([
            Input(shape=(input_shape, 1)),
            Bidirectional(SimpleRNN(128, activation='relu', return_sequences=True)),
            Dropout(dropout),
            Bidirectional(SimpleRNN(64, activation='relu')),
            Dropout(dropout),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        return model

    @staticmethod
    def get_ed_birnn(input_shape, dropout=0.1):
        """2. Encoder-Decoder BiRNN - Advanced architecture for latent mapping."""
        inputs = Input(shape=(input_shape, 1))
        # Encoder
        encoder = Bidirectional(SimpleRNN(64, activation='relu'))(inputs)
        encoder = Dropout(dropout)(encoder)
        # Latent Space (Bottleneck)
        latent = RepeatVector(input_shape)(encoder)
        # Decoder
        decoder = Bidirectional(SimpleRNN(64, activation='relu', return_sequences=True))(latent)
        outputs = TimeDistributed(Dense(1))(decoder)
        final_output = Flatten()(outputs)
        final_output = Dense(1, activation='linear')(final_output)
        return Model(inputs=inputs, outputs=final_output)

    @staticmethod
    def get_lstm(input_shape, dropout=0.2):
        """3. Long Short-Term Memory (LSTM) Network."""
        model = Sequential([
            Input(shape=(input_shape, 1)),
            LSTM(128, activation='relu', return_sequences=True),
            Dropout(dropout),
            LSTM(64, activation='relu'),
            Dropout(dropout),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        return model

    @staticmethod
    def get_rnn(input_shape, dropout=0.2):
        """4. Simple Recurrent Neural Network (RNN)."""
        model = Sequential([
            Input(shape=(input_shape, 1)),
            SimpleRNN(128, activation='relu', return_sequences=True),
            Dropout(dropout),
            SimpleRNN(64, activation='relu'),
            Dropout(dropout),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        return model

    @staticmethod
    def get_dnn(input_shape, dropout=0.2):
        """5. Deep Neural Network (DNN) / MLP baseline."""
        model = Sequential([
            Input(shape=(input_shape,)),
            Dense(128, activation='relu'), 
            Dropout(dropout),
            Dense(64, activation='relu'),
            Dropout(dropout),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        return model

    @staticmethod
    def get_catboost(iterations=500, learning_rate=0.05, depth=6):
        """6. CatBoost Regressor - Gradient Boosting baseline."""
        model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function='RMSE',
            verbose=0,
            random_seed=42
        )
        return model