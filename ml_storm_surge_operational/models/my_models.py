import tensorflow as tf
import keras

class KerasModels():
    "Define models and train them "
    
    def __init__(self, x_train_norm, y_train_norm, station:str, horizon:int, member:int=None, verbose=0):
        self.x_train_norm = x_train_norm
        self.y_train_norm = y_train_norm
        self.station = station
        self.horizon = horizon
        self.member = member
        self.verbose = verbose
        
        self.setup()
        
        physical_devices = tf.config.list_physical_devices('GPU') 
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        
    def setup(self):
        self.OUT_STEPS = self.horizon
        self.num_features = 1
        self.batch_size =32
        self.epochs=500

        
    def hindcast_keras_model(self):
        data_dir = (
            '/lustre/storeB/project/IT/geout/machine-ocean/workspace/paulinast'
            + '/storm_surge_results/ml/best_models/hindcast'
            )
        path_to_best_model = (
            data_dir
            + '/best_model_hindcast_' 
            + self.station 
            + '_t' + str(self.horizon) 
            + '.h5'
            )
        multi_dense_model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(rate=0.3),
            tf.keras.layers.Dense(self.OUT_STEPS*self.num_features)    
        ])
        
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                path_to_best_model, save_best_only=True, monitor="val_loss"
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
            ),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=self.verbose),
        ]
        
        multi_dense_model.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.MeanSquaredError(),
            metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.MeanSquaredError()],
        )
        
        
        history = multi_dense_model.fit(
            self.x_train_norm,
            self.y_train_norm,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            validation_split=0.3,
            verbose=self.verbose,
        )
        
        return multi_dense_model, history

        
    def operational_keras_model(self):
        
        history_list = []
        model_list = []
        
        data_dir = (
            '/lustre/storeB/project/IT/geout/machine-ocean/workspace/paulinast'
            + '/storm_surge_results/ml/best_models/operational'
            )
        
        for t in range(self.horizon):
            path_to_best_model = (
                data_dir 
                + '/best_model_operational_' 
                + self.station 
                + '_t' + str(t) 
                + '_m' + str(self.member)
                + '.h5'
                )
            
            multi_dense_model = tf.keras.Sequential([
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(rate=0.3),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(rate=0.4),
                tf.keras.layers.Dense(1)    
            ])
        
            callbacks = [
                keras.callbacks.ModelCheckpoint(
                    path_to_best_model, save_best_only=True, monitor="val_loss"
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
                ),
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=self.verbose),
            ]
        
            multi_dense_model.compile(
                optimizer=tf.optimizers.Adam(),
                loss=tf.losses.MeanSquaredError(),
                metrics=[tf.metrics.MeanAbsoluteError()],#, tf.metrics.MeanSquaredError()],
            )
        
        
            history = multi_dense_model.fit(
                self.x_train_norm,
                self.y_train_norm[:, t],
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=callbacks,
                validation_split=0.3,
                verbose=self.verbose,
            )
            
            history_list.append(history)
            model_list.append(multi_dense_model)
            del history
            del multi_dense_model
            
        return model_list, history_list
