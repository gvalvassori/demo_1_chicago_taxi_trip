
from typing import List, Text
from absl import logging
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
import keras_tuner
from tfx import v1 as tfx
from tfx_bsl.public import tfxio

# Specify features that we will use.
_FEATURE_KEYS = [
    "trip_seconds", "trip_miles", "work_day", "work_hour",
    "trip_speed", "pickup_community_area", "dropoff_community_area"
]

_LABEL_KEY = "trip_total"

_TRAIN_BATCH_SIZE = 64
_EVAL_BATCH_SIZE = 32

# NEW: TFX Transform will call this function.
def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.

    Args:
    inputs: map from feature keys to raw not-yet-transformed features.

    Returns:
    Map from string feature key to transformed feature.
    """
    outputs = {}

    # Uses features defined in _FEATURE_KEYS only.
    for key in _FEATURE_KEYS:
        if key in ["work_day", "work_hour"]:
            # Convert work_day and work_hour to float32
            outputs[key] = tf.cast(inputs[key], tf.float32)
        # elif key in ["pickup_community_area", "dropoff_community_area"]:
        #     # Keep these as int64, but normalize them
        #     outputs[key] = tft.scale_to_z_score(tf.cast(inputs[key], tf.float32))
        else:
            # For other features, apply z-score normalization
            outputs[key] = tft.scale_to_z_score(inputs[key])
    
    outputs[_LABEL_KEY] = inputs[_LABEL_KEY]

    return outputs


# NEW: This function will apply the same transform operation to training data
#      and serving requests.
# def _apply_preprocessing(raw_features, tft_layer):
#     transformed_features = tft_layer(raw_features)
#     if _LABEL_KEY in raw_features:
#         transformed_label = transformed_features.pop(_LABEL_KEY)
#         return transformed_features, transformed_label
#     else:
#         return transformed_features, None


# NEW: This function will create a handler function which gets a serialized
#      tf.example, preprocess and run an inference with it.
def _get_serve_tf_examples_fn(model, tf_transform_output):
    # We must save the tft_layer to the model to ensure its assets are kept and
    # tracked.
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
    ])
    def serve_tf_examples_fn(serialized_tf_examples):
                # Expected input is a string which is serialized tf.Example format.
        feature_spec = tf_transform_output.raw_feature_spec()
        # Remove label feature since these will not be present at serving time.
        feature_spec.pop(_LABEL_KEY)

        parsed_features = tf.io.parse_example(serialized_tf_examples,
                                              feature_spec)

        # Preprocess parsed input with transform operation defined in
        # preprocessing_fn().
        transformed_features = model.tft_layer(parsed_features)
        # Run inference with ML model.
        outputs = model(transformed_features)
        
        return {'predictions': outputs}        

    return serve_tf_examples_fn

def _get_serve_raw_examples_fn(model, tf_transform_output):
    # This function will create the custom serving signature
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_raw_examples_fn(trip_seconds, 
                              trip_miles, 
                              work_day, 
                              work_hour,
                              trip_speed, 
                              pickup_community_area, 
                              dropoff_community_area):
        features = {
            "trip_seconds": trip_seconds, 
            "trip_miles": trip_miles, 
            "work_day": tf.cast(work_day, tf.float32), 
            "work_hour": tf.cast(work_hour, tf.float32),
            "trip_speed": trip_speed, 
            "pickup_community_area": tf.cast(pickup_community_area, tf.float32), 
            "dropoff_community_area": tf.cast(dropoff_community_area, tf.float32)
        }
              
        transformed_features = model.tft_layer(features)
        outputs = model(transformed_features)
        
        return {'predictions': outputs}

    return serve_raw_examples_fn



def _input_fn(file_pattern: List[Text],
              data_accessor: tfx.components.DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 64) -> tf.data.Dataset:
    """Generates features and label for tuning/training.

    Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

    Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
    """
   
    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
            batch_size=batch_size, label_key=_LABEL_KEY),
        tf_transform_output.transformed_metadata.schema)
    dataset=dataset.repeat()
    return dataset.prefetch(tf.data.AUTOTUNE)


def _build_keras_model(hparams: keras_tuner.HyperParameters) -> tf.keras.Model:
    """Builds a Keras model for taxi trip fare prediction with hyperparameter tuning.

    Args:
        hparams: (Hyperparameter) Hyperparameter object for tuning.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """

    units_1 = hparams.get('units_1')
    units_2 = hparams.get('units_2')
    learning_rate = hparams.get('learning_rate')
    activation = "relu"
    dropout_rate = hparams.get('dropout_rate')
    l2_regularization = hparams.get('l2_regularization')

    # Create inputs for each feature
    inputs = {
        feature: tf.keras.Input(shape=(1,), name=feature, dtype=tf.float32)
        for feature in _FEATURE_KEYS
    }

    # Concatenate all inputs
    concat = tf.keras.layers.Concatenate()(list(inputs.values()))

    # Add first hidden layer with dropout and L2 regularization
    hidden_layer_1 = tf.keras.layers.Dense(
        units_1, 
        activation=activation,
        kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)
    )(concat)
    hidden_layer_1 = tf.keras.layers.Dropout(dropout_rate)(hidden_layer_1)

    # Add second hidden layer with dropout and L2 regularization
    hidden_layer_2 = tf.keras.layers.Dense(
        units_2, 
        activation=activation,
        kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)
    )(hidden_layer_1)
    hidden_layer_2 = tf.keras.layers.Dropout(dropout_rate)(hidden_layer_2)

    # Add output layer
    output = tf.keras.layers.Dense(1)(hidden_layer_2)

    # Create the functional model
    model = tf.keras.Model(inputs=inputs, outputs=output)

    # Compile the model using Adam optimizer with tuned learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='huber_loss', metrics=['mean_absolute_error'])
    
    model.summary(print_fn=logging.info)

    return model

def _get_hyperparameters() -> keras_tuner.HyperParameters:
    """Defines the search space for hyperparameters."""
    hp = keras_tuner.HyperParameters()
    hp.Int('units_1', min_value=32, max_value=512, step=32)
    hp.Int('units_2', min_value=16, max_value=256, step=16)
    hp.Float('learning_rate', min_value=1e-5, max_value=1e-1, sampling='log')
    hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    hp.Float('l2_regularization', min_value=1e-6, max_value=1e-1, sampling='log')
    return hp

# TFX Tuner will call this function.
def tuner_fn(fn_args: tfx.components.FnArgs) -> tfx.components.TunerFnResult:
    """Build the tuner using the KerasTuner API.

    Args:
    fn_args: Holds args as name/value pairs.
      - working_dir: working dir for tuning.
      - train_files: List of file paths containing training tf.Example data.
      - eval_files: List of file paths containing eval tf.Example data.
      - train_steps: number of train steps.
      - eval_steps: number of eval steps.
      - schema_path: optional schema of the input data.
      - transform_graph_path: optional transform graph produced by TFT.

    Returns:
    A namedtuple contains the following:
      - tuner: A BaseTuner that will be used for tuning.
      - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                    model , e.g., the training and validation dataset. Required
                    args depend on the above tuner's implementation.
    """
    # RandomSearch is a subclass of keras_tuner.Tuner which inherits from
    # BaseTuner.
    tuner = keras_tuner.RandomSearch(
      _build_keras_model,
      max_trials=20, 
      hyperparameters=_get_hyperparameters(),
      allow_new_entries=False,
      objective=keras_tuner.Objective('val_mean_absolute_error', 'min'),
      directory=fn_args.working_dir,
      project_name='demo_1_tuning')

    transform_graph = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = _input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      transform_graph,
      _TRAIN_BATCH_SIZE)

    eval_dataset = _input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      transform_graph,
      _EVAL_BATCH_SIZE)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, mode='auto')

    return tfx.components.TunerFnResult(
      tuner=tuner,
      fit_kwargs={
          'x': train_dataset,
          'validation_data': eval_dataset,
          'steps_per_epoch': fn_args.train_steps,
          'validation_steps': fn_args.eval_steps,
          'epochs': 16, 
          'callbacks':[early_stopping]
      })


# TFX Trainer will call this function.
def run_fn(fn_args: tfx.components.FnArgs):
    """Train the model based on given args.

    Args:
    fn_args: Holds args used to train the model as name/value pairs.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=_TRAIN_BATCH_SIZE)
    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=_EVAL_BATCH_SIZE)
    
    if fn_args.hyperparameters:
        hparams = keras_tuner.HyperParameters.from_config(fn_args.hyperparameters)
    else:
        # This is a shown case when hyperparameters is decided and Tuner is removed
        # from the pipeline. User can also inline the hyperparameters directly in
        # _build_keras_model.
        hparams = _get_hyperparameters()

    model = _build_keras_model(hparams)
    
    """
    # Write logs to path
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=fn_args.model_run_dir, update_freq='epoch')
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, mode='auto')
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=100,
        callbacks=[early_stopping])
    
    # NEW: Save a computation graph including transform layer.
    
    signatures = {
        'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output),
        'serving_raw_examples': _get_serve_raw_examples_fn(model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(shape=[None,1], dtype=tf.float32, name="trip_seconds"),
            tf.TensorSpec(shape=[None,1], dtype=tf.float32, name="trip_miles"),
            tf.TensorSpec(shape=[None,1], dtype=tf.float32, name="work_day"),
            tf.TensorSpec(shape=[None,1], dtype=tf.float32, name="work_hour"),
            tf.TensorSpec(shape=[None,1], dtype=tf.float32, name="trip_speed"),
            tf.TensorSpec(shape=[None,1], dtype=tf.float32, name="pickup_community_area"),
            tf.TensorSpec(shape=[None,1], dtype=tf.float32, name="dropoff_community_area"),
        )}

    # The result of the training should be saved in `fn_args.serving_model_dir`
    # directory.
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
