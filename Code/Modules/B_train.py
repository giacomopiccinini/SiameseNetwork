import os
import tensorflow as tf

from Code.Networks.siamese import SiameseNetwork
from Code.Modules.A_load_data import load_data


def train():

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # TensorFlow routine
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    session = tf.compat.v1.Session(config=config)
    gpus = tf.config.experimental.list_physical_devices("GPU")

    if gpus:

        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")

        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # Load configuration
    train_configuration = load("train_configuration.yaml")
    globals().update(train_configuration)

    # Prepare network
    siamese = SiameseNetwork()
    siamese.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    siamese.summary()

    # Load data
    (pairs_train, labels_train), (pairs_val, labels_val) = load_data()

    # Define early stopping
    earlystopping = EarlyStopping(patience=5)

    # Define checkpoints
    checkpoint = ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
        save_freq="epoch",
    )

    # Train model
    history = siamese.fit(
        [pairs_train[:, 0], pairs_train[:, 1]],
        labels_train[:],
        validation_data=([pairs_val[:, 0], pairs_val[:, 1]], labels_val[:]),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint, earlystopping],
    )

    # Define tensorboard
    # tensorboard_callback = TensorBoard(log_dir=logs_ID_directory)

    # Fit autoencoder
    # autoencoder.fit(train_data, epochs=epochs, validation_data=validation_data,
    # callbacks=[checkpoint, earlystopping, tensorboard_callback])

    # Clear session (autoencoder will be reloaded later on)
    # del autoencoder
    # K.clear_session()
    # gc.collect()

    # Save trained model
    # siamese.save(MODEL_PATH)

    return
