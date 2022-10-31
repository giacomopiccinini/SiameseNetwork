from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D


def sister_network(input_shape, latent_dim):

    # Specify the inputs for the feature extractor network
    inputs = Input(input_shape)

    # First set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(64, (2, 2), padding="same", activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    # Second set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    # Prepare the final outputs
    pooled_output = GlobalAveragePooling2D()(x)
    outputs = Dense(latent_dim)(pooled_output)

    # Build the model
    sister_net = Model(inputs, outputs)

    return sister_net
