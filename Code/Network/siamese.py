from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda

from Code.Utilities.loading_utils import load
from Code.Networks.Architectures.architectures import sister_network
from Code.Networks.Distances.euclidean import euclidean_distance


def SiameseNetwork():

    # Import configuration
    train_configuration = load("train_configuration.yaml")
    image_configuration = load("image_configuration.yaml")
    siamese_configuration = load("siamese_configuration.yaml")

    globals().update(train_configuration)
    globals().update(image_configuration)
    globals().update(siamese_configuration)

    # Construct shape of input images
    IMG_SHAPE = (SHAPE_Y, SHAPE_X, COLOURS)

    # Define the inputs of the network
    image_1 = Input(shape=IMG_SHAPE)
    image_2 = Input(shape=IMG_SHAPE)

    # Define the feature extractor (i.e. the sister network)
    feature_extractor = sister_network(IMG_SHAPE, LATENT)

    # Extract features from the two input images
    features_1 = feature_extractor(image_1)
    features_2 = feature_extractor(image_2)

    # Pass the features into a distance layer
    distance = Lambda(euclidean_distance)([features_1, features_2])

    # Pass the distance in a dense layer with sigmoid activation
    outputs = Dense(1, activation="sigmoid")(distance)

    # Construct the model
    model = Model(inputs=[image_1, image_2], outputs=outputs)

    return model
