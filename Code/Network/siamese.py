from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda

from Code.Network.sister import SisterNetwork
from Code.Metrics.euclidean import euclidean_distance


def SiameseNetwork(shape_y: int, shape_x: int, colours: int, latent: int):

    """Siamese network implementation based on a feature extractor defined as the
    Sister Network"""

    # Construct shape of input images
    image_shape = (shape_y, shape_x, colours)

    # Define the inputs of the network
    image_1 = Input(shape=image_shape)
    image_2 = Input(shape=image_shape)

    # Define the feature extractor (i.e. the sister network)
    feature_extractor = SisterNetwork(image_shape, latent)

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
