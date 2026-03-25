from tensorflow.keras import layers, models


def build_cnn(input_shape=(128, 128, 1)):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            # Block 1
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            # Block 2
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            # Block 3
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            # Classifier
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),  # 0=forged, 1=genuine
        ]
    )
    return model
