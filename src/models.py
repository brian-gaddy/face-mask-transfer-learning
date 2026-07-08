"""Transfer-learning model builders for three-class face-mask classification."""

from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetB0, ResNet50, VGG16
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D


BACKBONES = {
    "efficientnetb0": EfficientNetB0,
    "resnet50": ResNet50,
    "vgg16": VGG16,
}


def build_transfer_model(backbone_name: str, input_shape=(128, 128, 3),
                         num_classes: int = 3, dropout_rate: float = 0.2) -> Model:
    """Build a frozen-backbone transfer-learning classifier."""
    key = backbone_name.lower()
    if key not in BACKBONES:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    backbone = BACKBONES[key](
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    backbone.trainable = False

    x = GlobalAveragePooling2D()(backbone.output)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=backbone.input, outputs=outputs, name=f"{key}_mask_classifier")
