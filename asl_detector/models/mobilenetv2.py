import tensorflow as tf

class ASLMobilenetv2():
    def __init__(self, num_classes: int = 29, dropout_rate: float = 0.2):
        ## Base model
        self.base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet',            
        )
        self.base_model.trainable = False
    def build_classifier(self, num_classes: int = 29):
        ## TODO build the classifier head
        ...
        
    def unfreeze_top_n_layers(self, model: tf.keras.Model, n: int):
        """Phase 2: unfreeze the last N layers of the backbone."""
        self.base_model.trainable = True
        for layer in self.base_model.layers[:-n]:
            layer.trainable = False