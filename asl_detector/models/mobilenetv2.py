import tensorflow as tf

class ASLMobilenetv2():
    def __init__(self, num_classes: int = 29, dropout_rate: float = 0.2):
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        ## Base model
        self.base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet',            
        )
        self.base_model.trainable = False
        self.build_classifier(num_classes)
    
    def build_classifier(self, num_classes: int = 29):
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = self.base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        self.model = tf.keras.Model(inputs, outputs)
        
    def unfreeze_top_n_layers(self, n: int):
        """Phase 2: unfreeze the last N layers of the backbone."""
        self.base_model.trainable = True
        for layer in self.base_model.layers[:-n]:
            layer.trainable = False

model = ASLMobilenetv2()
model.model.summary()