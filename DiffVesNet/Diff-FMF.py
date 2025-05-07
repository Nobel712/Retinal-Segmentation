class GRUModel(layers.Layer):
    def __init__(self, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.bi_gru = layers.Bidirectional(layers.GRU(hidden_size, return_sequences=True))
        self.fc = layers.Dense(output_size)

    def call(self, inputs):
        x = self.bi_gru(inputs)
        output = self.fc(x)
        return output

class DiffusionFeatureMapFusion(layers.Layer):
    def __init__(self, out_channels):
        super(DiffusionFeatureMapFusion, self).__init__()
        
        # Learnable convolution to process the fusion of feature maps
        self.conv1x1 = layers.Conv2D(out_channels, (1, 1), activation="relu")
        
        # Diffusion-like filter: used to integrate features with spatial adaptation
        self.diffusion_filter = layers.Conv2D(out_channels, (3, 3), padding="same", activation="relu", use_bias=False)
        
        # Attention mechanism to focus on important features
        self.attention = layers.Attention()
        
        # A convolutional layer to process the attention-weighted fusion
        self.attention_weighted_conv = layers.Conv2D(out_channels, (1, 1), activation="relu")

    def call(self, encode_map, GRU_map, Decode_map):
        """
        Forward pass that integrates the feature maps using diffusion-like mechanisms.
        
        encode_map, GRU_map, Decode_map: These are the feature maps to be fused.
        """

        # Step 1: Resize GRU map to match the encoder's spatial dimensions
        GRU_map_resized = layers.Lambda(lambda x: tf.image.resize(x, size=(encode_map.shape[1], encode_map.shape[2])))(GRU_map)
        
        # Step 2: Concatenate feature maps for fusion
        fused_map = tf.concat([encode_map, GRU_map_resized, Decode_map], axis=-1)
        
        # Step 3: Apply diffusion filtering (spatial context integration)
        diffused_map = self.diffusion_filter(fused_map)
        
        # Step 4: Apply attention mechanism for adaptive fusion
        attention_weights = self.attention([diffused_map, diffused_map])  # Self-attention for feature adaptation
        
        # Step 5: Weighted fusion based on attention scores
        attention_fused_map = self.attention_weighted_conv(attention_weights * diffused_map)
        
        # Step 6: Final fusion through 1x1 convolution
        return self.conv1x1(attention_fused_map)
