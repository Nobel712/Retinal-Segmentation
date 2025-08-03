from tensorflow.keras import layers
class InterScale_Adaptive_Normalization(layers.Layer):
    """
    Layer Normalization conditioned on external features.
    
    The gamma and beta parameters are dynamically generated based on
    conditioning features, allowing context-aware normalization.
    """
    def _init_(self, epsilon=1e-6, **kwargs):
        super()._init_(**kwargs)
        self.epsilon = epsilon
        self.norm = layers.LayerNormalization(epsilon=epsilon, center=False, scale=False)
        self.gamma_dense = None
        self.beta_dense = None
        
    def build(self, input_shape):
        # Handle different input shape scenarios
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 2:
            # Case: Two inputs passed separately
            main_shape, cond_shape = input_shape
            if main_shape is None or cond_shape is None:
                raise ValueError(
                    f"ConditionalLayerNormalization received None for one of the input shapes. "
                    f"Main shape: {main_shape}, Conditioning shape: {cond_shape}"
                )
            feature_dim = main_shape[-1]
        else:
            # Case: Single input or incorrect number of inputs
            if input_shape is None:
                raise ValueError("ConditionalLayerNormalization received None for input_shape.")
            feature_dim = input_shape[-1]
            
        # Create the gamma and beta projection layers
        self.gamma_dense = layers.Dense(feature_dim)
        self.beta_dense = layers.Dense(feature_dim)
        super().build(input_shape)
        
    def call(self, inputs, conditioning_features=None):
        """
        Forward pass for ConditionalLayerNormalization.
        
        Args:
            inputs: The primary tensor to be normalized.
            conditioning_features: The features used to condition normalization.
                                   If None, standard normalization is applied.
        """
        # Apply standard normalization without learned gamma/beta
        normalized = self.norm(inputs)
        
        # If no conditioning features, return standard normalization
        if conditioning_features is None:
            return normalized
        
        # Generate dynamic gamma and beta from conditioning features
        gamma = self.gamma_dense(conditioning_features)
        beta = self.beta_dense(conditioning_features)
        
        # Apply the conditional scaling and shifting
        return normalized * (1 + gamma) + beta
        
    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config
