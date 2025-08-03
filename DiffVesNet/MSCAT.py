class MultiScaleContextAdaptiveTransformer(layers.Layer):
    """
    Multi-Scale Context-Adaptive Transformer (MSCAT)
    """

    def _init_(self,
                 embed_dim: int,
                 num_heads: int,
                 ff_dim: int,
                 window_size: int = 7,
                 dropout_rate: float = 0.1,
                 activation: str = "gelu",
                 **kwargs):
        super()._init_(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.window_size = window_size
        self.dropout_rate = dropout_rate
        self.activation = activation

        # 1. Context-Adaptive Normalization
        self.context_norm1 = InterScale_Adaptive_Normalization(epsilon=1e-6)
        self.context_norm2 = InterScale_Adaptive_Normalization(epsilon=1e-6)
        
        # 2. Global Attention
        self.global_attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate
        )
        
        # 3. Local Attention via Depth-wise Convolution
        self.local_conv = layers.Conv2D(
            embed_dim, 
            kernel_size=window_size, 
            padding='same', 
            groups=embed_dim,  # Depth-wise convolution
            use_bias=False
        )
        self.local_proj = layers.Dense(embed_dim)
        
        # 4. Dynamic Gating between Local and Global
        self.attn_gate = layers.Dense(embed_dim, activation="sigmoid")
        
        # 5. Cross-Scale Interaction
        self.cross_scale_attn = layers.MultiHeadAttention(
            num_heads=2,
            key_dim=embed_dim // 2,
            dropout=dropout_rate
        )
        
        # 6. Feed-Forward Network with Spatial Awareness
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation=activation),
            layers.Dropout(dropout_rate),
            # Adding spatial awareness to FFN
            layers.Reshape((-1, 1, ff_dim)),
            layers.DepthwiseConv2D(3, padding='same'),
            layers.Reshape((-1, ff_dim)),
            layers.Dense(embed_dim),
        ])
        
        # 7. Feature Gating for FFN
        self.ffn_gate = layers.Dense(embed_dim, activation="sigmoid")
        
        # 8. Dropout layers
        self.attn_dropout = layers.Dropout(dropout_rate)
        self.ffn_dropout = layers.Dropout(dropout_rate)

    def call(self, 
             inputs,
             conditioning_features=None,
             spatial_size=None,
             mask=None, 
             training=False):
        """
        Forward pass for MSCAT.
        """
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Use inputs as conditioning features if none provided
        if conditioning_features is None:
            conditioning_features = inputs
            
        # For local attention, we need spatial dimensions
        if spatial_size is None:
            # Default: assume square feature map
            height = width = int(tf.sqrt(tf.cast(seq_len, tf.float32)))
        else:
            height, width = spatial_size
            
        # 1. Global Self-Attention
        global_attn_out = self.global_attn(
            query=inputs,
            key=inputs,
            value=inputs,
            attention_mask=mask,
            training=training
        )
        
        # 2. Local Attention via Depthwise Convolution
        # Reshape to spatial dimensions for convolution
        inputs_spatial = tf.reshape(inputs, [batch_size, height, width, self.embed_dim])
        local_attn_out = self.local_conv(inputs_spatial)
        local_attn_out = self.local_proj(local_attn_out)
        # Reshape back to sequence form
        local_attn_out = tf.reshape(local_attn_out, [batch_size, seq_len, self.embed_dim])
        
        # 3. Dynamic Gating between Local and Global
        gate = self.attn_gate(inputs)
        attn_out = gate * local_attn_out + (1 - gate) * global_attn_out
        attn_out = self.attn_dropout(attn_out, training=training)
        
        # 4. Cross-Scale Interaction with conditioning features
        cross_scale_out = self.cross_scale_attn(
            query=attn_out,
            key=conditioning_features,
            value=conditioning_features,
            training=training
        )
        attn_out = attn_out + cross_scale_out
        
        # 5. First Context-Adaptive Normalization - FIXED: Pass as a list
        x = self.context_norm1(inputs + attn_out, conditioning_features)
        
        # 6. Feed-Forward Network with Spatial Awareness
        ffn_out = self.ffn(x, training=training)
        
        # 7. Gated Residual Connection
        gate = self.ffn_gate(x)
        gated_ffn = gate * ffn_out
        gated_ffn = self.ffn_dropout(gated_ffn, training=training)
        
        # 8. Second Context-Adaptive Normalization - FIXED: Pass as a list
        output = self.context_norm2(x + gated_ffn, conditioning_features)
        
        return output
