tf.keras.losses.BinaryCrossentropy()

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    # Convert y_true to float32 to match y_pred
    y_true = tf.cast(y_true, tf.float32)

    # Binary Cross-Entropy loss
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)

    # Compute p_t (probability of correct class)
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)

    # Compute Focal Weight
    focal_weight = alpha * tf.pow(1 - p_t, gamma)

    # Apply focal weight to BCE loss
    return focal_weight * bce

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))

# Tversky Loss Function
def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    TP = K.sum(y_true_f * y_pred_f)
    FP = K.sum((1 - y_true_f) * y_pred_f)
    FN = K.sum(y_true_f * (1 - y_pred_f))
    
    return 1 - ((TP + smooth) / (TP + alpha * FP + beta * FN + smooth))
