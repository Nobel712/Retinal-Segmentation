def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

# Custom True Detection Rate (TDR) function
# def true_detection_rate(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     false_positives = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
#     return true_positives / (true_positives + false_positives + K.epsilon())


#Custom Recall function
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    false_negatives = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
    return true_positives / (true_positives + false_negatives + K.epsilon())

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    false_positives = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    return true_positives / (true_positives + false_positives + K.epsilon())
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())
def BM(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    sensitivity = true_positives / (possible_positives + K.epsilon())

    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    specificity= true_negatives / (possible_negatives + K.epsilon())
    
    # Compute Bookmaker Informedness (BM)
    bm = sensitivity + specificity - 1
    
    return bm

def f1_score(y_true, y_pred):
    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    f1_score_val = 2 * (precision_val * recall_val) / (precision_val + recall_val + tf.keras.backend.epsilon())
    return f1_score_val
def dice_coef(y_true, y_pred, smooth=100):
    y_true = K.cast(y_true, dtype='float32')
    y_pred = K.cast(y_pred, dtype='float32')

    # Ensure shapes are the same
    y_true = K.reshape(y_true, (-1,))
    y_pred = K.reshape(y_pred, (-1,))

    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)

    return (2 * intersection + smooth) / (union + smooth)


def iou_coef(y_true, y_pred, smooth=100):
    y_true = K.cast(y_true, dtype='float32')
    y_pred = K.cast(y_pred, dtype='float32')

    intersection = K.sum(y_true * y_pred)
    total = K.sum(y_true + y_pred)
    
    iou = (intersection + smooth) / (total - intersection + smooth)
    return iou


from sklearn.metrics import matthews_corrcoef, confusion_matrix

def mcc(y_true, y_pred):
    # Convert tensors to numpy arrays if needed
    if isinstance(y_true, tf.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, tf.Tensor):
        y_pred = y_pred.numpy()

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Compute MCC
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = numerator / denominator if denominator != 0 else 0  # Avoid division by zero
    
    return mcc

def BM(y_true, y_pred):
    # Convert tensors to numpy arrays if needed
    if isinstance(y_true, tf.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, tf.Tensor):
        y_pred = y_pred.numpy()

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
  
    # Compute Sensitivity (Recall) and Specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Compute Bookmaker Informedness
    bm = sensitivity + specificity - 1
    
    return bm


