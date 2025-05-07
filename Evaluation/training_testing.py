import keras
class DisplayCallback(keras.callbacks.Callback):
    def __init__(self, dataset, epoch_interval=5):
        self.dataset = dataset
        self.epoch_interval = epoch_interval
    
    def display(self, display_list, extra_title=''):
        plt.figure(figsize=(15, 15))
        title = ['Input Image', 'True Mask', 'Predicted Mask']

        if len(display_list) > len(title):
            title.append(extra_title)

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(title[i])
            plt.imshow(display_list[i], cmap='gray')
            plt.axis('off')
        plt.show()
        
    def create_mask(self, pred_mask):
        pred_mask = (pred_mask > 0.5).astype("int32")
        return pred_mask[0]
    
    def show_predictions(self, dataset, num=1):
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            self.display([image[0], mask[0], self.create_mask(pred_mask)])
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch and epoch % self.epoch_interval == 0:
            self.show_predictions(self.dataset)
            print ('\nSample Prediction after epoch {}\n'.format(epoch+1))


# Create the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath='best_model_weights.h5',    # or use .keras for full model format
    save_weights_only=True,              # Set False to save the full model
    save_best_only=True,                 # Only save when val loss improves
    monitor='loss',                      # or 'val_loss' if you have validation
    mode='min',
    verbose=1
)

model.fit(
    train_dataset, 
    callbacks=[DisplayCallback(train_dataset),checkpoint],
    batch_size=batch_size,
    epochs=epochs
)


test_loss, test_accuracy, test_sensitivity,auc,test_precision, test_dice,test_iou,test_specificity,test_f1_score = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test dice: {test_dice}")
print(f"Test iou: {test_iou}")
print(f"Test prec: {test_precision}")
print(f"Test specificity: {test_specificity}")
print(f"Test f1: {test_f1_score}")
print(f"Test sensitivity: {test_sensitivity}")
