root = " "  # Change this to your dataset path
image_dir = os.path.join(root, '')
mask_dir = os.path.join(root, '')

# Define Image Parameters
IMAGE_SIZE = 256
BATCH_SIZE = 12
exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")  # Supported extensions

# Load Image & Mask Paths
images = sorted(
    [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.lower().endswith(exts) and not fname.startswith(".")]
)

masks = sorted(
    [os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir) if fname.lower().endswith(exts) and not fname.startswith(".")]
)

# Verify Number of Samples
print("Original counts - Images:", len(images), "Masks:", len(masks))

# Extract filenames without extensions for proper alignment
def get_filename(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

image_filenames = set(get_filename(f) for f in images)
mask_filenames = set(get_filename(f) for f in masks)

# Find mismatches
missing_masks = image_filenames - mask_filenames
extra_masks = mask_filenames - image_filenames

print("Missing masks for images:", missing_masks)
print("Extra masks without corresponding images:", extra_masks)

# Align Images and Masks by Filtering Out Unmatched Files
common_filenames = image_filenames.intersection(mask_filenames)

images = [f for f in images if get_filename(f) in common_filenames]
masks = [f for f in masks if get_filename(f) in common_filenames]

print("Final matched count - Images:", len(images), "Masks:", len(masks))

# Function to Read & Process Images
def read_files(image_path, mask=False):
    image = tf.io.read_file(image_path)

    # Convert tensor to string for file format check
    image_path_str = image_path.numpy().decode("utf-8").lower()

    if mask:
        if image_path_str.endswith(".tif") or image_path_str.endswith(".tiff"):
            image = tfio.experimental.image.decode_tiff(image)
        else:
            image = tf.io.decode_image(image, channels=4)  # Read as RGBA first

        # If the mask has 4 channels (RGBA), we convert it to a single channel (grayscale)
        if image.shape[-1] == 4:
            image = image[:, :, :1]  # Take the first channel (R) if RGBA

        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
        image = tf.cast(image, tf.uint8)  # Ensure mask values remain integer
    else:
        # Handle TIFF images separately
        if image_path_str.endswith(".tif") or image_path_str.endswith(".tiff"):
            image = tfio.experimental.image.decode_tiff(image)  # Decode TIFF
            image = image[:, :, :3]  # Keep only RGB channels
        else:
            image = tf.io.decode_image(image, channels=3)  # Auto-detect format for standard images
        
        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
        image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    
    return image

# Wrap read_files to use with TensorFlow Dataset API
def load_data(image_path, mask_path):
    image = tf.py_function(read_files, [image_path, False], tf.float32)
    mask  = tf.py_function(read_files, [mask_path, True], tf.uint8)

    # Fix Tensor Shapes
    image.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3))
    mask.set_shape((IMAGE_SIZE, IMAGE_SIZE, 1))  # Ensure mask has 1 channel

    return image, mask

# Dataset Generator
def data_generator(image_list, mask_list):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

# Create Training Dataset
test_dataset = data_generator(images, masks)


# Visualize Function
def visualize(**images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(20, 20))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    plt.show()

# Iterate and Check Shapes
image_batch, mask_batch = next(iter(test_dataset.take(1)))
print("Batch Shapes:", image_batch.shape, mask_batch.shape)

# Check Min/Max Values and Unique Labels in Mask
for img, msk in zip(image_batch[:10], mask_batch[:10]):
    print("Mask Min/Max:", msk.numpy().min(), msk.numpy().max())
    print("Unique Mask Values:", np.unique(msk.numpy()))
    visualize(image=img.numpy(), gt_mask=msk.numpy())
