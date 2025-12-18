import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def create_datasets(data_dir):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir + "/train",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir + "/test",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = train_ds.class_names
    print("Classes:", class_names)

    return train_ds, test_ds, class_names
