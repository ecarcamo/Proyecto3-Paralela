def make_dataset(tf, pixels, labels, batch_size, shuffle=False, seed=0):
    with tf.device("/CPU:0"):
        dataset = tf.data.Dataset.from_tensor_slices((pixels, labels))
        if shuffle:
            dataset = dataset.shuffle(len(labels), seed=seed, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(
            lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return dataset.prefetch(tf.data.AUTOTUNE)
