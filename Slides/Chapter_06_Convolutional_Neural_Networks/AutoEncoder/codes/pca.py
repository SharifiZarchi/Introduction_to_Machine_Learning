encoder = tf.keras.models.Sequential([tf.keras.layers.Dense(2, input_shape=input_shape)])

decoder = tf.keras.models.Sequential([tf.keras.layers.Dense(input_shape, input_shape=[2])])

pca_autoencoder = tf.keras.models.Sequential([encoder, decoder])

pca_autoencoder.compile(loss="mse", optimizer=tf.keras.optimizers.SGD())