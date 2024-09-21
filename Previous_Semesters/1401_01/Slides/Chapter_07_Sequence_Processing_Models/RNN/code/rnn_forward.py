import tensorflow as tf

W_hh, h = tf.random.normal((4, 4)), tf.random.normal((4, 3))
W_hx, x_t = tf.random.normal((4, 1)), tf.random.normal((1, 3))
b_h = tf.random.normal((4, 3))

print(tf.matmul(W_hh, h) + tf.matmul(W_hx, x_t) + b_h)

print(tf.matmul(tf.concat((W_hh, W_hx), axis=1), tf.concat((h, x_t), axis=0)) + b_h)