import numpy as np
import cv2
import tensorflow as tf


def get_gradcam_heatmap(model, img_array, last_conv_layer_name=None):

    # -------- AUTO FIND LAST CONV LAYER --------
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                last_conv_layer_name = layer.name
                break

    if last_conv_layer_name is None:
        raise ValueError("No Conv layer found in model")

    # -------- BUILD GRAD MODEL --------
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = 1 - predictions[:, 0]

    # -------- GRADIENTS --------
    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    # -------- WEIGHTED SUM --------
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # -------- SAFE NORMALIZATION --------
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros_like(heatmap.numpy())

    heatmap /= max_val

    return heatmap.numpy()


def overlay_heatmap(heatmap, image):

    # Resize heatmap
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Convert to 0-255
    heatmap = np.uint8(255 * heatmap)

    # Apply colormap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = np.uint8(image)

    # Blend image + heatmap
    superimposed_img = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    return superimposed_img