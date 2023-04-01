import tensorflow as tf
import cv2
import numpy as np
tf.compat.v1.disable_eager_execution()
# pip install tensorflow opencv-python numpy

# load the model in
def load_model(model_path):

    model = tf.keras.models.load_model(model_path)

    # print model information
    print("----- Model Info -----")
    print("-- summary:")
    print(model.summary())
    print("-- inputs & output:")
    print(model.inputs)
    print(model.outputs)
    print("-- loss & metrics:")
    print(model.loss)
    print(model.metrics)
    print("-- Model History:")
    print(model.history)

    return model

# preprocessing
def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    orig_img = image

    print("shape: ", image.shape)
    # Preprocess the image
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    return image, orig_img

def generate_heatmap(model, image):
    #print("image shapes", image.shape)
    # Get the model's predictions for the image
    preds = model.predict(image)
    print("Predictions:", preds)

    class_idx = np.argmax(preds[0])
    print("Prediction Label: ", class_idx)
    class_output = model.output[:, class_idx]

    # Get the last convolutional layer in the model
    last_conv_layer = model.get_layer('block5_conv3')

    # Calculate the gradients of the output with respect to the last convolutional layer
    grads = tf.keras.backend.gradients(class_output, last_conv_layer.output)[0]

    #TensorFlow 2.0 version that gives an error:
    #with tf.GradientTape() as tape:
        #grads = tape.gradient(class_output, last_conv_layer.output)[0]

    # Pool the gradients over all the axes leaving out the channel dimension
    pooled_grads = tf.keras.backend.mean(grads, axis=(0, 1, 2))

    # Define a function to generate the heatmap
    heatmap_fn = tf.keras.backend.function([model.input], [pooled_grads, last_conv_layer.output[0]])

    # Generate the heatmap and resize it to match the size of the original image
    pooled_grads_value, conv_layer_output_value = heatmap_fn([image])
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[2]))

    # Convert the heatmap to RGB
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    return heatmap

# Superimpose the heatmap on the original image
def superimpose_heatmap():
    superimposed_img = cv2.addWeighted(image[0].astype(np.float32), 0.5, heatmap.astype(np.float32), 0.5, 0.4, dtype=cv2.CV_32F)
    return superimposed_img

def visualize_heatmap(image_path, output_path, alpha=0.7, beta=0.3, gamma=0):
    model_path = 'modelTwo.h5'
    model = load_model(model_path)
    image, orig_img = preprocess_image(image_path)
    heatmap = generate_heatmap(model, image)

    width, height = orig_img.shape[0], orig_img.shape[1]

    orig_img = cv2.resize(orig_img, (224, 224))
    vis = cv2.addWeighted(orig_img, alpha, heatmap, beta, gamma, dtype=cv2.CV_32F)
    vis = cv2.resize(vis, (954, 896))

    cv2.imwrite(output_path, vis)


visualize_heatmap('input.jpg', 'output.jpg')