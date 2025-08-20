import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load and preprocess images
def load_and_process_img(path_to_img):
    img = load_img(path_to_img, target_size=(512, 512))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return vgg19.preprocess_input(img)

# Deprocess for display
def deprocess_img(processed_img):
    x = processed_img.copy()
    x = x.reshape((512, 512, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    return np.clip(x, 0, 255).astype('uint8')

# Load content and style images
content_path = "content.jpg"
style_path = "style.jpg"

content = load_and_process_img(content_path)
style = load_and_process_img(style_path)

# Display input images
def show_img(img, title=None):
    img = deprocess_img(img)
    plt.imshow(img)
    if title: plt.title(title)
    plt.axis('off')
    plt.show()

show_img(content, "Content Image")
show_img(style, "Style Image")

# Extract features using VGG19
def get_model():
    vgg = vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    content_layers = ['block5_conv2'] 
    style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
    outputs = [vgg.get_layer(name).output for name in style_layers + content_layers]
    model = tf.keras.Model([vgg.input], outputs)
    return model, style_layers, content_layers

model, style_layers, content_layers = get_model()

# Helper functions for loss calculation
def gram_matrix(tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    input_shape = tf.shape(tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / num_locations

def get_feature_representations(model, content, style):
    style_outputs = model(style)
    content_outputs = model(content)
    style_features = [style_layer for style_layer in style_outputs[:len(style_layers)]]
    content_features = [content_layer for content_layer in content_outputs[len(style_layers):]]
    return style_features, content_features

# Style and content weights
style_weight = 1e-2
content_weight = 1e4

# Calculate total loss
def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights
    model_outputs = model(init_image)
    style_output_features = model_outputs[:len(style_layers)]
    content_output_features = model_outputs[len(style_layers):]
    style_score = 0
    content_score = 0

    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += tf.reduce_mean((gram_matrix(comb_style) - target_style) ** 2)

    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += tf.reduce_mean((comb_content - target_content) ** 2)

    total_loss = style_weight * style_score + content_weight * content_score
    return total_loss

# Optimization
@tf.function()
def train_step(image, model, loss_weights, gram_style_features, content_features, opt):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, loss_weights, image, gram_style_features, content_features)
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, -103.939, 255.0 - 103.939))

# Run Style Transfer
def run_style_transfer(content, style, iterations=100):
    model, style_layers, content_layers = get_model()
    style_features, content_features = get_feature_representations(model, content, style)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
    init_image = tf.Variable(content, dtype=tf.float32)
    opt = tf.optimizers.Adam(learning_rate=5.0)
    best_img = None
    best_loss = float('inf')

    for i in range(iterations):
        train_step(init_image, model, (style_weight, content_weight), gram_style_features, content_features, opt)
        if i % 20 == 0:
            print(f"Iteration {i}")
    return init_image

# Run and display result
stylized_image = run_style_transfer(content, style)
show_img(stylized_image.numpy(), "Stylized Image")