# Image Style Transfer Using Convolutional Neural Network

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++=
# importing the required libraries
import tensorflow as tf
import numpy as np
import os

# Defining the required variables
ROOT = os.getcwd()  # project directory
CONTENT_IMAGE = ROOT + '/content.jpeg'  # image that is given to neural network as content
STYLE_IMAGE = ROOT + '/style.jpeg'  # styling image that needs to be transferred
STYLE_LAYERS = ['block1_conv1',  # layers in VGG to extract the features
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
                ]
CONTENT_LAYERS = 'block5_conv2'  # content features that will be extracted from higher layer according to research paper
OUTPUT_DIR = ROOT + '/outputs/'  # output dir to store generated outputs

width, height = tf.keras.preprocessing.image.load_img(
    CONTENT_IMAGE).size  # calculating the size for pre processing the images
img_rows = 400  # max rows in an image to be generated
img_cols = int(width * img_rows / height)  # calculate the max rows for the image with above defined rows
# using transfer learning where VGG-19 weight are initialized
model = tf.keras.applications.vgg19.VGG19(include_top=False,
                                          weights='imagenet')
output_dict = dict([(layer.name, layer.output) for layer in model.layers])  # defining the output for the network
feature_extractor = tf.keras.Model(model.input, output_dict)  # wrapping VGG-19 in keras.Model for forward propagation
optimizer = tf.keras.optimizers.Adam(learning_rate=2.0)  # initializing adam optimizer to optimize the gradient
total_variation_weight = 1e-6  # limit for total variation in the generated image
style_weight = 1e-6  # limit for style weights so they are not absolute zeros
content_weight = 2.5e-8  # same logic as the style weights check research paper for more info


# --------------------------------------------------------------------------------------------------------------

def process_images(image_path):  # handling the images before feeding them into network
    img = tf.keras.preprocessing.image.load_img(image_path,
                                                target_size=(img_rows, img_cols))  # load image from the disk
    img = tf.keras.preprocessing.image.img_to_array(img)  # converting them into array
    img = np.expand_dims(img,
                         axis=0)  # expanding the dimension so it looks something like (1 , height , width , channels)
    img = tf.keras.applications.vgg19.preprocess_input(img)  # pre processing the image before it is forward propagated
    return tf.compat.v1.convert_to_tensor(
        img)  # converting them to tensors you  could also use numpy but converting into tensor worked for me


def deprocess_image(image):  # converting the image back from the tensors
    image = image.reshape((img_rows, img_cols, 3))  # resizing them into 3D
    image[:, :, 0] += 103.939  # making sure there are not much noise
    image[:, :, 1] += 116.779  # same logic
    image[:, :, 2] += 123.68  # same logic
    image = image[:, :, ::-1]
    image = np.clip(image, 0, 255).astype('uint8')
    return image


def total_variation_loss(x):  # using variational loss so that generated image doesnt have much noise  in them
    a = tf.square(
        x[:, : img_rows - 1, : img_cols - 1, :] - x[:, 1:, : img_cols - 1, :]
    )
    b = tf.square(
        x[:, : img_rows - 1, : img_cols - 1, :] - x[:, : img_rows - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))


# calculating the loss between content image features and generated images feature so here we use mean squared
def content_cost(content_image_features,
                 noise_image_features):
    return tf.reduce_sum(tf.square(noise_image_features - content_image_features))


# generating gram matrix for each styling layers features
def gram_matrix(A):
    x = tf.transpose(A, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram


# calculating loss for each styling layer so here we calculate the difference between the actual style and generated
# style
def style_loss(style_feature, noise_style_feature):
    S = gram_matrix(style_feature)
    C = gram_matrix(noise_style_feature)
    channels = 3
    size = img_rows * img_cols
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))


# computing total loss for gradient descent
def compute_loss(combination_image, content_image, style_image):
    input_tensor = tf.concat([content_image, style_image, combination_image], axis=0)
    features = feature_extractor(input_tensor)
    loss = tf.zeros(shape=())
    layer_features = features[CONTENT_LAYERS]
    content_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * content_cost(content_image_features, combination_features)

    for layer_name in STYLE_LAYERS:
        layer_features = features[layer_name]
        style_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        style_J = style_loss(style_features, combination_features)
        loss += (style_weight / len(STYLE_LAYERS)) * style_J
    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss


# we change the input variable with respect to loss because the network is freezed so no trainable parameters are
# available for further information read research paper
@tf.function
def compute_loss_and_grads(combination_image, content_image, style_image):
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image, content_image, style_image)
    grads = tape.gradient(loss, combination_image)  # defining the gradient with respect to the generated image
    return loss, grads


if __name__ == '__main__':
    content_image = process_images(CONTENT_IMAGE)
    style_image = process_images(STYLE_IMAGE)
    combination_image = tf.compat.v1.Variable(process_images(CONTENT_IMAGE))
    iteration = 100
    for i in range(1, iteration + 1):
        loss, grads = compute_loss_and_grads(combination_image, content_image, style_image)
        optimizer.apply_gradients([(grads, combination_image)])  # applying gradient descent
        if i % 10 == 0:
            print("Iteration %d : loss= %.2f" % (i, loss))
            img = deprocess_image(combination_image.numpy())
            tf.keras.preprocessing.image.save_img(OUTPUT_DIR + str(i) + '.jpg', img)
