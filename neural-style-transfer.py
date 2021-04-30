# Neural Style Transfer
# takes in content image, style reference image, input image you want to style (initialized with the content image)
# blends them together so input image looks like content image but in style of style reference image

import matplotlib.pyplot as plt 
import matplotlib as mpl 
mpl.rcParams['figure.figsize'] = (10,10)
mpl.rcParams['axes.grid'] = False

import numpy as np 
from PIL import Image
import time
import functools

import tensorflow as tf 

from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

# eager execution enabled by default in Tensorflow version 2.x
print("Eager execution: {}".format(tf.executing_eagerly()))

# Set up global values
content_path = 'cafe-terrace-at-night.jpeg'
style_path = 'sunrise.jpeg'

# Visualize Input
def load_img(path_to_img):
    max_dim = 512
    img = Image.open(path_to_img)
    long = max(img.size)
    scale = max_dim / long
    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)

    img = kp_image.img_to_array(img)

    # we need to broadcast the image array such that it has a batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def imshow(img, title=None):
    # remove the batch dimension
    out = np.squeeze(img, axis=0)
    # normalize for display
    out = out.astype('uint8')
    plt.imshow(out)
    if title is not None:
        plt.title(title)
    plt.imshow(out)

plt.figure(figsize=(10,10))

content = load_img(content_path).astype('uint8')
style = load_img(style_path).astype('uint8')

plt.subplot(1, 2, 1)
imshow(content, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style, 'Style Image')
plt.show()

# Prepare the Data
# perform same preprocessing process as expected by VGG training process
# VGG networks are trained on image with each channel normalized by mean = [103.939, 116.779, 123.68] and with channels BGR
def load_and_process_img(path_to_img):
    img = load_img(path_to_img)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                               "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")
    
    # perform the inverse of the preprocessing so we can view outputs
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    # clip to maintain values between 0 and 255 since optimized image can have values from -infinity to infinity
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Define Content and Style Representations
# pull out intermediate layers from VGG19 model (pretrained image classification network)

# content layer where we'll pull our feature maps
content_layers = ['block5_conv2']

# style layer we're interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
                ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# Build the Model
# load VGG19 and feed in input tensor to the model
# to extract feature maps (and then content and style representations) of content, style, and generated images
# use Keras Functional API to define model with desired output activations
def get_model():
    """
    Creates our model with access to intermediate layers.

    This function will load the VGG19 model and access the intermediate layers.
    These layers will then be used to create a new model that will take input image
    and return the outputs from these intermediate layers from the VGG model.

    Returns: a keras model that takes image inputs and outputs the style and content intermediate layers.
    """
    # load our model: pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    # get output layers corresponding to style and content layers
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs

    # build model by setting the model’s inputs to an image and the outputs to the outputs of the style and content layers
    return models.Model(vgg.input, model_outputs)


# Define and Create Loss Functions (Content and Style Distances)

# compute content loss by comparing the intermediate layer outputs of the content image and the base input image
def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

# compute style loss by comparing the style representation (gram matrices) of the intermediate outputs
# of the base input image (x) and the style image (a)
def gram_matrix(input_tensor):
    # make the image channels first
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

def get_style_loss(base_style, gram_target):
    """expects two images of dimension h, w, c"""
    # height, width, num filters of each layer
    # scale the loss at a given layer by the size of the feature map and the number of filters
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)

    return tf.reduce_mean(tf.square(gram_style - gram_target)) # / (4. * (channels ** 2) * (width * height) ** 2)

# Run Gradient Descent
# Use Adam optimizer to minimize our loss, iteratively update output image
def get_feature_representations(model, content_path, style_path):
    """
    Helper function to compute our content and style feature representations.

    This function will simply load and preprocess both the content and style images
    from their path. Then it will feed them through the network to obtain the outputs
    of the intermediate layers.

    Arguments:
        model: The model that we are using.
        content_path: The path to the content image.
        style_path: The path to the style image.
    
    Returns: the style features and the content features
    """
    # load our images in
    content_image = load_and_process_img(content_path)
    style_image = load_and_process_img(style_path)

    # batch compute content and style features
    style_outputs = model(style_image)
    content_outputs = model(content_image)

    # get the style and content feature representations from our model
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
    return style_features, content_features

# Compute the Loss and Gradients
def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    """
    This function will compute the total loss.

    Arguments:
        model: The model that will give us access to the intermediate layers
        loss_weights: The weights of each contribution of each loss function.
            (style weight, content weight, and total variation weight)
        init_image: Our initial base image. This image is what we are updating with
            our optimization process. We apply the gradients wrt the loss we are
            calculating to this image.
        gram_style_features: Precomputed gram matrices corresponding to the
            defined style layers of interest.
        content_features: Precomputed outputs from defined content layers of interest.

    Returns: the total loss, style loss, content loss, and total variation loss
    """
    style_weight, content_weight = loss_weights

    # feed our init image through our model
    # this will give us the content and style representations at our desired layers
    # since we're using eager, our model is callable just like any other function!
    model_outputs = model(init_image)

    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    # accumulate style losses from all layers
    # here, we equally weight each contribution of each loss layer
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

    # accumulate content losses from all layers
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * get_content_loss(comb_content[0], target_content)
    
    style_score *= style_weight
    content_score *= content_weight

    # get total loss
    loss = style_score + content_score
    return loss, style_score, content_score

# compute the gradients
def compute_grads(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    
    # compute gradients wrt input image
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_image']), all_loss

# Optimization Loop
import IPython.display

def run_style_transfer(content_path,
                       style_path,
                       num_iterations=1000,
                       content_weight=1e3,
                       style_weight=1e-2):
    # we don't need to (or want to) train any layers of our model, so set trainable to false
    model = get_model()
    for layer in model.layers:
        layer.trainable = False
    
    # get the style and content feature representations (from our specified intermediate layers)
    style_features, content_features = get_feature_representations(model, content_path, style_path)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    # set initial image
    init_image = load_and_process_img(content_path)
    init_image = tf.Variable(init_image, dtype=tf.float32)

    # create our optimizer
    opt = tf.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)

    # for displaying intermediate images
    iter_count = 1

    # store our best result
    best_loss, best_img = float('inf'), None

    # create a nice config
    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }

    # for displaying
    num_rows = 2
    num_cols = 5
    display_interval = num_iterations / (num_rows * num_cols)
    start_time = time.time()
    global_start = time.time()

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    imgs = []
    for i in range(num_iterations):
        grads, all_loss = compute_grads(cfg)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)
        end_time = time.time()

        if loss < best_loss:
            # update best loss and best image from total loss
            best_loss = loss
            best_img = deprocess_img(init_image.numpy())
        
        if i % display_interval == 0:
            start_time = time.time()

            # use the .numpy() method to get the concrete numpy array
            plot_img = init_image.numpy()
            plot_img = deprocess_img(plot_img)
            imgs.append(plot_img)
            IPython.display.clear_output(wait=True)
            IPython.display.display_png(Image.fromarray(plot_img))
            print('Iteration: {}'.format(i))
            print('Total loss: {:.4e}, style loss: {:.4e}, content loss: {:.4f}, time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))
        
    print('Total time: {:.4f}s'.format(time.time() - global_start))
    IPython.display.clear_output(wait=True)
    plt.figure(figsize=(14,4))
    for i,img in enumerate(imgs):
        plt.subplot(num_rows,num_cols,i+1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    
    return best_img, best_loss

best, best_loss = run_style_transfer(content_path, style_path, num_iterations=1000)
Image.fromarray(best)

# Visualize Outputs
# deprocess output image to remove processing that was applied to it
def show_results(best_img, content_path, style_path, show_large_final=True):
    plt.figure(figsize=(10, 5))
    content = load_img(content_path)
    style = load_img(style_path)

    plt.subplot(1, 2, 1)
    imshow(content, 'Content Image')

    plt.subplot(1, 2, 2)
    imshow(style, 'Style Image')

    if show_large_final:
        plt.figure(figsize=(10, 10))

        plt.imshow(best_img)
        plt.title('Output Image')
        plt.show()
    
show_results(best, content_path, style_path)