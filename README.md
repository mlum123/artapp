# Anyone Can Draw

#### Michelle Lum

If you’re like me, you’d love to be able to whip up a quick sketch or a beautiful watercolor in just a few hours, but unfortunately, you lack the artistic talent to do so. That’s where my project, Anyone Can Draw, comes in.

Anyone Can Draw is a Flask web app that allows you to upload photos and turn them into artwork that looks like it was drawn or painted by a real person. You can upload a plain old photo, and the app will generate an oil painting, pencil sketch, pointillist painting, or posterized version of that photo for you.

The app can also perform style transfer. Upload one content image and one style image, and it'll produce an image with the content of the content image but in the style of the style image.

Deployed at https://michelles-art-app.herokuapp.com/, unfortunately without the pointillist painting and posterized results because those are too slow for Heroku's default request timeout of 30 seconds.

**_Flask App Demo: https://youtu.be/SUOt_b0j2bg_**

## Run this project locally

Download the GitHub repo.

In the project directory, run `pip3 install -r requirements.txt` to install the required Python libraries.

In the project directory, run `flask run` to start the web app. Note: you must re-run this command any time you make changes.

When running locally, if you'd like to try out the pointillizing and posterizing features, go to app.py and switch the uncommented art_functions variable line with the commented-out art_functions line.

## Technologies Used

Python, Flask (as a Python web development framework), HTML, CSS, Heroku (for deployment)

OpenCV (for some built-in functions that convert photos to art)

scikit-learn (for k-means clustering for pointillizing and posterizing)

NumPy (for working with images as numpy arrays)

Pillow (for opening images)

Matplotlib (for displaying and saving images)

TensorFlow (for accessing the pretrained VGG19 image classification model in my exploration of neural style transfer)

TensorFlow Hub (for the pretrained image stylization module I ended up using for style transfer in my Flask app)

## Project Directory Structure

artapp folder:
This folder holds the code for our app. Any routing for the app is handled in this folder, as well as functionality for the features of the app.

config.py:
This file initializes the configuration of the app and sets the static folder as the default upload folder for files.

app.py:
This script initializes the Flask app and contains all the routing for the app. Handles backend scripts and
renders templates from the templates folder.

art.py:
Script that contains all the functions used to convert an image to different styles of art.

templates folder:
Folder that holds the HTML code.

static folder:
Upload folder that will hold images uploaded by users and images produced after processing user images. Also contains styles.css stylesheet.

runtime.txt:
Lists the Python version to use for deployment by Heroku.

requirements.txt:
Lists the Python libraries required by this Flask app.

Aptfile:
Needed to fix library errors when using OpenCV on a Heroku app with heroku-buildpack-apt.

Procfile:
Specifies the command to be executed by the Heroku app on startup.

## Photo to Art Examples

_original photo_

<img width="300" alt="Food Original Photo" src="https://user-images.githubusercontent.com/14852932/116738897-e9ad1880-a9a7-11eb-8c32-dede8e2ad338.jpeg" />

_oil painting_

<img width="300" alt="Food Oil Painting" src="https://user-images.githubusercontent.com/14852932/116739201-414b8400-a9a8-11eb-83da-87eacfe151a0.jpeg" />

_watercolor_

<img width="300" alt="Food Watercolor" src="https://user-images.githubusercontent.com/14852932/116739150-3395fe80-a9a8-11eb-944f-b0e36e3acd2a.jpeg" />

_pointillist painting_

<img width="300" alt="Food Pointillist Painting" src="https://user-images.githubusercontent.com/14852932/116739043-119c7c00-a9a8-11eb-8f56-8fdd80a8961b.jpeg" />

_posterized version_

<img width="300" alt="Food Posterized Version" src="https://user-images.githubusercontent.com/14852932/116739086-1f520180-a9a8-11eb-9db4-815d7c64d653.jpeg" />

---

_original photo_

<img width="300" alt="Mudd Original Photo" src="https://user-images.githubusercontent.com/14852932/116739424-84a5f280-a9a8-11eb-80f8-d88b042121b7.jpeg" />

_black-and-white pencil sketch_

<img width="300" alt="Mudd Black-and-White Pencil Sketch" src="https://user-images.githubusercontent.com/14852932/116739489-98e9ef80-a9a8-11eb-9ec8-ebf9c901d719.jpeg" />

_color pencil sketch_

<img width="300" alt="Mudd Color Pencil Sketch" src="https://user-images.githubusercontent.com/14852932/116739545-a606de80-a9a8-11eb-8fcb-64df3d83e87f.jpeg" />

## Style Transfer Example Using Original Neural Style Transfer Process

_content image_

<img width="300" alt="Totoro Content Image" src="https://user-images.githubusercontent.com/14852932/116739912-1ada1880-a9a9-11eb-9a92-3a3e9360520e.jpeg" />

_style image_

<img width="300" alt="Composition 8 Style Image" src="https://user-images.githubusercontent.com/14852932/116739926-20376300-a9a9-11eb-857d-f9ca3639bdec.jpeg" />

_output image_

<img width="350" alt="Totoro Composition 8 Combined Image Original Process" src="https://user-images.githubusercontent.com/14852932/116741695-4cec7a00-a9ab-11eb-8244-55c6b81ae182.png">

## Style Transfer Example Using TensorFlow Hub

_content image_

<img width="300" alt="Totoro Content Image" src="https://user-images.githubusercontent.com/14852932/116739912-1ada1880-a9a9-11eb-9a92-3a3e9360520e.jpeg" />

_style image_

<img width="300" alt="Composition 8 Style Image" src="https://user-images.githubusercontent.com/14852932/116739926-20376300-a9a9-11eb-857d-f9ca3639bdec.jpeg" />

_output image_

<img width="350" alt="Totoro Composition 8 Combined Image TensorFlow Hub" src="https://user-images.githubusercontent.com/14852932/116740028-478e3000-a9a9-11eb-81c4-cb6f7f853641.png">

## Process and Reflections

I started this project by first exploring how to convert a single image into different styles of art like oil paintings, watercolors, and pencil sketches, using OpenCV functions. Then, I looked into two processes that were much more involved: pointillizing and posterizing artwork using k-means clustering.

Once I got my photo-to-art functions working, I set up a simple Flask web app as an interface for users to convert their own images into art. I deployed the Flask app on Heroku — lots of frustration was involved, but I got it to work after half a day on Stack Overflow.

With the Flask app up and running, I decided to pursue a stretch goal of performing style transfer: taking in a content image and a style image, and then producing an image with the content of the content image but in the style of the style image. Initially, I coded up the whole process of neural style transfer using TensorFlow's pretrained VGG19 image classification model. Certain intermediate layers of the model represent the content of an image, while others represent the style of an image. In neural style transfer, we start out with the content image. Then, in each iteration, we extract feature maps from the VGG19 neural network, and try to minimize the loss between the style image and current output image & the content image and current output image, using gradient descent. However, this process was extremely involved, with each style transfer running 1000 iterations. It took an hour to run on my Mac, which is way too long to be used with a Flask app.

I tried running neural style transfer on a Colab GPU, and it was much faster (took only about 5 minutes to run 1000 iterations).

However, I wouldn't have access to the Colab GPU on my Flask app, so I did some more digging and found TensorFlow Hub's pretrained image stylization module, which is what I ended up using in my Flask app for faster style transfer. TensorFlow Hub's pretrained image stylization module took less than a minute to perform style transfer! The style transfer results from TensorFlow Hub and my original neural style transfer process are different — as you can see above — but I think both results are quite interesting.

Ultimately, deploying to Heroku still proved to be a bit iffy. I was not able to get my photo-to-art pointillizing and posterizing features up on Heroku due to Heroku's default 30-second request timeout. Pointillizing and posterizing take longer than 30 seconds to run, so my Heroku app would break if I included those features on the deployed app.

Overall, though, I would say this project went pretty well. It was exciting to be able to successfully convert photos to so many different styles of art, and I was able to experiment with two methods of style transfer, both of which gave interesting results.

Next time, I'd try deploying my Flask app through a different service, since Heroku gave me quite a bit of trouble. I might find more success deploying my app on another platform!
