# Anyone Can Draw

If you’re like me, you’d love to be able to whip up a quick sketch or a beautiful watercolor in just a few hours, but unfortunately, you lack the artistic talent to do so. That’s where my project, Anyone Can Draw, comes in.

Anyone Can Draw is a Flask web app that allows you to upload photos and turn them into artwork that looks like it was drawn or painted by a real person. You can upload a plain old photo, and the app will generate an oil painting, pencil sketch, pointillist painting, or posterized version of that photo for you. In terms of the type of artwork the web app can generate, the possibilities are endless. This web app has lots of different styles of art incorporated so that truly, anyone can draw.

Deployed at https://michelles-art-app.herokuapp.com/, unfortunately without the pointillist painting and posterized results because those are too slow for Heroku's default request timeout of 30 seconds.

## Technologies Used

Python, Flask, OpenCV, HTML, CSS

## Run this project locally

In the project directory, run `source bin/activate` to enter the virtual environment.

In the project directory, run `flask run` to start the web app. Note: you must re-run this command any time you make changes.

When running locally, if you'd like to try out the pointillizing and posterizing features, go to app.py and switch the uncommented art_functions variable line with the commented-out art_functions line.

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
Upload folder that will hold images uploaded by users. Also contains styles.css stylesheet.
