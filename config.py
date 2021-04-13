# initialize configuration of app

import os

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'

    # set the upload folder as the static folder
    UPLOAD_FOLDER = 'static'