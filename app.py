from flask import Flask, render_template, redirect, request, send_from_directory, flash
from config import Config
import os
from werkzeug.utils import secure_filename
import cv2

import art

app = Flask(__name__, static_url_path='')
app.config.from_object(Config)

# home page is image uploading page
@app.route('/', methods=["GET", "POST"])
def index():
    """
    index page (home page) either displays a form for user to upload images
    or the resulting artwork from an uploaded image
    depending on if it's a GET or a POST request
    """
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        # if user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        # if the image is valid, do the following
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            # create a path to the image in the upload folder, save the upload file to this path
            save_old=(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file.save(save_old)
            
            # generate different styles of art from image

            # NOTE: use the commented-out art_functions line instead if you want to see the pointillist and posterizing results as well
            # however, I don't display those results here because if I did, Heroku would timeout the request at 30 seconds before the processes are done running, and we wouldn't be able to see any resulting images
            # art_functions = {"oil": art.oil_painting, "watercolor": art.watercolor, "pencil_gray": art.pencil_sketch_bw, "pencil_color": art.pencil_sketch_color, "pointillist": art.pointillize, "poster": art.posterize}
            
            art_functions = {"oil": art.oil_painting, "watercolor": art.watercolor, "pencil_gray": art.pencil_sketch_bw, "pencil_color": art.pencil_sketch_color}
            art_filenames = []
            img = cv2.imread(save_old)
            for style in art_functions:
                new_img = art_functions[style](img)
                new_filename = filename.rsplit('.', 1)[0] + '_' + style + '.' + filename.rsplit('.', 1)[1]
                save_new=(os.path.join(app.config['UPLOAD_FOLDER'], new_filename))
                art_filenames += [new_filename]
                cv2.imwrite(save_new, new_img)
            
            # render template with resulting artwork
            rt = render_template('results.html', filenames=art_filenames)
            
            return rt
    
    return render_template('index.html')

# used for uploading pictures
@app.route('/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# allowed image types 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['ALLOWED_EXTENSIONS']=ALLOWED_EXTENSIONS

def allowed_file(filename):
    """
    checks if file is allowed to be uploaded by checking its file extension
    against allowed extensions in config

    returns true if allowed, false if not
    """
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']