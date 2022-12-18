import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline
from flask import Flask,Response,render_template,request
import os

# Declare Global Variables
v, m = 0, 0

from pathlib import Path
print(Path.cwd())
# Dictionary with all Filters
filters = {"Grey Scale":1,"Bright Night":2,"Gloomy Day":3,"Blurry View":4,"Cartoonify":5,"Sharp View":6,"Nostalgia":7,"Pencil Sketch":8,"HDR Effect":9,"Pixel Invert":10,"Summer Effect":11,"Winter Effect":12}

# CV2 Video Capture 
video = cv2.VideoCapture(0)

# Get width and height ofe window
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(width,height)
# Declare default image for applying filters
image=cv2.imread("/app/cat_photos.jpg",cv2.IMREAD_UNCHANGED)
print(image.shape)
# Resize input image 
image1=cv2.resize(image, (width, height))

# Initialize Flask Object
app = Flask(__name__)

# Smart Phone Camera like Filters using OpenCV - Filters on Live Video
# Listed down methods that can apply most common camera filters on either live videl or images using OpenCV

# Grayscale conversion for Black & White effect
def grayScaleFilter(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

# Brighten image pixels 
def nightBrightenerFilter(image):
    alpha = 2 
    beta = 0 
    image = cv2.convertScaleAbs(image,alpha=alpha, beta=beta)
    return image

# Darken image pixels
def DayDarkenerFilter(image):
    alpha = 0.5
    beta = 0 
    image = cv2.convertScaleAbs(image,alpha=alpha, beta=beta)
    return image

# Blur an image 
def blurFilter(image):
    ksize = (10, 10)
    image = cv2.blur(image, ksize)
    return image

# Creating a cartoon effect on image
# Approach : 
# ---> Convert to grayscale
# ---> Medien Blur to reduce noise
# ---> Locating the edges using adaptiveThreshold
# ---> Getting a mask image by creating a bitwise and between original image and processed image
# This creates a cartoon like effect
def cartoonifyFilter(image):
    org_img = cv2.bilateralFilter(image, 9, 300, 300)
    gray = grayScaleFilter(image)
    image1= cv2.medianBlur(gray, 3)
    image2 = cv2.adaptiveThreshold(image1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    image = cv2.bitwise_and(org_img, org_img, mask=image2)
    return image

# Sharpen the image 
def sharpenImageFilter(image):
    kernel_array = np.array([[-1, -1, -1], [-1, 9.3, -1], [-1, -1, -1]])
    image = cv2.filter2D(image, -1, kernel_array)
    return image

# Creating a nostalgia like cool effect
def nostalgiaFilter(image):
    image = np.array(image, dtype=np.float64)
    image = cv2.transform(image, np.matrix([[0.365, 0.453, 0.284],[0.764, 0.234, 0.143],[0.354, 0.785, 0.167]])) 
    image[np.where(image > 255)] = 255
    image = np.array(image, dtype=np.uint8)
    return image

# Pencil Sketch effect
def pencilSketchFilter(image):
    #inbuilt function to create sketch effect in colour and greyscale
    _, image = cv2.pencilSketch(image, sigma_s=70, sigma_r=0.06, shade_factor=0.1) 
    return  image

# HDR Effect by enhancing details in the image
def hdrEffect(image):
    image = cv2.detailEnhance(image, sigma_s=13, sigma_r=0.17)
    return image

# Invert pixels in image
def pixelInvertFilter(image):
    image = cv2.bitwise_not(image)
    return image

# Increase warmth of image to give summer like light effect by increasing red channel values
def summerEffectFilter(image):
    inctable = UnivariateSpline([0, 64, 128, 256], [0, 70, 140, 256])(range(256))
    dectable = UnivariateSpline([0, 64, 128, 256], [0, 60, 120, 256])(range(256))
    blue, green,red  = cv2.split(image)
    red = cv2.LUT(red, inctable).astype(np.uint8)
    blue = cv2.LUT(blue, dectable).astype(np.uint8)
    image = cv2.merge((blue, green, red ))
    return image

# Decrease warmth of image to give winter like effect by increasing blue channel values
def winterEffectFilter(image):
    inctable = UnivariateSpline([0, 64, 128, 256], [0, 70, 140, 256])(range(256))
    dectable = UnivariateSpline([0, 64, 128, 256], [0, 60, 120, 256])(range(256))
    blue, green,red  = cv2.split(image)
    red = cv2.LUT(red, dectable).astype(np.uint8)
    blue = cv2.LUT(blue, inctable).astype(np.uint8)
    image = cv2.merge((blue, green, red ))
    return image

# Apply Filter based on user choise to a live video
def video_process(video):
    while True:
        success, image = video.read()
        if v==1:
            image = grayScaleFilter(image)
        elif v==2:
            image = nightBrightenerFilter(image)
        elif v==3:
            image = DayDarkenerFilter(image)
        elif v==4:
            image = blurFilter(image)
        elif v==5:
            image = cartoonifyFilter(image)
        elif v==6:
            image = sharpenImageFilter(image)
        elif v==7:
            image = nostalgiaFilter(image)
        elif v==8:
            image = pencilSketchFilter(image)
        elif v==9:
            image = hdrEffect(image)
        elif v==10:
            image = pixelInvertFilter(image)
        elif v==11:
            image = summerEffectFilter(image)
        elif v==12:
            image = winterEffectFilter(image)
        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Apply Filter based on user choise to image
def image_process(image):
    if v==1:
        image = grayScaleFilter(image)
    elif v==2:
        image = nightBrightenerFilter(image)
    elif v==3:
        image = DayDarkenerFilter(image)
    elif v==4:
        image = blurFilter(image)
    elif v==5:
        image = cartoonifyFilter(image)
    elif v==6:
        image = sharpenImageFilter(image)
    elif v==7:
        image = nostalgiaFilter(image)
    elif v==8:
        image = pencilSketchFilter(image)
    elif v==9:
        image = hdrEffect(image)
    elif v==10:
        image = pixelInvertFilter(image)
    elif v==11:
        image = summerEffectFilter(image)
    elif v==12:
        image = winterEffectFilter(image)
        
    ret, jpeg = cv2.imencode('.jpg', image)
    frame = jpeg.tobytes()
    yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Main Landing Page
@app.route('/')
def index():
    return render_template('home.html')

# Video Processing Landing PAge
@app.route("/video_filters/",methods=['POST','GET'])
def swap1():
    global m
    m=1
    return render_template('video_main.html')

# Video Process Filters Request
@app.route('/video_filters/video_feed')
def video_feed():
    return Response(video_process(video), mimetype='multipart/x-mixed-replace; boundary=frame')

# Image Processing Landing Page
@app.route("/image_filters/",methods=['POST','GET'])
def swap2():
    global m
    m=2
    return render_template('image_main.html')

# Image Process Filters Request
@app.route('/image_filters/img_share')
def img_share():
    global image
    image=cv2.resize(image, (480, 480))
    return Response(image_process(image), mimetype='multipart/x-mixed-replace; boundary=frame')

# Request to Upload Image from Local Directory
@app.route('/image_upload/',methods=['POST','GET'])
def take_img():
    global image
    if request.method == 'POST':
        f = request.files['image']
        print(f.filename)
        path = f.filename
        f.save(path)
        image=cv2.imread(path)
        print(image.shape)
    return render_template('image_main.html')

# Choose Filter Request
@app.route('/requests/',methods=['POST','GET'])
def tasks():
    global v ,image,filters
    
    if request.method == 'POST':
        try:
            req = list(request.form)[0]
        except:
            req = "Reset Filters"
        if req in filters.keys():
            print(req)
            v=filters[req]
            print(v)
        elif request.form.get('2'):
            return render_template('file.html')
        else:
            v=0
    if m==1:
        return render_template('video_main.html')
    elif m==2:
        if request.form.get('1'):
            success, image = video.read()
        return render_template('image_main.html')

    
if __name__ == '__main__':
    app.run()