# Install Flask on your system by writing
#!pip install Flask
#Import all the required libraries
#Importing Flask
#render_template--> To render any html file, template

from flask import Flask, Response,jsonify,request

# Required to run the YOLOv8 model
import cv2

# YOLO_Video is the python file which contains the code for our object detection model
#Video Detection is the Function which performs Object Detection on Input Video
from YOLO_Video import video_detection
app = Flask(__name__)

app.config['SECRET_KEY'] = 'muhammadmoin'
#Generate_frames function takes path of input video file and  gives us the output with bounding boxes
# around detected objects

#Now we will display the output video with detection
def generate_frames(path_x = ''):
    # yolo_output variable stores the output for each detection
    # the output with bounding box around detected objects

    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)
        # Any Flask application requires the encoded image to be converted into bytes
        #We will display the individual frames using Yield keyword,
        #we will loop over all individual frames and display them as video
        #When we want the individual frames to be replaced by the subsequent frames the Content-Type, or Mini-Type
        #will be used
        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(path_x='../Videos/bikes.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')
    #return Response(generate_frames(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam')
def webcam():
    return Response(generate_frames(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)