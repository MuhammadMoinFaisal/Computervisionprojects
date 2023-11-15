//Handle Image Input
function handleImageInput(event){
    const fileInput = event.target;
    const file = fileInput.files[0];
    if (file){
        const reader = new FileReader();
        reader.onload = function (e) {
            const imgMain = document.getElementById("img-main");
            imgMain.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
    
}
//Compute Color for Labels
function computeColorforLabels(className){
    if(className=='person'){
        color=[85, 45, 255,200];
      }
      else if (className='cup'){
        color=[255, 111, 0, 200]
      }
      else if (className='cellphone'){
        color=[200, 204, 255, 200]
      }
      else{
        color=[0,255,0,200];
      }
      return color;
}

function drawBoundingBox(predictions, image){
    predictions.forEach(
        prediction => {
            const bbox = prediction.bbox;
            const x = bbox[0];
            const y = bbox[1];
            const width = bbox[2];
            const height = bbox[3];
            const className = prediction.class;
            const confScore = prediction.score;
            const color = computeColorforLabels(className)
            console.log(x, y, width, height, className, confScore);
            let point1 = new cv.Point(x, y);
            let point2 = new cv.Point(x+width, y+height);
            cv.rectangle(image, point1, point2, color, 2);
            const text = `${className} - ${Math.round(confScore*100)/100}`;
            const font =  cv.FONT_HERSHEY_TRIPLEX;
            const fontsize = 0.70;
            const thickness = 1;
            //Get the size of the text
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            const textMetrics = context.measureText(text);
            const twidth = textMetrics.width;
            console.log("Text Width", twidth);
            cv.rectangle(image, new cv.Point(x, y-20), new cv.Point(x + twidth + 150,y), color, -1);
            cv.putText(image, text, new cv.Point(x, y-5), font, fontsize, new cv.Scalar(255, 255, 255, 255), thickness);
        }
    )
}
function OpenCVReady(){
    cv["onRuntimeInitialized"]=()=>{
        console.log("OpenCV Ready");
        let imgMain = cv.imread("img-main");
        cv.imshow("main-canvas", imgMain)
        imgMain.delete();
        
        //Handle Image Input
        document.getElementById("image-upload").addEventListener('change', handleImageInput);

        //RGB Image
        document.getElementById("RGB-Image").onclick = function(){
            console.log("RGB Image");
            let imgMain = cv.imread("img-main");
            cv.imshow("main-canvas", imgMain);
            imgMain.delete();
        }

        //Gray Scale Image
        document.getElementById("Gray-Scale-Image").onclick = function(){
            console.log("Gray Scale Image");
            let imgMain = cv.imread("img-main");
            let imgGray = new cv.Mat();
            cv.cvtColor(imgMain, imgGray, cv.COLOR_RGBA2GRAY);
            cv.imshow("main-canvas", imgGray);
            imgMain.delete();
            imgGray.delete();
        }

        //Object Detection Image
        document.getElementById("Object-Detection-Image").onclick = function(){
            console.log("Object Detection Image");
            const image = document.getElementById("img-main");
            let inputImage = cv.imread(image);
            cocoSsd.load().then(model => {
                model.detect(image).then(predictions =>{
                    console.log("Predictions", predictions)
                    console.log("Length of Predictions", predictions.length)
                    if (predictions.length > 0){
                        drawBoundingBox(predictions, inputImage);
                        cv.imshow("main-canvas", inputImage)
                        inputImage.delete();
                    
                    }
                    else{
                        cv.imshow("main-canvas", inputImage);
                        inputImage.delete();
                    }
                })
            })
        };

        //Object Detection Live Webcam Feed
        document.getElementById("Object-Detection-Video").onclick = function(){
            console.log("Object Detection Live Webcam Feed");
            const video = document.getElementById("webcam");
            const enableWebcamButton = document.getElementById("enable-webcam-button");
            let model = undefined;
            let streaming = false;
            let src;
            let cap;
            const FPS = 24;

            //Browser Feature Detection
            if (!!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)){
                enableWebcamButton.addEventListener('click', ()=>{
                    if (!streaming){
                        console.log("Streaming Started");
                        enableCam();
                        streaming = true;
                    }
                    else{
                        console.log("Streaming Paused");
                        video.pause();
                        video.srcObject = null;
                        streaming = false;
                    }
                })
            }
            else{
                console.log("getUserMedia is not suported in your browser")
            }

            //Enable Cam Function
            function enableCam(){
                if(!model){
                    return;
                }
                navigator.mediaDevices.getUserMedia({'video':true, 'audio':false}).then(function(stream){
                    video.srcObject = stream;
                    video.addEventListener('loadeddata', predictWebcam);
                })
            }

            setTimeout(function(){
                cocoSsd.load().then(function (loadedModel){
                    model = loadedModel;
                    console.log("Model Loaded");
                });
            }, 0);

            function predictWebcam(){
                if (video.videoWidth > 0 && video.videoHeight > 0){
                    const begin = Date.now();
                    src = new cv.Mat(video.videoHeight, video.videoWidth, cv.CV_8UC4);
                    cap = new cv.VideoCapture(video);
                    cap.read(src);
                    model.detect(video).then(function (predictions) {
                        console.log("Predictions", predictions)
                        if (predictions.length > 0){
                            drawBoundingBox(predictions, src);
                            cv.imshow("main-canvas", src);
                            const delay = 1000/FPS - (Date.now() - begin);
                            setTimeout(predictWebcam, delay);
                            src.delete();
                        }
                        else{
                            cv.imshow("main-canvas", src);
                            const delay = 1000/FPS - (Date.now() - begin);
                            setTimeout(predictWebcam, delay);
                            src.delete();
                        }

                    })
                } 

                else{
                    // Video dimensions are not valid yet, wait for the next frame
                    window.requestAnimationFrame(predictWebcam);
                }

            }
        };


    }
}