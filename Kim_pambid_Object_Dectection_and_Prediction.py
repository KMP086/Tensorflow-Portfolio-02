########################################################################################################################################
#Created by Kim Pambid 10/26/2023 2:11 pm MNL
#a. Use OOP in Programming
#b. Read Classes for labelling
#c. Create Bounding box
#d. Predict the video
########################################################################################################################################
import cv2, time, tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file
np.random.seed(42) #color random change

class Kim_Pambid_Object_Detector_and_Prediction:
    def __init__(self, loaded_model, video_path, video_threshold, class_file_path):
        self.pambid_model = loaded_model
        self.videopath = video_path   
        self.threshold = video_threshold
        self.classfilepath = class_file_path
        pass        
            
    
    #Create Bounding box
    def create_bounding_box(self, image):
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis,...]

        detections = self.pambid_model(inputTensor)

        bboxs = detections["detection_boxes"][0].numpy()
        
        classindexes = detections["detection_classes"][0].numpy().astype(np.int32)
        classscores = detections["detection_scores"][0].numpy()

        imH, imW, imC = image.shape #image height, weight , and width
        
        # max_output_size=50 ~~ 50 boxes
        bboxidx = tf.image.non_max_suppression(bboxs, classscores, max_output_size=20, 
        iou_threshold=0.5, score_threshold=0.5)

        print(bboxidx)    
        if len(bboxidx) !=0:
            for i in bboxidx:
                bbox = tuple(bboxs[i].tolist())
                classconfidence = round(100*classscores[i])
                classindex = classindexes[i]                
                #print(classindex, len(self.classesList))

                classlabeltext = self.classesList[classindex]
                classcolor = self.colorlist[classindex]

                displayText = '{}: {}%'.format(classlabeltext, classconfidence)

                ymin, xmin, ymax, xmax = bbox

                xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH)
                xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

                cv2.rectangle(image,(xmin, ymin), (xmax, ymax), color=classcolor, thickness=1)
                cv2.putText(image, displayText, (xmin, ymin -10), cv2.FONT_HERSHEY_PLAIN, 1, classcolor, 2)
                ##Border Top####################################################################
                lineWidth = min(int((xmax - xmin) * 0.2), int((ymax - ymin) * 0.2))
                ##Left##
                cv2.line(image, (xmin, ymin), (xmin + lineWidth, ymin), classcolor, thickness=5)
                cv2.line(image, (xmin, ymin), (xmin, ymin + lineWidth), classcolor, thickness=5)
                ##Right##
                cv2.line(image, (xmax, ymin), (xmax - lineWidth, ymin), classcolor, thickness=5)
                cv2.line(image, (xmax, ymin), (xmax, ymin + lineWidth), classcolor, thickness=5)
                ##Border End#####################################################################
                ##Left##
                cv2.line(image, (xmin, ymax), (xmin + lineWidth, ymax), classcolor, thickness=5)
                cv2.line(image, (xmin, ymax), (xmin, ymax - lineWidth), classcolor, thickness=5)
                ##Right##
                cv2.line(image, (xmax, ymax), (xmax - lineWidth, ymax), classcolor, thickness=5)
                cv2.line(image, (xmax, ymax), (xmax, ymax - lineWidth), classcolor, thickness=5)

        return image

    #Predict the video    
    def kim_pambid_predict_video(self):
        #read classes
        with open(self.classfilepath, 'r') as f:
            self.classesList = f.read().splitlines()            
        #color each boxes
        self.colorlist = np.random.uniform(low=0, high=255, size=(len(self.classesList),3))        
        print(f"Number classes / Folder Names:  {len(self.classesList)}, Number color field per class {len(self.colorlist)}" )        

        #Read video
        cap = cv2.VideoCapture(self.videopath)
        if(cap.isOpened() == False):
          print("Error opening file...")
          return
        
        (success, image) = cap.read()  
        starttime =0    
        while success:
            currenttime = time.time()
            
            fps = 1/(currenttime - starttime)
            starttime = currenttime            
            bboximage = self.create_bounding_box(image) 
            cv2.putText(bboximage, 
                        "FPS: " + str(int(fps)), 
                        (20, 70), 
                        cv2.FONT_HERSHEY_PLAIN,2,
                        (0, 255, 0), 2)   
            cv2.imshow("Result", bboximage)
            key = cv2.waitKey(1) & 0xFF
            print("Press Q to quit!")
            if key == ord("q"):
                break
            (success, image) = cap.read()
        cv2.destroyAllWindows()