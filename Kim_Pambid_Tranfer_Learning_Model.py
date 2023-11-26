import os, tensorflow as tf
from tensorflow.python.keras.utils.data_utils import get_file
#Source: https://www.tensorflow.org/lite/examples/object_detection/overview
#Source: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
#Source dataset of Coco Detection: https://cocodataset.org/#home
#Source: https://www.kaggle.com/code/meaninglesslives/efficientnet-eb0-eb5-model-comparisons
########################################################################
#Created by Kim Pambid 10/26/2023 1:30 pm MNL
#Create a model that use EfficientNet from tensorflow.hub
#a. Use OOP in Programming
#b. Download EfficientNet (Transfer learning)
#c. Load the model (Transfer learning)
#########################################################################

#download your model at  https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
#http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz

class Kims_Pambid_Transfer_Model:
    def __init__(self, model_url, transfer_model_path, model_name): 
        self.modelurl = model_url        
        self.transfer_model = transfer_model_path
        self.modelname = model_name                        

    # Download EfficientNet (Transfer learning)            
    def download_model(self):
        filename = os.path.basename(self.modelurl)
        self.modelname = filename[:filename.index('.')]
        
        print("File Name: " + filename)
        print("Model Name: " + self.modelname)
        
        self.cachedir="./Trained_Model" 
                 
        os.makedirs(self.cachedir, exist_ok=True)
        get_file(fname=filename,
                 origin=self.modelurl,
                 cache_dir=self.cachedir,
                 cache_subdir="checkpoints",
                 extract=True)
        return self.modelname
   
    # Load the model (Transfer learning)     
    def load_model(self):        
        #print("Loading Model: " + self.modelname)
        model_file_path = self.transfer_model
        print(os.path.exists(model_file_path))        
        print(model_file_path)
        tf.keras.backend.clear_session() # Resets all state generated by Keras.
        pambid_model = tf.saved_model.load(model_file_path)
        print("Model " + self.modelname + " loaded successfully")
        return pambid_model