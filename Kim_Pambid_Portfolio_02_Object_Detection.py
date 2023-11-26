from Kim_Pambid_Tranfer_Learning_Model import *
from Kim_pambid_Object_Dectection_and_Prediction import *
#################################################################################################################################################
#Created by Kim Pambid 10/26/2023 4:01 pm MNL
#a. Get the URL of the transfer Model or Pretrained model
#b. Set the video sample directory
#c. Set the classnames
#d. Set the OOP and Parameters
#e. Activate and predict the Sample
#################################################################################################################################################
#Get the URL of the transfer Model or Pretrained model
#Note no need to train and test images it was already included
modelurl = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz"

#Set the pretrained directory
transfermodel = "E:\\Programming\\Projects\\py_tensorflow\\Portfolio 02\\Trained_Model\\checkpoints\\efficientdet_d0_coco17_tpu-32\\saved_model"

#Set the video sample directory
#video should be at 1024x1024 pixel
videopath = "E:\\Programming\\Projects\\py_tensorflow\\Portfolio 02\\video sample\\Manila2_resized.mp4"

#Set the classnames
classfile = "E:\\Programming\\Projects\\py_tensorflow\\Portfolio 02\\Kim_Objects_Classes.names"

#############################################################################################################################
##Set the OOP and Parameters
###Load the pretrained model
modelname = Kims_Pambid_Transfer_Model(model_url = modelurl, 
                                       transfer_model_path = transfermodel, 
                                       model_name = "").download_model()


#The pre-trained model
model_of_pambid = Kims_Pambid_Transfer_Model(model_url = modelurl, 
                                             transfer_model_path= transfermodel,  
                                             model_name = modelname).load_model()

#############################################################################################################################
#Activate the Sample
Kim_Pambid_Object_Detector_and_Prediction(loaded_model= model_of_pambid, 
                                          video_path= videopath, 
                                          video_threshold=0.5,
                                          class_file_path= classfile).kim_pambid_predict_video()
