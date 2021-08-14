#ref: https://github.com/serengil/deepface.git
import random
import multiprocessing
import re
import dlib
from pandas.core import frame
#dlib.DLIB_USE_CUDA=False
print("dlib.DLIB_USE_CUDA: {}".format( dlib.DLIB_USE_CUDA ))
# if(dlib.DLIB_USE_CUDA==True):
#     import cupy as np
# else: 
#     import numpy as np
import numpy as np
from PIL import Image
import math
#from ast import dump
#from mtcnn import MTCNN

import os
import bz2
from uuid import UUID
import cv2
import gdown
from pathlib import Path

from sklearn.svm import SVC
import pickle
import json

from multiprocessing import Process, Queue
from threading import Thread
from queue import Empty,Full
import datetime
import time
print("import owner lib")


# #want to use CPU have to uncomment bellow to disable GPU
# try:
#     # Disable all GPUS
#     tf.config.set_visible_devices([], 'GPU')
#     visible_devices = tf.config.get_visible_devices()
#     for device in visible_devices:
#         assert device.device_type != 'GPU'
# except:
#     # Invalid device or cannot modify virtual devices once initialized.
#     # Disable all GPUS
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#     pass

class DlibMetaData:
    def __init__(self):
        self.input_shape = [[1, 150, 150, 3]]


# import tensorflow as tf
# tf_version = int(tf.__version__.split(".")[0])

# if tf_version == 1:
# 	import keras
# 	from keras.preprocessing.image import load_img, save_img, img_to_array
# 	from keras.applications.imagenet_utils import preprocess_input
# 	from keras.preprocessing import image
# elif tf_version == 2:
# 	from tensorflow import keras
# 	from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
# 	from tensorflow.keras.applications.imagenet_utils import preprocess_input
# 	from tensorflow.keras.preprocessing import image

class DlibResNet:

    def __init__(self):

        self.__FileFolder = os.path.dirname(os.path.realpath(__file__))
        self.file_weights = self.__FileFolder + \
            '/weights/dlib_face_recognition_resnet_model_v1.dat'
        # this is not a must dependency
        import dlib  # 19.20.0

        self.layers = [DlibMetaData()]

        # download pre-trained model if it does not exist
        if os.path.isfile(self.file_weights) != True:
            print("dlib_face_recognition_resnet_model_v1.dat is going to be downloaded")

            url = "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
            output = self.file_weights+url.split("/")[-1]
            gdown.download(url, output, quiet=False)

            zipfile = bz2.BZ2File(output)
            data = zipfile.read()
            #newfilepath = output[:-4]  # discard .bz2 extension
            open(self.file_weights, 'wb').write(data)

        # ---------------------

        model = dlib.face_recognition_model_v1(self.file_weights)
        self.__model = model

        # ---------------------

        return None  # classes must return None

    def predict(self, img_aligned):
        try:            
            #http://dlib.net/face_recognition.py.html
            # functions.detectFace returns 4 dimensional images
            
            if len(img_aligned.shape) == 4:
                img_aligned = img_aligned[0]

            # functions.detectFace returns bgr images
            img_aligned = img_aligned[:, :, ::-1]  # bgr to rgb

            # deepface.detectFace returns an array in scale of [0, 1] but dlib expects in scale of [0, 255]
            if img_aligned.max() <= 1:
                img_aligned = img_aligned * 255

            img_aligned = img_aligned.astype(np.uint8)

            model = self.__model
            #small 5 point pose
            
            #10: more detail but slow 10 times
            #img_representation = model.compute_face_descriptor(img_aligned,10)
            
            img_representation = model.compute_face_descriptor(img_aligned,1)

            img_representation = np.array(img_representation)
            img_representation = np.expand_dims(img_representation, axis=0)

            return img_representation
        except Exception as ex:
            raise Exception(ex)
            
    
    def face_vector(self,img_face_croped):
        try:
            #t1= datetime.datetime.now().timestamp()
            imgResized=self.normalize_face(img_face_croped, 150, 150)            
            #t2= datetime.datetime.now().timestamp()
            #print("{} resize".format(t2-t1))
            vector=self.predict(imgResized)[0].tolist()
            
            #t2= datetime.datetime.now().timestamp()
            #print("{} vector".format(t2-t1))
            # vectorR=[]
            # for i in vector:
            #     vectorR.append( round(i,8))
            
            # print(vectorR)
            # vector=self.predict(self.normalize_face(img_face_croped, 150, 150))[0,:]
            # print(vector)
            # exit(0)
            return (vector,imgResized)
        except:
            print("Error to get face vector")
            return None
    
    def normalize_face(self,img, w, h):
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgResized = cv2.resize(img, (w, h))

        return imgResized
       
        # img_pixels = image.img_to_array(imgResized)
        
        # img_pixels = np.expand_dims(img_pixels, axis = 0)
        # img_pixels /= 255 #normalize input in [0, 1]
      
        # return img_pixels
        
class DlibDetector:
    def __init__(self):

        self.__FileFolder = os.path.dirname(os.path.realpath(__file__))

        import dlib  # this requirement is not a must that's why imported here
        self.file_weights = self.__FileFolder + '/weights/shape_predictor_5_face_landmarks.dat'
        #self.file_weights = self.__FileFolder + '/weights/shape_predictor_68_face_landmarks.dat'
            #shape_predictor_68_face_landmarks.dat
        # print(os.path.isfile(self.file_weights))
        # exit(0)
        # check required file exists in the home/.deepface/weights folder
        self.gpuFileWeight= self.__FileFolder +'/weights/mmod_human_face_detector.dat'

        if os.path.isfile(self.file_weights) != True:

            print("shape_predictor_5_face_landmarks.dat.bz2 is going to be downloaded")

            url = "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2"
            output = self.file_weights+url.split("/")[-1]

            gdown.download(url, output, quiet=False)

            zipfile = bz2.BZ2File(output)
            data = zipfile.read()
            # newfilepath = output[:-4]  # discard .bz2 extension
            open(self.file_weights, 'wb').write(data)

        self.face_detector = dlib.get_frontal_face_detector()

        #http://dlib.net/cnn_face_detector.py.html
        self.face_detector_gpu= dlib.cnn_face_detection_model_v1( self.gpuFileWeight)
        
        self.object_detector = dlib.full_object_detections()

        self.shape_predict = dlib.shape_predictor(self.file_weights)

    def DetectFace(self, imgInput, align=True,padding=0.25):
        """[summary]
        http://dlib.net/face_alignment.py.html
        Args:
            imgInput ([type]): [cv2.imread]
            align (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [List of tuple(face,rect,keypoints)]
        """
        temp=[]
        try:
            if(dlib.DLIB_USE_CUDA):
                temp= self.detect_face_gpu(imgInput, align, padding) 
                print("GPU detector")
            else:
                temp= self.detect_face(imgInput, align, padding) 
        except Exception as ex:
            temp= self.detect_face(imgInput, align, padding) 
        # for f,r in temp:
        #     cv2.imshow("face_", f)
        #     cv2.waitKey(1)
        return temp

    def detect_face_gpu(self, imgInput, align=True,padding=0.25):
        """[summary]
        http://dlib.net/face_alignment.py.html
        Args:
            imgInput ([type]): [cv2.imread]
            align (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [List of tuple(face,rect,keypoints)]
        """
        #img = dlib.load_rgb_image("file path")
        detections=self.face_detector_gpu(imgInput,1)
        listDetected=[]
        
        if (len(detections) > 0):
            for idx, dv in enumerate(detections):     
                try:         
                    d=dv.rect
                    left = d.left()#x0
                    right = d.right()#x1
                    top = d.top()#y0
                    bottom = d.bottom()#y1
                    detected_face = imgInput[top:bottom, left:right]
                    detected_face_region = [left, top, right - left, bottom - top]  

                    points=[]
                    img_shape = self.shape_predict(imgInput, d)
                    for p in img_shape.parts():
                            points.append((p.x,p.y))
                    #         cv2.rectangle(imgInput,(p.x,p.y),(p.x+2,p.y+2),(0,255,0,0),2)
                    # cv2.imshow("xxxx",imgInput)
                    # cv2.waitKey(1)
                    if align:                                                    
                        detected_face = dlib.get_face_chip(
                            imgInput, img_shape, size=detected_face.shape[0],padding=padding)

                        """
                        shape_predictor_68_face_landmarks
                        "chin": points[0:17],
                        "left_eyebrow": points[17:22],
                        "right_eyebrow": points[22:27],
                        "nose_bridge": points[27:31],
                        "nose_tip": points[31:36],
                        "left_eye": points[36:42],
                        "right_eye": points[42:48],
                        "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
                        "bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [points[64]]
     
                        shape_predictor_5_face_landmarks
                            "nose_tip": [points[4]],
                        "left_eye": points[2:4],
                            "right_eye": points[0:2],
                        """

                    listDetected.append((detected_face, detected_face_region,points))
                except Exception as ex:
                    print("Error detect_face_gpu")
                    print(ex)
                    pass

        return listDetected

    def detect_face(self, imgInput, align=True,padding=0.25):
        """[summary]
        http://dlib.net/face_alignment.py.html
        Args:
            imgInput ([type]): [cv2.imread]
            align (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [List of tuple(face=cv2.Mat,rect=[x,y,h,w],keypoints)]
        """
        import dlib  # this requirement is not a must that's why imported here
        
        #print(sp)
        #exit(0)
        #img_region = [0, 0, imgInput.shape[0], imgInput.shape[1]]

        detections = self.face_detector(imgInput, 1)
        listDetected=[]
        
        if (len(detections) > 0):
            for idx, d in enumerate(detections):    
                try:            
                    left = d.left()#x0
                    right = d.right()#x1
                    top = d.top()#y0
                    bottom = d.bottom()#y1
                    detected_face = imgInput[top:bottom, left:right]
                    detected_face_region = [left, top, right - left, bottom - top]  
                    
                    points=[]
                    img_shape = self.shape_predict(imgInput, d)
                    for p in img_shape.parts():
                            points.append((p.x,p.y))
                    
                    if align:                        
                        # print(img_shape)
                        #https://github.com/davisking/dlib/blob/master/examples/dnn_face_recognition_ex.cpp
                        #https://github.com/davisking/dlib/blob/master/python_examples/face_landmark_detection.py
                        # print("Part 0: {}, Part 1: {} ...".format(img_shape.part(0),
                        #                           img_shape.part(1)))

                        detected_face = dlib.get_face_chip(
                            imgInput, img_shape, size=detected_face.shape[0],padding=padding)
                        #cv2.imshow("face", detected_face)
                        #cv2.waitKey(1)
                    listDetected.append((detected_face, detected_face_region,points))
                except Exception as ex:
                    print("Error detect_face")
                    print(ex)

        return listDetected

    # def normalize(self,image, fixed=False):
    #     if fixed:
    #         return (np.float32(image) - 127.5) / 127.5
    #     else:
    #         mean = np.mean(image)
    #         std = np.std(image)
    #         std_adj = np.maximum(std, 1.0 / np.sqrt(image.size))
    #         y = np.multiply(np.subtract(image, mean), 1 / std_adj)
    #         return y

class VectorCompare:
    def __init__(self):
        pass
    
    def Linalg(self, listVector, vector):
        np.linalg.norm(listVector- vector,axis=1)


    def findCosineDistance(self, source_representation, test_representation):
        try:
            if type(source_representation) == list:
                source_representation = np.array(source_representation)

            if type(test_representation) == list:
                test_representation = np.array(test_representation)
            
            a = np.matmul(np.transpose(source_representation), test_representation)
            b = np.sum(np.multiply(source_representation, source_representation))
            c = np.sum(np.multiply(test_representation, test_representation))
            return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
        except Exception as ex:
            print("Error findCosineDistance")
            print(ex)

    def findEuclideanDistance(self, source_representation, test_representation):
        try:
            if type(source_representation) == list:
                source_representation = np.array(source_representation)

            if type(test_representation) == list:
                test_representation = np.array(test_representation)
            
            source_representation= self.l2_normalize(source_representation)
            test_representation= self.l2_normalize(test_representation)

            euclidean_distance = source_representation - test_representation
            euclidean_distance = np.sum(np.multiply(
                euclidean_distance, euclidean_distance))
            euclidean_distance = np.sqrt(euclidean_distance)
            return euclidean_distance
        except Exception as ex:
            print("Error findEuclideanDistance")
            print(ex)

    def l2_normalize(self, x):
        return x / np.sqrt(np.sum(np.multiply(x, x)))
    
    def predictCosi(self,vector, arrayVectors, arrayLabels, takeTop=2, threshold=0):
        try:
            resCompare=[]
            for idx, fDec in enumerate(arrayVectors):
                distance = self.findCosineDistance(fDec, vector)            
                if(threshold>0 and distance>threshold):
                    continue
                resCompare.append((arrayLabels[idx],distance))   


            resCompare= sorted(resCompare,key= lambda tup:tup[1])
            
            resCompare=resCompare[:takeTop]
            
            lblArr={}
            for (l,d) in resCompare:
                lblArr[l]=d

            if(len(lblArr)>1):
                return (None,None)
            
            return (resCompare[0][0],round(resCompare[0][1],3))

        except Exception as ex:
            print("Error pridict cosi: ")
            print(ex)
            return (None,None)
            pass
    
    def predictEcli(self,vector, arrayVectors, arrayLabels,takeTop=2, threshold=0):
        try:
            resCompare=[]
            for idx, fDec in enumerate(arrayVectors):
                distance = self.findEuclideanDistance(fDec, vector)            
                if(threshold>0 and distance>threshold):
                    continue
                resCompare.append((arrayLabels[idx],distance))   

            resCompare= sorted(resCompare,key= lambda tup:tup[1])[:takeTop]
            
            # lblArr={}
            # for (l,d) in resCompare:
            #     lblArr[l]=d

            # if(len(lblArr)>1):
            #     return (None,None)
            
            return (resCompare[0][0],round(resCompare[0][1],3))

        except Exception as ex:
            print("Error pridict ecli: ")
            print(ex)
            return (None,None)
            pass

    def predictCosiByMin(self, vector, arrayVectors, arrayLabels):
        """[summary]

        Args:
            vector ([type]): [description]
            arrayVectors ([type]): [description]
            arrayLabels ([type]): [description]

        Returns:
            [type]: (lbl,disntace)
        """
        try:
            resCompare=[]
            for fDec in arrayVectors:
                distance = self.findCosineDistance(fDec, vector)                
                resCompare.append(distance)                
            
            resCompare=np.array(resCompare)        
            idxDlib = np.argmin( resCompare)
            minDistance =np.amin(resCompare)
            lbl=arrayLabels[idxDlib]

            return (lbl,round( minDistance,3))
        except Exception as ex:
            print("Error predictCosiByMin")
            print (ex)
            return (None,None)
    
    def predictEcliByMin(self, vector, arrayVectors, arrayLabels):
        """[summary]

        Args:
            vector ([type]): [description]
            arrayVectors ([type]): [description]
            arrayLabels ([type]): [description]

        Returns:
            [type]: (lbl,disntace)
        """
        try:
            resCompare=[]
            for fDec in arrayVectors:
                distance = self.findEuclideanDistance(fDec, vector)
                resCompare.append(distance)

            resCompare=np.array(resCompare)
            idxDlib = np.argmin( resCompare)
            minDistance =np.amin(resCompare)

            lbl=arrayLabels[idxDlib]
            return (lbl,round( minDistance,3))
        except Exception as ex:
            print("Error predictEcliByMin")
            print (ex)
            return (None,None)

class ImageAugmentation:
    def __init__(self) :
        pass

    def rotate(self, image, angle=90, scale=1.0):
        '''
        Rotate the image
        :param image: image to be processed
        :param angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
        :param scale: Isotropic scale factor.
        '''
        h,w,c = image.shape
        
        #rotate matrix
        M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
        #rotate
        image = cv2.warpAffine(image,M,(w,h))
        return image

    def flip(self, image, vflip=False, hflip=False):
        '''
        Flip the image
        :param image: image to be processed
        :param vflip: whether to flip the image vertically
        :param hflip: whether to flip the image horizontally
        '''
        if hflip or vflip:
            if hflip and vflip:
                c = -1
            else:
                c = 0 if vflip else 1
            image = cv2.flip(image, flipCode=c)
        return image 

class SvmFaceClassifier:
    def __init__(self, vectors=[], labels=[]):
        #self.model = SVC(kernel='rbf', probability=True,gamma='auto')
        self.model = SVC(kernel='linear', probability=True)
        self.faceVectors=vectors
        self.faceLabels=labels

    def Train(self,vectors=[], labels=[]):
        """[summary]

        Args:
            vectors [[],[]]: [same length and order to labels]. Defaults to [].
            labels ['a','b']: [same length and order to vectors]. Defaults to [].
        """
        if(len(vectors)>0 ):
            self.faceVectors = vectors
            self.faceLabels = labels        
        
        self.model.fit(np.array( self.faceVectors), np.array(self.faceLabels))

        pass

    def SaveModel(self,modelPath=""):

        self.__FileFolder = os.path.dirname(os.path.realpath(__file__))
        if(modelPath==""):
            modelPath= self.__FileFolder+"/svm.pkl"

        pickle.dump(self.model,open(modelPath, 'wb'))
    
    def LoadModel(self,modelPath="",labels=[] ):
        self.__FileFolder = os.path.dirname(os.path.realpath(__file__))
        if(modelPath==""):
            modelPath= self.__FileFolder+"/svm.pkl"
        
        self.model = pickle.load(open(modelPath, 'rb'))
        
        self.faceLabels = self.model.classes_ 

        #result = loaded_model.score(X_test, Y_test)
    
    def PredictProba(self, vector):
        """[summary]

        Args:
            vector ([type]): [array of 128 item of face encoding]

        Returns:
            [type]: [description]
        """
        probs= self.model.predict_proba([vector])
        
        return probs
    
    def Predict(self,vector):
        """[summary]

        Args:
            vector ([type]): [array of 128 item of face encoding]
        """
        try:
            probs =self.model.predict_proba([vector])[0]
            #print("-------------------")
            #print(self.model.predict([vector]))

            svmMaxDistanceIdx = np.argmax(probs)
            
            svmResult =self.faceLabels[svmMaxDistanceIdx]
            
            svmProbability= round( float( probs[svmMaxDistanceIdx]),3)
            
            return(svmResult,svmProbability)

        except Exception as ex:
            print("Error: SvmFaceClassifier.Predict")
            print(ex)
            return(None,None)
            pass
class DlibSingleThread:
    """DlibSingleThread Better version, 2 thread cpu
    """
    def __init__(self, cameraUrl) :
        
        #self.detector = MtcnnDetector()        
        self.detector = DlibDetector()

        self.encoderDlib = DlibResNet()        
        #self.faceNetEncoder = FaceNet()
        
        self.comparer = VectorCompare()
        self.svmFaceClassifierDlib = SvmFaceClassifier()
        
        self.arrVectorDlib=[]
        self.arrVectorFacenet=[]
        self.arrLabel=[]     

        self.last_sent_notification={}
        self._cameraUrl=cameraUrl

        self.ratio=1

        self.withRecoginze=True

        self._lastFrame=Queue()
        self._lastDetected=Queue()
        self._lastRecongize=Queue()

        self.lastHitGamma="1"
        self.smartGamma=["1","1.5","0.5"]
        self.smartGammaFunc={
            "1.5":(lambda f: DlibSingleThread.adjust_gamma(f,1.5)),
            "0.5":(lambda f: DlibSingleThread.adjust_gamma(f,0.5)),
            "1":(lambda f:f)
        }

        # self._frameDetectFaceProcess=Thread(target=self.LoopDetectAndRecognize
        # , args=(self._cameraUrl, self._lastFrame, self._lastDetected, self._lastRecongize)
        # ,  daemon=True)
        self.threadGetShowVideo = Process(target=DlibSingleThread.GetAndShowFrame 
        , args=(self._cameraUrl, self.withRecoginze,self.ratio, self._lastFrame, self._lastDetected, self._lastRecongize)
        , daemon=True)

        pass
   
    def InitDataTest(self):
        imgAugm= ImageAugmentation()
        currentDir = os.path.dirname(os.path.realpath(__file__)).replace("\\","/")

        id="imagetest"

        fileRawVectorLbl = currentDir +"/{}.vectorlbl.raw.txt".format(id)
        fileDlibModel= currentDir +"/{}.svc.dlib.model".format(id)
        fileFacenetModel= currentDir+"/{}.svc.facenet.model".format(id)

        folderFound=currentDir+"/{}.foundfaces".format(id)
        
        if( not os.path.exists(folderFound)):
            os.makedirs(folderFound)
 
        print("root app: "+  currentDir )
        print("faces to: "+ folderFound)
        print("fileRawVectorLbl: "+ fileRawVectorLbl)
        print("fileDlibModel: "+ fileDlibModel)
        print("fileFacenetModel: "+ fileFacenetModel)
        
        if(os.path.exists(fileRawVectorLbl)):
            rawVectorLbl = Path(fileRawVectorLbl).read_text()
            obj = json.loads(rawVectorLbl)
            self.arrVectorDlib=obj["dlibvectors"]
            self.arrVectorFacenet=obj["facenetvectors"]
            self.arrLabel=obj["labels"]
            
            self.svmFaceClassifierDlib.LoadModel(fileDlibModel, self.arrLabel)     
            #self.svmFaceClassifierFacenet.LoadModel(fileFacenetModel, self.arrLabel)       

            print("loaded model")
            print("Lbl: "+ str(len(self.arrLabel)))
            print("Dlib: "+str(len(self.arrVectorDlib)))
            print("Facenet: "+str(len(self.arrVectorFacenet)))
            
            return

        du = cv2.imread(currentDir+"/imgtest/du.png")        
        du1 = cv2.imread(currentDir+"/imgtest/dud123.jpg")
        lien = cv2.imread(currentDir+"/imgtest/kimlien3.jpg")        
        lien1 = cv2.imread(currentDir+"/imgtest/kimlien2.png")        
        lien2 = cv2.imread(currentDir+"/imgtest/kimlien.jpg")
        tanh = cv2.imread(currentDir+"/imgtest/aantt.png")

        listFaceImg=[du,du1,lien,lien1,lien2,tanh]
        self.arrLabelDefinition=["du","du","lien","lien","lien","tanh"]
        self.arrLabel=[]
        self.arrVectorDlib=[]
        self.arrVectorFacenet=[]
        # init data
        for idx,f in enumerate( listFaceImg):
            lbl= self.arrLabelDefinition[idx]
            h,w,c = f.shape
            ratio=0.5
            while(w>1800): 
                f= cv2.resize(f,(0,0),fx=ratio,fy=ratio)
                h,w,c = f.shape
                ratio= ratio*1.5
                if(ratio>1):
                    break           
            ffound=None
            ffound=self.detector.DetectFace(f)
            
            if(len(ffound)>0):
                fcropO,rrect,points = ffound[0]

                fcropAugm= [fcropO
                            #,imgAugm.rotate(fcropO,3),imgAugm.rotate(fcropO,-3),
                            #imgAugm.rotate(fcropO,10),imgAugm.rotate(fcropO,-10),
                            #imgAugm.rotate(fcropO,15),imgAugm.rotate(fcropO,-15)
                            ]

                for idxf,fcrop in enumerate( fcropAugm):
                    vector,imgresized =self.encoderDlib.face_vector(fcrop) #[[]]
                    #vectorFn= self.faceNetEncoder.face_vector(fcrop)
                    
                    self.arrVectorDlib.append(vector)         
                    self.arrLabel.append(lbl)  
                                      
                    fileToWrite="{}/{}_{}_agm_{}.jpg".format(folderFound,lbl,idx,idxf)
                    
                    print (fileToWrite)     
                    
                    cv2.imwrite(fileToWrite ,imgresized)
                #self.arrVectorFacenet.append(vectorFn)             
        
        print(len(self.arrVectorDlib))
        print(len(self.arrLabel))

        fileVectorLabel= open(fileRawVectorLbl,"w")
        fileVectorLabel.write(json.dumps({
            "dlibvectors": self.arrVectorDlib,
            "facenetvectors": self.arrVectorFacenet,
            "labels":self.arrLabel
        }))
        fileVectorLabel.close()

        self.svmFaceClassifierDlib.Train(self.arrVectorDlib, self.arrLabel)
        self.svmFaceClassifierDlib.SaveModel(fileDlibModel)

        pass

    def BuildData(self,id,folderRoot):

        imgAugm= ImageAugmentation()
        self.__FileFolder = os.path.dirname(os.path.realpath(__file__))
        fileRawVectorLbl = self.__FileFolder +"/{}.vectorlbl.raw.txt".format(id)
        fileDlibModel= self.__FileFolder +"/{}.svc.dlib.model".format(id)
        fileFacenetModel= self.__FileFolder +"/{}.svc.facenet.model".format(id)

        folderFound=self.__FileFolder+"/{}.foundfaces".format(id)

        if( not os.path.exists(folderFound)):
            os.makedirs(folderFound)

        print("dataset from: "+ folderRoot)        
        print("root app: "+  self.__FileFolder )
        print("faces to: "+ folderFound)
        print("fileRawVectorLbl: "+ fileRawVectorLbl)
        print("fileDlibModel: "+ fileDlibModel)
        print("fileFacenetModel: "+ fileFacenetModel)

        if(os.path.exists(fileRawVectorLbl)):
            rawVectorLbl = Path(fileRawVectorLbl).read_text()
            obj = json.loads(rawVectorLbl)
            self.arrVectorDlib=obj["dlibvectors"]
            self.arrVectorFacenet=obj["facenetvectors"]
            self.arrLabel=obj["labels"]
            
            self.svmFaceClassifierDlib.LoadModel(fileDlibModel, self.arrLabel)     
            #self.svmFaceClassifierFacenet.LoadModel(fileFacenetModel, self.arrLabel)       

            print("loaded model")
            print("Lbl: "+ str(len(self.arrLabel)))
            print("Dlib: "+str(len(self.arrVectorDlib)))
            print("Facenet: "+str(len(self.arrVectorFacenet)))
            
            return

        self.arrLabel=[]
        self.arrVectorDlib=[]
        self.arrVectorFacenet=[]
        listAll = os.listdir(folderRoot)

        print(folderRoot)
        allCounter= len(listAll)
        counter=0
        for fd in listAll:
            fd = folderRoot+"/"+fd
            if(os.path.isdir(fd)):
                print ("{}/{} {}".format(counter,allCounter, fd))
                files = os.listdir(fd)
                lbl=Path(fd).name
                print("label: "+lbl)
                for f in files:
                    f= fd+"/"+f
                    try:
                        if(os.path.isfile(f)):    
                            #print(f)                    
                            bmp = cv2.imread(f)     

                            ratio=0.5
                            h,w,c = bmp.shape

                            while(w>1800): 
                                bmp= cv2.resize(bmp,(0,0),fx=ratio,fy=ratio)
                                h,w,c = bmp.shape
                                ratio= ratio*1.5
                                if(ratio>1):
                                    break      
                            dt1=datetime.datetime.now().timestamp()
                            ffound=self.detector.DetectFace(bmp)    
                            dt2=datetime.datetime.now().timestamp()
                            print ("Time detected: {}".format(dt2-dt1))
                            if(len(ffound)<=0):
                                temp = cv2.rotate(bmp, cv2.ROTATE_90_CLOCKWISE)
                                ffound=self.detector.DetectFace(bmp)
                            if(len(ffound)<=0):
                                temp = cv2.rotate(bmp, cv2.ROTATE_180)
                                ffound=self.detector.DetectFace(bmp)
                            if(len(ffound)<=0):
                                temp = cv2.rotate(bmp, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                ffound=self.detector.DetectFace(bmp)

                            if(len(ffound)>0):
                                fcropO,rrect,points = ffound[0]
                                fcropOf=imgAugm.flip(fcropO,False,True)
                                
                                fcropAugm= [
                                fcropO,
                                fcropOf,
                                imgAugm.rotate(fcropO,5),imgAugm.rotate(fcropO,-5),
                                imgAugm.rotate(fcropOf,5),imgAugm.rotate(fcropOf,-5),
                                ]

                                for idxf,fcrop in enumerate( fcropAugm):
                                    fch,fcw,fcc=fcrop.shape
                                    if fcw< 50:
                                        continue
                                    #cv2.imshow("fff",fcrop)
                                    #cv2.waitKey(1)
                                    #print("-----------face vector")
                                    dt1=datetime.datetime.now().timestamp()
                                    vector,imgresized =self.encoderDlib.face_vector(fcrop) #[[]]
                                    dt2=datetime.datetime.now().timestamp()
                                    print ("Time recoginized: {}".format(dt2-dt1))
                                    #vectorFn= self.faceNetEncoder.face_vector(fcrop)

                                    #print(vector)
                                    
                                    self.arrVectorDlib.append(vector)           
                                    #self.arrVectorFacenet.append(vectorFn)    
                                    
                                    self.arrLabel.append(lbl)
                                    
                                    fileToWrite="{}/{}_agm_{}_{}".format(folderFound,lbl,idxf,Path(f).name)
                                    
                                    print (fileToWrite)     
                                    cv2.imwrite(fileToWrite ,imgresized)

                            else:
                                print("not found face: "+f)
                    except:
                        time.sleep(1)
                        pass
                counter=counter+1
            pass
        pass
        
        if(len(self.arrLabel)<=0):
            print("not found any face to train")
            raise
        
        fileVectorLabel= open(fileRawVectorLbl,"w")
        fileVectorLabel.write(json.dumps({
            "dlibvectors": self.arrVectorDlib,
            "facenetvectors": self.arrVectorFacenet,
            "labels":self.arrLabel
        }))
        fileVectorLabel.close()

        self.svmFaceClassifierDlib.Train(self.arrVectorDlib, self.arrLabel)
        self.svmFaceClassifierDlib.SaveModel(fileDlibModel)
                
        #self.svmFaceClassifierFacenet.Train(self.arrVectorFacenet, self.arrLabel)
        #self.svmFaceClassifierFacenet.SaveModel(fileFacenetModel)

        time.sleep(1)
        #reload from file        
        self.BuildDataKo1504()

    def DetectFrame(self, frame):

        t1= datetime.datetime.now().timestamp()
        
        xframe = self.smartGammaFunc[self.lastHitGamma](frame)
        
        foundFace = None
        t1= datetime.datetime.now().timestamp()
        foundFace = self.detector.DetectFace(xframe) 
       
        if(len(foundFace)==0):
            for igama in self.smartGamma:
                if(igama!=self.lastHitGamma):
                    xframe = self.smartGammaFunc[igama](xframe)

                    foundFace = self.detector.DetectFace(xframe) 

                    if(len(foundFace)>0):
                        self.lastHitGamma=igama
                        break
        
        t2= datetime.datetime.now().timestamp()
        print("{} Detect in {} gama {}".format(len(foundFace), t2-t1, self.lastHitGamma))
        
        foundFaceRectOrg=[]
        
        for fc,fr,points in foundFace:
            x=int(fr[0]*self.ratio)
            y=int(fr[1]*self.ratio)
            w=int(fr[2]*self.ratio)
            h=int(fr[3]*self.ratio)

            foundFaceRectOrg.append((fc,[x,y,w,h]))
            pass

        return foundFaceRectOrg
        pass

    def PredictFace(self,orginalFrame, face_croped, region_face):
        """[summary]

        Args:
            orginalFrame ([type]): [description]
           
            face_croped ([type]): [description]
            region_face ([type]): [description]

        Returns:
            ( dx0, dy0, dx1, dy1, 
                    svmLblDlib, svmProbDlib,
                    lblDlib , valDlib,
                    lblDlibEcli , valDlibEcli,
                    svmLblFn, svmProbFn, 
                    lblFn,valFn,   
                    lblFnEcli, valFnEcli,
                    lblPredictCombine,
                    fileImgPub,
                    urlImgPub,
                    self._cameraUrl,
                    OriginWidth,
                    OriginHeight
                    )
        """
        try:
            #(orginalFrame, face_croped, region_face) = inputTuple
            #(propPrint,propRelay)=self.livenessDetector.Predict(face_croped)
            #cv2.imshow("face found", face_croped)
            #cv2.waitKey(1)

            OriginHeight,OriginWidth,originc= orginalFrame.shape

            t1 = datetime.datetime.now().timestamp()
            dx0=region_face[0]
            dy0=region_face[1]
            facew=region_face[2]
            dx1=region_face[0]+region_face[2]
            faceh=region_face[3]
            dy1=region_face[1]+region_face[3]

            vector,imgresized= self.encoderDlib.face_vector(face_croped)
            
            #vectorFn= self.faceNetEncoder.face_vector(face_croped)

            # self.comparer.predictCosiByMin(vector,self.arrVectorDlib,self.arrLabel)
            # self.comparer.predictEcliByMin(vector,self.arrVectorDlib,self.arrLabel)

            (lblDlib,valDlib)= self.comparer.predictCosiByMin(vector,self.arrVectorDlib,self.arrLabel )
            #(lblFn,valFn)= self.comparer.predictCosi(vectorFn,self.arrVectorFacenet,self.arrLabel )
            (lblFn,valFn)=(None,None)

            (lblDlibEcli,valDlibEcli)= self.comparer.predictEcliByMin(vector,self.arrVectorDlib,self.arrLabel )
            #(lblFnEcli,valFnEcli)= self.comparer.predictEcli(vectorFn,self.arrVectorFacenet,self.arrLabel )
            (lblFnEcli,valFnEcli)=(None,None)

            (svmLblDlib,svmProbDlib) =self.svmFaceClassifierDlib.Predict(vector)                                        
            #(svmLblFn,svmProbFn) = self.svmFaceClassifierFacenet.Predict(vectorFn)


            (svmLblFn,svmProbFn) =(None,None)
            lblPredictCombine=""
            
            faceBound= face_croped
            fileImgPub=""
            urlImgPub=""
            if(lblDlib==lblDlibEcli and lblDlib==svmLblDlib):
                """
                thresholds = {
                'VGG-Face': {'cosine': 0.40, 'euclidean': 0.55, 'euclidean_l2': 0.75},
                'OpenFace': {'cosine': 0.10, 'euclidean': 0.55, 'euclidean_l2': 0.55},
                'Facenet':  {'cosine': 0.40, 'euclidean': 10, 'euclidean_l2': 0.80},
                'DeepFace': {'cosine': 0.23, 'euclidean': 64, 'euclidean_l2': 0.64},
                'DeepID': 	{'cosine': 0.015, 'euclidean': 45, 'euclidean_l2': 0.17},
                'Dlib': 	{'cosine': 0.07, 'euclidean': 0.6, 'euclidean_l2': 0.6},
                'ArcFace':  {'cosine': 0.6871912959056619, 'euclidean': 4.1591468986978075, 'euclidean_l2': 1.1315718048269017}
                }
                """
                if(valDlib<=0.035 and valDlibEcli<= 0.45 and svmProbDlib<=0.8 and lblDlib== lblDlibEcli and lblDlib==svmLblDlib):
                    lblPredictCombine=lblDlib
                    allowSent =True
                    
                    if( lblPredictCombine in self.last_sent_notification.keys()):
                        nowTime =datetime.datetime.now().timestamp()                                
                        lastSent= self.last_sent_notification[lblPredictCombine]

                        if(nowTime- lastSent>5):
                            allowSent=True
                        else:
                            allowSent=False
                        
            
            t2 = datetime.datetime.now().timestamp()

            print("{} -Predicted -------------------------".format(t2-t1))

            return (
                    dx0, dy0, facew, faceh, 
                    svmLblDlib, svmProbDlib,
                    lblDlib , valDlib,
                    lblDlibEcli , valDlibEcli,
                    svmLblFn, svmProbFn, 
                    lblFn,valFn,   
                    lblFnEcli, valFnEcli,
                    lblPredictCombine,
                    fileImgPub,
                    urlImgPub,
                    self._cameraUrl,
                    OriginWidth,
                    OriginHeight
                    )

        except Exception as ex:
            print("Error PredictFace")
            print(ex)
            return None 

    @staticmethod
    def adjust_gamma(image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    @staticmethod
    def DrawDetectedAndPredict(frame,ratio, tempFacesPredicted):
        
        lenLastPredicted=len(tempFacesPredicted)
        colorIdx=255
        if(lenLastPredicted>0) :     
            counter=0     
            colorIdx=int(255/lenLastPredicted)     
            
            for predicted in tempFacesPredicted:                        
                (
                dx0, dy0, facew, faceh, 
                svmLblDlib, svmProbDlib,
                lblDlib , valDlib,
                lblDlibEcli , valDlibEcli,
                svmLblFn, svmProbFn, 
                lblFn,valFn,   
                lblFnEcli, valFnEcli,
                lblPredictCombine,
                fileImgPub,
                urlImgPub,
                cameraUrl,
                OriginWidth,
                OriginHeight
                ) = predicted
                
                dx1=int( (dx0+facew)*ratio)
                dy1=int( (dy0+faceh)*ratio)
                dx0=int(dx0*ratio)
                dy0=int(dy0*ratio)

                colorIdx= colorIdx+colorIdx
                color=(0,colorIdx,colorIdx,0)
                posX= dx0
                
                # if(counter%2==0):
                #     posX= dx0-dx0
                #     color=(0,0,colorIdx,0)
                # else:
                #     posX=dx0

                counter=counter+1

                cv2.rectangle(frame,(dx0,dy0),(dx1,dy1),color,2)

                midDy0Dy1=int(dy0 +(dy1-dy0)/2)
                
                cv2.putText(frame, "{}".format(lblPredictCombine)
                                ,(posX,dy0- 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  color, 2) 

                cv2.putText(frame, "c: {} {} {} {}".format(lblDlib,valDlib,lblFn,valFn)
                                ,(posX,dy0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1) 

                cv2.putText(frame, "c: {} {} {} {}".format(lblDlib,valDlib,lblFn,valFn)
                                ,(posX+1,dy0+1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  ( 0,0, 255, 0), 1) 
                                                                        
                cv2.putText(frame, "e: {} {} {} {}".format(lblDlibEcli,valDlibEcli,lblFnEcli,valFnEcli)
                                ,(posX,midDy0Dy1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)    
                cv2.putText(frame, "e: {} {} {} {}".format(lblDlibEcli,valDlibEcli,lblFnEcli,valFnEcli)
                                ,(posX+1,midDy0Dy1+1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  ( 0,0, 255, 0), 1)                       

                cv2.putText(frame, "s: {} {} {} {}".format(svmLblDlib,svmProbDlib,svmLblFn,svmProbFn)
                                ,(posX,dy1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  color, 1) 
                cv2.putText(frame, "s: {} {} {} {}".format(svmLblDlib,svmProbDlib,svmLblFn,svmProbFn)
                                ,(posX+1,dy1+1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  ( 0,0, 255, 0), 1) 


        pass
    
    @staticmethod
    def DrawDetected( frame,ratio, tempDetectedFaces):
        lenLastPredicted=len(tempDetectedFaces)
        colorIdx=255
        if(tempDetectedFaces!=None):
            for fc,fr in tempDetectedFaces:
                
                colorIdx= colorIdx+colorIdx
                color=(0,colorIdx,colorIdx,0)
            
                #list of (facecroped, faceregion)=ff       
                colorIdx = colorIdx+colorIdx
                dx0=int(fr[0]*ratio)
                dy0=int(fr[1]*ratio)
                dx1=int ((fr[0]+fr[2])*ratio)
                dy1=int((fr[1]+fr[3])*ratio)
                cv2.rectangle(frame,(dx0,dy0),(dx1,dy1),(255,0,255,0),2)
        pass
   
    @staticmethod
    def GetAndShowFrame(cameraUrl,withRecoginze,ratio,lastFrame, lastDetected, lastRecongize):

        cameraUrl=cameraUrl

        #os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
        #vid = cv2.VideoCapture(cameraUrl,cv2.CAP_FFMPEG)
        vid = cv2.VideoCapture(cameraUrl,cv2.CAP_ANY)
        # vid.set(3, 640)
        # vid.set(4, 480)
        #vid.set(cv2.CAP_PROP_FPS, 5.0)
        myLastDetect=[]
        myLastRecognize=[]
        frame=None
        tlastfound =datetime.datetime.now().timestamp()
        while(True):
            try:                
                tnow =datetime.datetime.now().timestamp()

                if(tnow- tlastfound>20):                    
                    myLastDetect=[]
                    myLastRecognize=[]

                ret, frame = vid.read()
                #frame= cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
                
                fps =  vid.get(cv2.CAP_PROP_FPS)
                width = vid.get(cv2.CAP_PROP_FRAME_WIDTH) 
                    
                #orginalFrame = frame.copy()

                height, width, channels = frame.shape 

                ratio=1

                newWidth=width
                if(newWidth>1024):
                    newWidth=1024

                height= (int)((newWidth*height)/width)
                width=newWidth

                frame= cv2.resize(frame,(width,height),interpolation = cv2.INTER_AREA)                

                lastFrame.put_nowait(frame)
                
                #----

                qsize= lastDetected.qsize()      
                #print("{} lastDetected".format(qsize))         
                tempLastDetect=[]
                for i in range(0, qsize):
                    try:
                        tempLastDetect.append( lastDetected.get_nowait())
                    except:
                        pass

                if(len(tempLastDetect)>0):                     
                        tlastfound =datetime.datetime.now().timestamp()                     
                        myLastDetect=[tempLastDetect[-1]]

                DlibSingleThread.DrawDetected(frame,  ratio, myLastDetect) 

                if withRecoginze:       
                    tempLastRecognize=[]
                    qsize= lastRecongize.qsize()    
                    
                    #print("{} lastRecongize".format(qsize))  
                    for i in range(0, qsize):
                        try:
                            tempLastRecognize.append( lastRecongize.get_nowait())
                        except:
                            pass

                    if(len(tempLastRecognize)>0):
                        myLastRecognize=[tempLastRecognize[-1]]
                
                    DlibSingleThread.DrawDetectedAndPredict(frame, ratio,myLastRecognize)

                cv2.imshow("DlibSingleThread",frame)
                cv2.waitKey(1)

            except Exception as ex:
                print(ex)
                pass
            finally:
                pass

    def LoopDetectAndRecognize(self,withRecoginze,lastFrame, lastDetected, lastRecongize):
        
        tlastfound =datetime.datetime.now().timestamp()
        
        while(True):
            try:

                tnow =datetime.datetime.now().timestamp()            

                qsize = lastFrame.qsize()  
                if(qsize==0) :
                    #print("{} lastFrame".format(qsize))
                    continue

                #print("{} lastFrame".format(qsize))
                tempLastFrame=[]
                frame = None
                for i in range(0,qsize):
                    try:
                        frame= lastFrame.get_nowait()
                        tempLastFrame.append(frame)
                    except:
                        pass

                if(len(tempLastFrame)==0 ):
                    continue
                frame= tempLastFrame[-1]

                foundFaceRectOrg= self.DetectFrame(frame)
                
                for ffound in foundFaceRectOrg:  

                    tlastfound =datetime.datetime.now().timestamp()             

                    lastDetected.put_nowait(ffound)

                    if withRecoginze :
                        (face_croped, region_face)=ffound  
                        
                        predicted=self.PredictFace(frame,face_croped, region_face)
                        if(predicted!=None):        
                            lastRecongize.put_nowait( predicted)
                
                pass
            except Exception as ex:
                print("ERROR LoopDetectAndRecognize")
                print(ex)
                time.sleep(1000)
                pass
            finally:
                time.sleep(0.03)
                pass
        
    def Start(self):
        
        self.InitDataTest()

        self.threadGetShowVideo.start()

        self.LoopDetectAndRecognize(self.withRecoginze, self._lastFrame, self._lastDetected, self._lastRecongize)

if __name__ == '__main__':

    (DlibSingleThread(0)).Start()
    


