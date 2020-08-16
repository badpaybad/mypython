import cv2
import os
import numpy

__rootDir = os.getcwd()

print(cv2.data.haarcascades)

faceDetector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

eyeDetector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml")

pathFileImageInput = __rootDir+"/faces.jpg"

print(pathFileImageInput)

imgFacesInput = cv2.imread(pathFileImageInput)

imgFacesInputGray = cv2.cvtColor(imgFacesInput, cv2.COLOR_BGR2GRAY)

facesDetected = faceDetector.detectMultiScale(
    image=imgFacesInputGray, scaleFactor=1.68, minNeighbors=2, minSize=(10, 10), maxSize=(500, 500),  flags=cv2.CASCADE_SCALE_IMAGE)

print(len(facesDetected))

idx = 0
imgInputCroped = []
for (x, y, w, h) in facesDetected:
    idx = idx+1

    croped = imgFacesInput[y:y+h, x:x+w]
    cv2.imwrite(f"{__rootDir}/input_{idx}.jpg", croped)

    imgInputCroped.append(croped)

    #imgFacesInput = cv2.rectangle(
    #    imgFacesInput, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #roi_gray = imgFacesInputGray[y:y+h, x:x+w]
    #eyes = eyeDetector.detectMultiScale(roi_gray)
    # for (ex,ey,ew,eh) in eyes:
    #    cv2.rectangle(croped,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

#cv2.imshow('img', imgFacesInput)

_resizeTo=(300,300)

duImg = cv2.imread(__rootDir+"/du.png")

duImgGray = cv2.cvtColor(duImg, cv2.COLOR_BGR2GRAY)

duFaces = faceDetector.detectMultiScale(
    image=duImgGray, scaleFactor=1.68, minNeighbors=2, minSize=(10, 10), maxSize=(500, 500),  flags=cv2.CASCADE_SCALE_IMAGE)

idx = 0
duImgCroped = []
for (x, y, w, h) in duFaces:
    idx = idx+1
    # duImg = cv2.rectangle(
    #    duImg, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #roi_gray = duImgGray[y:y+h, x:x+w]
    #roi_color = duImg[y:y+h, x:x+w]
    subFace = duImg[y:y+h, x:x+w]
    duImgCroped.append(subFace)
    cv2.imwrite(f"{__rootDir}/duFace_{idx}.jpg", subFace)
    pass

#cv2.imshow('du', duImg)

lbphRecognitor = cv2.face_LBPHFaceRecognizer.create()

eigenRecogitor = cv2.face_EigenFaceRecognizer.create()

duIds = []
duFaceMat = []
for i in range(0, len(duImgCroped)):
    duIds.append(21)
    resized = cv2.resize(duImgCroped[i], _resizeTo, interpolation = cv2.INTER_CUBIC)
    duFaceMat.append(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY))
    pass

lbphRecognitor.train(duFaceMat, numpy.array(duIds))
eigenRecogitor.train(duFaceMat, numpy.array(duIds))

lbphRecognitor.save(f"{__rootDir}/model_lbph_trained.bin")
eigenRecogitor.save(f"{__rootDir}/model_eigen_trained.bin")

if os.path.isdir(f"{__rootDir}/result") == False:
    os.mkdir(f"{__rootDir}/result")
    pass

result = []
for du in duImgCroped:
    imgInputCroped.append(du)

for f in imgInputCroped:
    f = cv2.resize(f, _resizeTo, interpolation = cv2.INTER_CUBIC)
    fg = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    predictR = lbphRecognitor.predict(fg)
    result.append(predictR)
    cv2.imwrite(f"{__rootDir}/result/lbph_{predictR[0]}_{predictR[1]}.jpg", f)
    pass

for f in imgInputCroped:
    f = cv2.resize(f, _resizeTo, interpolation = cv2.INTER_CUBIC)
    fg = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    predictR = eigenRecogitor.predict(fg)
    result.append(predictR)
    cv2.imwrite(f"{__rootDir}/result/eigen_{predictR[0]}_{predictR[1]}.jpg", f)
    pass

cv2.waitKey(0)
cv2.destroyAllWindows()
