import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2
from time import time


############################################
classID = 0 # zero is fake and 1 is real
outputFolderPath = 'Dataset/DataCollect'
debug = False
confidence = 0
offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight = 640, 480
floatingPoint = 6
save = True
blurThreshold = 35
############################################
# Initialize the webcam
# '2' means the third camera connected to the computer, usually 0 refers to the built-in webcam
cap = cv2.VideoCapture(0)
cap.set(3,camWidth)
cap.set(4,camHeight)
detector = FaceDetector()

# Run the loop to continually get frames from the webcam
while True:
    listBlur = []
    listInfo = []
    success, img = cap.read()
    imgOut = img.copy()

    img, bboxs = detector.findFaces(img,draw=False)

    # Check if any face is detected
    if bboxs:
        # Loop through each bounding box
        for bbox in bboxs:


            x, y, w, h = bbox['bbox']
            score = bbox['score'][0]

#--------------------Check the score ------------------------------------#

            if score >confidence:

                # -------------------Adding and offset to the face detected --------------#

                offsetW = (offsetPercentageW / 100) * w
                x = int(x - offsetW)
                w = int(w + offsetW * 2)

                offsetH = (offsetPercentageH / 100) * h
                y = int(y - offsetW * 3)
                h = int(h + offsetW * 3.5)
                # -----------------------------------------------------------------#
                # ------  To avoid values below 0 --------
                if x < 0: x = 0
                if y < 0: y = 0
                if w < 0: w = 0
                if h < 0: h = 0

                # -------------------Find blureness ----------------#
                imgFace = img[y:y + h, x:x + w]
                cv2.imshow("Face", imgFace)
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                if blurValue > blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)

                #---------------------Normalize value -----------------------#
                ih, iw, _ = img.shape
                xc, yc = x + w / 2, y + h / 2

                xcn, ycn = round(xc / iw, floatingPoint), round(yc / ih, floatingPoint)
                wn, hn = round(w / iw, floatingPoint), round(h / ih, floatingPoint)
                # print(xcn, ycn, wn, hn)

                # ------  To avoid values above 1 --------
                if xcn > 1: xcn = 1
                if ycn > 1: ycn = 1
                if wn > 1: wn = 1
                if hn > 1: hn = 1

                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")

                # -------------------Drawing ----------------#

                cv2.rectangle(imgOut, (x, y, w, h), (255, 0, 0), 3)
                cvzone.putTextRect(imgOut, f"Score: {int (score * 100)}%Blur: {blurValue}", (x, y - 0),
                                   scale=2, thickness=3)
                if debug:
                    cv2.rectangle(img, (x, y, w, h), (255, 0, 0), 3)
                    cvzone.putTextRect(img, f"Score: {int(score * 100)}%Blur: {blurValue}", (x, y - 0),
                                       scale=2, thickness=3)
          # ------  To Save --------
        if save:
            if all(listBlur) and listBlur != []:
                # ------  Save Image  --------
                timeNow = time()
                timeNow = str(timeNow).split('.')
                timeNow = timeNow[0] + timeNow[1]
                print(timeNow)
                cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", img)

    #       -------------------Save label text file ------------------------------#
                for info in listInfo:
                    f = open(f"{outputFolderPath}/{timeNow}.txt", 'a')
                    f.write(info)
                    f.close()



    cv2.imshow("Image", imgOut)
    cv2.waitKey(1)

