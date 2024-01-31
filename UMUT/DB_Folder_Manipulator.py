import os
import re
import cv2
import shutil
import matplotlib.pyplot as plt

#Change these
dbName = 'YoutubeFace' #IBUG, LFPW, HELEN, AFW, IBUG, YoutubeFace
showFrontalFaceExamples = True #True for show, False for not show
isThereTrainTest = False #True for LFPW Dataset, False for anothers
inputOrAutoMod = False #True for auto, False for input



#Global Variables for decideWhichElementsWhichFeatures
file_id_index, inner_id_right_side_index, inner_id_left_side_index, learnType_index= 0, 0, 0, 0 

#This variable will be automatically changed according to the number of features
copyFlag = False

if dbName == 'YoutubeFace':
    YoutubeFaceDB = True
else:
    YoutubeFaceDB = False


def printFeatures( output_dict ):
    print("\n\nfile_name: " + output_dict["file_name"])
    print("file_name_withoutExtension: " + output_dict["file_name_withoutExtension"])
    print("extension: " + output_dict["extension"])
    print("learnType: " + str(output_dict["learnType"]))
    print("file_id: " + output_dict["file_id"])
    print("inner_id_right_side: " + output_dict["inner_id_right_side"])
    print("inner_id_left_side: " + output_dict["inner_id_left_side"])

def decideWhichElementsWhichFeatures( file_name_split ):
    file_id_index = 0
    inner_id_right_side_index = 0
    inner_id_left_side_index = 0
    learnType_index = 0

    for element in file_name_split:
        os.system('cls')
        print("File Name: " + file_name)
        print(file_name_split)
        inputTemp = input("\nWhat is the feature of '" + element + "' ? \n"+  
                        " n -> next \n"
                        " f -> file_id \n" +
                        " ir -> inner_id_right_side \n" +
                        " il -> inner_id_left_side \n" +
                        " l -> learnType : " )
        if inputTemp == "f":
            file_id_index = file_name_split.index(element)
        elif inputTemp == "ir":
            inner_id_right_side_index = file_name_split.index(element)
        elif inputTemp == "il":
            inner_id_left_side_index = file_name_split.index(element)
        elif inputTemp == "l":
            learnType_index = file_name_split.index(element)
        elif inputTemp == "n":
            continue
        else:
            print("Wrong Input!")
            exit()

    print("\nElement indexes: \n" + 
            "file_id_index: " + str(file_id_index) + "\n" +
            "inner_id_right_side_index: " + str(inner_id_right_side_index) + "\n" +
            "inner_id_left_side_index: " + str(inner_id_left_side_index) + "\n" +
            "learnType_index: " + str(learnType_index) + "\n" )
    
    return file_id_index, inner_id_right_side_index, inner_id_left_side_index, learnType_index

#You should change this function according to your dataset,if you want to use auto mod
#Adjusted for IBUG Dataset
def autoDetermineAccordingToFeatureCount( file_name_split ):
    global file_id_index, inner_id_right_side_index, inner_id_left_side_index, learnType_index

    integerFeatureSliceCount = 0
    #calculate feature count but only numbers
    for element in file_name_split:
        if re.match("^\d+$", element):
            integerFeatureSliceCount += 1
    print("Integer Feature Slice Count: " + str(integerFeatureSliceCount))
    if isThereTrainTest:
        if integerFeatureSliceCount == 2:
            file_id_index = 3
            inner_id_right_side_index = 4
            inner_id_left_side_index = 0
            learnType_index = 2
        else:
            print("Wrong Feature Count!")
            exit()
    else:
        if integerFeatureSliceCount == 3:
            file_id_index = 2
            inner_id_right_side_index = 4
            inner_id_left_side_index = 3
            learnType_index = False
        elif integerFeatureSliceCount == 2:
            file_id_index = 2
            inner_id_right_side_index = 3
            inner_id_left_side_index = False
            learnType_index = False
        else:
            print("Wrong Feature Count!")
            exit()
    return file_id_index, inner_id_right_side_index, inner_id_left_side_index, learnType_index

def extractFeaturesFromFileName(fileName): # this should change according to the dataset
    global file_id_index, inner_id_right_side_index, inner_id_left_side_index, learnType_index,makeDeceisonFlag  # Declare as global
    #Don't change this part
    file_name_split = fileName.split('_')
    file_name_withoutExtension = file_name.split('.')[0]    
    extension = file_name.split('.')[-1] # jpg or mat
    
    file_name_split = file_name_split[:-1] + file_name_split[-1].split('.') 
    numberOfSlices = len(file_name_split)

    #Number of slices changed, we should extract which feature is which
    if makeDeceisonFlag == True:
        if inputOrAutoMod:
            file_id_index, inner_id_right_side_index, inner_id_left_side_index, learnType_index = autoDetermineAccordingToFeatureCount(file_name_split)
        else:
            file_id_index, inner_id_right_side_index, inner_id_left_side_index, learnType_index = decideWhichElementsWhichFeatures(file_name_split)
        makeDeceisonFlag = False
        
    file_id = file_name_split[file_id_index]
    inner_id_right_side = file_name_split[inner_id_right_side_index]
    inner_id_left_side = file_name_split[inner_id_left_side_index]
    
    # train or test only for LFPW Dataset
    learnType = file_name_split[learnType_index]
    if learnType != 'train' and learnType != 'test':
        learnType = False

    #Only for YoutubeFaceDB
    output_dict = {
            "file_name": file_name,
            "file_name_withoutExtension": file_name_withoutExtension, 
            "extension": extension, 
            "inner_id_right_side": inner_id_right_side, 
            "learnType": learnType, 
            "file_id": file_id, 
            "inner_id_left_side": inner_id_left_side,
            "numberOfSlices": numberOfSlices
        }
    
    #Uncomment this for see the features
    #printFeatures(output_dict)
    return output_dict


def yunetDetectionDNN(img):
    height, width, _ = img.shape
    detector = cv2.FaceDetectorYN.create("face_detection_yunet.onnx",  "", (0, 0))
    detector.setInputSize((width, height))
    return detector.detect(img)

def DNNFrontalHandle(faces, image_cv2_yunet):
    if faces is not None: 
        for face in faces:
            
            # confidence
            confidence = face[-1]
            confidenceArray.append({'confidence': confidence, 'img': image_cv2_yunet})
    else:
        logString = "No face detected: " + file_name
        writeLog('./LOG/'+dbName+'/logNoFace.txt', logString)
        
def writeLog(log_file_path, log):
    with open(log_file_path, 'a') as log_file:
        log_file.write(log + '\n')

def clearLogs():
    log_file_path = './LOG/'+dbName
    if os.path.exists(log_file_path):
        shutil.rmtree(log_file_path)
        os.makedirs(log_file_path, exist_ok=True)
        
def findMaxFrontalFace(confidenceArr): 
    #find max confidence and write it to the folder
    maxConf = 0
    confidence = 0
    for conf in confidenceArr:
        if conf['confidence'] > maxConf:
            maxConf = float(conf['confidence'])
            bestImage = conf['img']
            confidence = conf['confidence']
    
    return bestImage,confidence

def showFrontalFaces(image, confidence, frontalCount):
    if frontalCount<40 and showFrontalFaceExamples:
        plt.subplot(4,10,frontalCount)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(str(round(float(confidence),2)))
        plt.axis('off')
    elif frontalCount == 40 and showFrontalFaceExamples:
        plt.show()    

#This function will write the frontal face to the folder
def writeFrontalFaceToFolder(image, confidence, frontalCount, destination):    
    os.makedirs( destination + 'frontal/', exist_ok=True)
    output_file_path_frontal = destination + 'frontal/' + file_name_withoutExtension + '.' + extension
    
    if YoutubeFaceDB == True:
        input_file_path_frontal = file
    else:
        input_file_path_frontal = './' + dbName + '/' + file_name

    #print("Frontal input file path: " + input_file_path_frontal)
    shutil.copy(input_file_path_frontal, output_file_path_frontal)

    logString = "Added Frontal Image Count: " + str(frontalCount) + " - " + str(file_id)+ " - " + str(confidence) + " - " + file_name
    if confidence < 0.9:
        logString = logString + " Low Confidence! "
    writeLog('./LOG/'+dbName+'/logFrontalFaceAdded.txt', logString)

def youtubeDBFilesConcat(inFiles):
    outFilesPaths = []
    for file in inFiles:
        #example ->AFW_815038_1_12.jpg
        file_name = file.name
        folder_flag = len(file_name.split('.')) == 1
        if folder_flag:
            innerFolder = os.scandir('./'+dbName+'/'+file_name)
            for inner_file in innerFolder:
                folder_flag = len(inner_file.name.split('.')) == 1
                if folder_flag == True:
                    inner_inner_folder = os.scandir('./'+dbName+'/'+file_name+'/'+inner_file.name)
                    for inner_inner_file in inner_inner_folder:
                        folder_flag = len(inner_inner_file.name.split('.')) == 1
                        if inner_inner_file.name.split('.')[-1] == "jpg":    
                            outFilesPaths.append('./'+dbName+'/'+file_name+'/'+inner_file.name+'/'+inner_inner_file.name)
    return outFilesPaths
    
def replaceEntersAndTabs(array):
    newArray = []
    for element in array:
        element = element.replace('\n', '')
        element = element.replace('\t', '')
        element = element.replace(' ', '')
        newArray.append(element)
    return newArray
        

##########################   MAIN   ##########################
clearLogs()
plt.figure(figsize=(20,10))
files = os.scandir('./'+dbName)
confidenceArray = []
firstFlag = True;makeDeceisonFlag = True
imageCounter = 0 # only for youtubeFaceDB
holdID = 0;holdFeaturesLen = 0;i=0;frontalCount = 0

os.makedirs('./LOG', exist_ok=True)
os.makedirs('./LOG/'+dbName, exist_ok=True)

if YoutubeFaceDB ==True:
    imageInformationsTxt = open('./output2.txt', 'r') # change this
    imageInformations = imageInformationsTxt.readlines()
    imageInformations = replaceEntersAndTabs(imageInformations)
    files = youtubeDBFilesConcat(files)

for file in files:
    #example ->AFW_815038_1_12.jpg
    if YoutubeFaceDB == True:
        file_name = imageInformations[imageCounter]+'.jpg'
        imageCounter += 1
        if imageCounter%1000 == 0:
            print("Image Counter: " + str(imageCounter))
    else:
        file_name = file.name
    
    features = extractFeaturesFromFileName(file_name)  
    
    file_name_withoutExtension = features["file_name_withoutExtension"]
    inner_id_right_side = features["inner_id_right_side"]
    inner_id_left_side = features["inner_id_left_side"]
    extension = features["extension"]
    learnType = features["learnType"]
    file_id = features["file_id"]
    numberOfSlices = features["numberOfSlices"]

    #When number of slices changed, we should extract features again
    if numberOfSlices != holdFeaturesLen and firstFlag == False:
        print("Number of features changed! Please check the features!")
        if inputOrAutoMod == False:
            input("Press Enter to continue...")
        makeDeceisonFlag = True
        features = extractFeaturesFromFileName(file_name)
    holdFeaturesLen = numberOfSlices
    
    # This part is important for the output folder structure
    # In this part, you can change the output folder structure according to your needs
    if learnType == False:
        output_folder = './' + dbName + '_FOLDERED/' + file_id + '/'
    else:
        output_folder = './' + dbName + '_FOLDERED/' + learnType + '/' + file_id + '/'
        
    if inner_id_left_side != False and inner_id_left_side.isdigit() == True:
        output_folder = output_folder + inner_id_left_side + '/'

    #print("File Name: " + file_name)
    #print("Output Folder: " + output_folder)

    #get match value as an integer

    # Create folders if they don't exist / COPY PROCESS
    # If len of files in output folder is less than 10, copy the file
    if copyFlag == False:
        try:
            if len(os.listdir(output_folder)) < 10 or copyFlag == True:
                copyFlag = True
        except:
            copyFlag = True
            print("Folder does not exist, creating folder: " + output_folder)    
        
    if copyFlag:
        os.makedirs(output_folder, exist_ok=True)
        output_file_path = output_folder + file_name
        
        if YoutubeFaceDB == True:
            input_file_path = file
            #print("Input File Path: " + input_file_path)
        else:
            input_file_path = './' + dbName + '/' + file_name


        #print("Output File Path: " + output_file_path)
        #copy input filepath to output filepath
        shutil.copy(input_file_path, output_file_path)


        logString = "Added Image: " + file_name
        writeLog('./LOG/'+dbName+'/logAddedImage.txt', logString)
    
    
    #DNN frontal detection
    if extension == 'jpg':
        if holdID != file_id and firstFlag == False:
            image_cv2_yunet, confidence = findMaxFrontalFace(confidenceArray)
            confidenceArray.clear()
            frontalCount += 1
            i+=1
            
            showFrontalFaces(image_cv2_yunet, confidence, frontalCount)
            #Our frontal image is ready
            #create a folder that named frontal, and copy this into
            writeFrontalFaceToFolder(image_cv2_yunet, confidence, frontalCount, output_folder)
            
        firstFlag = False
        holdID = file_id
        if YoutubeFaceDB == True:
            input_file_path = file    
        else:
            input_file_path = './' + dbName + '/' + file_name
        
        image_cv2_yunet = cv2.imread(input_file_path)
        _, faces = yunetDetectionDNN(image_cv2_yunet)
        DNNFrontalHandle(faces, image_cv2_yunet)


