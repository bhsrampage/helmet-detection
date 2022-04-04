import cv2 as cv
import boto3
import numpy as np

from PIL import Image, ImageDraw, ExifTags, ImageColor

capture = cv.VideoCapture("F:/Python/helmet_detector/sample2.mp4") #if you provide integer value it will capture from webcam 0:- primary , 1:-secondary and so on
client=boto3.client('rekognition') 
s3 = boto3.client('s3', region_name='ap-south-1')
count=0

while True:
    isTrue , frame = capture.read()
    cv.imwrite(f'live{count}.jpg', frame)
    local_image = open('./'+f'live{count}.jpg', 'rb')
    res = s3.put_object(Bucket="folktel-app-user-images", Key = f'live{count}.jpg', Body = local_image, ContentType= 'image/png')  
    image = Image.open(f'live{count}.jpg')
    width = image.size[0]
    height = image.size[1]
    bucket='folktel-app-user-images'
    draw = ImageDraw.Draw(image) 
    fileName=f'live{count}.jpg'
    threshold = 99
    maxFaces=1
    response=client.detect_protective_equipment(
        
    Image={
        "S3Object": {
            "Bucket": bucket,
            "Name": f'live{count}.jpg'
        }
    },
   SummarizationAttributes= {
        "MinConfidence": 80,
        "RequiredEquipmentTypes": [
            "HEAD_COVER"
        ]
    },
    

    )
    count+=1
                                
    for person in response["Persons"]:
        #print(person)
        for BodyPart in person['BodyParts']:
            if(BodyPart["Name"] == 'HEAD'):
                for i in BodyPart['EquipmentDetections']:
                    if i["Type"] == 'HEAD_COVER' and i["CoversBodyPart"]['Value'] :
                        position = i['BoundingBox']
                        similarity = str(i["Confidence"])
                        left = position['Left'] * width
                        top = position['Top'] * height
                        width_1 = position['Width'] * width
                        height_1 = position['Height'] * height
                        print(f'\nlive{count}.jpg')
                        print('Left: ' + '{0:.0f}'.format(left))
                        print('Top: ' + '{0:.0f}'.format(top))
                        print('Face Width: ' + "{0:.0f}".format(width_1))
                        print('Face Height: ' + "{0:.0f}".format(height_1))
                    print('Covers Body Part: ' + str(i["CoversBodyPart"]['Value']))

                points = (
                    (left,top),
                    (left + width_1, top),
                    (left + width_1, top + height_1),
                    (left , top + height_1),
                    (left, top)

                )
                draw.line(points, fill= '#00d400' if i["CoversBodyPart"]['Value'] else  '#FF0000', width=5)
            # if len(BodyPart['EquipmentDetections']) == 0 :   
            #     draw.text((10,10),"Helmet not worn !" , fill='#FF0000')
        cv.imshow('Analysis',cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR))
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release() #release capture instance
cv.destroyAllWindows() #close all windows