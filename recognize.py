import datetime
import os
import time
import cv2
import pandas as pd

def recognize_attendence():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  
    recognizer.read("TrainingImageLabel" + os.sep + "Trainner.yml")
    harcascadePath = "haarcascade_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    df = pd.read_csv("StudentDetails" + os.sep + "StudentDetails.csv")
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Time']
    attendance = pd.DataFrame(columns=col_names)

    # Get today's date for the filename
    today_date = datetime.datetime.now().strftime('%Y-%m-%d')
    attendance_file_path = f"Attendance{os.sep}Attendance_{today_date}.csv"

    # start realtime video capture
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(3, 640) 
    cam.set(4, 480) 
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5,
                minSize=(int(minW), int(minH)), flags=cv2.CASCADE_SCALE_IMAGE)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x+w, y+h), (10, 159, 255), 2)
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 100:
                name_array = df.loc[df['Id'] == Id]['Name'].values
                confstr = "  {0}%".format(round(100 - conf))
                name = name_array[0] if len(name_array) > 0 else 'Unknown'
            else:
                Id = '  Unknown  '
                name = 'Unknown'
                confstr = "  {0}%".format(round(100 - conf))

            # Update attendance condition: threshold set to 30
            if (100 - conf) > 30:
                ts = time.time()
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                attendance.loc[len(attendance)] = [Id, name, timeStamp]

            display_text = f"{Id} - {name}"
            if (100 - conf) > 30:
                display_text += " [Pass]"
                cv2.putText(im, display_text, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            else:
                cv2.putText(im, display_text, (x + 5, y - 5), font, 1, (255, 255, 255), 2)

            if (100 - conf) > 30:
                cv2.putText(im, str(confstr), (x + 5, y + h - 5), font, 1, (0, 255, 0), 1)
            elif (100 - conf) > 50:
                cv2.putText(im, str(confstr), (x + 5, y + h - 5), font, 1, (0, 255, 255), 1)
            else:
                cv2.putText(im, str(confstr), (x + 5, y + h - 5), font, 1, (0, 0, 255), 1)

        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        cv2.imshow('Attendance', im)
        if (cv2.waitKey(1) == ord('q')):
            break

    # Save attendance to a single CSV file for the day
    if not os.path.exists(attendance_file_path):
        attendance.to_csv(attendance_file_path, index=False, mode='w', header=True)
    else:
        attendance.to_csv(attendance_file_path, index=False, mode='a', header=False)

    print("Attendance Successful")
    cam.release()
    cv2.destroyAllWindows()
