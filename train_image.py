import os
import time
import cv2
import numpy as np
from PIL import Image

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    faces = []
    Ids = []

    for imagePath in imagePaths:
        try:
            pilImage = Image.open(imagePath).convert('L')  # Convert image to grayscale
            imageNp = np.array(pilImage, 'uint8')

            if imageNp.size == 0:
                print(f"Warning: Image {imagePath} is empty and will be skipped.")
                continue

            Id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces.append(imageNp)
            Ids.append(Id)
        except Exception as e:
            print(f"Error processing image {imagePath}: {e}")

    return faces, Ids

def counter_img(num_images):
    for imgcounter in range(1, num_images + 1):
        print(f"{imgcounter} Images Trained", end="\r")
        time.sleep(0.1)

def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)

    faces, Ids = getImagesAndLabels("TrainingImage")

    if len(faces) == 0:
        print("No training images found. Please add images to the 'TrainingImage' directory.")
        return
    
    try:
        recognizer.train(faces, np.array(Ids))
        
        os.makedirs("TrainingImageLabel", exist_ok=True)
        recognizer.save("TrainingImageLabel" + os.sep + "Trainner.yml")

        num_images = len(faces)
        counter_img(num_images)
        print("\nTraining complete. Model saved as Trainner.yml")
    
    except Exception as e:
        print(f"An error occurred during training: {e}")

