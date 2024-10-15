import os
import check_camera
import capture_image
import train_image
import recognize

def title_bar():
    os.system('cls')  # or 'clear' for Unix-based systems
    print("\t***** Face Recognition Attendance System *****")

def mainMenu():
    while True:
        title_bar()
        print()
        print(10 * "*", "WELCOME MENU", 10 * "*")
        print("[1] Check Camera")
        print("[2] Capture Faces")
        print("[3] Train Images")
        print("[4] Recognize & Attendance")
        print("[5] Quit")
        
        try:
            choice = int(input("Enter Choice: "))
            if choice == 1:
                checkCamera()
            elif choice == 2:
                CaptureFaces()
            elif choice == 3:
                Trainimages()
            elif choice == 4:
                recognizeFaces()
            elif choice == 5:
                print("Thank You")
                break
            else:
                print("Invalid Choice. Enter 1-5\nTry Again")
        except ValueError:
            print("Invalid Choice. Enter 1-5\nTry Again")

def checkCamera():
    check_camera.camer()
    input("Enter any key to return to the main menu")
    mainMenu()

def CaptureFaces():
    capture_image.take_images()  # Ensure this function is defined correctly in capture_image.py
    input("Enter any key to return to the main menu")
    mainMenu()

def Trainimages():
    train_image.TrainImages()  # Ensure this function is defined correctly in train_image.py
    input("Enter any key to return to the main menu")
    mainMenu()

def recognizeFaces():
    recognize.recognize_attendence()  # Call the correct function name as defined in recognize.py
    input("Enter any key to return to the main menu")
    mainMenu()

# Main driver
if __name__ == "__main__":
    mainMenu()
