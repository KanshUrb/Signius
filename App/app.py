import tkinter
import cv2
from tkinter import *
from LearningModule.ImageProcessing import *
from LearningModule.DetectObject import *
from PIL import Image, ImageTk

"""here we getting path to parent drectory, where we have logo"""
LOGO_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/Assets/Signius_logo.png'

"""It seems to me that comments in this part of the code are unnecessary"""


class InitWindow:
    def __init__(self, master: tkinter.Tk):
        self.app = None
        self.master = master
        self.master.geometry("1080x720")
        self.master.title("Signius - A.I. for sing language")
        self.master.iconphoto(True, PhotoImage(file=LOGO_PATH))
        self.master.config(background='#122C34')
        self.start_button = Button(self.master,
                                   text="CLICK ME!",
                                   font=("notosansmono nerd font", 40),
                                   command=self.new_window,
                                   bg="red",
                                   borderwidth=10,
                                   )

        self.logo_label = Label(self.master,
                                text="Signius",
                                font=("inconsolata condensed bold", 60))

        self.info_label = Label(self.master,
                                text="Version: Alpha 1.02, Author: Piotr Urbański, 321815")

        self.logo_label.place(relx=0.5, rely=0.2, anchor=CENTER)
        self.info_label.place(relx=1.0, rely=1.0, anchor=SE)
        self.start_button.place(relx=0.5, rely=0.5, anchor=CENTER)

    def new_window(self):
        new_window = Toplevel(self.master)
        self.app = MainWindow(new_window)
        self.master.withdraw()


class MainWindow:

    def __init__(self, master: tkinter.Toplevel):
        self.text = "Your text:"
        self.master = master
        self.master.geometry("1080x720")
        self.master.config(background='#122C34')
        self.logo_label = Label(self.master, text="Signius", font=("inconsolata condensed bold", 60))
        self.info_label = Label(self.master, text="Version: Alpha 1.02, Author: Piotr Urbański, 321815")

        self.info_label.place(relx=1.0, rely=1.0, anchor=SE)

        self.f1 = LabelFrame(self.master, bg="red")
        self.f1.place(relx=0.5, rely=0.6, anchor=CENTER)
        self.f1.pack()
        self.L1 = Label(self.f1, bg="red")
        self.L2 = Label(self.master, font=("notosansmono nerd font", 20))
        self.L1.pack()
        self.L2.pack()
        self.L2.place(relx=0.5, rely=0.95, anchor=CENTER)

        img_processing = ImageProcessing()

        """Here we have dict with id of each gesture with corresponding letter"""
        iterator = 0
        prv_gesture = 0
        gestures_dict = {0: "",
                         1: "a",
                         2: "b",
                         3: "c",
                         4: "d",
                         5: "e",
                         6: "f",
                         7: "g",
                         8: "h",
                         9: "i",
                         10: ""}

        cap = cv2.VideoCapture(0) #if video capturing is not working, change the value for 1/2/3 etc. basically it should work with id 0 which is default id for camera
        cap.set(3, 320)
        cap.set(4, 320)
        cap.set(10, 100)

        while True:
            ret, frame = cap.read()
            processed_data = img_processing.ret_processed_image(frame)
            """Scaving image and most accurate gesture id from ImageProcessing into two variables"""
            img = processed_data[0]
            gesture_id = processed_data[1]


            if iterator == 45:
                if gesture_id == 10 and len(self.text) > 10:
                    self.text = self.text[:-1]
                elif gesture_id != prv_gesture:
                    self.text += gestures_dict[gesture_id]
                prv_gesture = gesture_id
                iterator = 0

            iterator += 1
            img = cv2.resize(img, (800, 600))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(img))
            self.L1['image'] = img
            self.L2['text'] = self.text
            self.master.update()


if __name__ == '__main__':
    root = Tk()
    InitWindow(root)
    root.mainloop()
