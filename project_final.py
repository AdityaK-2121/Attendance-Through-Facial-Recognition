import cv2
import numpy as np
import face_recognition
import os
import csv
import datetime
from tkinter import *
from tkinter import messagebox

def func1():
    x=var.get()
    return x


def exit():
    ans=messagebox.askyesno("Exit","Do you want to exit?")
    if ans==True:
        win.destroy()
def func():
    x=var.get()
    if x!="":
        messagebox.showinfo("Success","Name added")
    else:
        messagebox.showwarning("Warning","Enpty box")
    print(x)
    return x


def reg_face():
    name=func1()
    print(name)
    cap=cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    while True:
        path="C:\Users\DELL\OneDrive\Desktop\Projects\Attendance through facial recognition"
        ret,frame=cap.read()
        frame=cv2.flip(frame,1)
        if ret:
            cv2.imshow("Frame",frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cv2.imwrite(f"{path}/{name}.jpg",frame)


def attendance():
    today = str(datetime.date.today())
    att_file=open("attendance.csv","a+")
    att_writ=csv.writer(att_file)
    att_writ.writerow(['Name',today])


    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    path = "images"
    img_names = []
    img_labels = []
    scan_name=[]
    un_name=[]
    folders = os.listdir(path)
    for folder in folders:
        img = cv2.imread(f'{path}/{folder}')
        img_names.append(img)
        t=folder.replace('.jpg','')
        img_labels.append(t)  # Save the name of the image file for labeling

    def img_trainer(l):
        encode_train = []
        for img in l:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encoding = face_recognition.face_encodings(img)[0]
            encode_train.append(encoding)
        return encode_train

    encoding_known = img_trainer(img_names)

    cap = cv2.VideoCapture(0)

    while True:
        ret, f = cap.read()
        if ret == True:
            frame1 = cv2.flip(f, 1)
            frame = cv2.resize(frame1, (500, 500))
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faceloccur = face_recognition.face_locations(frame_rgb)  # list
            encodecur = face_recognition.face_encodings(frame_rgb, faceloccur)  # array

            if len(faceloccur) > 0 and len(encodecur) > 0:
                for i in range(len(faceloccur)):
                    matches = face_recognition.compare_faces(encoding_known, encodecur[i])
                    dis = face_recognition.face_distance(encoding_known, encodecur[i])
                    mat_ind = np.argmin(dis)

                    if matches[mat_ind]:
                        name = img_labels[mat_ind]
                        print(name)

                        top, right, bottom, left = faceloccur[i]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame, name, (left, top - 10), font, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        if name not in scan_name:
                            scan_name.append(name)
                            att_writ.writerow([name, 1])
                        
                        

                                


            faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow("frame", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    l3=list(set(img_labels)-set(scan_name))
    for k in l3:
        att_writ.writerow([k,0])
    cap.release()
    cv2.destroyAllWindows()
    att_file.close()


win = Tk()
win.title("Attendance")
win.geometry('600x300')
# # file=PhotoImage(file="bg2.png")
# # lab=Label(win,image=file)
# lab.pack()
lbl=Label(win,text="Name")
lbl.place(x=40,y=150)
btn1=Button(win,text="Register new face",command=reg_face)
btn1.place(x=40, y=220)
var=StringVar()
ent=Entry(win,textvariable=var)
ent.place(x=90,y=150)
btn2 = Button(win, text="Start Attendance", command=attendance)
btn2.place(x=350, y=220)
btn3 = Button(win, text="Submit",command=func)
btn3.place(x=220, y=145)
btn4 = Button(win, text="Exit",command=exit)
btn4.place(x=490, y=220)

win.mainloop()