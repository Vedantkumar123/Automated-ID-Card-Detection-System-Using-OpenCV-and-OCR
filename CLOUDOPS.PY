import tkinter as tk
from tkinter import PhotoImage
from tkinter import StringVar
from tkinter import OptionMenu
from tkinter import Text
from tkinter import Entry
from tkinter import ttk
import cv2
import numpy as np
import time
import pytesseract as tess
from PIL import Image, ImageTk

import mysql.connector
import random
import csv
from pytesseract import Output
import matplotlib.pyplot as plt
import os
import id_detect

tess.pytesseract.tesseract_cmd=r'C:\\Users\\Acer\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'
database=mysql.connector.connect(host="localhost",user="root",password="1234",database="CLOUD_OPS_ATTENDANCE")
def main():

 root = tk.Tk()
 root.title("CLOUD OPS")
 root.geometry("400x400") 
 img2 = PhotoImage(file='img2.jpeg')
 root.iconphoto(False, img2)

 root.configure(bg="pink")
 global img1
 img1=PhotoImage(file='img1.png')
 background=tk.Label(root,font=("times new roman",50),image=img1)
 background.place(x=0,y=0)

 frame = tk.Frame(root, bg="pink")
 frame.pack(expand=True)
 l_1=tk.Label(root,text="WELCOME TO CLOUDOPS",font=("times new roman",20),fg="black",bg="yellow").place(x=35,y=0)

 start_registration_button = tk.Button(root,text="START REGISTRATION",font=("times new roman",12),fg="yellow",bg="blue",command=start_registration1).place(x=110,y=70)

 view_data_button = tk.Button(root,text="VIEW DATA",font=("times new roman",12),fg="yellow",bg="blue",command=view_data).place(x=150,y=140)

 delete_data_button = tk.Button(root,text="DELETE DATA",font=("times new roman",12),fg="yellow",bg="blue",command=delete_data_function).place(x=150,y=210)

 generate_file_button = tk.Button(root,text="GENERATE FILE",font=("times new roman",12),fg="yellow",bg="blue",command=generate_file).place(x=140,y=280)
 exit_file_button = tk.Button(root,text="EXIT",font=("times new roman",12),fg="yellow",bg="blue",command=root.destroy).place(x=180,y=350)
 root.mainloop()


def start_registration():
   cursor_obj=database.cursor()
   query_2="select * from ATTENDEE"
   cursor_obj.execute(query_2)
   l=cursor_obj.fetchall()
   count=l[-1][0]
   name=input("enter name:")
   reg_no=input("enter reg_no:")
   if(len(reg_no)==9 and reg_no[0:2].isnumeric() and reg_no[2:5].isalpha() and reg_no[5:9].isnumeric() and reg_no.isalnum()):
     print("valid")
     cursor_obj.execute("insert into ATTENDEE values(%s,%s,%s)",(count,reg_no.upper(),name.upper()))
     database.commit()
   else:
     print("invalid") 


def view_data():
     
    data_window = tk.Toplevel()
    data_window.title("ATTENDANCE DATA")
    data_window.geometry("600x400")
  
    # Create a treeview widget to display data
    tree = ttk.Treeview(data_window, columns=("SL.NO","REGISTRATION NUMBER","NAME"), show="headings")
    z = ttk.Style(data_window)
    z.theme_use("alt")
    tree.heading("SL.NO", text="SL.NO")
    tree.heading("REGISTRATION NUMBER", text="REGISTRATION NUMBER")
    tree.heading("NAME", text="NAME")
    tree.pack(fill="both", expand=True)

    # Fetch data from the SQL table and insert it into the treeview
    def refresh_tree():
        for item in tree.get_children():
            tree.delete(item)
    refresh_tree()    
    cursor_obj = database.cursor()
    query_2 = "SELECT * FROM ATTENDEE"
    cursor_obj.execute(query_2)
    data = cursor_obj.fetchall()

    for row in data[1:]:
            tree.insert("", "end", values=(row[0], row[1], row[2]))
    
    

def generate_file():
    def display():
       display_window = tk.Toplevel()
       display_window.title("GENERATION SUCCESS")
       display_window.geometry("500x150")
       display_window.configure(bg="pink")
       l_1=tk.Label(display_window,text="STATUS: FILE GENERATED SUCCESSFULLY !",font=("times new roman",15),fg="green",bg="pink").place(x=35,y=0)
       l_2=tk.Label(display_window,text="FILENAME: ATTENDANCE.CSV",font=("times new roman",15),fg="green",bg="pink").place(x=35,y=50)
       exit_file_button = tk.Button(display_window,text="EXIT",font=("times new roman",12),fg="yellow",bg="blue",command=display_window.destroy).place(x=35,y=100)
    cursor_obj = database.cursor()
    query_2 = "SELECT * FROM ATTENDEE"
    cursor_obj.execute(query_2)
    data = cursor_obj.fetchall()[1:]  # Exclude the first row

   
    csv_file_path = "attendance.csv"

    # Write data to the CSV file
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Sl.No", "Name", "Reg No"])  # Write the header row
        for index, row in enumerate(data, start=1):
            csv_writer.writerow([index, row[2], row[1]])

    print(f"CSV file '{csv_file_path}' generated successfully!")
    display()


def start_registration1():
  cursor_obj=database.cursor()
  query_2="select * from ATTENDEE"
  cursor_obj.execute(query_2)
  l=cursor_obj.fetchall()
  count=l[-1][0]
  root = tk.Toplevel()
# Create a frame
  app = tk.Frame(root, bg="white")
  app.grid()

# Create a label in the frame
  lmain = tk.Label(app)
  lmain.grid()

# Capture from camera
  cap = cv2.VideoCapture(0)

# Function for video streaming
  def video_stream():
    _, frame = cap.read()
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(1, video_stream) 

  video_stream()

# Function to capture and save an image
  def capture_image():
    def insert_data():
       def dummyfun():
           
        def insert_sql_data(count):
           def success():
              root = tk.Tk()
              root.title("CLOUD OPS")
              root.geometry("300x100")
              root.configure(bg="pink")
              l_2=tk.Label(root,text="Record Created !!",font=("times new roman",15),fg="green",bg='pink').place(x=75,y=10)
              exit_file_button = tk.Button(root,text="OK",font=("times new roman",12),fg="yellow",bg="blue",command=root.destroy).place(x=115,y=50)
           
           cursor_obj = database.cursor()
           count=count+1
           name=name_text_box.get(1.0, "end-1c").upper()
           reg_no=reg_text_box.get(1.0, "end-1c").upper()
           print(count)
           cursor_obj.execute("insert into ATTENDEE values(%s,%s,%s)",(count,reg_no,name))
           database.commit()
           success()
           root.destroy()
        insert_sql_data(count)
       root = tk.Tk()
       root.title("CLOUD OPS")
       root.geometry("400x150") 
       root.configure(bg='pink')
       l_1=tk.Label(root,text="NAME:",font=("times new roman",12),fg="black",bg='pink').place(x=0,y=10)
       name_text_box=tk.Text(root,height = 1, width = 30,font=("times new roman",12))
       name_text_box.place(x=150,y=10)
       name_text_box.insert("1.0",k[1].upper().replace('\n', ' '))
       l_2=tk.Label(root,text="REGISTRATION NO:",font=("times new roman",12),fg="black",bg='pink').place(x=0,y=40)
       reg_text_box=tk.Text(root,height = 1, width = 30,font=("times new roman",12))
       reg_text_box.place(x=150,y=40)
       reg_text_box.insert("1.0",k[0].upper())
       submit_data_button = tk.Button(root,text="SUBMIT",font=("times new roman",12),fg="yellow",bg="blue",command=dummyfun).place(x=120,y=80)
       exit_file_button = tk.Button(root,text="EXIT",font=("times new roman",12),fg="yellow",bg="blue",command=root.destroy).place(x=200,y=80)
    _, frame = cap.read()
    cv2.imwrite("test_img.jpg", frame)
    baseImg = cv2.imread("base_img.jpg")
    img2=cv2.imread("test_img.jpg") 
    k=id_detect.finalrun(baseImg,img2)
    
    img=Image.open('test_img.png') 
    
    insert_data()

# Create a "Capture" button
  capture_button = tk.Button(root, text="Capture", command=capture_image)
  capture_button.grid()

  root.mainloop()

def delete_data_function():
   root = tk.Tk()
   root.title("CLOUD OPS")
   root.geometry("300x200")
   root.configure(bg="pink")
   def test_data():
     inp = text_box.get(1.0, "end-1c").upper()
     cursor_obj = database.cursor()
     query_2 = "select * from ATTENDEE"
     cursor_obj.execute(query_2)
     items = cursor_obj.fetchall()
     for i in items:
        if i[1] == inp:
            cursor_obj.execute("delete from attendee where REG_NO=%s",(inp,))
            database.commit()
            del_success_function()
            break  
     else:
        del_unsuccess_function()
   l_2=tk.Label(root,text="Enter Registration No:",font=("times new roman",12),fg="black",bg='pink').place(x=30,y=10)
   text_box=tk.Text(root,height = 1, width = 10,font=("times new roman",12))
   text_box.place(x=180,y=10)
   delete_button = tk.Button(root,text="DELETE",font=("times new roman",12),fg="yellow",bg="blue",command=test_data).place(x=115,y=50)
   root.mainloop()

      
  
  
       
def del_success_function():
      root = tk.Tk()
      root.title("CLOUD OPS")
      root.geometry("300x100")
      root.configure(bg="pink")
      l_2=tk.Label(root,text="Record Deleted Successfully !!",font=("times new roman",15),fg="green",bg='pink').place(x=20,y=10)
      exit_file_button = tk.Button(root,text="OK",font=("times new roman",12),fg="yellow",bg="blue",command=root.destroy).place(x=115,y=50)

def del_unsuccess_function():
          root = tk.Tk()
          root.title("CLOUD OPS")
          root.geometry("300x100")
          root.configure(bg="pink")
          l_2=tk.Label(root,text="Data Not Found !!",font=("times new roman",15),fg="red",bg='pink').place(x=75,y=10)
          exit_file_button = tk.Button(root,text="OK",font=("times new roman",12),fg="yellow",bg="blue",command=root.destroy).place(x=115,y=50)

#main


main()

