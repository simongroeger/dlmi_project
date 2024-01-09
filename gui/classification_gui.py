import tkinter as tk
from tkinter import *
from tkinter import messagebox, filedialog
from PIL import ImageTk, Image
import os
import pandas as pd
from nn_classification import *
from baseline_classification import *


def load_data(path_1, path_2):
    """Load testdata"""

    filename_path_dict = {}
    for root, _, files in os.walk(path_1):
        for fname in files[:10]:
            fpath = root+"/"+fname
            filename_path_dict["Ampulla of Vater,"+fname] = [fpath, fname, "Ampulla of Vater"]

    for root, _, files in os.walk(path_2):
        for fname in files[:10]:
            fpath = root+"/"+fname
            filename_path_dict["Blood - fresh,"+fname] = [fpath, fname, "Blood - fresh"]

    print(filename_path_dict)
    return filename_path_dict


def create_window():
    """Create root window"""

    # create root window
    root = Tk()  
    root.title("Frame Example")
    root.config(bg="skyblue")
    root.geometry("1400x900")

    # caption_frame = Frame(root, width=700, height=40)
    # caption_frame.place(x=350, y=20)
    # create_caption(root)    

    # samples_header_frame = Frame(root, width=250, height=40)
    # samples_header_frame.place(x=50, y=140)

    # samples_frame = Frame(root, width=250, height=380)
    # samples_frame.place(x=50, y=185)
    # select_sample(root)

    # class_header_frame = Frame(root, width=250, height=40)
    # class_header_frame.place(x=50, y=610)

    # class_frame = Frame(root, width=250, height=100)
    # class_frame.place(x=50, y=655)
    # select_algorithm(root)

    # button_frame = Frame(root, width=250, height=50)
    # button_frame.place(x=50, y=790)

    # image_frame = Frame(root, width=700, height=700)
    # image_frame.place(x=350, y=140)
    # display_image(root)

    # predition_header_frame = Frame(root, width=250, height=40)
    # predition_header_frame.place(x=1090, y=140)

    # predition_frame = Frame(root, width=250, height=40)
    # predition_frame.place(x=1090, y=185)

    # add_data_header_frame = Frame(root, width=250, height=40)
    # add_data_header_frame.place(x=1100, y=270)

    # add_data_frame = Frame(root, width=250, height=520)
    # add_data_frame.place(x=1100, y=315)
    # display_img_data(root)

    return root
    # root.mainloop()


def create_caption(root):
    """Create the caption of the window"""

    # create cation frame
    caption_frame = Frame(root, width=700, height=40)
    caption_frame.place(x=350, y=20)

    # set caption
    tk.Label(caption_frame, font=("Arial", 25), 
                text="Bleeding detection for Wireless Capsule Endoscopy").pack()
    

def select_sample(root, image_data, metadata_path, image_data_field, prediction_field):
    """Select a sample for classification
    https://www.geeksforgeeks.org/scrollable-listbox-in-python-tkinter/ ???"""

    # create samples header frame
    samples_header_frame = Frame(root, width=250, height=40)
    samples_header_frame.place(x=50, y=140)

    # set header
    tk.Label(samples_header_frame, font=("Arial", 12), text="Image samples").pack()


    # create samples frame
    samples_frame = Frame(root, width=250, height=380)
    samples_frame.place(x=50, y=185)

    # create listbox, add it left-justified to samples frame
    listbox = Listbox(samples_frame, height=20, width=30, font=("Arial", 11))
    listbox.pack(side=LEFT, fill=BOTH) 
    
    # create scrollbar, add it right-justified to samples frame
    scrollbar = Scrollbar(samples_frame)
    scrollbar.pack(side=RIGHT, fill=BOTH) 
    
    # insert elements in listbox
    image_names = list(image_data.keys())
    for img_name in image_names: 
        listbox.insert(END, img_name) 
        
    # add vertical scrollbar to listbox
    listbox.config(yscrollcommand=scrollbar.set) 
    
    # set vertical/yview of listbox
    scrollbar.config(command=listbox.yview) 

    # selection - click event
    def show(event):
        prediction_field.delete("1.0","end")
        selection = listbox.curselection()
        if selection:
            selected_image_name = listbox.get(selection[0])
            fpath, fname, image_class = image_data[selected_image_name]
            display_image_data(root, fname, image_class, metadata_path, image_data_field)
            display_image(root, fpath)
            
    listbox.bind("<<ListboxSelect>>", show)

    # init: select first item of listbox 
    listbox.select_set(0)
    show("")

    return listbox


def select_sample_from_file_explorer(root, data_path, metadata_path, image_data_field):
    """Select sample from file explorer"""

    # create frames
    samples_header_frame = Frame(root, width=250, height=40)
    samples_header_frame.place(x=50, y=140)

    samples_button_frame = Frame(root, width=250, height=40)
    samples_button_frame.place(x=50, y=190)

    samples_text_frame = Frame(root, width=250, height=40)
    samples_text_frame.place(x=50, y=240)

    def select_sample():
        fpath = filedialog.askopenfilename(initialdir=data_path, title="Select Sample")
        fname = fpath.split("/")[-1]
        image_class = fpath.split("/")[-2].replace("_", " ")
        print("!!!"+image_class)

        # update sample field
        sample_field.delete("1.0","end")
        sample_field.insert(END, fpath)

        # update image data
        display_image_data(root, fname, image_class, metadata_path, image_data_field)

        # update image
        display_image(root, fpath)

    # set header
    tk.Label(samples_header_frame, font=("Arial", 12), text="Image sample").pack()

    # create file explorer button
    button = Button(samples_button_frame, text="Select Sample", font=("Arial", 11), width=27, command=select_sample)
    button.pack()

    # create sample text field
    sample_field = Text(samples_text_frame)
    sample_field.configure(font=("Arial", 11), height=3, width=33)
    sample_field.pack()

    return sample_field


def display_image(root, fpath):
    """Display the selected WCE image"""

    # create image frame
    image_frame = Frame(root, width=700, height=700)
    image_frame.place(x=400, y=190)

    # create Tkinter PhotoImage object
    image_path = fpath
    img = ImageTk.PhotoImage(Image.open(image_path).resize((600,600)))

    # show image with label widget
    label = Label(image_frame, image=img)
    label.image = img
    label.pack()


def create_image_data_field(root):
    """Create image data field"""

    # create additional img data header frame
    add_data_header_frame = Frame(root, width=250, height=40)
    add_data_header_frame.place(x=1090, y=270)

    # set additional img data header
    tk.Label(add_data_header_frame, font=("Arial", 12), text="Additional image data").pack()

    # create additional img data frame
    add_data_frame = Frame(root, width=250, height=520)
    add_data_frame.place(x=1090, y=315)

    # create image data field
    image_data_field = Text(add_data_frame)
    image_data_field.configure(font=("Arial", 11), height=28, width=33)
    image_data_field.pack()

    return image_data_field


def display_image_data(root, fname, image_class, metadata_path, image_data_field):
    """Display additional data to selected image"""

    # set additional img data text
    df = pd.read_csv(metadata_path, sep=";")
    d = df[df["filename"].str.match(fname) & df["finding_class"].str.match(image_class)]
    data = "Filename: {0}\nVideo_id: {1}\nFrame_number: {2}\nFinding_category: {3}\nFinding_class: {4}\nX1: {5}\nY1: {6}\nX2: {7}\nY2: {8}\nX3: {9}\nY3: {10}\nX4: {11}\nY4: {12}"\
        .format(d.iloc[0,0], d.iloc[0,1], d.iloc[0,2], d.iloc[0,3], d.iloc[0,4], d.iloc[0,5], d.iloc[0,6], d.iloc[0,7], d.iloc[0,8], d.iloc[0,9], d.iloc[0,10], d.iloc[0,11], d.iloc[0,12])
    image_data_field.delete("1.0","end")
    image_data_field.insert(END, data)


def select_algorithm(root):
    """Select algorithm which should be used for the WCE image classification"""

    # create class header frame
    class_header_frame = Frame(root, width=250, height=40)
    class_header_frame.place(x=50, y=610)

    # set header
    tk.Label(class_header_frame, font=("Arial", 12), text="Classifier").pack()

    # create class frame
    class_frame = Frame(root, width=250, height=100)
    class_frame.place(x=50, y=655)

    # create algorithm radio button
    neural_network_selected = IntVar()   
    baseline_selected = IntVar() 
    checkbutton_1 = Checkbutton(class_frame, text="Neural Network", font=("Arial", 11), width=25, variable=neural_network_selected, onvalue = 1, offvalue = 0)# , height = 2, width = 10) 
    checkbutton_1.pack()
    checkbutton_2 = Checkbutton(class_frame, text="Baseline", font=("Arial", 11), width=25, variable=baseline_selected, onvalue = 1, offvalue = 0)# , height = 2, width = 10) 
    checkbutton_2.pack()
    return neural_network_selected, baseline_selected


def create_prediction_field(root):
    """Create Prediciton header and field"""

    # create predition header frame
    predition_header_frame = Frame(root, width=250, height=40)
    predition_header_frame.place(x=1090, y=140)
    
    # set header
    tk.Label(predition_header_frame, font=("Arial", 12), text="Image class prediction").pack()

    # create predition frame
    predition_frame = Frame(root, width=250, height=40)
    predition_frame.place(x=1090, y=185)

    # create prediciton field
    prediction_field = Text(predition_frame)
    prediction_field.configure(font=("Arial", 11), height=3, width=33)
    prediction_field.pack()

    return prediction_field


def classify_image(root, sample_field, neural_network_selected, baseline_selected, prediction_field):
    """Classifies the selected image with the selected algorithm and shows class"""

    def prediction():

        # get filepath from sample field
        fpath = sample_field.get()

        # predict image class
        image_class_1 = ""
        image_class_2 = ""
        if neural_network_selected.get() == 1:
            image_class_1 = "Neural Network prediction: "+nn_classify(fpath, "")
        if baseline_selected.get() == 1:
            image_class_2 = "Baseline prediciton: "+baseline_classify(fpath)
        if neural_network_selected.get() == 0 and baseline_selected.get() == 0:
             messagebox.showerror('Python Error', 'Error: Select a classifier.')

        # update predicion field
        prediction_field.delete("1.0","end")
        if image_class_1 != "" and image_class_2 != "":
            image_classes = image_class_1+"\n"+image_class_2
        elif image_class_1 != "" and image_class_2 == "":
            image_classes = image_class_1
        elif image_class_1 == "" and image_class_2 != "":
            image_classes = image_class_2
        prediction_field.insert(END, image_classes)
    
    # create button frame
    button_frame = Frame(root, width=250, height=50)
    button_frame.place(x=50, y=765)

    # create prediction button
    button = Button(button_frame, text="Classify", font=("Arial", 11), width=27, command=prediction)
    button.pack()





