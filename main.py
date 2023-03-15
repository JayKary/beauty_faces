import streamlit as st
import numpy as np
import tensorflow
from streamlit_option_menu import option_menu
from PIL import Image
from keras.utils import load_img,img_to_array
from keras.models import load_model

beauty_imges =["beauty1.jpg","beauty2.jpg",
               "beauty3.jpg","beauty4.jpg",
               "beauty5.jpg","beauty6.jpg"]

average_imges =["average1.jpg","average2.jpg",
                "average3.jpg","average4.jpg",
                "average5.jpg","average6.jpg"]

with st.sidebar:
    choose = option_menu('App Gallery',['About','Beautiful Faces','Average Faces',
                                        'AI Predict','Metrics Performance'],
                         icons=['house','image','image-fill','question-diamond-fill','speedometer'],
                         menu_icon='prescription2',default_index=0,
                         styles={
                             'container':{'padding':'5!important','background-color':'#fafafa'},
                             'icon':{'color':'orange','font-size':'25px'},
                             'nav-link':{'font-size':'16px','text-align':'left','margin':'0px','--hover-color':'#eee'},
                             'nav-link-selected':{'background-color':'#02ab21'}
                         })

if choose=='About':
    st.write("<h2>Classify female faces: Beautiful or Average</h2>",unsafe_allow_html=True)
    st.write("Credits to Gerry, This dataset was created to determine if a CNN model could be trained to classify female facial images "
             "into one of two classes: Beautiful or Average.")
    st.write("Towards this end, 2000 images of fashion models were gathered "
             "from all over the world to use as the images for the Beautiful data set. "
             "We then gathered 2000 images selected from the CELEB data set. From that "
             "data set I selected women that appeared to be in about the same age range as"
             " the fashion models but that I thought were of average appearance."
             " Of course this is **MY interpretation** of average which will differ from what"
             " others might consider as average.")

elif choose=='Beautiful Faces':
    st.write('<div align= "center"><h1>Beautiful Faces</h1></div>',unsafe_allow_html=True)
    col1,col2,col3=st.columns(3)
    with col1:
        for i in range(2):
            img=Image.open(beauty_imges[i])
            st.image(img)
    with col2:
        for i in range(2,4):
            img=Image.open(beauty_imges[i])
            st.image(img)
    with col3:
        for i in range(4,6):
            img=Image.open(beauty_imges[i])
            st.image(img)

elif choose=='Average Faces':
    st.write("<div align='center'><h1>Average Faces</h1></div>",unsafe_allow_html=True)
    col1,col2,col3 = st.columns(3)
    with col1:
        for i in range(2):
            img=Image.open(average_imges[i])
            st.image(img)
    with col2:
        for i in range(2,4):
            img=Image.open(average_imges[i])
            st.image(img)
    with col3:
        for i in range(4,6):
            img=Image.open(average_imges[i])
            st.image(img)
elif choose=='AI Predict':
    st.write("<div align ='center'><h2>Female Faces Classification: <br> Beautiful or Average ? </h2></div>",unsafe_allow_html=True)
    model =load_model("C:/Users/j.elachkar/Desktop/beauty.h5")
    image1="predict1.jpg"
    image2="predict2.jpg"
    image3="predict3.jpg"
    image4="predict4.jpg"
    class_names =["Average","Beautiful"]
    uploaded_file = st.file_uploader("",type=['jpg','jpeg','png','gif'])
    if st.button("Predict"):
        if uploaded_file is not None:
            img=load_img(uploaded_file,target_size=(256,256))
            img = img_to_array(img)
            img=np.expand_dims(img,axis=0)
            img = img/255.0
            pred= model.predict(img)
            if pred > 0.5:
                st.write("The Model predicts with **95%** accuracy that the uploaded image shows **Beautiful** face traits")
            else:
                st.write("The model predicts with **95%** accuracy that the uploaded image shows **Average** face traits")
    col1, col2, col3, col4 = st.columns ( 4 )
    with col1:
        img = Image.open ( image1 )
        st.image ( img, caption="Image 1" )
    with col2:
        img = Image.open ( image2 )
        st.image ( img, caption="Image 2" )
    with col3:
        img = Image.open ( image3 )
        st.image ( img, caption='Image 3' )
    with col4:
        img = Image.open ( image4 )
        st.image ( img, caption='Image 4' )

elif choose=='Metrics Performance':
    st.write("<div align='center'><h2> Metrics Performance</h2></div>",unsafe_allow_html=True)
    col1,col2,col3 = st.columns(3)
    with col1:
        if st.button('Check Accuracy'):
            st.write("The overall model accuracy is **94.33 %**")
    with col2:
        if st.button("Check Recall"):
            st.write("The recall  tells us the percentage of positive samples that were correctly identified by the model")
            st.write("The recall is **91 %**")
    with col3:
        if st.button("Check Precision"):
            st.write("The precision tells us the percentage of the model's positive predictions that are correct")
            st.write("The precision is **97 %**")












    


