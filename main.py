import streamlit as st
from streamlit_option_menu import option_menu

import pandas as pd
import numpy as np
import pickle

import timm
import torch
import torch.nn as nn
import albumentations as A
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from albumentations.pytorch import ToTensorV2
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# import firebase_admin
# from firebase_admin import auth, credentials 


transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914,0.4822,0.4655),(0.2023,0.1994,0.2010))])
pickle_in =open("classifier.pkl","rb")
model=pickle.load(pickle_in)

classes = ["","Grassy","Marshy","Rocky","Sandy"]


def predict_terrain(image):
   img=transform(np.array(Image.open(image)))
   img=torch.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2]))
   prediction=model.forward(img)
   print(prediction)
   idx=torch.argmax(prediction,dim=1)
#    print(type(idx))
   return idx.item()
    
def main():
    st.title("upload image")
    html_temp = """
    <div>classification</div>
    """

    st.markdown(html_temp,unsafe_allow_html=True)
    image=st.file_uploader("Please upload an image")
    result=0
    if image is not None:
        st.image(image)
    if st.button("Predict"):
        result=predict_terrain(image)
        result=result+1
    st.success('The terrain is {}'.format(classes[result]))

if __name__=='__main__':
    main()


# if not firebase_admin._apps:
#     cred = credentials.Certificate('firebase.json') 
#     default_app = firebase_admin.initialize_app(cred)

# st.set_page_config(
#     page_title="Streamlit",
#     page_icon="🧊",
# )

st.title('welcome')


choice = st.selectbox('Login/Signup', ['Login','Sign Up'])
# def state():
#     st.write(auth.current_user['uid'])


# state()

# def f():
#     try:
#         user = auth.get_user_by_email(email)
#         # state()
#         st.write(user.uid+' login successfull')
#     except:
#         st.warning('Login Failed')


# if choice == 'Login':
#     email = st.text_input('Email Address')
#     password = st.text_input('Password', type = 'password')
#     st.button('Login',on_click=f)

# else: 
#     email = st.text_input('Email Address')
#     password= st.text_input('Password', type='password')
#     username = st.text_input('Enter your unique username')
#     if st.button('Create my account'):
#         user = auth.create_user(email = email,password=password,uid=username)

#         st.success('Account created successfully, Please Login')
#         st.balloons()

# hide_streamlit_style = """
# <style>
# #MainMenu {visibility: hidden;}
# footer {visibility: hidden;}
# .stDeployButton {visibility: hidden;}
# </style>

# """



# # 1=sidebar menu, 2=horizontal menu, 3=horizontal menu w/ custom menu
# EXAMPLE_NO = 2


# def streamlit_menu(example=1):
#     if example == 2:
#         # 2. horizontal menu w/o custom style
#         selected = option_menu(
#             menu_title=None,  # required
#             options=["Home", "Predict", "Profile"],  # required
#             icons=["house", "book", "envelope"],  # optional
#             menu_icon="cast",  # optional
#             default_index=0,  # optional
#             styles={
#                 "container": {"padding": "0!important", "background-color": "#fafafa"},
#                 "icon": {"color": "orange", "font-size": "25px"},
#                 "nav-link": {
#                     "text-align": "center",
#                     "margin": "5px",
#                     "--hover-color": "#eee",
#                 },
#                 "nav-link-selected": {"background-color": "green"},
#             },
#             orientation="horizontal",
#         )
#         return selected


# selected = streamlit_menu(example=EXAMPLE_NO)

# if selected == "Home":
#     st.title(f"You have selected {selected}")
# if selected == "Predict":

    
#     st.title("upload image")
#     # image=st.file_uploader("Please upload an image")
#     # if image is not None:
#     #     st.image(image)
    
# if selected == "Profile":
#     st.title(f"You have selected {selected}")


# st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
