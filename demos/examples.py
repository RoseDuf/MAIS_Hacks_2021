import streamlit as st
import pandas as pd
from PIL import Image


def show():

    st.subheader('Database')
    st.write('https://www.kaggle.com/grassknoted/asl-alphabet')
    st.write('For each letter `a` `b` `...` `z` `del` `space`, we included 2800 images.')
#     with st.echo():
#         st.download_button(
#             label="DOWNLOAD!",
#             data="trees",
#             file_name="string.txt",
#             mime="text/plain"
#         )
    st.markdown('---')

    st.subheader('Preprocessing')
    st.write('`Mediapipe` was adpated to capture shapes of hands by returning landmarks, as shown in the following figures. Since each hand has 21 landmarks, we processed each image so that it comes with a label for letter and a feature list of length 63 (with x, y, z indices for each).  ')

   
    img1 = Image.open("pic/k2969.png")
    img2 = Image.open("pic/k2980.png")
    img3 = Image.open("pic/k2977.png")
    img4 = Image.open("pic/k2975.png")
    img5 = Image.open("pic/k2971.png")
    img6 = Image.open("pic/k2979.png")
    #st.image(img)
    #st.image(img)
    
    st.image([img1,img2,img3,img4,img5,img6])
    original_title = '<p style="font-size: 13px;">landmarks on images for letter K </p>'
    st.markdown(original_title, unsafe_allow_html=True)

    
    st.markdown('---')
    st.subheader('Models trained')
    st.write('With data splitting in to `train` and `test` by 80/20, following are the a few models we tried:')
    st.markdown('* Kth Nearest Neighbours')
    st.markdown('* Linear Regression')
    st.markdown('* Decision Trees')
    st.markdown('* Random Forest')
    st.markdown('* Gaussian Naive Bayes')
    st.write('where it turns out `Random Forest` outperforms the others.')
    img7 = Image.open("pic/acc.png")
    st.image(img7)
    original_title = '<p style="font-size: 13px;">Random forest accuracy for each letter </p>'
    st.markdown(original_title, unsafe_allow_html=True)
    
    st.markdown('---')
    st.subheader('Auto correction')
    st.write('We used the `space` as a stop sign, then do auto correction of the letters in the stack.')
    st.markdown('* word is marked `green` if correct upon inputting')
    st.markdown('* word is marked `yellow` if auto-corrected')


