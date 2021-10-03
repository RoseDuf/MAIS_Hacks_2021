import streamlit as st
import pandas as pd


def show():

    st.subheader('Database we choosed')
    st.write('https://www.kaggle.com/grassknoted/asl-alphabet')
#     with st.echo():
#         st.download_button(
#             label="DOWNLOAD!",
#             data="trees",
#             file_name="string.txt",
#             mime="text/plain"
#         )
    st.markdown('---')

    st.subheader('Preprocessing')

    with st.echo():
        binary_contents = b'whatever'

        # Defaults to the mimetype 'application/octet-stream'
        st.download_button('Download binary file', binary_contents)

    st.write('👉 Note:')
    st.markdown('---')

