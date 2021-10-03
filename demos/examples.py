import streamlit as st
import pandas as pd


def show():

    st.subheader('Database we choosed')
    st.write('With a single command, you can now create a button which downloads data of the form: strings, bytes or file pointers')
    with st.echo():
        st.download_button(
            label="DOWNLOAD!",
            data="trees",
            file_name="string.txt",
            mime="text/plain"
        )
    st.markdown('---')

    st.subheader('Preprocessing')

    with st.echo():
        binary_contents = b'whatever'

        # Defaults to the mimetype 'application/octet-stream'
        st.download_button('Download binary file', binary_contents)

    st.write('👉 Note: If a output file_name is missing, Streamlit creates an output file name for you')
    st.markdown('---')

    st.subheader('')
    with st.echo():
        text_contents = '''
        Foo, Bar
        123, 456
        789, 000
        '''
        st.download_button('Download CSV', text_contents, 'aaa.csv', 'text/csv')