import streamlit as st
from . import examples
from PIL import Image



def show_examples():

    st.write(
            """
        ### ASL table generated from the database we used
        """
    )
    
    st.write("---")
    img1 = Image.open("pic/asl.png")
    st.image(img1)


if __name__ == "__main__":
    pass