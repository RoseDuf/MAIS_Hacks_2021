import streamlit as st
from . import examples



def show_examples():

    st.write(
            """
        # Method overview 
        """
    )

    st.write("---")
    examples.show()


if __name__ == "__main__":
    pass