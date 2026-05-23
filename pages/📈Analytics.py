import streamlit as st
import streamlit.components.v1 as components
import pygwalker as pyg


def main():

    # Set up Streamlit interface
    st.set_page_config(
        page_title="📈 Interactive Visualization Tool", page_icon="📈", layout="wide"
    )

    st.header("📈 Interactive Visualization Tool")
    st.write("### Welcome to interactive visualization tool. Please enjoy !")

    # Render pygwalker
    if st.session_state.get("df") is not None:
        pyg_html = pyg.to_html(st.session_state.df)
        components.html(pyg_html, height=1000, scrolling=True)

    else:
        st.info("Please upload a dataset to begin using the interactive visualization tools")


if __name__ == "__main__":
    main()