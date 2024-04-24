import streamlit as st
import PyPDF2

def read_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""

    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
        
    return text

def main():
    st.title("PDF Viewer")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Read the contents of the uploaded PDF file
        pdf_contents = read_pdf(uploaded_file)
        
        # Display the contents on the web page
        st.header("PDF Contents:")
        st.text(pdf_contents)

if __name__ == "__main__":
    main()
