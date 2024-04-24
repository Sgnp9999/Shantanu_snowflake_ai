import base64
import streamlit as st
import plotly.express as px

df = px.data.iris()
@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("side_bar.jpg")
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://source.unsplash.com/random");
background-size: 100%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

#sidebar
st.sidebar.title('Access Keys')
hf_token = st.sidebar.text_input("**HF TOKEN**", type="password")
answer_length = st.sidebar.slider("**ANSWER LENGTH**", min_value=0, max_value=100)


import streamlit as st
import pandas as pd
import streamlit_pdf_reader
import mistral

st.title("AI ChatBot")
data_type = st.radio("Select Input Type:", ("Chat", "CSV", "Personal", "Snowflake"))

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import faiss
from langchain.text_splitter import CharacterTextSplitter
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2", cache_folder="emb")

if data_type == "Chat":
  user_input = st.text_input("Ask question:", placeholder="What is the currency of India?")
  if st.button("Send"):
    if user_input:
      output = mistral.ask_mistral(user_input)
      st.write("Output:")
      st.success(output)
    else:
      st.error("Please enter some text to process.")

elif data_type == "Personal":
  question = st.text_input("Ask question:", placeholder="What is the currency of India?")
  uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
  try:
    text=streamlit_pdf_reader.read_pdf(uploaded_file=uploaded_file)
    text_splitter = CharacterTextSplitter()
    texts= text_splitter.split_text(text=text)
    db = faiss.FAISS.from_texts(texts, embeddings)
  except:
    pass
  if st.button("Send"):
    if question:
        output=mistral.personal_mistral(question=question, db=db)
        st.write("Output:")
        st.success(output)
    else:
        st.error("Please enter some text to process.")

elif data_type == "Snowflake":
  question = st.text_input("Ask question:", placeholder="Create a employee table in Snowflake")
  uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
  try:
    text=streamlit_pdf_reader.read_pdf(uploaded_file=uploaded_file)
    text_splitter = CharacterTextSplitter()
    texts= text_splitter.split_text(text=text)
    db = faiss.FAISS.from_texts(texts, embeddings)
  except:
    pass
  if st.button("Send"):
    if question:
        output=mistral.personal_mistral_snowflake(question=question, db=db)
        st.write("Output:")
        st.success(output)
    else:
        st.error("Please enter some text to process.")

elif data_type == "CSV":
  question=st.text_input("Ask question on CSV:", placeholder="What is the maximum value in the 'Age' column?")
  uploaded_file = st.file_uploader("Upload CSV File:")
  try:
    df=pd.read_csv(uploaded_file)
  except:
    pass
  if uploaded_file is not None:
    if st.button("Send"):
      output = mistral.mistral_csv(df=df, question=question)
      st.write("Output:")
      st.success(output)