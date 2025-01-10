import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from googleapiclient.discovery import build
# from pytubefix import YouTube

# import time
import os
from dotenv import load_dotenv
load_dotenv()  # Loading all environment variables

## Streamlit App Configuration
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

## API Keys and Inputs
with st.sidebar:
    groq_api_key = os.getenv("GROQ_API_KEY")
    youtube_api_key = os.getenv("YOUTUBE_API_KEY")

generic_url = st.text_input("URL", label_visibility="collapsed")

## Initialize the Gemma Model
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

## Helper Function Using YouTube Data API (Replaces PyTube)
def get_youtube_video_title(video_url, api_key):
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        video_id = video_url.split("v=")[-1]
        request = youtube.videos().list(part="snippet", id=video_id)
        response = request.execute()
        return response['items'][0]['snippet']['title']
    except Exception as e:
        st.error(f"Error fetching video title: {e}")
        return "Unknown Title"

## Button to Trigger Summarization
if st.button("Summarize the content from YT or website"):
    if not groq_api_key.strip() or groq_api_key == "" or not generic_url.strip():
        st.error("Please provide the required information.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Fetching and summarizing content..."):
                # Use YouTube API to fetch the title instead of PyTube
                if "youtube.com" in generic_url:
                    video_title = get_youtube_video_title(generic_url, youtube_api_key)
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                    st.write(f"Video Title: {video_title}")
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False,
                                                   headers={"User-Agent": "Mozilla/5.0"})
                
                # Load Content and Summarize
                docs = loader.load()
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.invoke({"input_documents": docs})['output_text']
                
                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception: {e}")
