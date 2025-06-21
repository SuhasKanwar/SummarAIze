import os
from dotenv import load_dotenv
import streamlit as st
import validators
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

st.set_page_config(page_title="SummarAIze", page_icon=":robot:", layout="wide")
st.title("SummarAIze - Summarize YouTube Videos and Web Pages")
st.subheader("Enter a YouTube video URL or a web page URL to get a summary")

url = st.text_input("URL", label_visibility="collapsed")
llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)

prompt_template = """
Provide a concise summary of the following content:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize"):
    if not url or not url.strip():
        st.error("Please enter a URL.")
    elif not validators.url(url):
        st.error("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Loading..."):
                if "youtube.com" in url or "youtu.be" in url:
                    # loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                    loader = YoutubeLoader.from_youtube_url(url)
                else:
                    loader = UnstructuredURLLoader(urls=[url], ssl_verify=False, headers=HEADERS)
                
                documents = loader.load()
                
                if not documents:
                    st.error("No content found at the provided URL.")
                else:
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    summary = chain.run(documents)
                    
                    st.success("Summary generated successfully!!!")
                    st.write(summary)
        except Exception as e:
            st.error("Request failed. Check URL or headers.")
            st.write(e)