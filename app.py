import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.tools import tool
import re

def render_math_text(text: str):
    if not text:
        return
    
    text = text.replace("<br>", "\n")
    
    text = re.sub(r'\\\)\$', r'$', text)
    text = re.sub(r'\$\\\)', r'$', text)
    text = re.sub(r'\|\)', r')', text)
    
    text = re.sub(r'\(([^)]*?(?:\\in\vert{}\\mathbb\vert{}\\times\vert{}\\frac\vert{}\\sqrt\vert{}[_^])[^)]*?)\)', r'$\1$', text)
    
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    
    lines = text.split("\n")
    
    in_code_block = False
    code_content = []
    
    for line in lines:
        if line.strip().startswith("```"):
            if not in_code_block:
                in_code_block = True
                code_content.append(line)
            else:
                in_code_block = False
                code_content.append(line)
                st.markdown("\n".join(code_content))
                code_content = []
            continue
            
        if in_code_block:
            code_content.append(line)
        else:
            st.markdown(line)

load_dotenv()

st.set_page_config(
    page_title="AI Research Paper Intelligence System",
    layout="centered",
)

st.markdown("""
    <style>
    [data-testid="stAppDeployButton"],
    .stDeployButton,
    [data-testid="collapsedControl"], 
    [data-testid="stConnectionStatus"], 
    #MainMenu {
        display: none !important;
    }
    
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    
    [data-testid="stAppViewContainer"] {
        border: 4px solid #024B75 !important;
        box-sizing: border-box;
        min-height: 100vh;
    }

    [data-testid="stMainBlockContainer"] {
        padding: 45px !important;
        margin-top: 30px !important;
        margin-bottom: 30px !important;
        background-color: #FFFFFF !important;
    }

    .main-title {
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-size: 2.4rem;
        font-weight: 500;
        color: #000000;
        margin-bottom: 12px;
        line-height: 1.2;
    }
    .sub-title {
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        font-weight: 400;
        color: #000000;
        margin-bottom: 35px;
    }

    div[data-baseweb="input"] {
        background-color: #F3F4F6 !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 8px !important;
        padding: 4px !important;
    }
    div[data-baseweb="input"] input {
        color: #000000 !important;
        background-color: #F3F4F6 !important;
        font-size: 1.05rem !important;
        caret-color: #000000 !important;
    }
    div[data-baseweb="input"] input::placeholder {
        color: #6B7280 !important;
    }
    
    div[data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
    }

    div[data-testid="stFormSubmitButton"] {
        width: 100% !important;
        display: block !important;
    }

    div[data-testid="stFormSubmitButton"] button {
        background-color: #024B75 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 0px !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        margin-top: 10px;
        width: 100% !important;
        box-shadow: none !important;
        display: block !important;
    }
    div[data-testid="stFormSubmitButton"] button:hover,
    div[data-testid="stFormSubmitButton"] button:focus,
    div[data-testid="stFormSubmitButton"] button:active {
        background-color: #023859 !important;
        color: #FFFFFF !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }
    
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 8px !important;
        padding: 20px !important;
        margin-bottom: 15px !important;
    }

    .article-title {
        color: #000000 !important;
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 5px;
        margin-bottom: 15px;
    }

    div[data-testid="stExpander"] {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        margin-top: 15px !important;
        margin-bottom: 15px !important;
    }
    
    div[data-testid="stExpander"] summary {
        background-color: #024B75 !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
        padding: 10px 16px !important;
        width: fit-content !important;
        min-width: 280px;
    }
    
    div[data-testid="stExpander"] summary p,
    div[data-testid="stExpander"] summary svg {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }
    
    div[data-testid="stExpander"] summary:hover {
        background-color: #023859 !important;
    }

    div[data-testid="stExpanderDetails"] {
        background-color: #EBF5FB !important; 
        margin-top: 12px !important; 
        border-radius: 6px !important;
        padding: 15px !important;
        border: 1px solid #AED6F1 !important;
        color: #000000 !important;
    }

    .custom-keyword-tag {
        background-color: #E5E7EB !important;
        color: #000000 !important;
        padding: 4px 10px !important;
        border-radius: 6px !important;
        font-size: 0.9rem !important;
        font-family: monospace !important;
        display: inline-block !important;
        margin-right: 6px !important;
        margin-top: 4px !important;
        border: 1px solid #D1D5DB !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_heavy_resources():
    df = pd.read_csv("processed_papers.csv") 
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    faiss_index = faiss.read_index("paper_faiss.index")
    llm = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=os.getenv("GROQ_API_KEY"))    
    return df, embedding_model, faiss_index, llm

df, embedding_model, faiss_index, llm = load_heavy_resources()

@tool
def extract_keywords(text: str, top_n: int = 5) -> str:
    """Extract keywords from the given text snippet or title."""
    prompt = f"Extract the top {top_n} most important technical keywords or key phrases from the following text. Return the keywords formatted strictly as a Python list of strings (e.g., ['keyword1', 'keyword2']):\n\n{text}"
    response = llm.invoke(prompt)
    return response.content

@tool
def search_and_summarize(query: str) -> str:
    """Search FAISS and summarize abstracts."""
    query_embedding = embedding_model.encode([query]).astype(np.float32)
    scores, indices = faiss_index.search(query_embedding, k=3)
    matched_papers = df.iloc[indices[0]]
    
    context = ""
    for idx, row in matched_papers.iterrows():
        context += f"Title: {row['title']}\nAbstract: {row['abstract']}\n\n"
        
    prompt = f"Based on the following research papers, answer the query: {query}\n\n{context}"
    summary_response = llm.invoke(prompt)
    return summary_response.content

@tool
def compare_papers(paper1: str, paper2: str) -> str:
    """Compare two papers."""
    emb1 = embedding_model.encode([paper1]).astype(np.float32)
    emb2 = embedding_model.encode([paper2]).astype(np.float32)
    
    _, idx1 = faiss_index.search(emb1, k=1)
    _, idx2 = faiss_index.search(emb2, k=1)
    
    p1_data = df.iloc[idx1[0][0]]
    p2_data = df.iloc[idx2[0][0]]
    
    prompt = (
        f"Compare and contrast the following two research papers in a detailed Markdown format. "
        f"Include a summary comparison table analyzing their methodologies, key findings, and applications.\n\n"
        f"Paper 1: {p1_data['title']}\nAbstract: {p1_data['abstract']}\n\n"
        f"Paper 2: {p2_data['title']}\nAbstract: {p2_data['abstract']}"
    )
    comparison_response = llm.invoke(prompt)
    return comparison_response.content

tools = [search_and_summarize, extract_keywords, compare_papers]
llm_with_tools = llm.bind_tools(tools)

st.markdown('<div class="main-title">AI Research Paper Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">To start, enter a paper identifier</div>', unsafe_allow_html=True)

with st.form("search_form", clear_on_submit=False):
    user_query = st.text_input(
        "Search Input Field",
        label_visibility="collapsed",
        placeholder="Search by keywords, paper title, DOI or another identifier"
    )
    search_button = st.form_submit_button("Run Agent Query", use_container_width=True)

if search_button and user_query.strip():
    with st.spinner("🤖 Agent is reasoning and selecting the right tool..."):
        response = llm_with_tools.invoke(user_query)
        
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if tool_name == "compare_papers":
                final_output = compare_papers.invoke(tool_args)
                with st.container(border=True):
                    st.markdown(f'<div class="article-title">Comparison Analysis</div>', unsafe_allow_html=True)
                    render_math_text(final_output)
                    
            elif tool_name == "extract_keywords":
                final_output = extract_keywords.invoke(tool_args)
                try:
                    keywords_list = eval(final_output)
                    kw_html = "".join([f'<span class="custom-keyword-tag">{kw}</span>' for kw in keywords_list])
                except:
                    kw_html = f'<span class="custom-keyword-tag">{final_output}</span>'
                
                with st.container(border=True):
                    st.markdown(f'<div class="article-title">Extracted Keywords</div>', unsafe_allow_html=True)
                    st.markdown(f"**Keywords:** {kw_html}", unsafe_allow_html=True)
                    
            elif tool_name == "search_and_summarize":
                final_output = search_and_summarize.invoke(tool_args)
                with st.container(border=True):
                    st.markdown(f'<div class="article-title">Search Summary</div>', unsafe_allow_html=True)
                    render_math_text(final_output)
                    with st.expander("Show Background Source Abstract Context"):
                        st.write("Retrieved related resources and contexts are aggregated within this query execution pipeline.")
        else:
            with st.container(border=True):
                st.markdown(f'<div class="article-title">Agent Response</div>', unsafe_allow_html=True)
                render_math_text(response.content)

elif search_button:
    st.warning("Please type a topic or phrase into the search box above.")