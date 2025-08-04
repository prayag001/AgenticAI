"""This is a lightweight LLM-powered assistant that extracts the Product Name and its Tentative Price in USD from a product description. 
It uses Groq LLMs (via Langchain) and provides an easy-to-use Streamlit UI."""

from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

#load Groq env
load_dotenv()

#Product model
class Product(BaseModel):
    product_id:Optional[str]= Field(default=None, description="Product ID")
    product_name:Optional[str]= Field(default=None, description="Product Name")
    descripton:Optional[str]= Field(default=None, description="Product Description")
    tentative_price_usd:Optional[float]= Field(default=None, description="Tentative Price in USD")
    category:Optional[str]= Field(default=None, description="Product Category such as Electronics, Clothing, etc.")
    rating:Optional[float]= Field(default=None, ge=0, le=5, description="Average Product Rating (0-5)")
    

#Prompt template with system and humand messages
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant with deep domain knowledge in"
                " product analysis and pricing. When the user gives about any"
                " product details, provide\n1. The **Product Name** \n2. The"
                " **Tentative Product Price in USD**.\n\nOnly return valid and"
                " structured information.") ,
        ("human", "{input}"),
    ]
)

#Streamlit UI
st.set_page_config(page_title="Product Analysis Assistant", page_icon=":robot:")
st.title("Product Analysis Assistant")

st.markdown("Welcome to the product **price analysis assistant** Powered by Groq LLMs. ")

#column Layout
col1, col2 = st.columns([1,2])

with col1:
    st.markdown("select LLM Model")
    model_otions=[
        "deepseek-r1-distill-llama-70b",
        "qwen-qwq-32b",
        "llama-3.1-8b-instant",
        "groq-llama-65b-v1",
        "groq-llama-70b-v2",
        "groq-qwen-14b",
        "groq-llama-2-70b-chat",
        "groq-llama-13b-chat",
    ]    
    model_choice = st.selectbox("Groq Hosted Model", model_otions)

with col2:
    st.markdown("**Enter product descriptin**")
    product_input= st.text_area("Describe the product below:",
                                placeholder=("e.g. This is a 55-inch 4K Ultra HD Smart TV with HDR support, "),
                                )
    
#Run prediction
if st.button("**Get product Info**"):
    if product_input and model_choice:
        with st.spinner("Analyzing product details..."):
            model = ChatGroq(model=model_choice)
            structure_output = model.with_structured_output(Product)
            chain = prompt| structure_output
            
            try:
                result = chain.invoke({"input": product_input})
                st.success("Product information extracted successfully!")
                st.markdown("### Result:")
                st.write(f"**Product Name:** {result.product_name}")
                st.write(f"**Tentative Price in USD:** ${result.tentative_price_usd}")
                st.write(f"**Product ID:** {result.product_id}")
                st.write(f"**Category:** {result.category}")
                st.write(f"**Rating:** {result.rating}")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a product description and select a model before running the analysis.")
                