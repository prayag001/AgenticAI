import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
import streamlit as st

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

st.title("🛍️ Product Assistant 🚀")
user_input = st.text_input("Please enter your product query here...")


class ProductAssistant(BaseModel):
    """
    A Pydantic model representing product details including name, description, price, and category.
    """
    product_name: str = Field(description="The name of the product.")
    product_description: str = Field(
        description="A brief description of the product.", max_length=500)
    product_price: float = Field(
        description="The price of the product in USD.")
    product_category: str = Field(description="The category of the product.")


model_list = ["gemma2-9b-it", "llama-3.1-8b-instant"]
parser = JsonOutputParser(pydantic_object=ProductAssistant)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a product assistant that provides detailed information about products.Pleasr answer the user question professionally about the product\n"),
        ('user', '{product_query} \n {format_instructions}')
    ]
)
selected_model = st.selectbox("Select OpenAI Model", model_list)
button = st.button("Fetch Product Details", type="primary",
                   icon="🔍", use_container_width=True)
if selected_model:
    model = ChatGroq(model=selected_model, temperature=0.7, max_tokens=500)

if selected_model:
    model = ChatGroq(model=selected_model, temperature=0.7, max_tokens=500)
    chain = prompt | model | parser
    if user_input:
        if user_input.isnumeric():
            st.error(
                "Please enter a valid product query.Numeric input is not allowed.")
        else:
            response = chain.invoke(
                {"product_query": user_input, "format_instructions": parser.get_format_instructions()})
            response["product_price"] = f"${response['product_price']:.2f}"

            st.write("### Product Details")
            st.write(f"**Name:** {response['product_name']}")
            st.write(f"**Details:** {response['product_description']}")
            st.write(f"**Product Price:** {response['product_price']}")
            st.write(f"**Product Category:** {response['product_category']}")
            st.success("Product details fetched successfully!")
else:
    st.write("Please enter a product query to get started.")
