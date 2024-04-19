# from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv
load_dotenv()
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.schema.runnable import RunnableMap
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chat_models import ChatOpenAI
# from pprint import pprint
from google.generativeai.types.generation_types import StopCandidateException
import os
import csv

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap = 20
)
pdf_path = r"Khaki-Files-Inside-Stories-of-Police-Missions-Kumar-Neeraj-Z-Library.pdf"


# embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")


loader = PyMuPDFLoader(pdf_path)

pages = loader.load_and_split()

total_splits = []
for page in pages[4:]:
    text = page.page_content
    a = text_splitter.split_text(text=text)
    total_splits.extend(a)

# template = """
# You are a Paragraph reader who generates questions based on the provided
# paragraph as context inside triple backticks below.
# Context : ```{context}```
# Provide question in such a way that it incorporates every information provided
# in the given context.  
# """
template = """
You are a Paragraph reader who generates questions based on the provided
paragraph as context inside triple backticks below.
Context : ```{context}```
Provide question in such a way that it incorporates every information provided
in the given context.  
Provide question in string rather than Numbered list.
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatGoogleGenerativeAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-1.0-pro",
    temperature=0.4
)

chain = RunnableMap(
{
    "context": lambda x: x['context']
}
) | prompt | model | StrOutputParser()



# result = chain.invoke({'context': total_splits[0]})
# print(result)
# print()
# print(total_splits[0])

csv_filename = "dataset_gemini.csv"

if not os.path.isfile(csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header row
        writer.writerow(['context', 'question', 'input'])

with open(csv_filename, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    i = 0
    for context in total_splits[i:]:
        try:
            print(i)
            i = i+1
            question = chain.invoke({'context':context})
            context = context.split("\n")
            context = " ".join(context)
            # input = fr"###Human:\n{question}\n\n###Assistant:\n{context}"
            input = fr"<s> [INST] {question} [/INST] {context} </s>"
            # print("Context = ",context)
            # print("Question = ", question)
            writer.writerow([f"{context}", question, input])
        except StopCandidateException:
            continue

print(len(total_splits))





