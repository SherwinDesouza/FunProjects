from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import json
from CleaningData import JsonData
from langchain_text_splitters import RecursiveJsonSplitter

load_dotenv()

sheet_id = "17xIKTDiz8KRnbkggR7lOPjJPIidew4WVCrNPPhrGalM"
sheet_names = ["Sales", "Inventory"]
data = JsonData(sheet_id, sheet_names)
data = json.loads(data)


def split_large_json(data, key, chunk_size=5):  # Adjust chunk_size as needed
    if key in data and isinstance(data[key], list):
        chunks = [data[key][i:i + chunk_size] for i in range(0, len(data[key]), chunk_size)]
        return [{key: chunk} for chunk in chunks]
    return [data]  # Return as-is if no chunking needed

sales_chunks = split_large_json(data, "Sales", chunk_size=3)  # Adjust chunk size
inventory_chunks = split_large_json(data, "Inventory", chunk_size=3)

final_chunks = []
for s_chunk, i_chunk in zip(sales_chunks, inventory_chunks):
    final_chunks.append({**s_chunk, **i_chunk})  



template = (
    "You are a data analyst. Your task is to generate a detailed sales report based on the following dataset.\n\n"
    "### Sales Data:\n"
    "{data}\n\n"
    "### Instructions:\n"
    "1. Provide a summary of total sales, best-selling items, and trends over time.\n"
    "2. Identify the top-performing categories or products and the least performing ones.\n"
    "3. Highlight any seasonal patterns or fluctuations in sales.\n"
    "4. Compare the current period's sales to the previous period (if applicable).\n"
    "5. Identify which product is doing great in which regions so that in future we can increase our supplies for that product in that area."
    "### Format:\n"
    "- **Total Revenue:** $XXX\n"
    "- **Top-Selling Products:** Product A ($X sales), Product B ($X sales)\n"
    "- **Least-Selling Products:** Product X ($X sales), Product Y ($X sales)\n"
    "- **Sales Trends:** [Mention patterns, seasonal spikes]\n"
    "- **Comparative Analysis:** [Month-over-month growth, year-over-year insights]\n"
    "- **Future Plans for Sales:** [Mention Product Sales per Regions]\n"
    "Generate a structured and insightful report."
)

groq_api_key = os.getenv("groq_api_key")
model = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=groq_api_key)

parsed_reports = []
for i, chunk in enumerate(final_chunks, start=1):
    prompt = ChatPromptTemplate.from_template(template).format(data=chunk)
    response = model.invoke(prompt)
    print(f"Processed chunk {i}/{len(final_chunks)}")
    parsed_reports.append(response.content)

final_report = "\n\n".join(parsed_reports)
print(final_report)
