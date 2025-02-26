from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from CleaningData import JsonData

sheet_id = "google_sheet_id"
data = JsonData(sheet_id)
load_dotenv()

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
    "- **Future Plans for Sales:** [Mention Product Sales per Regions]"
    "Generate a structured and insightful report."
)

groq_api_key = os.getenv("groq_api_key")
model = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=groq_api_key)
prompt = ChatPromptTemplate.from_template(template).format(data=data)

report = model.invoke(prompt)

print(report.content)
