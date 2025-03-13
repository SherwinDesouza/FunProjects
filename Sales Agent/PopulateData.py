import gspread
from google.oauth2.service_account import Credentials
import random
import datetime
import pandas as pd
import time
json_keyfile = "creds.json" #Enter name of your json file

# Authenticate with correct scope
scopes = ["https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_file(json_keyfile, scopes=scopes)
client = gspread.authorize(creds)

sheet_id = "17xIKTDiz8KRnbkggR7lOPjJPIidew4WVCrNPPhrGalM"
sheet = client.open_by_key(sheet_id)

def PopulateData(sheet,data_exists=False):
    product_names = [
    "Wireless Noise-Canceling Headphones", "Smart LED Desk Lamp", "Portable Power Bank",
    "Gaming Mechanical Keyboard", "Ergonomic Office Chair", "4K Ultra HD Monitor",
    "Bluetooth Smartwatch", "Smartphone Tripod Stand", "Wireless Charging Pad",
    "USB-C Multiport Adapter"
    ]
    categories = ["Electronics", "Home Office", "Gaming", "Accessories"]
    suppliers = ["Tech Solutions", "Future Gadgets", "NextGen Supplies", "Innovative Retail"]
    regions = ["New York", "California", "Texas", "Florida", "Illinois"]

    data = [
        [
            product_names[i],  # Product Name
            f"P-{1000 + i}",  # Product ID
            datetime.date.today().strftime("%Y-%m-%d"),  # Sales Date
            random.choice(categories),  # Category
            random.choice(suppliers),  # Supplier
            "Yes" if random.randint(0, 1) else "No",  # Discount Applied
            round(random.uniform(3.0, 5.0), 1),  # Customer Ratings
            random.choice(regions),  # Region Sold In
            time.strftime("%H:%M:%S")# Sales time

        ]
        for i in range(10)
    ]

    # Add header row
    if not(data_exists):
        headers = [
            "Product Name", "Product ID", "Sales Date", "Category", "Supplier",
            "Discount Applied (Y/N)", "Customer Ratings (1-5)", "Region Sold In","Sale Time"
        ]
        sheet.append_row(headers)  # Add headers only once

    # Add generated data to Google Sheets
    for row in data:
        sheet.append_row(row)

    print("✅ Spreadsheet successfully populated with sample data!")

sheet1 = sheet.get_worksheet(0)
first_cell = sheet1.acell("A1").value

if first_cell is None or first_cell.strip() == "":
    PopulateData(sheet1,data_exists=False)
else:
   PopulateData(sheet1,data_exists=True)


def CreateInventory(sheet_id):
    sheet = client.open_by_key(sheet_id)
    sheet1 = sheet.get_worksheet(0)
    sheet2 = sheet.get_worksheet(1)
    first_cell = sheet2.acell("A1").value
    sales_data = sheet1.get_all_records()
    sales_df = pd.DataFrame(sales_data)
    items_sold = sales_df['Product Name'].value_counts().to_dict()  # Convert Series to dict
    InventoryLeft = {key: 100 - value for key, value in items_sold.items()}  # Calculate Inventory Left
    Products = list(items_sold.keys())  # Unique product names
    ProductID = sales_df.drop_duplicates(subset=['Product Name'])['Product ID'].tolist()  # Unique Product IDs
    AverageRating = sales_df.groupby('Product Name')['Customer Ratings (1-5)'].mean()

    PricePerUnit = [round(random.uniform(10, 500), 2) for _ in range(len(Products))]
    Revenue = [PricePerUnit[i] * items_sold[Products[i]] for i in range(len(Products))]

    # Headers
    Headers = ['Product Name', 'Product ID', 'Inventory Left', 'Items Sold', 'Price Per Unit', 'Revenue','Average Rating']
    sheet2.append_row(Headers)

    for i in range(len(Products)):
        row = [Products[i], ProductID[i], InventoryLeft[Products[i]], items_sold[Products[i]], PricePerUnit[i], Revenue[i],round(AverageRating[i],1)]
        sheet2.append_row(row)  # Append row by row

sheet2 = sheet.get_worksheet(1)
first_cell = sheet2.acell("A1").value

if first_cell is None or first_cell.strip() == "":
    CreateInventory('17xIKTDiz8KRnbkggR7lOPjJPIidew4WVCrNPPhrGalM')
else:
   sheet2.clear()
   CreateInventory('17xIKTDiz8KRnbkggR7lOPjJPIidew4WVCrNPPhrGalM')
