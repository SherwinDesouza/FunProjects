import gspread
from google.oauth2.service_account import Credentials
import random
import datetime
import time
json_keyfile = "creds.json" #Enter name of your json file

# Authenticate with correct scope
scopes = ["https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_file(json_keyfile, scopes=scopes)
client = gspread.authorize(creds)

sheet_id = "google_sheet_id"
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
            random.randint(5, 50),  # Items Sold
            random.randint(10, 100),  # Inventory Left
            round(random.uniform(10, 500), 2),  # Price per Unit
            0,  # Placeholder for Total Revenue
            datetime.date.today().strftime("%Y-%m-%d"),  # Sales Date
            random.choice(categories),  # Category
            random.choice(suppliers),  # Supplier
            "Yes" if random.randint(0, 1) else "No",  # Restock Alert
            "Yes" if random.randint(0, 1) else "No",  # Discount Applied
            round(random.uniform(3.0, 5.0), 1),  # Customer Ratings
            random.choice(regions),  # Region Sold In
            time.strftime("%H:%M:%S")# Sales time

        ]
        for i in range(10)
    ]

    # Calculate Total Revenue for each row
    for row in data:
        row[5] = round(row[2] * row[4], 2)  # Total Revenue = Items Sold * Price per Unit

    # Add header row
    if not(data_exists):
        headers = [
            "Product Name", "Product ID", "Items Sold", "Inventory Left", "Price per Unit",
            "Total Revenue", "Sales Date", "Category", "Supplier", "Restock Alert (Y/N)",
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
