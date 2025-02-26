import requests
import json

def JsonData(sheet_id):
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:json"

    response = requests.get(url)
    raw_data = response.text[47:-2] 

    data_json = json.loads(raw_data)

    columns = [col["label"] for col in data_json["table"]["cols"]]

    # Extract row values
    rows = []
    for row in data_json["table"]["rows"]:
        row_data = [cell.get("v", "") if cell else "" for cell in row["c"]]
        rows.append(dict(zip(columns, row_data)))
    return json.dumps(rows, indent=2)



