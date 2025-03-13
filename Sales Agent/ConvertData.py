import requests
import json


def JsonData(sheet_id, sheet_names):
    all_data = {}

    for sheet_name in sheet_names:
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:json&sheet={sheet_name}"
        
        response = requests.get(url)
        raw_data = response.text[47:-2]

        data_json = json.loads(raw_data)
        columns = [col["label"] for col in data_json["table"]["cols"]]

        rows = []
        for row in data_json["table"]["rows"]:
            row_data = [cell.get("v", "") if cell else "" for cell in row["c"]]
            rows.append(dict(zip(columns, row_data)))

        all_data[sheet_name] = rows

    return json.dumps(all_data, indent=2)




