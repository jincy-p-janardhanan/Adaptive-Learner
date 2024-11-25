# helper file to ignore incomplete data entries from the json file for dataset

import json
with open('dataset.json', 'r') as f:
    data = json.load(f)
    
filtered_data = [
    item for item in data if not (item['difficulty'] is None or item['type'] == "" or not item['exercises'])
]

with open('dataset.json', 'w') as f:
    json.dump(filtered_data, f, indent=4)

# print()
# print(json.dumps(filtered_data, indent=4))
print("File updated successfully.")
