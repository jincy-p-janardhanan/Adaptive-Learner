import json

# Load data from dataset.json
with open('dataset.json', 'r') as f:
    data = json.load(f)

# Filter the data
filtered_data = [
    item for item in data if not (item['difficulty'] is None or item['type'] == "" or not item['exercises'])
]

# Write the filtered data back to the same file
with open('dataset.json', 'w') as f:
    json.dump(filtered_data, f, indent=4)

print()
print(json.dumps(filtered_data, indent=4))
print("File updated successfully.")
