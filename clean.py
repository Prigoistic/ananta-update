import json

file_path = "formatted_math_dataset.json"
output_path = "cleaned_math_dataset.json"

fixed_data = []

with open(file_path, "r", encoding="utf-8") as f:
    try:
        # Load the entire JSON to check structure
        data = json.load(f)
        
        # Ensure it's a list
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict):  # Ensure valid JSON object
                    fixed_data.append(entry)
                else:
                    print(f"⚠️ Skipping invalid entry: {entry}")
        else:
            print("🚨 JSON does not contain a valid list of objects!")

    except json.JSONDecodeError as e:
        print(f"🚨 JSON Decode Error: {e}")

# Save cleaned JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(fixed_data, f, indent=4)

print(f"✅ Cleaned dataset saved as {output_path}. Valid entries: {len(fixed_data)}")
