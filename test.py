import json

file_path = "formatted_math_dataset.json"

# Open JSON file in streaming mode
with open(file_path, "r") as f:
    print("✅ Reading JSON line by line to avoid MemoryError...")
    first_line = f.readline()  # Read opening "["
    data_list = []

    for line in f:
        line = line.strip()
        if line in ["]", "["]:  # Ignore opening/closing brackets
            continue

        if line.endswith(","):  # Remove trailing comma
            line = line[:-1]

        try:
            entry = json.loads(line)  # Load only one entry at a time
            data_list.append(entry)

            if len(data_list) % 1000 == 0:  # Print progress every 1000 samples
                print(f"Loaded {len(data_list)} samples so far...")
        except json.JSONDecodeError as e:
            print(f"⚠️ Skipping invalid JSON entry: {e}")

print(f"✅ Successfully loaded {len(data_list)} samples!")
print("Example sample:", data_list[0])  # Print first sample
