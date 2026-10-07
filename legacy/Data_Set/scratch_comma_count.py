with open(r"e:\Data_Set\integrated_output\full_integrated_dataset.csv", "r", encoding="utf-8") as f:
    header = f.readline()
    for idx in range(1, 150168):
        line = f.readline()
        if idx == 150167:
            print(f"Line length: {len(line)}")
            # Let's search for "Female"
            fem_idx = line.find("Female")
            print(f"Index of 'Female': {fem_idx}")
            if fem_idx != -1:
                print(f"Surrounding 'Female': '{line[max(0, fem_idx-10):fem_idx+20]}'")
            
            # Let's search for "34.0"
            bmi_idx = line.find("34.0")
            print(f"Index of '34.0': {bmi_idx}")
            if bmi_idx != -1:
                print(f"Surrounding '34.0': '{line[max(0, bmi_idx-10):bmi_idx+20]}'")
            break
