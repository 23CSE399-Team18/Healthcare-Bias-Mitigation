with open(r"e:\Data_Set\integrated_output\full_integrated_dataset.csv", "r", encoding="utf-8") as f:
    header = f.readline()
    for idx in range(1, 150167):
        line = f.readline()
        if idx == 150166:
            print(f"Tabs in line: {line.count('\t')}")
            print(f"Semicolons in line: {line.count(';')}")
            print(f"Commas in line: {line.count(',')}")
            # Let's see if there are any other characters or if the file is truly comma-separated
            break
