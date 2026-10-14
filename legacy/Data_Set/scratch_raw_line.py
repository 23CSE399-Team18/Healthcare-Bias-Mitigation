with open(r"e:\Data_Set\integrated_output\full_integrated_dataset.csv", "r", encoding="utf-8") as f:
    # Read header
    header = f.readline()
    print("RAW HEADER (first 1000 chars):")
    print(header[:1000])
    
    # Read lines to row 150166 (which is line index 150166 since line 0 is header)
    for idx in range(1, 150167):
        line = f.readline()
        if idx == 150166:
            print(f"\nRAW LINE 150166 (first 1000 chars):")
            print(line[:1000])
            print(f"\nRAW LINE 150166 (last 500 chars):")
            print(line[-500:])
            break
