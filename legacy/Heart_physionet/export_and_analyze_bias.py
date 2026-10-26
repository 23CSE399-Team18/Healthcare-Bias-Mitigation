import os
import glob
import re
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

def parse_hea_file(filepath, base_dir):
    rel_path = os.path.relpath(filepath, base_dir).replace('\\\\', '/').replace('\\', '/')
    rec_id = os.path.splitext(rel_path)[0]
    patient_id = rel_path.split('/')[0]
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    if not lines:
        return fields

    header_info = lines[0].strip().split()
    num_signals = header_info[1] if len(header_info) > 1 else None
    fs = header_info[2] if len(header_info) > 2 else None
    num_samples = header_info[3] if len(header_info) > 3 else None
    
    fields = {
        'record': rec_id,
        'patient_id': patient_id,
        'num_signals': int(num_signals) if num_signals and num_signals.isdigit() else None,
        'sampling_freq_hz': int(fs) if fs and fs.isdigit() else None,
        'num_samples': int(num_samples) if num_samples and num_samples.isdigit() else None,
        'duration_sec': round(int(num_samples)/int(fs), 2) if num_samples and fs and num_samples.isdigit() and fs.isdigit() else None
    }
    
    current_key = None
    for line in lines:
        line_s = line.strip()
        if line_s.startswith('#'):
            content = line_s[1:].strip()
            if ':' in content:
                key, val = content.split(':', 1)
                key = key.strip().lower()
                val = val.strip()
                fields[key] = val
                current_key = key
            elif current_key:
                fields[current_key] += ' ' + content

    return fields

def main():
    base_dir = r'd:\Heart'
    hea_files = glob.glob(os.path.join(base_dir, '**', '*.hea'), recursive=True)
    print(f"Found {len(hea_files)} .hea files.")

    # Read CONTROLS list
    controls_file = os.path.join(base_dir, 'CONTROLS')
    controls_set = set()
    if os.path.exists(controls_file):
        with open(controls_file, 'r') as f:
            controls_set = set(line.strip() for line in f if line.strip())

    records_data = []
    for fp in hea_files:
        parsed = parse_hea_file(fp, base_dir)
        rec = parsed['record']
        parsed['is_in_controls_list'] = 'Yes' if rec in controls_set else 'No'
        records_data.append(parsed)

    df_records = pd.DataFrame(records_data)

    # Standardize & clean key fields
    if 'reason for admission' in df_records.columns:
        df_records['reason_for_admission_clean'] = df_records['reason for admission'].fillna('Unspecified / Missing')
        df_records['reason_for_admission_clean'] = df_records['reason_for_admission_clean'].replace({'n/a': 'Unspecified / Missing', '': 'Unspecified / Missing'})
    else:
        df_records['reason_for_admission_clean'] = 'Unspecified / Missing'

    # Clean sex
    if 'sex' in df_records.columns:
        df_records['sex_clean'] = df_records['sex'].astype(str).str.lower().str.strip()
        df_records['sex_clean'] = df_records['sex_clean'].replace({'male': 'Male', 'female': 'Female', 'n/a': 'Unknown/Missing', '': 'Unknown/Missing', 'none': 'Unknown/Missing'})
    else:
        df_records['sex_clean'] = 'Unknown/Missing'

    # Clean age
    if 'age' in df_records.columns:
        df_records['age_years'] = pd.to_numeric(df_records['age'], errors='coerce')
    else:
        df_records['age_years'] = None

    # Sort records cleanly
    df_records.sort_values(by=['patient_id', 'record'], inplace=True)

    # Prepare Patient Level Summary
    patient_grp = df_records.groupby('patient_id')
    patient_summary = []
    for pid, group in patient_grp:
        records_cnt = len(group)
        sex_val = group['sex_clean'].iloc[0]
        age_val = group['age_years'].iloc[0]
        diag_val = group['reason_for_admission_clean'].iloc[0]
        control_val = group['is_in_controls_list'].iloc[0]
        
        # Check additional info
        add_diag = group['additional diagnoses'].iloc[0] if 'additional diagnoses' in group.columns else 'n/a'
        smoker = group['smoker'].iloc[0] if 'smoker' in group.columns else 'n/a'
        
        patient_summary.append({
            'patient_id': pid,
            'primary_diagnosis': diag_val,
            'sex': sex_val,
            'age_years': age_val,
            'total_ecg_records': records_cnt,
            'is_control_group': control_val,
            'additional_diagnoses': add_diag,
            'smoker_status': smoker
        })
    
    df_patients = pd.DataFrame(patient_summary)

    # -----------------------------
    # BIAS CALCULATIONS & METRICS
    # -----------------------------

    # 1. Class Imbalance
    rec_class_counts = df_records['reason_for_admission_clean'].value_counts()
    pat_class_counts = df_patients['primary_diagnosis'].value_counts()

    df_class_bias = pd.DataFrame({
        'Diagnostic Category': rec_class_counts.index,
        'ECG Record Count': rec_class_counts.values,
        'ECG Record %': (rec_class_counts.values / len(df_records) * 100).round(2),
        'Patient Count': [pat_class_counts.get(cat, 0) for cat in rec_class_counts.index],
        'Patient %': [(pat_class_counts.get(cat, 0) / len(df_patients) * 100) for cat in rec_class_counts.index]
    })
    df_class_bias['Patient %'] = df_class_bias['Patient %'].round(2)

    # 2. Sex Disparity
    sex_diag_ct = pd.crosstab(df_records['reason_for_admission_clean'], df_records['sex_clean'], margins=True, margins_name='Total')
    sex_diag_pct = pd.crosstab(df_records['reason_for_admission_clean'], df_records['sex_clean'], normalize='index') * 100
    sex_diag_pct = sex_diag_pct.round(2)

    # 3. Age Stats by Diagnosis
    age_stats = df_records.groupby('reason_for_admission_clean')['age_years'].agg(
        Count='count',
        Mean='mean',
        Std='std',
        Min='min',
        Q25=lambda x: x.quantile(0.25),
        Median='median',
        Q75=lambda x: x.quantile(0.75),
        Max='max'
    ).reset_index()
    for col in ['Mean', 'Std', 'Q25', 'Median', 'Q75']:
        age_stats[col] = age_stats[col].round(1)

    # 4. Recording Frequency per Patient
    rec_freq = df_patients['total_ecg_records'].value_counts().sort_index().reset_index()
    rec_freq.columns = ['ECGs per Patient', 'Number of Patients']
    rec_freq['% of Patients'] = (rec_freq['Number of Patients'] / len(df_patients) * 100).round(2)
    rec_freq['Total ECG Records'] = rec_freq['ECGs per Patient'] * rec_freq['Number of Patients']

    # 5. Missingness Rates
    missing_rates = []
    key_fields = ['age', 'sex', 'reason for admission', 'acute infarction (localization)', 
                  'additional diagnoses', 'smoker', 'number of coronary vessels involved',
                  'hemodynamics', 'chest x-ray', 'peripheral blood pressure (syst/diast)', 
                  'echocardiography', 'medication pre admission', 'in hospital medication']
    
    for field in key_fields:
        if field in df_records.columns:
            missing_cnt = df_records[field].isna().sum() + (df_records[field].astype(str).str.lower().isin(['n/a', '', 'none', 'nan'])).sum()
            missing_pct = round((missing_cnt / len(df_records)) * 100, 2)
            
            # Control missingness
            ctrl_records = df_records[df_records['reason_for_admission_clean'] == 'Healthy control']
            ctrl_missing = ctrl_records[field].isna().sum() + (ctrl_records[field].astype(str).str.lower().isin(['n/a', '', 'none', 'nan'])).sum()
            ctrl_missing_pct = round((ctrl_missing / len(ctrl_records)) * 100, 2) if len(ctrl_records) > 0 else 0
            
            missing_rates.append({
                'Metadata Field': field,
                'Total Missing Records': missing_cnt,
                'Overall Missing %': missing_pct,
                'Control Missing Records': ctrl_missing,
                'Control Missing %': ctrl_missing_pct
            })
    df_missing = pd.DataFrame(missing_rates)

    # -----------------------------
    # EXCEL EXPORT WITH OPENPYXL
    # -----------------------------
    output_excel = os.path.join(base_dir, 'PTB_ECG_Database_Metadata.xlsx')
    
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df_records.to_excel(writer, sheet_name='Record_Level_Metadata', index=False)
        df_patients.to_excel(writer, sheet_name='Patient_Level_Summary', index=False)
        
        # Write Bias Summary Sheet manually with formatting
        wb = writer.book
        ws_bias = wb.create_sheet(title='Bias_Metrics_Summary')
        
        # Header Style
        title_font = Font(name='Calibri', size=14, bold=True, color='1F497D')
        section_font = Font(name='Calibri', size=12, bold=True, color='1F497D')
        header_font = Font(name='Calibri', size=11, bold=True, color='FFFFFF')
        header_fill = PatternFill(start_color='1F497D', end_color='1F497D', fill_type='solid')
        sub_header_fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
        border_thin = Border(left=Side(style='thin', color='D9D9D9'),
                             right=Side(style='thin', color='D9D9D9'),
                             top=Side(style='thin', color='D9D9D9'),
                             bottom=Side(style='thin', color='D9D9D9'))

        current_row = 1

        # Title
        ws_bias.cell(row=current_row, column=1, value="PTB Diagnostic ECG Database - Comprehensive Bias Metrics Summary").font = title_font
        current_row += 2

        # 1. Class Imbalance Table
        ws_bias.cell(row=current_row, column=1, value="1. Diagnostic Class Imbalance (Selection Bias)").font = section_font
        current_row += 1
        
        headers_1 = df_class_bias.columns.tolist()
        for col_idx, h in enumerate(headers_1, 1):
            cell = ws_bias.cell(row=current_row, column=col_idx, value=h)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal='center', vertical='center')
        current_row += 1
        
        for r_idx, row in df_class_bias.iterrows():
            for c_idx, val in enumerate(row, 1):
                cell = ws_bias.cell(row=current_row, column=c_idx, value=val)
                cell.border = border_thin
                if c_idx > 1:
                    cell.alignment = Alignment(horizontal='right')
            current_row += 1
        
        current_row += 2

        # 2. Sex Disparity Table
        ws_bias.cell(row=current_row, column=1, value="2. Gender Disparity across Diagnostic Groups").font = section_font
        current_row += 1
        
        ws_bias.cell(row=current_row, column=1, value="Diagnostic Category").font = header_font
        ws_bias.cell(row=current_row, column=1).fill = sub_header_fill
        
        sex_cols = sex_diag_ct.columns.tolist()
        for c_idx, sc in enumerate(sex_cols, 2):
            cell = ws_bias.cell(row=current_row, column=c_idx, value=f"Count: {sc}")
            cell.font = header_font
            cell.fill = sub_header_fill
            cell.alignment = Alignment(horizontal='center')
        
        for c_idx, sc in enumerate(sex_cols[:-1], len(sex_cols) + 2):
            cell = ws_bias.cell(row=current_row, column=c_idx, value=f"% {sc}")
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal='center')
        current_row += 1

        for cat in sex_diag_ct.index:
            ws_bias.cell(row=current_row, column=1, value=cat).border = border_thin
            col_pos = 2
            for sc in sex_cols:
                val = sex_diag_ct.loc[cat, sc]
                c = ws_bias.cell(row=current_row, column=col_pos, value=val)
                c.border = border_thin
                c.alignment = Alignment(horizontal='right')
                col_pos += 1
            
            for sc in sex_cols[:-1]:
                val_pct = sex_diag_pct.loc[cat, sc] if cat in sex_diag_pct.index else 0
                c = ws_bias.cell(row=current_row, column=col_pos, value=val_pct)
                c.border = border_thin
                c.alignment = Alignment(horizontal='right')
                col_pos += 1
            current_row += 1

        current_row += 2

        # 3. Age Stats Table
        ws_bias.cell(row=current_row, column=1, value="3. Age Distribution Statistics by Diagnostic Group (Age Bias)").font = section_font
        current_row += 1
        
        headers_3 = age_stats.columns.tolist()
        for col_idx, h in enumerate(headers_3, 1):
            cell = ws_bias.cell(row=current_row, column=col_idx, value=h)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal='center')
        current_row += 1

        for r_idx, row in age_stats.iterrows():
            for c_idx, val in enumerate(row, 1):
                cell = ws_bias.cell(row=current_row, column=c_idx, value=val)
                cell.border = border_thin
                if c_idx > 1:
                    cell.alignment = Alignment(horizontal='right')
            current_row += 1

        current_row += 2

        # 4. Repeat Recording Frequency Table
        ws_bias.cell(row=current_row, column=1, value="4. Patient Repeat Recording Frequency (Data Leakage / Sampling Bias)").font = section_font
        current_row += 1

        headers_4 = rec_freq.columns.tolist()
        for col_idx, h in enumerate(headers_4, 1):
            cell = ws_bias.cell(row=current_row, column=col_idx, value=h)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal='center')
        current_row += 1

        for r_idx, row in rec_freq.iterrows():
            for c_idx, val in enumerate(row, 1):
                cell = ws_bias.cell(row=current_row, column=c_idx, value=val)
                cell.border = border_thin
                cell.alignment = Alignment(horizontal='right') if c_idx >= 1 else Alignment(horizontal='left')
            current_row += 1

        current_row += 2

        # 5. Missingness Rates Table
        ws_bias.cell(row=current_row, column=1, value="5. Metadata Missingness & Information Bias").font = section_font
        current_row += 1

        headers_5 = df_missing.columns.tolist()
        for col_idx, h in enumerate(headers_5, 1):
            cell = ws_bias.cell(row=current_row, column=col_idx, value=h)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal='center')
        current_row += 1

        for r_idx, row in df_missing.iterrows():
            for c_idx, val in enumerate(row, 1):
                cell = ws_bias.cell(row=current_row, column=c_idx, value=val)
                cell.border = border_thin
                if c_idx > 1:
                    cell.alignment = Alignment(horizontal='right')
            current_row += 1

        # Auto-adjust column widths across sheets
        for sheet in wb.worksheets:
            for col in sheet.columns:
                max_len = 0
                col_letter = get_column_letter(col[0].column)
                for cell in col:
                    if cell.value:
                        val_str = str(cell.value)
                        if len(val_str) > max_len:
                            max_len = len(val_str)
                sheet.column_dimensions[col_letter].width = min(max(max_len + 3, 12), 50)

    print(f"Successfully generated Excel workbook at: {output_excel}")

if __name__ == '__main__':
    main()
