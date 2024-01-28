import os
import pandas as pd

def get_excel_files(directory):
    excel_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            file_path = os.path.join(directory, filename)
            excel_files.append(file_path)
    return excel_files



def get_metrics(df):
    TN = df['TN'].sum()
    FP = df['FP'].sum()
    FN = df['FN'].sum()
    TP = df['TP'].sum()
    return TN, FP, FN, TP

def main():
    directory_path = './resultados_script_dumb'
    excel_files = get_excel_files(directory_path)

    all_metrics = []

    for file_path in excel_files:
        df = pd.read_excel(file_path)
        TN, FP, FN, TP = get_metrics(df)
        all_metrics.append({'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP})

    print(all_metrics)

    total_TN = sum(metric['TN'] for metric in all_metrics)
    total_FP = sum(metric['FP'] for metric in all_metrics)
    total_FN = sum(metric['FN'] for metric in all_metrics)
    total_TP = sum(metric['TP'] for metric in all_metrics)

    total_metrics= total_TN+ total_FP+total_FN+total_TP
    #print(total_metrics)
    pred_final = (total_metrics)/len(excel_files)
    #print(pred_final)
    # Check if more than half of the algorithms indicate a True Positive
    if pred_final >total_metrics/2:
        print("More than half of the algorithms indicate a True Positive. Possible attack detected!")
    else:
        print("No attack detected.")

if __name__ == "__main__":
    main()
