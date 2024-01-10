import os
import pandas as pd

def get_excel_files(directory):
    excel_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            file_path = os.path.join(directory, filename)
            excel_files.append(file_path)
    return excel_files

def find_best_f1_score(df):
    best_f1_index = df['F1 Score'].idxmax()
    return df.iloc[best_f1_index]

def main():
    directory_path = './resultados_script_smart'
    excel_files = get_excel_files(directory_path)

    best_f1_scores = []

    for file_path in excel_files:
        df = pd.read_excel(file_path)
        best_result = find_best_f1_score(df)
        best_f1_scores.append(best_result['F1 Score'])

    mean_f1_score = sum(best_f1_scores) / len(best_f1_scores)
    print(f"Mean F1 Score: {mean_f1_score}")

    if mean_f1_score > 0.5:
        print("The mean F1 Score is above 50%. Possible attack detected!")
    else:
        print("The mean F1 Score is not above 50%. No attack detected.")

if __name__ == "__main__":
    main()
