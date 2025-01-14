""" Read file drop collum and put file back """
import pandas as pd

def drop_column(input_file: str, output_file: str, column_to_drop: str) -> None:

    try:
        # Read the input file
        data = pd.read_csv(input_file)
        
        # Check if the column exists
        if column_to_drop not in data.columns:
            print(f"Column '{column_to_drop}' does not exist in the file.")
            return
        
        # Drop the specified column
        data = data.drop(columns=[column_to_drop])
        print(f"Column '{column_to_drop}' has been dropped.")
        
        # Save the modified data to the output file
        data.to_csv(output_file, index=False)
        print(f"Updated file saved to: {output_file}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    input_path = "Data/test_class_financial_distress_smote.csv"        # Replace with your input file path
    output_path = "Data/test_class_financial_distress_smote.csv"       # Replace with your desired output file path
    input2_path = "Data/train_class_financial_distress_smote.csv"       # Replace with your input file path
    output2_path = "Data/train_class_financial_distress_smote.csv"      # Replace with your desired output file path
    input1_path = "Data/class_financial_distress.csv"       # Replace with your input file path
    output1_path = "Data/class_financial_distress.csv"      # Replace with your desired output file path
    column_name = "x79" # Replace with the column name to drop
    
    drop_column(input_path, output_path, column_name)
    drop_column(input1_path, output1_path, column_name)
    drop_column(input2_path, output2_path, column_name)
