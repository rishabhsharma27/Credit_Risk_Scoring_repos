import pandas as pd
from pathlib import Path

def load_loan_data(filename):
    """
    Loads historical loan data from the data/ directory.
    
    Args:
        filename (str): The name of the CSV file.
        
    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    # Get the directory of this current script (src/)
    current_dir = Path(__file__).resolve().parent
    
    # Go up one level to the project root, then into the data folder
    project_root = current_dir.parent
    file_path = project_root / "data" / filename
    
    try:
        # Load the CSV into a pandas DataFrame
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df):,} rows and {len(df.columns)} columns from {file_path.name}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{filename}' in the {project_root / 'data'} folder.")
        print("Please make sure your CSV file is placed inside the 'data/' directory.")
        #return 0

# Test the function if running data_loader.py directly
if __name__ == "__main__":
    # Change 'credit_score.csv' to whatever your actual CSV file is named
    df = load_loan_data("credit_score.csv")
    
    if df is not None:
        print("\n--- First 5 Rows of Data ---")
        print(df.head())