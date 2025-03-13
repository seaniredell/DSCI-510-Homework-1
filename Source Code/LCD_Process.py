import os
import re
import numpy as np
import pandas as pd

def merge_csv_files(input_folder: str, output_folder: str, output_filename: str = "LCD_RawMaster.csv"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, input_folder)
    output_path = os.path.join(script_dir, output_folder, output_filename)

    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Get all CSV files in the input folder
    all_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.csv')]

    if not all_files:
        print("No CSV files found in the directory.")
        return

    # Read and concatenate CSV files
    dataframes = [pd.read_csv(file, low_memory=False) for file in all_files]
    master_df = pd.concat(dataframes, ignore_index=True)

    # Save to the output folder
    master_df.to_csv(output_path, index=False)
    print(f"Master CSV file saved to {output_path}")


def add_state_column(csv_path, excel_path, output_dir):
    df_csv = pd.read_csv(csv_path, dtype={'STATION': str}, low_memory=False)
    df_excel = pd.read_excel(excel_path, dtype={'CODE': str})
    output_path = os.path.join(output_dir, "LCD_RawMaster_State.csv")

    # Ensure 'CODE' column is exactly 5 characters long for correct matching
    df_excel['CODE'] = df_excel['CODE'].astype(str).str.zfill(5)
    # Extract last 5 digits from STATION column
    df_csv['STATION_CODE'] = df_csv['STATION'].str[-5:]

    # Create a mapping from code to state
    code_to_state = dict(zip(df_excel['CODE'], df_excel['STATE']))
    # Map STATE based on STATION_CODE
    df_csv['STATE'] = df_csv['STATION_CODE'].map(code_to_state)
    # Reorder columns to insert STATE before STATION
    cols = df_csv.columns.tolist()
    cols.insert(cols.index('STATION'), cols.pop(cols.index('STATE')))
    df_csv = df_csv[cols]
    # Drop the temporary STATION_CODE column
    df_csv.drop(columns=['STATION_CODE'], inplace=True)

    # Save the updated CSV
    df_csv.to_csv(output_path, index=False)
    print(f"Updated file saved at: {output_path}")


def filter_columns(csv_path, output_dir, output_filename="LCD_Filtered.csv"):
    """
    Drops all columns except for STATE, DATE, DailyAverageWindSpeed, MonthlyMeanTemperature,
    MonthlyTotalLiquidPrecipitation.
    """
    os.makedirs(output_dir, exist_ok=True)
    required_columns = ["STATE", "DATE", "DailyAverageWindSpeed", "MonthlyMeanTemperature",
                        "MonthlyTotalLiquidPrecipitation"]

    df = pd.read_csv(csv_path, low_memory=False)
    # Check if all required columns exist
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns in dataset: {missing_columns}")
    # Keep only the required columns in order
    df_filtered = df[required_columns]
    # Drop rows where all key column is empty (NaN)
    df_filtered = df_filtered.dropna(
        subset=["DailyAverageWindSpeed", "MonthlyMeanTemperature", "MonthlyTotalLiquidPrecipitation"],
        how="all"
    )

    output_path = os.path.join(output_dir, output_filename)
    df_filtered.to_csv(output_path, index=False)
    print(f"Filtered dataset saved at: {output_path}")


def process_lcd(csv_path, output_dir, output_filename="LCD_Processed.csv"):
    """
    Processes climate data by:
    1. Adding 'MonthlyAverageWindSpeed' by averaging 'DailyAverageWindSpeed' per state-month.
    2. Dropping rows where 'MonthlyMeanTemperature' is empty.
    3. Formatting 'DATE' to show only the month (1-12).
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path, low_memory=False)

    # Extract only the month from the "DATE" column
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.month  # Converts to month number (1-12)
    # Remove non-digit characters from "MonthlyTotalLiquidPrecipitation"
    df["MonthlyTotalLiquidPrecipitation"] = df["MonthlyTotalLiquidPrecipitation"].astype(str).apply(
        lambda x: re.sub(r"[^0-9.]", "", x) if pd.notna(x) else x
    )

    # Replace empty strings with np.nan and convert to float
    df["MonthlyTotalLiquidPrecipitation"] = (
        df["MonthlyTotalLiquidPrecipitation"]
        .replace("", np.nan)
        .astype(float)
    )

    # Calculate Monthly Average Wind Speed for each state-month
    monthly_avg_wind = df.groupby(["STATE", "DATE"])["DailyAverageWindSpeed"].mean().round(1).reset_index()
    monthly_avg_wind.rename(columns={"DailyAverageWindSpeed": "MonthlyAverageWindSpeed"}, inplace=True)
    # Merge the new calculated column back into the original DataFrame
    df = df.merge(monthly_avg_wind, on=["STATE", "DATE"], how="left")
    df.drop(columns=["DailyAverageWindSpeed"], inplace=True, errors="ignore")
    # Drop rows where all key column is empty (NaN)
    df = df.dropna(
        subset=[ "MonthlyMeanTemperature", "MonthlyTotalLiquidPrecipitation"],
        how="all"
    )

    # Define the output file path & Save the processed data
    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path, index=False)
    print(f"Processed dataset saved at: {output_path}")


#Step 1 merge groups of state LCD files into one master file
merge_csv_files("../../Data/Climate Data/LCD_Raw", "../../Data/Climate Data")
#Step 2 Add State full name
add_state_column("../../Data/Climate Data/LCD_RawMaster.csv",
                 "../../Data/Climate Data/StateStationCode.xlsx",
                 "../../Data/Climate Data")
#Step 3 Filter relevant metrics
filter_columns("../../Data/Climate Data/LCD_RawMaster_State.csv",
                        "../../Data/Climate Data")
#Step 4 Keep only monthly data, missing data for Washington.D.C. will be manually added
process_lcd("../../Data/Climate Data/LCD_Filtered.csv",
                     "../../Data/Climate Data")
