import pandas as pd
import os


def join_csv_files():
    base_path = "/Users/Mallory_X/Desktop/2025 Spring/DSCI550/Assignment 1"
    data_path = os.path.join(base_path, "Data")
    climate_data_path = os.path.join(data_path, "Climate Data")

    haunted_file = os.path.join(data_path, "haunted_new.csv")
    alcohol_abuse_file = os.path.join(data_path, "AlcoholAbuseDeathbyState.csv")
    lcd_file = os.path.join(climate_data_path, "LCD_Featurized.csv")
    output_file = os.path.join(data_path, "haunted_joined.csv")

    haunted_df = pd.read_csv(haunted_file)
    alcohol_abuse_df = pd.read_csv(alcohol_abuse_file)
    lcd_df = pd.read_cssv(lcd_file)

    # Extract 'MONTH' column from 'Haunted Places Date'
    haunted_df["MONTH"] = pd.to_datetime(haunted_df["Haunted Places Date"], errors='coerce').dt.month

    # Merge with AlcoholAbuseDeathbyState.csv on 'state' column
    merged_df = haunted_df.merge(alcohol_abuse_df, left_on="state", right_on="State", how="left")

    # Merge with LCD_Featurized.csv on 'STATE' and 'MONTH'
    final_df = merged_df.merge(lcd_df, left_on=["state", "MONTH"], right_on=["STATE", "MONTH"], how="left")

    # Drop redundant columns
    final_df = final_df.drop(columns=["STATE", "MONTH", "State"], errors='ignore')

    final_df.to_csv(output_file, index=False)
    print(f"File saved as {output_file}")


join_csv_files()
