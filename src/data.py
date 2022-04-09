import pandas as pd
import numpy as np
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')


def fetch_data():
    file_path = "../data/insurance.csv"
    if Path(file_path).is_file():
        df = pd.read_csv(file_path)
        def bmi_cat(row):
            if row["bmi"] <= 25:
                return "healthy"
            elif row["bmi"] <= 30 and row["bmi"] > 25:
                return "overweight"
            else:
                return "obeese"
        df["bmi_cat"] = df.apply(lambda x: bmi_cat(x), axis=1)
        return df

    else:
        print("Input filepath does not exist!!")