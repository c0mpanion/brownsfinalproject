import pandas as pd
import time

def main():
    start_time = time.time()
    data_df = pd.read_csv('NYPD_Motor_Vehicle_Collisions.csv')
    print("Reading csv file took ", time.time() - start_time, "to run");
    print(data_df.columns)
main()
