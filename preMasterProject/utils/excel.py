import pandas as pd
import torch
from   torch.utils.data import DataLoader


def saveAsExcelFile(data_loader,title):

    data_list = []

    for data in data_loader:
        data_list.append(data)

    df              = pd.DataFrame(data_list)
    excel_file_path = title + '.csv'
    df.to_csv(excel_file_path, index=False)

    print(f"Data saved to {excel_file_path}")
