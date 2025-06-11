"""
Script to parse GLEIF 4GB csv file and extract only required columns for ML training.
Saves the cleansed and jurisdiction data (IN hardcoded) in the data folder.
"""

import os
from pathlib import Path

import pandas as pd
from lenu.data import DataRepo
from lenu.data.lei import (COL_ELF, COL_JURISDICTION, COL_LEGALNAME,
                           get_legal_jurisdiction, load_lei_cdf_data)

data_dir = Path("data")
data_repo = DataRepo(data_dir)
data_loader = data_repo.from_data_dir(data_dir)


def load_lei_cdf_data(url, usecols=None):
    return pd.read_csv(
        url,
        compression="zip",
        low_memory=False,
        dtype=str,
        # the following will prevent pandas from converting strings like 'NA' to NaN.
        na_values=[""],
        keep_default_na=False,
        usecols=usecols,
    )

lei_data = load_lei_cdf_data(
            url="data\\sample_test_data.csv",
            usecols=[
                "LEI",
                COL_LEGALNAME,
                COL_JURISDICTION,
                COL_ELF,
                "Entity.LegalAddress.Region",
            ],
        ).assign(Jurisdiction=lambda d: d.apply(get_legal_jurisdiction, axis=1))

jurisdiction_data = lei_data[lei_data["Jurisdiction"] == "IN"]

lei_data.to_csv(data_dir, "cleansed_data.csv")
jurisdiction_data.to_csv(os.path.join(data_dir, "jurisdiction_data.csv"))