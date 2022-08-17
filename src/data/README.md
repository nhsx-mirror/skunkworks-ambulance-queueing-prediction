# Generating Fake Data

## Overview

This directory contains` generate_fake_data.py` and `acquisiton.py`. The purpose of `generate_fake_data.py` is to create `.parquet` files with three fake data sets.

The purpose of the fake data is to be used as an example of how the data needs to be formatted. It should be noted that the outputs of the model using the fake data are unrealistic and should not be used to test the performance of the model.

The data is generated completely randomly. 

`acquisition.py` contains functions to enrich the fake data with additional datasets. Please note that this script uses the `bank_holiday_data_1953_to_1954.csv` which lists all of the bank holidays in those years. This is purely for example purposes and if the model were to be ran on other data, this would need to be updated. This `.csv` has been created by carrying out internet research on historic dates. Note that these dates are not necessarily rigorously verified since we are creating fake data. They are only used to show the end-to-end process of this project.