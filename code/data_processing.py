import pandas as pd

# columns information
COLNAMES = [
    "Contract date", "Latitude", "Longitude", "Altitude", "1st class region id",
    "2nd class region id", "Road id", "Apartment id", "Floor",
    "Angle: The direction the house was built",
    "Area: Exclusive area for the traded house", "The limit number of car of parking lot",
    "The total area of parking lot", "Whether an external vehicle can enter the parking lot",
    "The average management fee of the apartment",
    "The number of households in the apartment", "Average age of residents", "Builder id",
    "Construction completion date", "Built year",
    "The number of schools near the apartment", "The number of bus stations near the apartment",
    "The number of subway stations near the apartment", "The actual transaction price"]

USING_INDEX = list(range(23))
SECOND_LAYER_INDEX = USING_INDEX
DATE_INDEX = [0, 18, 19]
LABEL_COLUMN = COLNAMES[-1]

# recover missing data
def feature_managing(data):
    from datetime import datetime
    data[COLNAMES[0]] = pd.to_datetime(data[COLNAMES[0]].fillna("1980-01-01")) - datetime(1980,1,1)
    data[COLNAMES[18]] = pd.to_datetime(data[COLNAMES[18]].fillna("1946-01-01")) - datetime(1946,1,1)
    data[COLNAMES[19]] = pd.to_datetime(data[COLNAMES[19]].fillna(1970.)) - datetime(1970,1,1)
    for i in DATE_INDEX:
        data[COLNAMES[i]] = data[COLNAMES[i]].dt.days
    for i in USING_INDEX:
        data[COLNAMES[i]] = data[COLNAMES[i]].fillna(0.)

# get numpy train data
def get_train_data(filename = 'data_train.csv'):
    data = pd.read_csv(filename, names=COLNAMES)
    feature_managing(data)
    return data.values

# get numpy test data
def get_test_data(filename = 'data_test.csv'):
    data = pd.read_csv(filename, names=COLNAMES[:-1])
    feature_managing(data)
    return data.values
