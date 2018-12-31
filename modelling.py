# This script will train models on the data from https://archive.ics.uci.edu/ml/datasets/Census+Income, and
# perform a grid search selection to select the best scoring one.
# We will do that for different algorithms, eventually keeping the one tied to the model with the highest
# overall score.

import urllib, pandas, pickle, matplotlib, sklearn, numpy
from io import StringIO

matplotlib.use("agg") # We use a non-interactive backend as we are generating PNG files, reduces dependencies

print("[ ] Retrieving the dataset...")
#train_data = pandas.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", header=None, sep=", ", engine="python")
test_data = pandas.read_csv( # The file has an ugly first line we need to get rid of
    StringIO(
        urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test").read().decode("utf-8")[21:]
    ),
    header=None,
    sep=", ",
    engine="python"
)

# Let's select our features and clean NAs

def clean_columns(df):
    df.rename(index=str, inplace=True, columns={
            0: "age",
            1: "workclass",
            2: "fnlwgt",
            4: "education_years",
            5: "marital_status",
            6: "occupation",
            7: "relationship",
            8: "race",
            9: "gender",
            12: "hours_per_week",
            13: "native_country",
            14: "salary_over_fiftyk"
    })

    return df[list(col for col in df.columns if not isinstance(col, int))] # As the columns we did not rename are not to be kept

# If a row contains one "?" value, drop it. Also let's replace the salary <=50k and >50k by 0 or 1
clean_values = lambda df: df.replace(
        {
            "?": numpy.nan,
            "<=50K.": 0,
            ">50K.": 1

        }).dropna()

print("[ ] Cleaning the data...")
#train_data = clean_values(clean_columns(train_data))
test_data = clean_values(clean_columns(test_data))

print(test_data)
