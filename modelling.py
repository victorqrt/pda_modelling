# This script will train models on the data from https://archive.ics.uci.edu/ml/datasets/Census+Income, and
# perform a grid search selection to select the best scoring one.
# We will do that for different algorithms, eventually keeping the one tied to the model with the highest
# overall score.

import urllib, pandas, pickle, matplotlib, sklearn, numpy
from io import StringIO
from sklearn import preprocessing, linear_model, tree, model_selection
from sklearn.model_selection import GridSearchCV

matplotlib.use("agg") # We use a non-interactive backend as we are generating PNG files, reduces dependencies

print("[ ] Retrieving the dataset (may take some time)...")
train_data = pandas.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", header=None, sep=", ", engine="python")
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
# We are replacing regex matches because the test/train sets are inconsistent in the way they write the salary
clean_values = lambda df: df.replace(
        regex={
            r"^\?$": numpy.nan,
            r"<=50K.?": 0,
            r">50K.?": 1
        }).dropna()

# We need to turn our str values to numeric ones before normalizing them. Some algos from sklearn will do that for
# us, but not all of them
numerize_values = lambda df: pandas.DataFrame({col: df[col].astype('category').cat.codes for col in df}, index=df.index)

prepare_set = lambda df: numerize_values(clean_values(clean_columns(df)))

print("[ ] Preparing the data...")

train_data = prepare_set(train_data)
test_data = prepare_set(test_data)

# Normalize the data using a scaler
tmp_data = train_data.copy()
features_cols = list(col for col in train_data.columns if col != "salary_over_fiftyk") # We want to keep 0 and 1
tmp_data = tmp_data[features_cols]

# We fit our scaler on the data, and save it for future use (prediction via our API) along with the trained model
model = dict()
model["scaler"] = preprocessing.StandardScaler().fit(tmp_data.values.astype(float))

# Normalize the training set, then the testing set (without refitting the scaler)
tmp_data = model["scaler"].transform(tmp_data.values.astype(float))
train_data[features_cols] = tmp_data

tmp_data = test_data.copy()
tmp_data = tmp_data[features_cols]
tmp_data = model["scaler"].transform(tmp_data.values.astype(float))
test_data[features_cols] = tmp_data

x_train, y_train, x_test, y_test = train_data[features_cols], train_data["salary_over_fiftyk"], test_data[features_cols], test_data["salary_over_fiftyk"]

print("[ ] Training and selecting models by grid search over hyperparameters on different algorithms:")

print("    [ ] Algorithm 1: Ridge regression classifier...")
rrc = GridSearchCV(estimator=linear_model.RidgeClassifier(),
    param_grid={
        "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
        "fit_intercept": [True, False]
    },
    cv=3
)
rrc.fit(x_train, y_train)

print("    [ ] Algorithm 2: Stochastic Gradient Descent classifier...")
sgdc = GridSearchCV(estimator=linear_model.SGDClassifier(tol=0.001, max_iter=1000, n_jobs=-1),
    param_grid={
        "loss": ["hinge", "log", "modified_huber", "perceptron"],
        "penalty": ["l2", "elasticnet"]
    },
    cv=3
)
sgdc.fit(x_train, y_train)

# We keep the model with the best testing score over all algos
models = [rrc, sgdc]
model["model"] = list(e for e in models if e.score(x_test, y_test) == max(map(lambda x: x.score(x_test, y_test), models)))[0]
print("[ ] The selected model is: " + str(model["model"].estimator))

print("[ ] Saving the model to model.bin...")
pickle.dump(model, open("model.bin", "wb"))

print("[+] Done.")
