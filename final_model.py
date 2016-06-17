import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# Read in dataframes
print("Reading and merging the datasets...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Merge in store info
stores = pd.read_csv("stores.csv")
train = train.merge(stores, how='left', on='Store')
test = test.merge(stores, how='left', on='Store')

# Create indexes for submission
train["Id"] = train["Store"].astype(str) + "_" + train["Dept"].astype(str) + "_" + train["Date"].astype(str)
train = train.set_index("Id")
test["Id"] = test["Store"].astype(str) + "_" + test["Dept"].astype(str) + "_" + test["Date"].astype(str)
test = test.set_index("Id")

# Also make an index by store_dept to split up the dataset
train["Index"] = train["Store"].astype(str) + "_" + train["Dept"].astype(str)
test["Index"] = test["Store"].astype(str) + "_" + test["Dept"].astype(str)

# Add column for year
train["Year"] = pd.to_datetime(train["Date"], format="%Y-%m-%d").dt.year
test["Year"] = pd.to_datetime(test["Date"], format="%Y-%m-%d").dt.year

# Add column for day
train["Day"] = pd.to_datetime(train["Date"], format="%Y-%m-%d").dt.day
test["Day"] = pd.to_datetime(test["Date"], format="%Y-%m-%d").dt.day

# Add column for days to next Christmas
train["Days to Next Christmas"] = (pd.to_datetime(train["Year"].astype(str)+"-12-31", format="%Y-%m-%d") -
                                   pd.to_datetime(train["Date"], format="%Y-%m-%d")).dt.days.astype(int)
test["Days to Next Christmas"] = (pd.to_datetime(test["Year"].astype(str) + "-12-31", format="%Y-%m-%d") -
                                   pd.to_datetime(test["Date"], format="%Y-%m-%d")).dt.days.astype(int)


# Create store_dept dictionaries
print("Splitting the datasets into subsets...")
traindict = {}
testdict = {}
for index in set(test["Index"].tolist()):
    traindict[index] = train[train["Index"]==index]
    testdict[index] = test[test["Index"]==index]

# Function for formatting each dataset
def featureprep(train, test, dropaxis, splitset):

    # Function for getting labeled dummies
    if splitset == True:
        def xdums(df):
            dums = pd.get_dummies(pd.to_datetime(df["Date"], format="%Y-%m-%d").dt.week)
            dums.columns = map(lambda x: "Week_" + str(x), dums.columns.values)
            return dums
    else:
        def xdums(df):
            dums = pd.get_dummies(df["Store"])
            dums = dums.set_index(df.index)
            dums.columns = map(lambda x: "Store_" + str(x), dums.columns.values)
            out = dums
            dums = pd.get_dummies(df["Dept"])
            dums.columns = map(lambda x: "Dept_" + str(x), dums.columns.values)
            out = out.join(dums)
            dums = pd.get_dummies(df["Type"])
            dums.columns = map(lambda x: "Type_" + str(x), dums.columns.values)
            out = out.join(dums)
            dums = pd.get_dummies(pd.to_datetime(df["Date"], format="%Y-%m-%d").dt.week)
            dums.columns = map(lambda x: "Week_" + str(x), dums.columns.values)
            out = out.join(dums)
            return out

    train_x = xdums(train).join(train[["IsHoliday", "Size", "Year", "Day", "Days to Next Christmas"]])
    test_x = xdums(test).join(test[["IsHoliday", "Size", "Year", "Day", "Days to Next Christmas"]])

    # Deal with NAs
    train_x = train_x.dropna(axis=dropaxis)
    test_x = test_x.dropna(axis=dropaxis)
    train_y = train.dropna(axis=dropaxis)["Weekly_Sales"]

    # Remove any train features that aren't in the test features
    for feature in train_x.columns.values:
        if feature not in test_x.columns.values:
            train_x = train_x.drop(feature, axis=1)

    # Remove any test features that aren't in the train features
    for feature in test_x.columns.values:
        if feature not in train_x.columns.values:
            test_x = test_x.drop(feature, axis=1)

    return train_x, train_y, test_x

#Define estimator
estimator = GradientBoostingRegressor(loss="huber")

# Function for returning estimates
def estimates(train, test, splitset):
    # Get estimates for columns that have no NAs
    train_x, train_y, test_x = featureprep(train, test, 1, splitset)
    estimator.fit(train_x, train_y)
    out = pd.DataFrame(index=test_x.index)
    out["Weekly_Sales"] = estimator.predict(test_x)
    out["Id"] = out.index

    # Create a dataframe for plotting the training feature regression
    plot = pd.DataFrame(index=train_x.index)
    plot["Weekly_Sales"] = train_y
    plot["Weekly_Predicts"] = estimator.predict(train_x)
    plot["Date"] = plot.index.str.split("_").str[-1]
    plot = plot.groupby("Date")[["Weekly_Sales", "Weekly_Predicts"]].sum()

    return out, plot

# Run the individual store-departments models
print("Beginning main model...")
out = pd.DataFrame()
plot = pd.DataFrame()
count = 0
for key in testdict.keys():
    count += 1
    try:
        ot, pt = estimates(traindict[key], testdict[key], True)
        out = pd.concat([out, ot])
        plot = pd.concat([plot, pt])
    except:
        print("No training data available for {}".format(key))
    if count%20 == 0:
        print("Modeling... {}%".format(list(testdict.keys()).index(key)/len(testdict.keys())*100))

# Run a model of all the data to fill in anything that was NA
print("Creating giant model to fill in for those pesky missing datas... Probably going to take a while.")
sout, splot = estimates(train, test, False)
sout = sout.join(out, how="left", lsuffix="_Backup")
sout["Weekly_Sales"] = sout["Weekly_Sales"].fillna(sout["Weekly_Sales_Backup"])

# Format for submission
sout["Id"] = sout["Id"].fillna(sout["Id_Backup"])
sout = sout.drop(["Weekly_Sales_Backup", "Id_Backup"], axis=1)
splot = splot.join(plot, how="left", lsuffix="_Backup")
splot["Weekly_Sales"] = splot["Weekly_Sales"].fillna(splot["Weekly_Sales_Backup"])
splot["Weekly_Predicts"] = splot["Weekly_Predicts"].fillna("Weekly_Predicts_Backup")
splot = splot.drop(["Weekly_Sales_Backup", "Weekly_Predicts_Backup"], axis=1)

sout.to_csv("kpalm_submission.csv", index=False)

# Save the plotting file for plotting later
sout["Date"] = sout.index.str.split("_").str[-1]
plot = sout.groupby("Date")[["Weekly_Sales"]].sum()
plot["Weekly_Predicts"] = plot["Weekly_Sales"]
plot = plot.drop("Weekly_Sales", axis=1)
splot = splot.append(plot)
splot = splot.reset_index().groupby("Date")[["Weekly_Sales", "Weekly_Predicts"]].sum()
splot.to_csv("plot.csv")