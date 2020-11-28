import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib


"""Data pre processing"""
# Files import and string-int/float conversion
training_set = pd.read_csv('The end.csv')  # Training data: 2019 ka data not included
training_set = training_set[["Current Score", "Wickets", "Balls Remaining", "Total Runs", "diff", "Home", "Toss",  "batting_team", "bowling_team"]]
testing_set = pd.read_csv("testcasei1_4.csv")  # Test data
testing_set = testing_set[["Current Score", "Wickets", "Balls Remaining", "diff", "Home", "Toss", "batting_team", "bowling_team"]]
team_names = [pd.read_csv("team_names.csv").loc[i][1] for i in range(14)]
training_set["batting_team"] = training_set["batting_team"].replace("Rising Pune Supergiant", "Rising Pune Supergiants")
training_set["bowling_team"] = training_set["bowling_team"].replace("Rising Pune Supergiant", "Rising Pune Supergiants")

for i in team_names:
    training_set["batting_team"] = training_set["batting_team"].replace(i, team_names.index(i))
    testing_set["batting_team"] = testing_set["batting_team"].replace(i, team_names.index(i))
    training_set["bowling_team"] = training_set["bowling_team"].replace(i, team_names.index(i))
    testing_set["bowling_team"] = testing_set["bowling_team"].replace(i, team_names.index(i))

# Data frame is created
X = training_set.iloc[:, [0, 1, 2, 4, 5, 6, 7, 8]].values
y = training_set.iloc[:, 3].values

# Splitting the data set into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


"""Training the model"""
# # Training
# rf = RandomForestRegressor(n_estimators=500, max_features=None)
# rf.fit(X_train, y_train)
# joblib.dump(rf, 'RandomForest2008_2018p2.pkl')

# Model loading - Comment out the top three lines and use the saved model .pkl
rf_from_joblib = joblib.load('RandomForest2008_2018p2.pkl')  # Loading the data


"""Testing the model"""
x_values = []
y_values_prediction = []
y_values_runs = []

for row in range(len(testing_set)):
    new_prediction = int(rf_from_joblib.predict(sc.transform(np.array([testing_set.iloc[row]])))[0])
    x_values.append(120 - [testing_set.iloc[row]][0][2])
    y_values_prediction.append(new_prediction)
    y_values_runs.append([testing_set.iloc[row]][0][0])
    print("Prediction score:", new_prediction, y_values_runs[row])


"""Plotting the graphs"""
total_runs_scored = [testing_set.iloc[len(testing_set)-1]][0][0]

plt.plot(x_values, y_values_runs, color="r", linestyle="--", marker=".", linewidth=2)
plt.plot(x_values, y_values_prediction, marker=".", linewidth=2)
plt.plot(x_values, [total_runs_scored]*len(x_values), color="black", linewidth=1)
# plt.plot(x_values, [sum(y_values_prediction)//len(y_values_prediction)]*len(x_values), color="black", linewidth=1)

max_dev = [max(y_values_prediction)]*len(x_values)
min_dev = [min(y_values_prediction)]*len(x_values)

plt.legend(["Current runs", "Predicted total runs"])
plt.title('Prediction of IPL match')
plt.ylabel('Runs')
plt.xlabel('Balls')

plt.plot(x_values, max_dev, alpha=0)
plt.plot(x_values, min_dev, alpha=0)
plt.xlim(xmin=0, xmax=121)
plt.ylim(ymin=0, ymax=max(y_values_runs + y_values_prediction) + 10)
plt.fill_between(x_values, max_dev, min_dev, where=(max_dev > min_dev), interpolate=True, alpha=0.25)
plt.text(0, total_runs_scored, str(total_runs_scored), horizontalalignment='right')

plt.tight_layout()
plt.show()