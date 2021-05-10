import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
data = pd.read_csv('healthcare-dataset-stroke-data.csv')
data = data.dropna()

minmax = preprocessing.MinMaxScaler()
age = data['age'].values.reshape(-1,1)
data['bmi'] = data['bmi'].apply(lambda x: 50 if x>50 else x)
bmi = data['bmi'].values.reshape(-1,1)
glucose = data['avg_glucose_level'].values.reshape(-1,1)

age = minmax.fit_transform(age)
bmi = minmax.fit_transform(bmi)
glucose = minmax.fit_transform(glucose)
data[['age', 'bmi', 'avg_glucose_level']] = [age, bmi, glucose]

data['gender'] = data['gender'].apply(lambda x: 1 if x=='Male' else 0)
data['ever_married'] = data['ever_married'].apply(lambda x: 1 if x=='yes' else 0)
data['Residence_type'] = data['Residence_type'].apply(lambda x: 1 if x=='Urban' else 0)
data = pd.get_dummies(data=data, columns=['smoking_status'])
data = pd.get_dummies(data=data, columns=['work_type'])
data = data.drop(columns='id', axis=1)

X = data.drop(columns='stroke', axis=1).values
Y = data['stroke'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1337)
sm = SMOTE(random_state=1337, k_neighbors=10)
X_train, Y_train = sm.fit_resample(X_train, Y_train)
#X_test, Y_test = sm.fit_resample(X_test, Y_test)

print('\n'+'Logistic Regression')
model = LogisticRegression(random_state=1337)
'''
C = [0.09]
solver = ['lbfgs', 'liblinear']
grid = dict(C=C, solver=solver)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1337)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='f1',error_score=0)
grid_result = grid_search.fit(X_train, Y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
'''
model.fit(X_train, Y_train)
Y_predict = model.predict(X_test)
aucroc = roc_auc_score(Y_test,Y_predict)
print("AUC score:\n" + str(aucroc))
score = f1_score(Y_test, Y_predict)
print("F1 score:\n" + str(score))
matrix = confusion_matrix(Y_test, Y_predict)
print('Confusion matrix:\n' + str(matrix))
'''
#print('\n'+'Random Forest')
model = RandomForestClassifier(random_state=1337)
model.fit(X_train, Y_train)
Y_predict = model.predict(X_test)
aucroc = roc_auc_score(Y_test,Y_predict)
print("AUC score:\n" + str(aucroc))
score = f1_score(Y_test, Y_predict)
print("F1 score:\n" + str(score))
matrix = confusion_matrix(Y_test, Y_predict)
print('Confusion matrix:\n' + str(matrix))
'''