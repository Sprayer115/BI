import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import re
from itertools import combinations

# Laden des Datensatzes
train = pd.read_csv("./train.csv")

# Entfernen irrelevanter Spalten
train.drop(['Cabin', 'PassengerId', 'Ticket'], axis=1, inplace=True)

# Ersetzen der Geschlechtswerte: male=1, female=2
train['Sex'] = train['Sex'].str.lower().map({'male': 1, 'female': 2}).fillna(train['Sex'])

# Auffüllen fehlender Werte für Embarked und Age
train['Embarked'] = train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
train['Age'] = train['Age'].fillna(train['Age'].median())

# Erstellen neuer Features
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
train['IsAlone'] = np.where(train['FamilySize'] == 1, 1, 0)

# Extrahieren des Titels aus dem Namen
def get_title(name):
    title_search = re.search(r' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

train['Title'] = train['Name'].apply(get_title)
train['Title'] = train['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train['Title'] = train['Title'].replace('Mlle', 'Miss')
train['Title'] = train['Title'].replace('Ms', 'Miss')
train['Title'] = train['Title'].replace('Mme', 'Mrs')
train['Title'] = train['Title'].map({"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4})
train['Title'] = train['Title'].fillna(0)

# Droppen von Name
train.drop('Name', axis=1, inplace=True)

# Überprüfen auf fehlende Werte
nan_columns = train.isna().sum()
print(nan_columns)

# Liste der Features
all_features = ["Sex", "Pclass", "Fare", "Age", "FamilySize", "Title", "IsAlone"]

# Funktion zum Evaluieren der Modelle
def evaluate_model(x, y):
    # Aufteilen des Datensatzes in Trainings- und Testdaten
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Skalieren der Features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    # Hyperparameter-Tuning für Logistic Regression
    lr_params = [
        {'penalty': ['l1'], 'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'saga'], 'max_iter': [10000]},
        {'penalty': ['l2'], 'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear', 'saga'], 'max_iter': [10000]}
    ]
    lr_grid = GridSearchCV(LogisticRegression(random_state=42), lr_params, cv=5, n_jobs=-1)
    lr_grid.fit(x_train, y_train)
    lr_best = lr_grid.best_estimator_

    # Hyperparameter-Tuning für Random Forest
    rf_params = {
        'n_estimators': [100, 120, 150, 200, 300],
        'max_features': [None, 'sqrt', 'log2'],
        'max_depth': [4, 6, 8],
        'criterion': ['gini', 'entropy']
    }
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, n_jobs=-1)
    rf_grid.fit(x_train, y_train)
    rf_best = rf_grid.best_estimator_

    # Evaluieren der besten Modelle
    lr_best.fit(x_train, y_train)
    lr_y_pred = lr_best.predict(x_test)
    
    rf_best.fit(x_train, y_train)
    rf_y_pred = rf_best.predict(x_test)
    
    # Ergebnisse speichern
    lr_acc = accuracy_score(y_test, lr_y_pred)
    rf_acc = accuracy_score(y_test, rf_y_pred)
    
    return lr_acc, rf_acc, lr_grid.best_params_, rf_grid.best_params_

# Ausprobieren verschiedener Kombinationen von Features
results = []
for L in range(1, len(all_features) + 1):
    for subset in combinations(all_features, L):
        x = train[list(subset)]
        y = train["Survived"]
        lr_acc, rf_acc, lr_params, rf_params = evaluate_model(x, y)
        results.append((subset, lr_acc, rf_acc, lr_params, rf_params))

# Ergebnisse anzeigen
results_df = pd.DataFrame(results, columns=['Features', 'Logistic Regression Accuracy', 'Random Forest Accuracy', 'Best LR Params', 'Best RF Params'])
print(results_df.sort_values(by='Random Forest Accuracy', ascending=False).head(10))

# Plotten der besten Feature-Kombinationen
plt.figure(figsize=(14, 7))
sns.barplot(data=results_df.sort_values(by='Random Forest Accuracy', ascending=False).head(10), x='Features', y='Random Forest Accuracy')
plt.xticks(rotation=90)
plt.title('Top 10 Feature Combinations by Random Forest Accuracy')
plt.show()


'''
                                            Features  Logistic Regression Accuracy  Random Forest Accuracy                                     Best LR Params                                     Best RF Params
103              (Sex, Pclass, Fare, Title, IsAlone)                      0.782123                0.849162  {'C': 0.01, 'max_iter': 10000, 'penalty': 'l1'...  {'criterion': 'entropy', 'max_depth': 8, 'max_...
101           (Sex, Pclass, Fare, FamilySize, Title)                      0.765363                0.843575  {'C': 0.1, 'max_iter': 10000, 'penalty': 'l2',...  {'criterion': 'gini', 'max_depth': 6, 'max_fea...
122  (Sex, Pclass, Fare, FamilySize, Title, IsAlone)                      0.776536                0.843575  {'C': 0.1, 'max_iter': 10000, 'penalty': 'l2',...  {'criterion': 'gini', 'max_depth': 6, 'max_fea...
76                    (Sex, Fare, FamilySize, Title)                      0.770950                0.843575  {'C': 1, 'max_iter': 10000, 'penalty': 'l1', '...  {'criterion': 'entropy', 'max_depth': 8, 'max_...
96                (Fare, FamilySize, Title, IsAlone)                      0.782123                0.843575  {'C': 0.01, 'max_iter': 10000, 'penalty': 'l1'...  {'criterion': 'entropy', 'max_depth': 8, 'max_...
56                         (Fare, FamilySize, Title)                      0.782123                0.837989  {'C': 0.01, 'max_iter': 10000, 'penalty': 'l1'...  {'criterion': 'gini', 'max_depth': 6, 'max_fea...
111          (Sex, Fare, FamilySize, Title, IsAlone)                      0.776536                0.837989  {'C': 10, 'max_iter': 10000, 'penalty': 'l1', ...  {'criterion': 'entropy', 'max_depth': 8, 'max_...
65                        (Sex, Pclass, Fare, Title)                      0.770950                0.837989  {'C': 0.1, 'max_iter': 10000, 'penalty': 'l2',...  {'criterion': 'entropy', 'max_depth': 8, 'max_...
86                 (Pclass, Fare, FamilySize, Title)                      0.782123                0.832402  {'C': 0.01, 'max_iter': 10000, 'penalty': 'l1'...  {'criterion': 'entropy', 'max_depth': 6, 'max_...
20                                     (Fare, Title)                      0.782123                0.832402  {'C': 0.01, 'max_iter': 10000, 'penalty': 'l1'...  {'criterion': 'gini', 'max_depth': 8, 'max_fea...
'''