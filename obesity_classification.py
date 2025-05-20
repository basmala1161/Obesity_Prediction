#import part
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier

#import data
df_train = pd.read_csv('data/train_dataset.csv')
df_test = pd.read_csv('data/test_dataset.csv')

#----------------show data at first----------------------
print("Train Head:")
print(df_train.head())
print("-----")
print("Test head:")
print(df_test.head())
print("-----")
print("Train Info:")
print(df_train.info())
print("-----")
print("Test Info:")
print(df_test.info())
print("-----")
print("Train describe:")
print(df_train.describe())
print("-----")
print("Test describe:")
print(df_test.describe())



#-------------Data Preprocessing---------------

df_train['BMI'] = df_train['Weight'] / (df_train['Height'] / 100)**2
df_test['BMI'] = df_test['Weight'] / (df_test['Height'] / 100)**2

#----------Handle Null Values And Dublicates-----------
df_train.isnull().sum().sort_values(ascending=False)
df_test.isnull().sum().sort_values(ascending=False)

mean1=df_train['FCVC'].mean()
mode1=df_train['CALC'].mode()[0]
df_train['FCVC'].fillna(mean1,inplace=True)
df_train['CALC'].fillna(mode1,inplace=True)
df_train.drop_duplicates(inplace=True)
df_train.isnull().sum().sort_values(ascending=False)

mean2=df_test['FCVC'].mean()
mode2=df_test['CALC'].mode()[0]
df_test['FCVC'].fillna(mean2,inplace=True)
df_test['CALC'].fillna(mode2,inplace=True)
df_test.drop_duplicates(inplace=True)
df_test.isnull().sum().sort_values(ascending=False)


#-------------Encoding Categorical Data--------------
categorical_one_hot  = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
categorical_label = ['CAEC', 'CALC', 'MTRANS','NObeyesdad']
#One-hot encode only the categorical columns
df_train = pd.get_dummies(df_train, columns=categorical_one_hot, drop_first=True)
df_test = pd.get_dummies(df_test, columns=categorical_one_hot, drop_first=True)
for col in categorical_label:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.transform(df_test[col])
print(df_train.head())
print("---------------------------------------------------------------------------------------------")
print(df_test.head())




# ------------------Handle Outliers using IQR method-------------------
def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.clip(df[column], lower_bound, upper_bound)
    return df

# Numerical columns to handle outliers (excluding those already one-hot encoded or label encoded)
numerical_cols = ['Age', 'Height', 'Weight', 'FCVC','NCP','CH2O','FAF','TUE']

# Multiple boxplots side by side
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_train[numerical_cols])
# plt.xticks(rotation=45)
plt.title('Boxplots of Numerical Features')
plt.show()

for col in numerical_cols:
    df_train = handle_outliers(df_train, col)
    df_test = handle_outliers(df_test, col)

print(df_train.head())
print("---------------------------------------------------------------------------------------------")
print(df_test.head())


#---------------Feature Scaling (Standardization)-----------------
cols_to_scale = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

df_train_scaled = df_train.copy()
df_test_scaled = df_test.copy()

scaler = StandardScaler()

df_train_scaled[cols_to_scale] = scaler.fit_transform(df_train[cols_to_scale])
df_test_scaled[cols_to_scale] = scaler.transform(df_test[cols_to_scale])

df_train[cols_to_scale] = df_train_scaled[cols_to_scale]
df_test[cols_to_scale] = df_test_scaled[cols_to_scale]

print(df_train.head())
print(df_test.head())


#-----------------Feature selection------------------
X = df_train.drop('NObeyesdad', axis=1)
y = df_train['NObeyesdad']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

importances = model.feature_importances_
feat_importances = pd.Series(importances, index=X.columns)

feat_importances.sort_values(ascending=False).plot(kind='bar', figsize=(10,5))
plt.title("Feature Importances")
plt.show()

selected_features = ['Weight', 'Height', 'FCVC', 'Age', 'Gender_Male', 'TUE',
                     'NCP', 'CH2O', 'FAF', 'family_history_with_overweight_yes',
                     'CAEC', 'CALC']

X_selected = df_train[selected_features]
y = df_train['NObeyesdad']

model = RandomForestClassifier(random_state=42)
model.fit(X_selected, y)

df_train = df_train[selected_features + ['NObeyesdad']]
df_test = df_test[selected_features + ['NObeyesdad']]

print(df_train.head())



#--------------Split Features And Target----------
X_train = df_train.drop('NObeyesdad', axis=1)
y_train = df_train['NObeyesdad']
X_test = df_test.drop('NObeyesdad', axis=1)
y_test = df_test['NObeyesdad']


#-----------Function To Evaluate The Model-----------
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


    #------------Logistic Regression Model---------
    print("\n--- Logistic Regression ---")
log_reg = LogisticRegression(max_iter=3000)
evaluate_model(log_reg, X_train, y_train, X_test, y_test)

#-----------Random Forest Model-------------
print("\n--- Random Forest ---")
rf = RandomForestClassifier(random_state=42)
evaluate_model(rf, X_train, y_train, X_test, y_test)

#---------Decision Tree Model----------
print("\n--- Decision Tree ---")
dt = DecisionTreeClassifier(random_state=42)
evaluate_model(dt, X_train, y_train, X_test, y_test)

#--------Neural Network Model-----------
print("\n--- Neural Network ---")
nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
evaluate_model(nn, X_train, y_train, X_test, y_test)

#-------Support Vector Machine Model---------
print("\n--- Support Vector Machine (SVM) ---")
svm = SVC()
evaluate_model(svm, X_train, y_train, X_test, y_test)

#---------K-Nearest Neighbors (KNN) Model----------
print("\n--- K-Nearest Neighbors (KNN) ---")
knn = KNeighborsClassifier()
evaluate_model(knn, X_train, y_train, X_test, y_test)

#-------Gradient Boosting Classifier Model-------
print("\n--- Gradient Boosting Classifier ---")
gbc = GradientBoostingClassifier(random_state=42)
evaluate_model(gbc, X_train, y_train, X_test, y_test)

#------Hyper Parameter Logistic Regression------
HyperParameter_LogReg = {
    'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['saga']
}

logreg = LogisticRegression(max_iter=3000)
grid_logreg = GridSearchCV(logreg, HyperParameter_LogReg, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_logreg.fit(X_train, y_train)

print("\n--- Logistic Regression ---")
print("The Best: ", grid_logreg.best_params_)
evaluate_model(grid_logreg.best_estimator_, X_train, y_train, X_test, y_test)

#--------Hyper Parameter KNN-----------
param_grid_knn = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 15],
    'weights': ['uniform', 'distance'],
     'p': [1, 2],
    'metric': ['minkowski', 'euclidean', 'chebyshev'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree']
}

knn = KNeighborsClassifier()
grid_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_knn.fit(X_train, y_train)

print("\n--- KNN ---")
print("The Best: ", grid_knn.best_params_)
evaluate_model(grid_knn.best_estimator_, X_train, y_train, X_test, y_test)


#--------------ManualPractical KNN  -----------

knn1 = KNeighborsClassifier(n_neighbors=3, weights='uniform')
evaluate_model(knn1, X_train, y_train, X_test, y_test)

knn2 = KNeighborsClassifier(n_neighbors=9, weights='distance')
evaluate_model(knn2, X_train, y_train, X_test, y_test)

#-----------ManualPractical Logistic Regression ------------
print("\ 1:Manual practical: ")
logreg1 = LogisticRegression(C=0.01, penalty='l2', solver='saga', max_iter=1000)
evaluate_model(logreg1, X_train, y_train, X_test, y_test)

print(" 2Manual Practical ::")
logreg2 = LogisticRegression(C=10, penalty='l1', solver='saga', max_iter=1000)
evaluate_model(logreg2, X_train, y_train, X_test, y_test)


#----------Stacking for Models----------
base_models = [
    ('logreg', grid_logreg.best_estimator_),
    ('knn', grid_knn.best_estimator_),
    ('rf', rf),
    ('dt', dt),
    ('nn', nn),
    ('svm', svm),
    ('gbc', gbc)
]

# استخدام Logistic Regression كـ meta-model
meta_model = LogisticRegression(max_iter=1000)

# بناء Stacking Classifier
stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,             # cross-validation
    n_jobs=-1,        # استخدام كل الأنوية المتاحة
    passthrough=False # لو True بيضيف الـ features الأصلية للمدخلات
)

# تدريب وتقييم الموديل
print("\n--- Stacking Model ---")
evaluate_model(stack_model, X_train, y_train, X_test, y_test)


#*****************************************
# Calculate BMI from Height and Weight
df_train['BMI'] = df_train['Weight'] / ((df_train['Height']/100) ** 2)

# Define numerical columns including the newly calculated BMI
numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI']

# Verify each column exists before plotting
existing_cols = [col for col in numerical_cols if col in df_train.columns]
missing_cols = [col for col in numerical_cols if col not in df_train.columns]
if missing_cols:
    print(f"Warning: These columns were not found in the DataFrame: {missing_cols}")

# Create plots for existing columns
plt.figure(figsize=(15, 12))  # Made taller to accommodate more plots
for idx, col in enumerate(existing_cols):
    plt.subplot(3, 3, idx+1)
    sns.histplot(df_train[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()



#----------------- Correlation Heatmap------------
plt.figure(figsize=(12,8))
corr_matrix = df_train.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()



#***************************************
# Boxplots: Feature vs Obesity Level
categorical_features = ['Gender_Male', 'family_history_with_overweight_yes', 'CAEC', 'CALC']

for feature in categorical_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='NObeyesdad', y=feature, data=df_train)
    plt.title(f'{feature} vs Obesity Level')
    plt.xticks(rotation=45)
    plt.show()

# -----------------Countplot of Target------------
plt.figure(figsize=(8,5))
sns.countplot(x='NObeyesdad', data=df_train, order=df_train['NObeyesdad'].value_counts().index)
plt.title('Distribution of Obesity Levels')
plt.xticks(rotation=45)
plt.show()

#********************************
feat_importances.sort_values(ascending=False).plot(kind='barh', figsize=(10,7))
plt.title("Feature Importances - Random Forest")
plt.show()



# -----------------Average BMI by Obesity Level
mean_bmi_by_obesity = df_train.groupby('NObeyesdad')['BMI'].mean()
print(mean_bmi_by_obesity)

# preparing gui


import joblib


joblib.dump(stack_model, "stack_model.pkl")      
joblib.dump(le, "label_encoder.pkl")              