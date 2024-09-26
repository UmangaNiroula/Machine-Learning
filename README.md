# Understanding Student Flexibility in Online Learning using Machine Learning Algorithms .

This project explores how students adapt to online learning environments using exploratory data analysis (EDA) and machine learning. A dataset containing demographic and technological attributes is analyzed to predict student flexibility levels. Several classification models are evaluated, with the RandomForestClassifier showing the highest accuracy. The findings provide insights that can help educational institutions optimize online learning strategies and support student engagement.

## Models Used

##### RandomForestClassifier
##### KNeighborsClassifier
##### SVC 
##### LogisticRegression 
##### XGBClassifier 

## Images
<img width="448" alt="Screenshot 2024-09-26 at 8 32 56 AM" src="https://github.com/user-attachments/assets/6fba34f4-abf0-45be-ae17-bff301f53f02">

<img width="448" alt="Screenshot 2024-09-26 at 8 33 04 AM" src="https://github.com/user-attachments/assets/0ca8d161-aeca-458c-9e7f-26bfb82b7ed7">

<img width="448" alt="Screenshot 2024-09-26 at 8 33 09 AM" src="https://github.com/user-attachments/assets/80974c08-f2a8-470e-82ab-ea79488ceee0">

<img width="448" alt="Screenshot 2024-09-26 at 8 33 19 AM" src="https://github.com/user-attachments/assets/2727d428-e86d-4bf5-a394-10cba6995151">

## Code
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


df = pd.read_csv('students_adaptability_level_online_education.csv')

df.Age = df.Age.astype(str)
df.info()

df.isnull().sum()

df.describe()

scaler = OrdinalEncoder()
names = df.columns
d = scaler.fit_transform(df)

scaled_df = pd.DataFrame(d, columns=names)
scaled_df.head()

oversample = SMOTE()
features, labels=  oversample.fit_resample(scaled_df.drop(["Flexibility Level"],axis=1),scaled_df["Flexibility Level"])

X_train, X_test, y_train, y_test=train_test_split(features, labels,test_size=0.33,random_state=42)

models = {
    "Random Forest Classifier": RandomForestClassifier(),
    "K-Nearest Neighbors Classifier": KNeighborsClassifier(),
    "Support Vector Classifier (SVC)": SVC(),
    "Logistic Regression": LogisticRegression()
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, zero_division=1, output_dict=True)

    print(f"**{model_name}:**")
    print("-" * 30)
    print(f"- **Accuracy:** {report['accuracy']:.0%}")
    for class_label, metrics in report.items():
        if class_label.isdigit():
            print(f"- **Class {int(class_label)}:**")
            print(f"  - **Precision:** {metrics['precision']:.0%}")
            print(f"  - **Recall:** {metrics['recall']:.0%}")
            print(f"  - **F1-score:** {metrics['f1-score']:.0%}")
    print("-" * 30)
    print()
models = [RandomForestClassifier(), KNeighborsClassifier(), SVC(), LogisticRegression()]
scores = dict()

for m in models:
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)

    print(f'model: {str(m)}')
    print(classification_report(y_test,y_pred, zero_division=1))
    print('-'*30, '\n')

    model=RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test,y_pred, zero_division=1))

cm = confusion_matrix(model.predict(X_test),y_test)
disp = ConfusionMatrixDisplay(cm, display_labels=["High","Low","Moderate"])
disp.plot()
plt.title("Confusion Matrix")
plt.show()

correlation_matrix = scaled_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
