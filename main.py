# TODO
# Clean up data
# remove uneeded features (cross validation)
# Assess results of tuned parameters



from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from time import process_time

warnings.filterwarnings('ignore')

def plot_relationships(df):

    # Normalize the numerical columns except 'diagnosis'
    columns_to_plot = [col for col in df.columns if col != 'diagnosis']

    # Determine the number of plots
    num_columns = len(columns_to_plot)

    # Create subplots: adjust rows and columns for better layout
    rows = 6
    cols = (num_columns + rows - 1) // rows  # Round up to fit all plots
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), sharey=True)
    axes = axes.flatten()

    # Loop through the columns and create KDE plots
    for i, category in enumerate(columns_to_plot):
        sns.kdeplot(data=df, x=category, hue='diagnosis', fill=True, ax=axes[i])
        axes[i].set_title(f'{category} by Diagnosis', fontsize=10)
        axes[i].set_xlabel(category)
        axes[i].set_ylabel('Density')

    # Hide unused subplots if columns don't fill all grid spaces
    for ax in axes[num_columns:]:
        ax.axis('off')

    # Adjust layout
    plt.tight_layout()

    plt.show()

def main():

    df = pd.read_csv("data.csv")
    # print(pd.head())

    # Remove the last column
    if df.columns[-1].startswith("Unnamed"):
            df = df.iloc[:, :-1]  # Removes the last column

    # Extract target variable
    target = df.pop('diagnosis')

    # Convert results to 0 or 1
    target.replace("M", 1, inplace=True)
    target.replace("B", 0, inplace=True)

    # Return diagnosis to the data
    df['diagnosis'] = target

    # Normalize the data
    scale = StandardScaler()
    df_norm = scale.fit_transform(df)
    df_norm = pd.DataFrame(df_norm, columns=df.columns)
    df_norm['diagnosis'] = df['diagnosis']


    # TODO clean up data
    # # Setup data to recognize 0s as NAN values
    # df.replace(0, np.nan, inplace=True)
    # # check for NAN
    # print(df.isna().sum())

    #Showing how balanced the results are (relatively balanced in this case)
    # print(pd.crosstab(target,target,normalize='all')*100)


    y = df_norm['diagnosis']
    x = df_norm.drop('diagnosis', axis = 1)
    x_true, x_test, y_true, y_test = train_test_split(x,y,test_size=0.3, random_state=42)
    SVM_classification = SVC()
    SVM_classification.fit(x_true, y_true)

    y_prediction = SVM_classification.predict(x_test)
    predictions = pd.DataFrame({'y_true': y_test, 'y_prediction': y_prediction})


    print(classification_report(y_test, y_prediction))


    # Hyperparameters optimization
    # C is cost of misclassification. Higher C leard to lower margins and overfitting
    # Gamma controls how tight/detailed the islands surrounding the dataset are

    parameters = {'C': [10, 100, 1000], 'gamma': ['scale', 0.01, 0.001], 'kernel': ['rbf']}

    # 10 fold cross-vailidation
    param_optimization = GridSearchCV(estimator=SVC(), param_grid=parameters, refit=True, verbose=1, cv=10)


    param_optimization.fit(x_true,y_true)

    print('Optimal hyperparameters:')
    print(param_optimization.best_params_)


    # print(process_time())

if __name__ == "__main__":
    main()
