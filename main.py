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
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from time import process_time

warnings.filterwarnings('ignore')

def prepareData():
    # Reads the file "dasta.csv" and returns a normalized Dataframe free of null entries
    
    df = pd.read_csv("data.csv")
    # print(pd.head())

    # Remove the last column which is empty
    if df.columns[-1].startswith("Unnamed"):
            df = df.iloc[:, :-1]  # Removes the last column
            
    # Remove unneeded collumns
    df.drop(['id'], axis = 1)

    # Remove rows containing any 0 value
    df = df[(df != 0).all(axis=1)] 
    # Need to reset indeces because it still had the original indeces after removing the 0s
    df.reset_index(drop=True, inplace=True)  
    
    # Extract target variable
    target = df.pop('diagnosis')

    # Convert target variable to 0 or 1
    target.replace("M", 1, inplace=True)
    target.replace("B", 0, inplace=True)

    # Normalize the data
    scale = StandardScaler()
    df_norm = scale.fit_transform(df)
    df_norm = pd.DataFrame(df_norm, columns=df.columns)
            
    # Reinsert "diagnosis" to the data
    df_norm['diagnosis'] = target 
    
    return df_norm   

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

def plot_heatmap(results):
    # Receives results from GridSearchCV
    # Visualisation of gridsearch performance
    
    # Create a DataFrame using results for plotting
    df_results = pd.DataFrame(results)

    # Pivot table to reshape data for heatmap
    heatmap_data = df_results.pivot(index='param_gamma', columns='param_C', values='mean_test_score')
    
    x_labels = [f"{x:.3g}" for x in heatmap_data.columns]  # Columns are 'C' values
    y_labels = [f"{y:.3g}" for y in heatmap_data.index]    # Indexes are 'gamma' values

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis", cbar_kws={'label': 'Mean Test Score'},xticklabels=x_labels,yticklabels=y_labels)
    plt.title("GridSearchCV Heatmap")
    plt.xlabel("C")
    plt.ylabel("Gamma")
    plt.show()
    
def gridSearch(SVM, c_range, gamma_range, x, y):
    # Receives a SVC() object
    # Hyperparameters optimization
    # C is cost of misclassification. Higher C leads to lower margins and overfitting
    # Gamma controls how tight/detailed the islands surrounding the dataset are
    
    # Setup parameters for the girdsearch
    parameters = {'C': c_range, 'gamma': gamma_range, 'kernel': ['rbf']}

    # 5 fold cross-vailidation
    param_optimization = GridSearchCV(estimator=SVM, param_grid=parameters, refit=True, verbose=0, cv=5)
    param_optimization.fit(x,y)
    
    # Extract the best C and gamma
    best_params = param_optimization.best_params_
    best_C = best_params['C']
    best_gamma = best_params['gamma']
    
    # print(f"Best C: {float(best_C)}, Best Gamma: {float(best_gamma)}")        
    # plot_heatmap(param_optimization.cv_results_)  
    
    
    return param_optimization   


def main():

    data = prepareData()    

    # Divide features and target variable
    y = data['diagnosis']
    x = data.drop('diagnosis', axis = 1)
    
    # Default SVM parameters
    SVM = SVC()
    
    # Perform 5-fold cross-validation
    cv_scores = cross_validate(SVM, x, y, cv=5, scoring='accuracy')

    # Do an initial gridsearch to find the optimal hyperparameters
    
    # Create a hyperparameter search range
    grid_size = 5
    C = np.logspace(0, 3, grid_size)  # values from 10^0 (1) to 10^3 (1000)
    gamma = np.logspace(-3, 1, grid_size)  # values from 10^-3 (0.001) to 10^1 (10)
    
    # Do gridsearch to get best parameters
    model = gridSearch(SVM, c_range = C, gamma_range = gamma, x = x, y = y)
    best_params = model.best_params_
    best_C = best_params['C']
    best_gamma = best_params['gamma']
            
    # Second gridsearch to further refine parameters
    
    # Create a new range using the previous best parameters
    C = np.linspace(best_C / 2, best_C * 2, grid_size)
    gamma = np.linspace(best_gamma / 2, best_gamma * 2, grid_size)
    
    # Do gridsearch and get best parameters    
    model = gridSearch(SVM, c_range = C, gamma_range = gamma, x = x, y = y) 
    best_params = model.best_params_
    best_C = best_params['C']
    best_gamma = best_params['gamma']
    
    SVM_optimized = SVC(C = best_C, gamma = best_gamma)
         
    # Perform 5-fold cross-validation with new parameters
    cv_scores_optimized = cross_validate(SVM_optimized, x, y, cv=5, scoring='accuracy')

    # Print the optimization results
    print("Default HyperParameters: C = 1.0, Gamma = ", float(1/x.columns.size) )
    print("Optimized HyperParameters: C = ", float(best_C), ", Gamma = ", float(best_gamma), "\n")
    print("Default 5-Fold Model Accuracy:", cv_scores['test_score'].mean())
    print("Optimized 5-Fold Model Accuracy:", cv_scores_optimized['test_score'].mean(), "\n")

    # print(process_time())
    
    #TODO
    # Use "data" to create 5 random subsets of data containing 50,175,300,425,550 entries. For each dataset, do a 5-fold cross-validation. 
    # At the end show the accuracy of each model.
    # Also show the TIME is took to test each model
    

if __name__ == "__main__":
    main()
