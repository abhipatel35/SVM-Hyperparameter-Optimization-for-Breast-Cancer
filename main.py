from sklearn.datasets import load_breast_cancer  # for breast cancer dataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV  # for ML model which will used for Hyper-parameters tuning
from sklearn.model_selection import train_test_split  # To train the data for cleaning and testing part

# load the dataset fot load_breast_cancer dataset library
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# fit the model first without using GridSearchCV to see how it performs
model = SVC()
print(model.fit(x_train, y_train))
predictions = model.predict(x_test)
print(classification_report(y_test, predictions))  # classification_report without using the gridsearchCV

# Define parameter range that we will use for GridSearchCV
# Create dictionary parm-grid for various parameter for svm can be the C value gamma
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear']  # I take linear kernel so it will not take lots of time.
}

# make grid
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3, n_jobs=-1, cv=2)  # the model we are using is SVC
# let's fit this grid
print(grid.fit(x_train, y_train))  # this will take sometime to completely run for greatest fitting
# get the list of the best parameter after tuning
print(grid.best_params_)
grid_predictions = grid.predict(x_test)
print(classification_report(y_test, grid_predictions))
