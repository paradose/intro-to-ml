# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# pandas is used to load the data
url = "iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url,names=names)

# returns how many rows and attributes are contained with the shape property.
print(dataset.shape)

# looking at the data .head(number of entries)
print(dataset.head(20))

# getting statistics of each attribute (column)
print(dataset.describe())

# number of rows that belong to each class
print(dataset.groupby('class').size())

# box and whisker plots
# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()

# histograms
# dataset.hist()
# pyplot.show()

# scatter plot matrix
# scatter_matrix(dataset)
# pyplot.show()

# Split-out validation dataset
array = dataset.values
# numpy method of slicing [rows, columns]
X = array[:,0:4]
y = array[:,4]

# our aim is to take flower measurements and be able to predict species of those flowers, therefore, we
# need to create models based on the existing data to estimate the accuracy on unseen measurements.
print(X)
print(y)
# here 80% is used to train, evaluate and select amongst the models,
# 20% is held back for testing and as a validation dataset.
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
