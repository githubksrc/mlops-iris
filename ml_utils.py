from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier #Murthy

# define a Gaussain NB classifier
clf = GaussianNB()
knnmodel=KNeighborsClassifier(n_neighbors=3) #Murthy

# define the class encodings and reverse encodings
classes = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica"}
r_classes = {y: x for x, y in classes.items()}

# function to train and load the model during startup
def load_model():
    # load the dataset from the official sklearn datasets
    X, y = datasets.load_iris(return_X_y=True)
    # do the test-train split and train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2)

    clf.fit(X_train, y_train)
    knnmodel.fit(X_train1, y_train1) #Murthy

    # calculate the print the accuracy score
    acc = accuracy_score(y_test, clf.predict(X_test)) 
    print(f"Model trained with accuracy (GaussianNB()): {round(acc, 3)}")

    knnacc = accuracy_score(y_test1, knnmodel.predict(X_test1))  #Murthy
    print(f"Model trained with accuracy (KNeighborsClassifier()): {round(knnacc, 3)}")  #Murthy


# function to predict the flower using the model
def predict(query_data):
    x = list(query_data.dict().values())
    
    #prediction = clf.predict([x])[0]   #Commented by Murthy
    
    prediction = knnmodel.predict([x])[0] #Murthy
    
    print(f"Model prediction: {classes[prediction]}")
    
    return classes[prediction]

# function to retrain the model as part of the feedback loop
def retrain(data):
    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.flower_class] for d in data]
    # fit the classifier again based on the new data obtained
    clf.fit(X, y)  
    knnmodel.fit(X,y) #Murthy