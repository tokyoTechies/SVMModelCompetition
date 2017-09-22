from flask import Flask,render_template,request
from sklearn import svm,datasets,metrics
from sklearn.model_selection import train_test_split
import numpy as np
import datetime
import matplotlib.pyplot as plt
import sqlite3

app = Flask(__name__)

@app.route('/')
def index():
    """Render Index Page, and supply list of feature names so 
       client can choose which features to use in SVM model"""

    return render_template('test.html',
            features=list(datasets.load_iris()['feature_names']))

@app.route('/handle_data', methods=['POST'])
def handle_data():
    """Render Leader-Board Page can only access this
       page through posting data from index form"""

    #Create Model based on form information
    best_model = svm.SVC(kernel=request.form['kernel'],
            gamma=float(request.form['gamma']),
    C=float(request.form['c']),degree=int(request.form['poly']))
    #Load features based on form feature selection
    X = datasets.load_iris().data[:,[int(request.form['feature1']),
        int(request.form['feature2'])]]
    #Load targe variables
    Y = datasets.load_iris().target
    #get accuracy and train/test sets
    accuracy,x_sets,y_sets = get_accuracy(best_model,X,Y)
    #get parameters for visualizing the decision boundries for model
    xs,ys,classification = get_classification_boundry_params(best_model,X,Y)
    #get image name of graph plotted
    image_name = save_plot(xs,ys,classification,x_sets,y_sets,
            int(request.form['feature1']),int(request.form['feature2']))
    data = append_table(accuracy,image_name)
    return render_template('leaderboard.html',data=data) 


def get_accuracy(best_model,X,Y):
    """Params:
            best_model - SVM model
            X          - all features
            Y          - all targets
       Return:
            Tuple of (accuracy of model, [training features,
            testing features],[training target, testing target])"""

    #split training and testing randomly with testing size being 40%
    train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.4)
    #train model
    best_model.fit(train_x,train_y)
    return (metrics.accuracy_score(test_y,best_model.predict(test_x))
            ,[train_x,test_x],[train_y,test_y])

def get_classification_boundry_params(best_model,X,Y):
    """Params:
            best_model - SVM model
            X          - all features
            Y          - all targets
       Return:
            Tuple of meshplot coordinates and the respective
            classification based on model"""
    #get boundries for meshgrid with padding
    X_min,X_max = X[:,0].min(),X[:,0].max()
    padding = (X_max - X_min) / 10
    X_max,X_min = X_max + padding, X_min - padding
    Y_min,Y_max = X[:,1].min(),X[:,1].max() 
    padding = (Y_max - Y_min) / 10
    Y_max, Y_min = Y_max + padding, Y_min - padding

    #get meshgrid
    xs,ys = np.meshgrid(np.arange(X_min,X_max,0.02),
            np.arange(Y_min,Y_max,0.02))
    #predict classifcation boundries based on model
    classification = best_model.predict(np.c_[xs.ravel(),ys.ravel()])
    classification = classification.reshape(xs.shape)
    return (xs,ys,classification)

def save_plot(xs,ys,classification,x_sets,y_sets,f1,f2):
    """Params:
            xs             - meshgrid x vales
            ys             - meshgird y vales
            classification - classification for respective (x,y) value
            x_sets         - set of training and testing features
            y_sets         - set of training and testing target
            f1             - index of feature 1 
            f2             - index of feature 2
       Return:
            image name of saved visualization"""

    #plot the boundries
    plt.pcolormesh(xs,ys,classification)

    for marker,x,y,test in zip(['.','x'],x_sets,y_sets,['',' Test']) :
        #seperate marker types and labels for training and 
        #testing data-points
        for target,label,color in zip(range(3),
            ['I. setosa','I. versicolor','I. virginica'],['y','r','b']) :
            #plot each flower with differnt color
            indices = np.argwhere(y == target)
            plt.scatter(x[indices,0],x[indices,1],
                label=label+test,c=color,alpha=0.5,marker=marker)
            
    #plot axies with respect to chosen features 
    plt.xlabel(datasets.load_iris().feature_names[f1])
    plt.ylabel(datasets.load_iris().feature_names[f2])
    plt.legend()

    #save plot as png
    fig = plt.gcf()
    plt.draw()
    #image name is timestamp
    image_name = format('{:%Y%m%d%H%M%S}'.format(datetime.datetime.now()))+'.png'
    fig.savefig('static/'+image_name)
    plt.close()
    return image_name

def append_table(accuracy,image_name):
    """Params:
            accuracy   - accuracy of model 
            image_name - image name for visualization of model
       Return:
            a list of dictionaries for all data-points in
            database"""

    #create database connection
    conn = sqlite3.connect('./databases/accuracy_student.db')
    #craete cursor for accessing the database
    c = conn.cursor()
    #schema:
    #(name text,accuracy real, image text)

    #create sql command
    command = 'INSERT INTO accuracy_student VALUES (\'' +\
            request.form['student'] +'\','+str(accuracy)+\
            ',\''+image_name+'\')'

    #execute command
    c.execute(command)
    #commit change
    conn.commit()

    #execute sql command to get ordered table
    table = c.execute('select * from accuracy_student order by accuracy desc');
    data = []
    #save data in form of list of dictionary 
    [data.append({'name':name,'accuracy':accuracy,'image':image})
            for (name,accuracy,image) in table]
    #terminate connection to database
    conn.close()
    return data
