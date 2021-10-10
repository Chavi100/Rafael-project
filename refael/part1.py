import math
import pandas as pd
import  matplotlib.pyplot  as plt
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score
from sklearn.model_selection import train_test_split


def num1(data):
    print(data.groupby("class").size())
    d1=data.isnull().sum(axis=1).tolist()
    d2=[]
    for i in range(len(data)):
        dist=29-int(d1[i]/7)
        d2.append(math.sqrt(data.loc[i,"posX_{}".format(dist)]**2+data.loc[i,"posY_{}".format(dist)]**2))
    data["dist"]=d2
    for i in range(25):
        types=data[data["class"]==i+1]
        plt.hist(types["dist"],bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.show()

def num2(data,num_of_rockets=100,classes=None,time=None,max_time=30,shape=None):
    if classes==None:
        classes = [i for i in range(1, 26)]
    color=["red","yellow","green","pink","orange","blue","black"]
    count=0
    for i in range(len(data)):
        if count==num_of_rockets:
            break
        my_type=int(data.loc[i,"class"])
        if my_type in classes:
            my_x = []
            my_y = []
            rocket_time=0
            for j in range(max_time):
                x = data.loc[i, "posX_{}".format(j)]
                y = data.loc[i, "posZ_{}".format(j)]
                if str(x) == "nan":
                    if time != None:
                        if j == time:
                            rocket_time += 1
                            break
                        break
                    break
                if time != None:
                    if j==time:
                        print(rocket_time)
                        break
                my_x.append(x)
                my_y.append(y)
                rocket_time=j
            print(rocket_time)
            if time==None:
                rocket_time=None
            if rocket_time==time:
                count+=1
                plt.plot(my_x, my_y,color=color[(my_type-1)%7])
    plt.show()






data=pd.read_csv("train.csv")
# data=data.drop("targetName",axis=1)
# num2(data,1)
# num2(data,50,[1,2,3,4,5,6])
# num2(data,50,[1,6],15)
num2(data,classes=[1,4,7,10])

def filter_data(data,class1,class2):
    condition1 = data['class'] == class1
    condition2 = data['class'] == class2
    filtered_table = data[condition1 | condition2]
    return filtered_table



def kineticEnergy(M, V):
    # Stores the Kinetic Energy
    KineticEnergy = 0.5 * M * V * V

    return KineticEnergy


# Function to calculate Potential Energy
def potentialEnergy(M, H):
    # Stores the Potential Energy
    PotentialEnergy = M * 10 * H

    return PotentialEnergy



def energy_graph(data):
    for k in range(len(data)):
        energy_map = []
        my_type = int(data.iloc[k]["class"])
        for j in range(30):
            h = data.iloc[k]["posZ_{}".format(j)]
            x = data.iloc[k]["velX_{}".format(j)]
            y = data.iloc[k]["velY_{}".format(j)]
            z = data.iloc[k]["velZ_{}".format(j)]
            if str(h) == "nan":
                break
            energy = potentialEnergy(1,h) + kineticEnergy(1,math.sqrt(x * x + y * y + z * z))
            energy_map.append(energy)
        if my_type == 5:
            plt.plot(energy_map, color="red")
        else:
            plt.plot(energy_map, color="blue")
    plt.show()

def peak_graph(data):
    vel_cols = [col for col in data.columns if 'velZ' in col]
    vel_values = data[vel_cols]
    vel_values = vel_values.loc[((vel_values > 0).any(axis=1)) & ((vel_values < 0).any(axis=1))]
    print(vel_values)
    index=np.array(vel_values.index)
    for k in index:
        x_map = []
        y_map=[]
        my_type = int(data.loc[k,"class"])
        for j in range(30):
            x = data.loc[k,"posX_{}".format(j)]
            y = data.loc[k,"posZ_{}".format(j)]
            if str(x) == "nan":
                break
            x_map.append(x)
            y_map.append(y)
        if my_type == 5:
            plt.plot(x_map,y_map, color="red")
        else:
            plt.plot(x_map,y_map, color="blue")
    plt.show()

# data = pd.read_csv('train.csv')
# condition1 = data['class'] == 1
# condition2 = data['class'] == 4
# condition3 = data['class'] == 7
# condition4 = data['class'] == 10
# filtered_table = data[condition1 | condition2|condition3|condition4]
# #energy_graph(filtered_table)
#
#
#
# from sklearn.ensemble import RandomForestClassifier
# filtered_table=filtered_table.fillna(0)
# filtered_table=filtered_table.drop("targetName", axis=1)
# X_train, X_test, y_train, y_test = train_test_split(filtered_table, filtered_table['class'], test_size=0.2,random_state=0)
# clf = RandomForestClassifier(n_estimators=100,max_depth=10, random_state=0)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print(confusion_matrix(y_test,y_pred))
# print(y_pred)
# print(accuracy_score(y_test, y_pred))
# print(f1_score(y_test,y_pred,average="weighted"))
#


# m=filter_data(data,5,6)
# peak_graph(m)