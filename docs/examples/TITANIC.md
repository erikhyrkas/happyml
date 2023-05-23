# Titanic
[Back to the table of contents](../README.md)

This example uses the Titanic dataset to predict whether a passenger survived or not.

## CODE

The current code is at [titanic.happyml](../../happyml_repo/scripts/titanic.happyml). 

## Data
The data is from [Kaggle](hhttps://www.kaggle.com/datasets/hesh97/titanicdataset-traincsv). It is a list of passengers on the Titanic, and whether they survived or not.

```happyml
create dataset titanic
       with header
       with expected label Survived at 1
       with given label    Pclass   at 2
       with given label    Sex      at 4
       with given number   Age      at 5
       with given number   SibSp    at 6
       with given number   Parch    at 7
       with given number   Fare     at 9
       with given label    Embarked at 11
       using file://../happyml_data/titanic/train.csv              
```


## Task
The task is to predict whether a passenger survived or not.

```happyml
create task label predict_survivor using titanic       
```

## Using the Task
```happyml
execute task predict_survivor using input ("Pclass": "2", "Sex": "female", "Age": "55", "SibSp": "0", "Parch": "0", "Fare": "16", "Embarked": "S")
```