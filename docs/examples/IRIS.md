# Titanic Survivor Example
[Back to the table of contents](../README.md)

This example uses the species of an Iris flower.

## CODE

The current code is at [iris.happyml](../../happyml_repo/scripts/iris.happyml). 

## Data
The data is from [Kaggle](https://www.kaggle.com/datasets/uciml/iris). It gives a list of flowers with their species and measurements.

```happyml
create dataset iris
       with header
       with expected label Species       at 5
       with given number   SepalLengthCm at 1
       with given number   SepalWidthCm  at 2
       with given number   PetalLengthCm at 3
       with given number   PetalWidthCm  at 4
       using file://../happyml_repo/raw/iris.csv            
```


## Task
Predict which species of iris flower you are looking at.

```happyml
create task label predict_iris using iris   
```

## Using the Task
```happyml
execute task predict_iris using input (SepalLengthCm: 5.2, SepalWidthCm: 2.7, PetalLengthCm: 3.9, PetalWidthCm: 1.4)
```