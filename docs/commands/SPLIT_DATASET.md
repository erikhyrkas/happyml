![happyml](../../happyml.png)

# SPLIT DATASET (future)
[Back to the table of contents](../README.md)

Disclaimer: happyml is constantly changing. If I see the chance for improvement, I'm going to take it. I hope that you get value out of it, 
but don't be shocked if you see changes. I'm trying to make it better, and if I find something doesn't feel right or good to me, I won't hessitate
to rewrite how it works. I hope you can appreciate the spirit of this project.

## Syntax

```happyml
split dataset <dataset name> at <percent>
```

## Description and Notes
**This is a future feature.**

Splits a dataset into two datasets. The first dataset will have the specified percentage of the original dataset. The second dataset will have the remaining percentage.

`dataset name` is the name of the dataset you want to split.

`percent` is the percentage of the original dataset that will be in the first dataset. The second dataset will have the remaining percentage.

You cannot specify a percentage greater than 100% or less than 0%. The names of the two new datasets will be the same as the original dataset, but with the percentage appended to the end.

## Example

Creates two new datasets: `titanic_70` and `titanic_30`. `titanic_70` will have 70% of the original dataset. `titanic_30` will have the remaining 30%.
```happyml
split dataset titanic at 70
```
