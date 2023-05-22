![happyml](../../happyml.png)

# DESCRIBE DATASET (future)
[Back to the table of contents](../README.md)

Disclaimer: happyml is constantly changing. If I see the chance for improvement, I'm going to take it. I hope that you get value out of it,
but don't be shocked if you see changes. I'm trying to make it better, and if I find something doesn't feel right or good to me, I won't hessitate
to rewrite how it works. I hope you can appreciate the spirit of this project.

## Syntax

```happyml
describe dataset <dataset name>
```

## Description and Notes
**This is a future feature.**

Describes a dataset. This includes:
* The number of rows
* The number of columns
* Columns:
  * Name
  * Type (number, label, image, or text)
  * Statistics (min, max, mean, standard deviation, etc.)
  * Labels (if applicable)
  * Shape

`dataset name` is the name of the dataset you are describing.

## Example

```happyml
describe dataset titanic
```

