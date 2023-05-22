# CREATE TASK

Disclaimer: happyml is constantly changing. If I see the chance for improvement, I'm going to take it. I hope that you get value out of it,
but don't be shocked if you see changes. I'm trying to make it better, and if I find something doesn't feel right or good to me, I won't hessitate
to rewrite how it works. I hope you can appreciate the spirit of this project.

## Syntax

```happyml
create task {label} <task name>
[with goal [speed|accuracy|memory]]
[with test <test dataset name>]
using <dataset name>
```

## Description and Notes

Creates a task from a dataset. A task is backed by a machine learning model.

Valid Task Types:
  * `label` -- A task that predicts a label or category.
  * (more coming soon)

`task name` is the name you will use to refer to the task in other commands. The name must be unique.

`with goal` indicates the goal of the task. Valid goals are:
  * `speed` -- The task should be optimized for speed.
  * `accuracy` -- The task should be optimized for accuracy. **Default if not specified.**
  * `memory` -- The task should be optimized for memory usage.

`with test` indicates the name of the dataset to use for testing.
  * `test dataset name` is the name of the dataset to use for testing. If not specified, the task is not tested, which may lead to worse results.

`using` indicates the name of the dataset to use for training.
  * `dataset name` is the name of the dataset to use for training.


## Example

```happyml
create task label predict_survivor 
with goal accuracy 
with test titanic_test 
using titanic
```
