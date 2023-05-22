![happyml](../../happyml.png)

# REFINE TASK (future)
[Back to the table of contents](../README.md)

Disclaimer: happyml is constantly changing. If I see the chance for improvement, I'm going to take it. I hope that you get value out of it,
but don't be shocked if you see changes. I'm trying to make it better, and if I find something doesn't feel right or good to me, I won't hessitate
to rewrite how it works. I hope you can appreciate the spirit of this project.

## Syntax

```happyml
refine task <task name>
[with label <task label>]
using dataset <dataset name>
```

## Description and Notes
**This is a future feature.**

Refines (or fine-tunes) a task using a dataset. 

`task name` is the name of the task you want to refine.

`task label` is the label of the task you want to refine. If you have multiple tasks with the same name, you can use this to specify which one you want to refine.

`dataset name` is the name of the dataset you want to use to refine the task.

The new task will be saved with the same name as the original task, but with a new label. The new label will include the name of the dataset used to refine the task.

## Example

Refines `my_tasks` and creates a new label for `my_task` called `my_dataset`:

```happyml
refine task my_task using dataset my_dataset
```



