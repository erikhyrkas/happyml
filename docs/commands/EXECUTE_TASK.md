# EXECUTE TASK (future)

Disclaimer: happyml is constantly changing. If I see the chance for improvement, I'm going to take it. I hope that you get value out of it,
but don't be shocked if you see changes. I'm trying to make it better, and if I find something doesn't feel right or good to me, I won't hessitate
to rewrite how it works. I hope you can appreciate the spirit of this project.

## Syntax

```happyml
execute task <task name>
[with label <task label>]
using dataset <dataset name>
```

```happyml
execute task <task name>
[with label <task label>]
using input ("key": "value", "key": "value", ...)
```

## Description and Notes
**This is a future feature.**

Executes a task on a dataset. This will run the task on the dataset and return the results.

`task name` is the name of the task you want to execute.

`task label` is the label of the task you want to execute. If you have multiple tasks with the same name, you can use this to specify which one you want to execute.

`dataset name` is the name of the dataset you want to execute the task on.

`input` is a list of key-value pairs as input for the task. This is an alternative to using a dataset. If you use this, you don't need to specify a dataset.

## Example

```happyml
execute task my_task using dataset my_dataset
```

```happyml
execute task my_task with label my_label using dataset my_dataset
```

```happyml
execute task my_task using input ("color": "red", "age": "5")
```


