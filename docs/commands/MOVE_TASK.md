![happyml](../../happyml.png)

# MOVE TASK (future)
[Back to the table of contents](../README.md)

Disclaimer: happyml is constantly changing. If I see the chance for improvement, I'm going to take it. I hope that you get value out of it,
but don't be shocked if you see changes. I'm trying to make it better, and if I find something doesn't feel right or good to me, I won't hessitate
to rewrite how it works. I hope you can appreciate the spirit of this project.

## Syntax

```happyml
move task <original task name> [with label <original task label>] to <new task name> [with label <new task label>]
```

## Description and Notes
**This is a future feature.**

Renames a task or label.

`original task name` is the name of the task you want to move.

`original task label` is the label of the task you want to move. If not specified, it assumes all labels. If you have multiple tasks with the same name, you can use this to specify which one you want to move.

`new task name` is the name of the new task you want to create.

`new task label` is the label of the new task you want to create. Only used if you specified an original task label.

If you don't specify a label, it will move all labels of the task. You cannot move the default task label if there are other task labels. You must first delete the other task labels. 


## Example

```happyml
copy task my_task to my_task_copy
```



