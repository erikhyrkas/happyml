# DELETE TASK (future)

Disclaimer: happyml is constantly changing. If I see the chance for improvement, I'm going to take it. I hope that you get value out of it,
but don't be shocked if you see changes. I'm trying to make it better, and if I find something doesn't feel right or good to me, I won't hessitate
to rewrite how it works. I hope you can appreciate the spirit of this project.

## Syntax

```happyml
delete task <task name> [with label <task label>]
```

## Description and Notes
**This is a future feature.**

Deletes a task.

`task name` is the name of the task you want to delete.

`task label` is the label of the task you want to delete. If you have multiple tasks with the same name, you can use this to specify which one you want to delete.

WARNING: If you don't specify a label, it will delete all tasks with the given name, even if there are multiple labels.

## Example

```happyml
delete task my_task
```



