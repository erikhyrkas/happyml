# DESCRIBE TASK (future)

Disclaimer: happyml is constantly changing. If I see the chance for improvement, I'm going to take it. I hope that you get value out of it,
but don't be shocked if you see changes. I'm trying to make it better, and if I find something doesn't feel right or good to me, I won't hessitate
to rewrite how it works. I hope you can appreciate the spirit of this project.

## Syntax

```happyml
describe task <task name> [with label <task label>]
```

## Description and Notes
**This is a future feature.**

Describes a task. This includes:
* Given columns
  * Name
  * Type (number, label, image, or text)
  * Shape
  * Labels (if applicable)
* Target columns
  * Name
  * Type (number, label, image, or text)
  * Shape

`task name` is the task you are describing.

`task label` is the label of the task you are describing. If you have multiple tasks with the same name, you can use this to specify which one you want to describe.

## Example

```happyml
describe task titanic
```

