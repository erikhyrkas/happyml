![happyml](../../happyml.png)

# PRINT
[Back to the table of contents](../README.md)

Disclaimer: happyml is constantly changing. If I see the chance for improvement, I'm going to take it. I hope that you get value out of it,
but don't be shocked if you see changes. I'm trying to make it better, and if I find something doesn't feel right or good to me, I won't hessitate
to rewrite how it works. I hope you can appreciate the spirit of this project.

## Syntax

```happyml
print {pretty|raw} <dataset name> [limit <limit number>]
```

## Description and Notes
Prints a dataset to the console.

Valid print options:
* `pretty` -- Prints the dataset in a human-readable format.
* `raw` -- Prints the dataset as tensors that are potentially normalized and standardized.

`dataset name` is the name you will use to refer to the dataset in other commands.

`limit` indicates the maximum number of rows to print. If not specified, happyml will print all rows.


## Example

```happyml
print raw titanic limit 5

print pretty titanic limit 5
```
