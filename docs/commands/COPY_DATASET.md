# COPY DATASET (future)

Disclaimer: happyml is constantly changing. If I see the chance for improvement, I'm going to take it. I hope that you get value out of it, 
but don't be shocked if you see changes. I'm trying to make it better, and if I find something doesn't feel right or good to me, I won't hessitate
to rewrite how it works. I hope you can appreciate the spirit of this project.

## Syntax

```happyml
copy dataset <source dataset name> to <destination dataset name>
```

## Description and Notes
**This is a future feature.**

Copies a dataset. If `destination dataset` doesn't exist, the new dataset will have the same data as the original dataset, but will have a different name.
However, `source dataset` exists, the two will be merged.

`source dataset name` is the name of the dataset you want to copy.

`destination dataset name` is the name of the destination dataset. If the destination exists, the two will be merged.

The destination must be compatible with the source. Duplicates are removed. 

## Example

```happyml
copy dataset titanic to titanic_copy
```
