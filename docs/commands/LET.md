![happyml](../../happyml.png)

# LET (future)
[Back to the table of contents](../README.md)

Disclaimer: happyml is constantly changing. If I see the chance for improvement, I'm going to take it. I hope that you get value out of it,
but don't be shocked if you see changes. I'm trying to make it better, and if I find something doesn't feel right or good to me, I won't hessitate
to rewrite how it works. I hope you can appreciate the spirit of this project.

## Syntax

```happyml
let <variable> <value>
```

## Description and Notes
**This is a future feature.**

Assigns a local value to a local variable




## Example

```happyml
let my_file_path file://my_file.txt

create dataset my_dataset
       with expected label favorite_color at 0
       with given label    school   at 1
       with given number   age      at 2
       using %my_file_path%
```
