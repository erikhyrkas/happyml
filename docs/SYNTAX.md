![happyml](../happyml.png)

# Basic Syntax 
[Back to the table of contents](README.md)

happyml is case-insensitive. I generally prefer lowercase, but you can use whatever you want.

## Comments

Use the hash symbol to write comments.

```happyml
# This is a comment
```

## Strings

Strings are surrounded by double quotes, and can be escaped with the backslash.

```happyml
"This is a string with \"quotes\""
```

## Numbers

Numbers are just numbers.

```happyml
123.3
```

## Whitespace

happyml ignores whitespace, so you can use it to make your code more readable.

```happyml
print  pretty       titanic 
  limit 5
```
Is the same as:
```happyml
print pretty titanic limit 5
```

## Line continuation

In scripts, commands can be continued on the next line without any special indicator, but if you are using the CLI, you must use the backslash to indicate that the command continues on the next line.

```happyml
> print pretty titanic \
limit 5
```

This is because the CLI immediately runs commands when it sees a newline, so you need to indicate that the command continues on the next line.

