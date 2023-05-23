![happyml](../happyml.png)

# Types
[Back to the table of contents](README.md)

The data types that happyml supports might feel different than other languages, but they are designed for what matters to
machine learning. If you are looking for a 'boolean' type, you won't find it here. Instead, you will find a 'label' type,
which is a text value that represents a category. If you are looking for a 'string' type, you won't find it here. Instead,
you will find a 'text' type, which is a string that is byte-pair encoded. If you are looking for a 'float' type, you won't
find it here. Instead, you will find a 'number' type, which is a numeric value that the happyml has the option of standardizing,
normalizing, and regularizing. Image might seem like the biggest odd ball, because it is a type that is not found in most
languages, but it exists in happyml because we have the opportunity to optimize the way that images are stored and processed.

| Type   | Description                                                                                                                                                                                                                                                                              |
|--------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| label  | As a programmer, you can think of a label as an enum. In the file, it is text or a number that represents a category or label, different than "text" in that the label is an indicator of what type of thing we are looking at, the model doesn't treat this as english, but as a number |
| number | A numeric value that the happyml has the option of standardizing, normalizing, and regularizing.                                                                                                                                                                                         |
| text   | text that should be byte-pair encoded and that the model will attempt to understand                                                                                                                                                                                                      |
| image  | pixels that make up the specified rows, columns and channels.                                                                                                                                                                                                                            |


