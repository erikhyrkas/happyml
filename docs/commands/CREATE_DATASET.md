# CREATE DATASET

Disclaimer: happyml is constantly changing. If I see the chance for improvement, I'm going to take it. I hope that you get value out of it, 
but don't be shocked if you see changes. I'm trying to make it better, and if I find something doesn't feel right or good to me, I won't hessitate
to rewrite how it works. I hope you can appreciate the spirit of this project.

## Syntax

```happyml
create dataset <dataset name>
[with header]
[with given {label|number|text|image} <given name> [<rows>, <columns>, <channels>] at <column position>]+
[with expected {label|number|text|image} <expected name> [<rows>, <columns>, <channels>] at <column position>]*
using <file://path/>
```

## Description and Notes
Creates a dataset from a file, putting it into a binary format efficient to use for training a task. If you have multiple channels, like in the case of an image, it is populates the channels in order. So if you have a 8x8 image with 3 channels, you would specify `with given image pixels 8, 8, 3 at 0` to read the first 64 values into the first channel, the next 64 into the second channel, and the last 64 into the third channel.


`dataset name` is the name you will use to refer to the dataset in other commands. The name must be unique.

`with header` indicates that the first row of the file is a header row, and should not be interpreted as data.

`with given` indicates that the column is a given value. 
* Valid data types are:
  * `label` -- Text that labels or categorizes the record. These values will be one-hot encoded.
  * `number` -- A numeric quantity. Will be normalized and standardized as needed.
  * `text` -- Text that is not a label. Will be encoded with byte-pair encoding.
  * `image` -- A set of pixels that make up an image. Stored in efficient pixel format tensors.
* `given name` is the name you will use to refer to the given value in other commands.
* `rows` -- The number of rows in the image. If not specified, happyml will assume 1.
* `columns` -- The number of columns in the image. If not specified, happyml will assume 1.
* `channels` -- The number of channels in the image. If not specified, happyml will assume 1.
* `column position` is the column offset it the original csv. It starts with 0. If you have a 8x8 image, all 64 pixels must be in the same row of the CSV, you just point at the starting offset where the first pixel is. happyml will assume the first pixel is row 0, column 0, of the first channel and read values until it has filled the dimensions that you specified.

`with expected` indicates that the column is a given value.
* Valid data types are:
  * `label` -- Text that labels or categorizes the record. These values will be one-hot encoded.
  * `number` -- A numeric quantity. Will be normalized and standardized as needed.
  * `text` -- Text that is not a label. Will be encoded with byte-pair encoding.
  * `image` -- A set of pixels that make up an image. Stored in efficient pixel format tensors.
* `expected name` is the name you will use to refer to the expected value in other commands.
* `rows` -- The number of rows in the image. If not specified, happyml will assume 1.
* `columns` -- The number of columns in the image. If not specified, happyml will assume 1.
* `channels` -- The number of channels in the image. If not specified, happyml will assume 1.
* `column position` is the column offset it the original csv. It starts with 0. If you have a 8x8 image, all 64 pixels must be in the same row of the CSV, you just point at the starting offset where the first pixel is. happyml will assume the first pixel is row 0, column 0, of the first channel and read values until it has filled the dimensions that you specified.

`using` indicates the path to the file to read. At the moment, the file must be a CSV or tSV and must be local.

## Example

```happyml
create dataset titanic
       with header
       with expected label Survived at 1
       with given label    Pclass   at 2
       with given label    Sex      at 4
       with given number   Age      at 5
       with given number   SibSp    at 6
       with given number   Parch    at 7
       with given number   Fare     at 9
       with given label    Embarked at 11
       using file://../data/titanic/train.csv
```