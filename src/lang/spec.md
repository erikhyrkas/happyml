# Spec v2

So, I've been pondering the language more and more now that there's a working interpreter, and it's simply a matter
of typing the mappings between commands and actual code that runs. 

The initial language spec wasn't to my liking for a few reasons, namely because while it was declarative in nature, I 
could see an issue with people not really knowing how to use it. The purpose of a declarative language is to let
the user state what their goal is and for the language to then accomplish it. And while the v1 spec does that,
it didn't seem useful, or at least it seemed like it would take too much understanding of the underlying ideas.

So, here's v2. I reserve the right to just change my mind again, but for the moment, this idea seems slightly better.

# Comments

# Current Commands:
General syntax:
```happyml
# Comments are indicated by a # at the beginning of a line

exit

help [{dataset|task|future}]
```
Future commands:
```happyml
set <property> <value>
```

Dataset commands:
```happyml
create dataset <dataset name>
[with header]
[with given {label|number|text|image} <given name> [<rows>, <columns>, <channels>] at <column position>]+
[with expected {label|number|text|image} <expected name> [<rows>, <columns>, <channels>] at <column position>]*
using <file://path/>

print {pretty|raw} <dataset name> [limit <limit number>]
```
Future dataset commands:
```happyml
describe dataset <dataset name>

list datasets [starting with <start string>]

copy dataset <original dataset name> to <new dataset name>

delete dataset <dataset name>

move dataset <original dataset name> to <new dataset name>
```

Task commands:
```happyml
create task {label} <task name>
[with goal [speed|accuracy|memory]]
[with test <test dataset name>]
using <dataset name>
```
Future task commands:
```happyml
describe task <task name> [with label <task label>]

list tasks [starting with <start string>]

execute task <task name>
[with label <task label>]
using dataset <dataset name>

execute task <task name>
[with label <task label>]
using input ("key": "value", "key": "value", ...)

refine task <task name>
[with label <task label>]
using dataset <dataset name>

copy task <original task name> [with label <original task label>] to <new task name> [with label <new task label>]

delete task <task name> [with label <task label>]

move task <original task name> [with label <original task label>] to <new task name> [with label <new task label>]
```


# Keywords

You can use these keywords to setup syntax highlighting in your editor. CLion has 4 groups of keywords, so I'm going
to split them into four groups.

Add *.happyml to the file type pattern and then add the keywords to the appropriate group. Here's what it looks like:

![](C:\Users\erikh\CLionProjects\happyml\happyml_file_type_pattern.png)

Hit ctrl-alt-s while in a happyml file to bring up the settings or navigate to the happyml file type manually, then
add the keywords:

![](C:\Users\erikh\CLionProjects\happyml\happyml_file_type_config.png)

Current keywords are:
### Control Keywords
```happyml
copy
create
delete
describe
execute
exit
help
list
move
print
refine
set
```
### Types
```happyml
accuracy
dataset
datasets
future
image
input
label
memory
number
speed
task
tasks
text
```
### Operational Keywords
```happyml
at
expected
given
goal
header
limit
pretty
raw
starting
test
to
using
with
```
### Path indicators
```happyml
file://
http://
https://
```

# Basics

There are 4 help menus. If you type "help" on its own, it will show you the top level menu with basic commands,
and if you type with a submenu name, it will show you details from that submenu. 
```
  help [dataset|task|future]
```

You are done with your happyml session and want to exit.
```
  exit
```

Comments:
```
  # This is a comment. I am to remind us later why we wrote it this way.
```

For the CLI, commands are all contained on a single line, but if you need to continue the line you can use the `\` character.
When describing syntax options below, I don't include the `\` because I'm just trying to show everything as 
neatly as possible. Here's an example of having a line continue:
```
  create task label my_task_name \
  using my_dataset
```

# Tasks

I decided that creating "models" may be too math-y. And not really focused on the end user. I think the goal is to
allow the users to define tasks they want to perform and support those tasks. Yes, under the cover, these are models
but maybe even slightly more than that, because they may automate some of the logic needed to use the model for that
purpose. In the original v1 spec, I even mentioned that I really wasn't a fan of having users create models, but at
the time, I hadn't thought of another approach. Now, I have.

Generate the task (which is more or less training a model and creating a default label) using a dataset:
```
  create task <task type> <task name> 
  [with goal <speed|accuracy|memory>]
  using <dataset name>
```

Brainstorming some task types below, but the names and such could all be refined. There could be so many more. This
is not a definitive list, and it's going to change. I'm just going off the top of my head of tasks I might want to 
try my hand at, but this list could get really long. I'd prioritize them based on what functionality already exists
then what I find most interesting and then what I think would be most useful to others. Why? Because there's a good
chance that I'm the only one that uses this, and so it makes sense to build it for my interests first.

My observation is that my entire initial list involves supervised learning methods (you teach the model by showing it
a large set of data with each row in the format of "given input and expected output" pair, and then later just give
the trained model the input, and it spits out the output.) There are many great unsupervised learning tasks and are
interesting, but they get off the beaten path of what I've been building so far. Maybe in the future, we can circle
back for them.

Possible task types:

| task type | description                                                                        | priority  | Why                         |
|-----------|------------------------------------------------------------------------------------|-----------|-----------------------------|
| label     | given an image or text, apply a label from a list                                  | immediate | Uses existing functionality |
| estimate  | given data, estimate one or more values                                            | immediate | Uses existing functionality |
| generate  | given text and/or an image generate text and/or an image                           | high      | Personal interest           |
| summarize | Paraphrase/summarize text                                                          | moderate  | Personally useful           |
| win       | given game state, find best move (chess, go, and others?)                          | low       | Personally fun              |
| perfect   | given image/text that is imperfect, perfect it. (image filter is a simple example) | low       | Personally fun              |
| forecast  | given past data points, predict future data points                                 | low       | Common use case             |
| recommend | given historical data, what are recommended things                                 | low       | Common use case             | 
| match     | given a primary record and X other records, return match probabilities             | low       | Common use case             |
| rank      | given viewer metadata plus X records, return record rankings                       | low       | Common use case             |

I didn't get very granular with the tasks, except "win", which I imagine using a similar process as
label, with a monte carlo algorithm on top. I mean, a lot of tasks could be seen as subtasks. For example, you
might want to "score" data/documents in a specific way. You could just look at this applying labels and looking 
at the probability of a label being true to act as a score. One could argue that "win" shouldn't be an option
and that the caller should just use label first and then implement their own monte carlo algorithm, but I
think that there aren't enough people who understand how to do this and yet there are a large group of people who
would benefit from being able to apply it to the game of their choice. 

I think match and rank are basically the same thing with different engineering around it. You send in two records 
and a score comes out, repeat for remaining records. Return either the records ordered by score for ranking or 
return the records plus score. If I get to the point of implementing these, maybe I'll find a better or more 
interesting way of solving them. I could theoretically have the models take in more than 2 things at a time
and then use binary cross entropy vs categorical cross entropy depending on if it is match or rank. None of this
is suitable for a large number of entries. If you were building a search engine (something that I have actually 
helped with), ranking millions of documents is more complex. However, for the simple case of you have maybe 100
items, and you want to order them or pick the things that are most similar to your criteria, then this would
possibly be handy. 

Random thought experiment: If I wanted to only use happyml to create a search engine with minimal external
code, I'd probably create different happyml tasks for categorizing items (whatever we are searching), categorizing 
search text (with the goal of filtering down results by best category matches compared to items and the search text), 
ranking the categories that the user or entire user-base was most likely to be interested in (using past user 
behavior, but clearly this has a bootstrap issue where there will be no user behavior if your search engine is 
new, but I can't solve all of your imaginary problems), and then a final task to rank items. I would use a 
combination of keyword searches, item category matches with the search text categories and the user's and
user-base's category preferences, and maybe an item popularity factor (how often do people view a given item that 
has matching search terms) to find reasonable top results that I could then do final ranking related to 
that user. Maybe. The hardest part of this is bootstrapping: getting enough data to start and train. You could 
start off treating all categories as having equal weight and rank evenly, then slowly let user-base and 
user-specific click-through feed into the process. Keyword search, search text classification, and item 
classification might be enough to get you reasonable initial results to then refine later with more weighting 
around the types of classifications and items that a user might be more likely to click on. There's probably also 
the need to evaluate the use cases of whether it's important to show user new items or new categories to evaluate 
whether they might click it. Clearly this would be a lot of work, and not be easy. Building an industrial strength 
search engine that performs well in both elapsed time and quality of matches for the specific user is challenging, 
and I'm not confident that my off-the-cuff ramblings would work. And so much of it would depend on the quality and 
granularity of your categorization. We're talking about huge amounts of manual labor to build training datasets if 
you can't find a way to utilize something you have. And honestly, you'd probably want multiple granularities of 
categorization so that poor searches that don't categorize well fall into a parent category. You need a way to
find an initial pool of results quickly with your keywords and category matching, so you spend the bulk of your time
sorting the top X results that they user will actually look at. I'm willing to bet that people want the answer
to their query to be in the top 3 results, and they want it there a milliseconds. So you can't afford to check
every document against the search text as specifically as you'd like.


Recommend is something that would generally use one of the popular recommendation approaches, but we could probably 
also solve this with a neural network just because that's what happyml does. This problem is super common and a well 
traveled path, but if I solved it, I don't know if I would want to leave the neural network path for it. I'm not 
going to rebuild xgboost. I'd rather re-use as much as I can of what I built, even if the solution isn't as
efficient.

Complete is something that's also another time series problem solver. This has been done a billion times. If
I do it, I'll still use a neural network, even though that's not optimal solution. Why? Because this library is
(mostly) about making neural networks. (Yeah, I know that BPE is technically a model, but I'm not exposing BPE
to the end users directly.) I know that neural networks are a hammer and not everything is a nail, but I also
only have so much time to implement things and creating something new that integrates with what I've built so
far makes me tired just thinking about. Especially when I'm not even that excited about solving this exact 
problem. Everybody has solved this problem before hundreds of times. I'm pretty sure you could use xgboost
to forecast a time series.


Refine the underlying model from a checkpoint (label) using a dataset.
```
  refine task <task name> 
  [with label [label]] 
  using dataset <dataset name>
```

List the existing tasks and checkpoints. 
```
  list tasks [<starting with x>]
```

Execute a task. I'll start off with doing so against a dataset, since I have the needed code immediately, but 
supporting input directly will enable other more real-time possibilities. The interpreter has been updated to 
handle lines ending with a backslash (\) character by removing the backslash and keeping the newline in the text. 
This enables the interpreter to process a CSV row with newlines. If a line ends with a backslash, the interpreter 
reads in another line before attempting to parse the combined text as a complete command. Be sure to use quotes
around csv text columns.
```
  execute task <task name> 
  [with label <label>] 
  using dataset <dataset>
  
  # Future
  execute task <task name>
  [with label <label>]
  using input <csv encoded row>    
```

Copy, move, and delete.
```
  # Future
  copy <task name> [<label>] to [<task name>] [<label>]
  
  # Future
  delete <task name> [<label>]

  # Future
  move <task name> [<label>] to [<task name>] [<label>]
```

## Create Dataset

Allows us to create a dataset. This will inform models what input and output should look like and
what the valid range of output is. We also need a way as part of this to split an original dataset
into a training set and a testing set. 

While the C++ implementation allows you to utilize delimited data directly and even do so completely 
in-memory, for the scripting language, we'll use this to import in delimited data into a binary
format that has a training and testing set and can easily be shuffled without being fully in-memory.

This gives us the most robust, reusable, and reliable form for creating and managing datasets.

Right now, datasets and tasks are backed by simple files, but eventually, it would be nice to support backing 
by a database or cloud storage. 

It's also important to note that I'm not including file utilities like "union", "copy", or "delete" because 
those operations get into managing resources in a way you could do outside happyml easily enough. I 
considered supporting union as part of making a dataset, but the problem because validating that the 
union was right. I'm storing the datasets in a binary format, and if things were unioned wrong, it 
would be hard to see without making even more utilities. Instead, I decided that the caller was 
responsible for making sure the incoming file was good before it was turned it into a dataset.

Note: I'm supporting "file://" initially. Eventually I'd like to support http and maybe https, or even a 
sql query against a database. Each option comes with a certain amount of work when it comes to ensuring 
a cross-platform solution. Right now, making it the caller's responsibility to make a good file is
the easiest option to support and shouldn't be terribly onerous. It will require more disk space, but I
don't imagine that being the biggest issue, even for large datasets. If you can do those operations to
prepare the data because of disk space, happyml isn't going to be able to do it for you.

Also, for this initial release, I'm only supporting CSVs with header rows. Strings will be trimmed and it
will assume the delimiter is a comma. I know this is more restrictive than the C++ api, but I'd rather keep
the syntax simple for now. Maybe in the future there can be a "with format" criteria that helps configure
how the data is loaded.

```
  create dataset <name>
  [with expected label at <column> ]*
  [with expected text at <column> ]*
  [with expected number(<rows>, <columns>, <channels>) at <column> ]*
  [with expected image(<rows>, <columns>, <channels>) at <column> ]*
  [with given <label|number|text|image> [(<rows>, <columns>, <channels>)] at <column> ]*   
  using <file://path/>
```

By default, it's assumed that the data is in one row with one channel, but you can optionally specify the dimensions of the 
data. For example, an RGB image would have 3 channels: one for red, one for green, and one for blue. That image would also
have rows equal to the picture's height and columns equal to the pictures width. By having multiple columns, you are also
hinting to happyml that this should be a convolutional neural network. A chess engine would use numbers rather than pixels,
but it works in a similar fashion as the image, in which it might have 30 or more channels, each with 8 rows and 8 columns.

Types:

| Type   | Description                                                                                                                                                                                                                                                                              |
|--------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| label  | As a programmer, you can think of a label as an enum. In the file, it is text or a number that represents a category or label, different than "text" in that the label is an indicator of what type of thing we are looking at, the model doesn't treat this as english, but as a number |
| number | A numeric value that the happyml has the option of standardizing, normalizing, and regularizing.                                                                                                                                                                                         |
| text   | text that should be byte-pair encoded and that the model will attempt to understand                                                                                                                                                                                                      |
| image  | pixels that make up the specified rows, columns and channels.                                                                                                                                                                                                                            |



List existing datasets:
```
  list datasets [<starting with x>]
```

Copy, move, and delete.
```
  # Future
  copy <dataset name> to [<dataset name>]
  
  # Future
  delete <dataset name>
 
  # Future
  move <dataset name> to [<dataset name>] [<label>]
```
