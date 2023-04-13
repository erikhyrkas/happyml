# Spec v2

So, I've been pondering the language more and more now that there's a working interpreter, and it's simply a matter
of typing the mappings between commands and actual code that runs. 

The initial language spec wasn't to my liking for a few reasons, namely because while it was declarative in nature, I 
could see an issue with people not really knowing how to use it. The purpose of a declarative language is to let
the user state what their goal is and for the language to then accomplish it. And while the v1 spec does that,
it didn't seem useful, or at least it seemed like it would take too much understanding of the underlying ideas.

So, here's v2. I reserve the right to just change my mind again, but for the moment, this idea seems slightly better.

# Comments

# Keywords

This will probably change, but after writing up the sample commands that I think I want with the syntax 
that I think I want, here are the keywords I'm currently planning:

* at
* config
* copy
* create
* dataset
* datasets
* delete
* execute
* expected
* exit
* given
* help
* input
* label
* list
* move
* pixel
* refine
* scalar
* task
* tasks
* through
* to
* using
* value
* with

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

| task type | description                                                  | priority  | Why                         |
|-----------|--------------------------------------------------------------|-----------|-----------------------------|
| label     | given an image or text, apply a label from a list            | immediate | Uses existing functionality |
| estimate  | given data, estimate one or more values                      | immediate | Uses existing functionality |
| generate  | given text and/or an image generate text and/or an image     | high      | Personal interest           |
| summarize | Paraphrase/summarize text                                    | moderate  | Personally useful           |
| win       | given game state, find best move (chess, go, and others?)    | low       | Personally fun              |
| complete  | given past data points, predict future data points           | low       | Common use case             |
| recommend | given historical data, what are recommended things           | low       | Common use case             | 
| match     | given a primary record and X return match probabilities      | low       | Common use case             |
| rank      | given viewer metadata plus X records, return record rankings | low       | Common use case             |

I didn't get very granular with the tasks, except "win", which I imagine using a similar process as
label, with a monte carlo algorithm on top. I mean, a lot of tasks could be seen as subtasks. For example, you
might want to "score" data/documents in a specific way. You could just look at this applying labels and looking 
at the probability of a label being true to act as a score. One could argue that "win" shouldn't be an option
and that the caller should just use label first and then implement their own monte carlo algorithm, but I
think that there aren't enough people who understand how to do this and yet there are a large group of people who
would benefit from being able to apply it to the game of their choice. 


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

```
  create dataset <name>
  [with expected <label|number|text|image> at <column> [through <column>] ]*
  [with given <label|number|text|image> at <column> [through <column>] ]*   
  using <file://path/>
```

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
