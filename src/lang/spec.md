# Spec v2

So, I've been pondering the language more and more now that there's a working interpreter, and it's simply a matter
of typing the mappings between commands and actual code that runs. 

The initial language spec wasn't to my liking for a few reasons, namely because while it was declarative in nature, I 
could see an issue with people not really knowing how to use it. The purpose of a declarative language is to let
the user state what their goal is and for the language to then accomplish it. And while the v1 spec does that,
it didn't seem useful, or at least it seemed like it would take too much understanding of the underlying ideas.

So, here's v2. I reserve the right to just change my mind again, but for the moment, this idea seems slightly better.

# Tasks

I decided that creating "models" may be too math-y. And not really focused on the end user. I think the goal is to
allow the users to define tasks they want to perform and support those tasks. Yes, under the cover, these are models
but maybe even slightly more than that, because they may automate some of the logic needed to use the model for that
purpose. In the original v1 spec, I even mentioned that I really wasn't a fan of having users create models, but at
the time, I hadn't thought of another approach. Now, I have.


Generate the task (which is more or less training a model and creating a default label) using a dataset:
```
create task <task name> for <task type> with parameters <task specific parameters> using <dataset name>
```

Brainstorming some task types below, but the names and such could all be refined. There could be so many more. This
is not a definitive list, and it's going to change. I'm just going off the top of my head of tasks I might want to 
try my hand at, but this list could get really long. I'd prioritize them based on what functionality already exists
then what I find most interesting and then what I think would be most useful to others. Why? Because there's a good
chance that I'm the only one that uses this, and so it makes sense to build it for my interests first.

| task type       | description                                                            | priority  | Why                    |
|-----------------|------------------------------------------------------------------------|-----------|------------------------|
| label image     | given an image, apply a label from a list                              | immediate | Existing functionality |
| label text      | given text, apply a label from a list                                  | immediate | Existing functionality |
| complete text   | given text, complete it. (Causal language models.)                     | high      | Personal interest      |
| summarize text  | given text, create a summary.                                          | moderate  | Personally useful      |
| best move       | given state, find best move (chess for sure, but maybe other options?) | moderate  | Personally fun         |
| image from text | given text, generate an image                                          | moderate  | Personally fun         |
| time series     | given past data points, predict future data points                     | low       | Common use case        |
| recommendations | given historical data, what are recommended things                     | low       | Common use case        | 
| matching        | given a primary record and X return match probabilities                | low       | Common use case        |


Refine the underlying model from a checkpoint (label) using a dataset.
```
refine task <task name> [label] using <dataset name> [<label>]
```

List the existing tasks and checkpoints. 
```
list tasks [starting with <x>]
```

Execute a task. I'll start off with doing so against a dataset, since I have the needed code immediately, but 
supporting input directly will enable other more real-time possibilities. Note: I'm pondering how to handle 
embedded newlines with a syntax like this, since the interpreter currently assumes that a new line should execute
the command. Sender might need to encode newlines with \n and the parser may need to handle that. We'll cross that
bridge when we are working on that command.
```
execute task <task name> [with <label>] for dataset <dataset>
execute task <task name> [with <label>] for input <csv encoded row>
```

Maybe a copy, move, and delete.
```
copy <task name> [<label>] to task [<task name>] label [<label>]
move <task name> [<label>] to task [<task name>] label [<label>]
delete <task name> [<label>]
```

## Create Dataset

Allows us to create a dataset. This will inform models what input and output should look like and
what the valid range of output is. We also need a way as part of this to split an original dataset
into a training set and a testing set. 

While the C++ implementation allows you to utilize delimited data directly and even do so completely 
in-memory, for the scripting language, we'll use this to import in delimited data into a binary
format that has a training and testing set and can easily be shuffled without being fully in-memory.

This gives us the most robust, reusable, and reliable form for creating and managing datasets.

```
create dataset <name> from <local file or folder|url> 
[with expected [<scalar|category|pixel>] at <column> [through <column>] ] 
[with given [<scalar|category|pixel>] at <column> [through <column>] ]
[with testing set of <x>%]
```
