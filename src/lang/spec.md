# Forward

"Dude, it feels like you just slapped together a language and didn't think this through at all."

I sort of did. I have iterated on it some and plan to iterate on it more as I go. My goal isn't to start with the 
perfect syntax, but rather to end with it. This is, more or less, my starting point and I'll improve on it.

If you iterate, won't you break everybody's code that uses it? Yeah, I will. That's unfortunate, but I don't have the
time or energy to devote years to finding the perfect way express a language like this and I expect things to break.

That said, since happyml is so lightweight, if you are happy with the version that works for you, keep using it. You 
could always move to a different version when you are ready. You could theoretically have multiple versions of it running
alongside your code and the impact should be low.

I don't want to be trapped into bad early decisions and not evolve as needed, just because somebody might be using it. And, 
I don't want to spend enormous amounts of time and energy trying to get it right in the first try.


# Current key word plans

| keyword  | type      | support priority | notes                                                                      |
|----------|-----------|------------------|----------------------------------------------------------------------------|
| model    | object    | initial          | type of object                                                             |
| dataset  | object    | initial          | type of object                                                             |
| row      | object    | future           | type of object                                                             |
| rows     | object    | initial          | type of object                                                             |
| column   | object    | future           | type of object                                                             |
| columns  | object    | future           | type of object                                                             |
| create   | action    | initial          | make a dataset                                                             |
| train    | action    | initial          | make a model based on a dataset                                            |
| retrain  | action    | future           | create a model shaped like another model but with a new or updated dataset |
| tune     | action    | future           | append to existing training using a different dataset                      |
| predict  | action    | initial          | use model                                                                  |
| infer    | action    | future           | synonym for predict                                                        |
| set      | action    | future           | set session parameters                                                     |
| let      | action    | future           | set script-local variables                                                 |
| copy     | action    | future           | copy a model or dataset                                                    |
| with     | criteria  | initial          | used to specify statement option                                           |
| at       | criteria  | initial          | used to specify column offset                                              |
| through  | criteria  | initial          | used to specify range                                                      |
| add      | criteria  | initial          | used to append values to a dataset                                         | 
| using    | criteria  | initial          | used to specify object relationship                                        |
| given    | criteria  | initial          | pass input to action                                                       |
| fast     | adjective | future           | Value performance over memory and accuracy                                 |
| small    | adjective | future           | Value small memory footprint over performance and accuracy                 |
| accurate | adjective | future           | Value quality results over memory and speed                                |


## Create Dataset
Allows us to create a dataset. This will inform models what input and output should look like and 
what the valid range of output is.

```
create dataset <name> from <local file or folder|url> 
[with format <delimited|image>]
[with expected [<scalar|category|pixel>] at <column> [through <column>] ] 
[with given [<scalar|category|pixel>] at <column> [through <column>] ]
```

## Train Model

I hate the word "model" because it isn't meaningful to most people, but I also don't have a better word yet. In the
code, I use the word "neural network", but I may want to support non-neural network models.

```
train [<adjective>*] <model type> model <model name> [<knowledge label>] using <data set name>
```

## Predict

`predict using <model name> [<model version>] given <input>`

or

`infer using <model name> [<model version>] given <input>`

## Add

```
add rows to dataset <name> using delimited data:
1, 2, 3, 4
(empty line to denote end of data)
```

## Set
Set's a session-scoped parameter.

`set <parameter> <value>`

| Parameter | Valid Values   | Default Value  | Comments                                                      |
|-----------|----------------|----------------|---------------------------------------------------------------|
| output    | human, machine | human          | Used to make machine parsable output or friendly human output |

Example:
```
set output machine
```

## Let
Assigns a value to a script-local variable. Maybe eventually support other scopes. Exists at this stage of the language 
development largely with the goal of making it easier to generate configurable and reusable scripts. My thought being 
that programs sending a script may have cleaner code if they can reuse the body, but just send some configuration
before running the script that leverages the variables.

`let <variable> = <value>`

Example:
```
let dataset_file = "/myfiles/mydataset.csv"
```

## Copy

```
copy <model name> [<knowledge label>] [as [<model name>] [<knowledge label>]]
```

## Tune

```
tune <model name> [<knowledge label>] [as [<model name>] [<knowledge label>]] using <data set name>
```

## Retrain

```
retrain <model name> [<knowledge label>] [as [<model name>] [<knowledge label>]] using <data set name>
```
# Rough thoughts on the protocol

syntax might be something like:

`set output <human/machine>`

```
create dataset <name> from <local file|local folder|url> 
[with format <delimited|image>]
[with expected [<scalar|category|pixel>] at <column> [through <column>] ] 
[with given [<scalar|category|pixel>] at <column> [through <column>] ]

add rows to dataset <name> using delimited data:
1, 2, 3, 4
(empty line to denote end of data)
```



```
create [<adjective>*] <model type> model <model name> [<knowledge label>] using <data set name>
tune <model name> [<knowledge label>] [as [<model name>] [<knowledge label>]] using <data set name>
retrain <model name> [<knowledge label>] [as [<model name>] [<knowledge label>]] using <data set name>
copy <model name> [<knowledge label>] [as [<model name>] [<knowledge label>]]
```

`predict using <model name> [<model version>] given <input>`

or

`infer using <model name> [<model version>] given <input>`

### Maybe for monte carlo search? See [Tasks] below.

queue input/predictions to be marked as success or failure:

`tracking start <model name> [<knowledge label>]`

mark the recorded input/predictions as success or failure and set them aside, clearing the prediction queue:

`tracking set [success|failure] to <model name> [<knowledge label>]`

using the success/failure rates of our input/predictions we build a tensor that represents what we learned and back
propagate:

`tracking apply <model name> [<knowledge label>]`

# Tasks

TLDR; I want a way for programs to manage weights using the monte carlo tree search pattern. I think
that I can potentially embed some of the most common aspects of that search tree into happyml, but
the adapter programs would be responsible for the interaction with other programs (whether that be
running those programs or screen scraping or whatever.)

Full stream of consciousness:
Somehow, I want to support the same sort of monte carlo tree search pattern used in the alpha go engine,
but I want to do it in a way that lets happyml interact with other programs.

This would give it a way of learning to do tasks like play tic-tac-toe, chess, go, poker, or
really any game. This functionality could extend to doing other activities that aren't games
but still have positive and negative outcomes we can analyze. For example, watching a web
page for a change in status and then playing an alert sound or sending a text message,
"Favorite Pop Singer Tickets are now available!"

I'd also like an observe-only training mode where it watched you do the task to try to
learn events and responses as a starting point. Humans can understand the rules of complex
games, and they start with a much better strategy than an ML algorithm will start with.
For activities that can't be done at incredible self-play speeds, this could save
hours or days of training, by just giving it a reasonable starting point. This isn't needed
for chess, because the computer could play itself and learn fast. Think about an endless
survival game where you can't play faster than a normal human could play. The algorithm would
be hopelessly bad for a long time if you didn't jump start its learning.

I think you'd have it make predictions on what it would do and then the feedback would be
immediate based on what the human did. If the AI did what the human did, then the response was
"good" and if the AI didn't do what the human did, then the response was bad. This feedback
is much more frequent than the typical end-of-game feedback, so the training pattern is different.
I need to think on this more. You might want to couple these micro feedback elements with
the bigger macro feedback on whether the human "won" or not.

Some games have an easy text-only interface and a simple event/response pattern, but
other games are graphical and have time-sensitive mouse responses. Feedback happens at the
end of a game or bigger period of time than the typical event/response loop. This feedback
tells us if we had a positive or negative outcome from our strategy.

As I recall:
The basic process is, we start with a strategy (that is probably bad), we use that strategy
to respond to events and eventually receive feedback for the outcome. We might use a strategy
for a number of times before we compare it to our past best strategy to see if our new change
was good. If newest strategy was better or worse than our best past strategy, we can use that
knowledge to find our next new strategy.

For chess or tic-tac-toe, the feedback is at the end of the game, but for an endless survival game,
the feedback might be seeing the next day.

With graphical games or programs, we need the ability to mask parts of the screen
it shouldn't click, so it doesn't change configuration of the game or exit the game
or do something else miserable.

With text-only games, we need it to not type random text that might be dangerous commands.

We might need to even use more than one ml model. One model to classify events, another to
infer responses, and another to classify feedback.

With complicated graphical games, think something like a first-person shooter, there is a
steady stream of events, but not all of them would trigger a response. Just because there
was a random bird that flew by, you don't have to change your plan.

Turn-based games like chess are the simplest to build an event/response/feedback configuration
for. And that would be where I started, but I'd like the plan to be generic enough that
it could be applied to any learnable task.

