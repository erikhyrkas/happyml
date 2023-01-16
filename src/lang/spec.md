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

| keyword  | type      | support priority | notes                                                                                             |
|----------|-----------|------------------|---------------------------------------------------------------------------------------------------|
| model    | object    | initial          | type of object                                                                                    |
| dataset  | object    | initial          | type of object                                                                                    |
| row      | object    | future           | type of object                                                                                    |
| rows     | object    | initial          | type of object                                                                                    |
| column   | object    | future           | type of object                                                                                    |
| columns  | object    | future           | type of object                                                                                    |
| create   | action    | initial          | make a dataset                                                                                    |
| train    | action    | initial          | make a model based on a dataset                                                                   |
| retrain  | action    | future           | create a model shaped like another model but with a new or updated dataset                        |
| tune     | action    | future           | append to existing training using a different dataset                                             |
| predict  | action    | initial          | use model                                                                                         |
| infer    | action    | future           | synonym for predict                                                                               |
| validate | action    | future           | check loss against a dataset                                                                      |
| set      | action    | future           | set session parameters                                                                            |
| let      | action    | future           | set script-local variables                                                                        |
| copy     | action    | future           | copy a model or dataset                                                                           |
| log      | action    | future           | write text to log file                                                                            |
| with     | criteria  | initial          | used to specify statement option                                                                  |
| at       | criteria  | initial          | used to specify column offset                                                                     |
| through  | criteria  | initial          | used to specify range                                                                             |
| add      | criteria  | initial          | used to append values to a dataset                                                                | 
| using    | criteria  | initial          | used to specify object relationship                                                               |
| given    | criteria  | initial          | pass input to action                                                                              |
| fast     | adjective | future           | Probably doesn't need to be a keyword. Value performance over memory and accuracy                 |
| small    | adjective | future           | Probably doesn't need to be a keyword. Value small memory footprint over performance and accuracy |
| accurate | adjective | future           | Probably doesn't need to be a keyword. Value quality results over memory and speed                |
| comma    | adjective | future           | Probably doesn't need to be a keyword. adjective for delimiter                                    |
| tab      | adjective | future           | Probably doesn't need to be a keyword. adjective for delimiter                                    |


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

| adjective | immediate plan                                                                                                                                                          | future plan                                 | eventual plan                                                                              |
|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------|--------------------------------------------------------------------------------------------|
| fast      | Goal of good enough results. Materialize when possible to avoid duplicate operations. Always 32-bit. Only use bias on final output. Minimum necessary model complexity. | Leverage vectorization on cpu memory.       | Use GPU when possible and support quantization (16/8 bit) when available.                  |
| small     | Goal of good enough results. Materialize only when necessary. Use 8-bit or 16-bit when possible. Only use bias on final output. Minimum necessary model complexity.     | Page large tensors to disk when not in use. | Distributed execution? Maybe there's a way to have different layers on different machines. |
| accurate  | Goal of great results. Materialize only when necessary. Always 32-bit. Use bias on more layers when it increases accuracy. Maximum model complexity.                    |                                             |                                                                                            |

Accurate will be the default. 

A model is only useful if the results are usually right. How usually? Well, it depends on the model and how it is used, but a model that is wrong most of the time is not useful in nearly all cases. So, no matter which option you pick, I would strive to achieve a fairly high accuracy. Accuracy defined the percentage of time that we get the right result. I think an accuracy of 70% if a fairly realistic goal in most real-world situations. An accuracy of 90% is very good.

I'm going to start off by having the model guess defaults based on my experience, but eventually support having the training do experiments to find a configuration that meets your goals.

Rambling thoughts:
My first inclination is to make the training do experiments to find the best complexity to get good accuracy that still meets your goals, but that might be slow. It might have to do 10 experiments to pick settings that meet the goals and still gives at least 70% accuracy. The time it takes to do an experiment might depend on the nature of the model and might even take days. I might make it so that experimentation only continues until we find something that gives us the minimum best results, so we don't always have to wait for X experiments.

The alternative is to include a "best" keyword that signals to do the experimentation or otherwise just uses default settings and hope they are good enough. So, "best fast" would do experiments to find the fastest model that still gave 70% accurate results. Where "best accurate" might do experiments to find the configuration that gave the highest accuracy. But, if you said "fast", it would just take a guess at configuration, do that training, and you'd have to hope that it was accurate enough.

The argument against the "best" keyword is that people don't have a way to do fine-grained controls of a model through the happyml language and there's a big difference between each goal. People who don't specify best would often get bad results. The argument for the "best" keyword is that it gives you a way to disable the experiments and go with a close-enough configuration. 

I may also have the adverb "very", which would give an extra lever to push further toward a goal extreme and would impact the model's complexity -- by either making it more or less complex. So, "very accurate" would increase complexity of the model, but "very fast" or "very small" would decrease the complexity of the model with an impact on the quality of the result.

If I did include "very", then I would make the default complexity closer to the "middle of the road", so that there was more room to push boundaries.

Then we get into "best very accurate", where our starting settings in an experiment are equivalent to "very accurate", but we continue to experiment if we don't meet our goal of 90% accuracy.

People might also want a "sort of" adverb, like "sort of fast" or "sort of small", but I don't think I'll go that far.


## Predict

`predict using <model name> [<model version>] given <input>`

or

`infer using <model name> [<model version>] given <input>`

## Validate

I originally was only going to do validate as part of training, but I think there are use cases where
revalidating a model might be handy. For example, you could validate against last month's data to see 
how well your model is performing and whether you should retrain it. This is only different than `predict`
in that you aren't interested in each individual prediction, but the loss across a full set of predictions.

```
validate model <model name> [<knowledge label>] using <data set name>
```

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

