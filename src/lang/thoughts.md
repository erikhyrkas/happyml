# Rough thoughts on the protocol

syntax might be something like:

`set output <human/machine>`

```create <data set type> dataset <name> from <location> [with <format>]
add rows to dataset <name> using delimited data:
1, 2, 3, 4

```
(empty line to denote end of data)

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

