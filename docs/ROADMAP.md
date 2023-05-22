![happyml](../happyml.png)

# Roadmap
[Back to the table of contents](README.md)

Nice-to-haves for alpha:
* [ ] Finish interpreter commands to handle interfacing with happyml through a dsl.
    * [x] exit
    * [x] help
    * [x] create dataset
    * [x] print dataset
    * [x] create task
    * [ ] execute tasks

Issues to address later:
* The save format could be more efficient and compact. (The file support utilities exist as of 3/23, but the models aren't using it.)
* OpenMP is not helping performance, if anything it is hurting it. I will likely remove it and rewrite concurrency to use std::thread and pick better places to use concurrency.
* Apply the style guide. I'm all over the place. I'll try to do better.
* I need to adopt CMAKE tests and add more tests.
* I need to add more comments to the code for educational purposes.
* I need to fix the build process for Linux and Mac on GitHub.

Future:
* Batch normalization
* Flesh out more tasks from the happyml scripting language
    * Encoder-Decoder Transformer Model support
* Reinforcement Learning from Human Feedback (RLHF) support
* Example usage of scripts to achieve real-world solutions
    * Generate text from a prompt
    * Summarize text
    * Grammar checker
    * Image labeling
    * Text labeling
    * Image generation from a prompt
    * Audio to text
    * Text to audio


## Possible Task Types

In the future, happyml will support more task types. Here are some ideas:

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

