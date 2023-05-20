![happyml](happyml.png)

# What is happyml?

happyml is a small machine learning library with a simple interface intended for everybody. 

### What can it do?

**In the (hopefully) near future**, you'll be able to use happyml to define tasks backed by a 
machine learning model and train it to solve the task by giving it a set of example inputs 
and the corresponding outputs. The syntax is still in flux, but it will look something like this:

```happyml
create dataset my_chat_base using file://chat_base.csv

create dataset my_chat_tasks using file://chat_tasks.csv

create task generate my_chat_task using my_chat_base

refine task my_chat_task using my_chat_tasks

execute my_chat_task using input "How tall is a giraffe?"
```

**At the present**, you still need to use C++ instead of the scripting language, and it does
require a deeper understanding of how machine learning works, but happyml still tries to make
it as simple as possible.

Here are two examples:
* [MNIST example](src/example/example_mnist_model_convolution.cpp).
* [BPE example](src/test/test_byte_pair_encoding.cpp).

### What makes happyml special?

1. EDUCATIONAL: I've made an effort to document what I've learned throughout, so that you can learn from it as well.
2. SMALL MEMORY FOOTPRINT: It supports 8-bit quantization for training, not just inference.
3. OPTIMIZED FOR CHEAP HARDWARE: The entire architecture is optimized for the CPU rather than the GPU, which allows the use of cheaper hardware with acceptable performance.
4. DECLARATIVE INTERFACE: (Still a work in progress, but the basics of it exist.)

So, I used a fancy term "quantization", which more or less is another way of saying that I let you use less memory by using approximate values. You are trading accuracy for lower memory consumption.

What I've found is that, if you are careful about which layers of the model are quantized, there is very little impact on the quality of the end results. That said, supporting models like this means that the performance is worse because of the techniques I used to reduce the amount of estimation.

For a small model, you don't have to use 8-bit or 16-bit features and can use 32-bit features, but for models that wouldn't fit on commodity hardware? This makes it possible to train models that would otherwise require a considerable amount of expensive, specialized hardware.

# Using

There are options as to how you use happyml. 

1. C++ CODE: (Current) You could use the code directly, like in the examples.
2. SCRIPTING: (Incomplete) The happyml scripting language is still a work in progress. See [my proposed language specification](src/lang/spec.md). 
3. COMPILED LIBRARY: (Future) You could compile it to a library. I didn't provide an example of this, yet, it's on my todo list.

We have a working lexer and parser to make scripting a reality, but there is still work to be done on filling out functionality. The work remaining on the interpreter is not so much hard as plentiful and a little boring.

# Compiling C++
I use CLion with Visual Studio 2022 community for the C++ runtime libraries.
* https://visualstudio.microsoft.com/downloads/

Look at the [clion_settings.png](clion_settings.png) for an example on configuring CLion. Pay special attention to the x64 setting. I had to type that value, since it was not in the dropdown options.

I didn't include the idea project settings files, so you'll have to make a project from source, but everything should work
based on the [CMakeLists.txt](CMakeLists.txt).

# Current State
This project is pre-alpha.

Recent Notable additions:
* Scripting language support for creating label tasks and training the supporting models (5/2023)
* L2 Regularization (5/2023)
* Layer Normalization (5/2023)
* Improved Profiling (5/2023)
* Improved Exit Strategy (5/2023)
* Improved Batch handling (5/2023)
* Xavier weight initialization (5/2023)
* C++ Example using Titanic dataset (5/2023)
* Support for multiple inputs when creating and training models (5/2023)
* New Concatenate Layer (5/2023)
* Support for creating and using binary datasets from happyml scripting language (5/2023)
* Categorical Cross Entropy and other Functions (5/2023)
* Softmax Activation Function (5/2023)
* Default Trained Byte Pair Encoding Model for happyml scripting language (4/2023)
* New program "create_tokenizer" to train default BPE model from a file (4/2023)
* Edge cases for Half float and tests (4/2023)
* BPE encoding/decoding optimization (4/2023)
* Dataset Shuffle is now in-place and the shuffler can be shared between datasets to keep them in sync. (4/2023)
* SGDM Optimizer with demon (4/2023)
* Adam Optimizer with demon (4/2023)
* Rotary Positional Embedding (4/2023)
* One Hot Encoding (4/2023)
* New Logo (4/2023)
* Byte Pair Encoding (3/2023)
* Basic Binary File Support (more to come for datasets) (3/2023)
* New Style Guide (that I'm occasionally following--I'll try to do better) (3/2023)
* Working Parser/Interpreter (but integration with framework still needs most features) (3/2023)
* Documentation (2/2023)
* Working Lexer (1/2023)

Nice-to-haves for alpha:
* [ ] Finish interpreter commands to handle interfacing with happyml through a dsl.
  * [x] exit
  * [x] help
  * [x] create dataset
  * [x] print dataset
  * [x] create task
  * [ ] list tasks
  * [ ] execute tasks
* [ ] Possibly support a decoder-only transformer model.

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


Right now, this project is mostly focused on making large models run on commodity hardware and making a scripting 
language that gives a simple interface. And finally, I hope that everything I build is educational.

# Why did you build this?

Three reasons I built happyml:
1. LEARNING: I want to understand every aspect of what goes into the science and algorithms fueling the future, and share what I learn along the way. The next generation can pick up this torch and build their own dreams.
2. DEMOCRATIZING POWER: The most powerful machine learning done today is done on millions of dollars of hardware by a relatively few people. My hope is that this framework can be used by anybody on commodity hardware to build their own dreams. 
3. STUBBORNNESS: There are a million reasons not to build this, which is enough to harden my resolve to build it.

# MIT License

See the [LICENSE](LICENSE) file for details.
