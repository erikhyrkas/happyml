![happyml](happyml.png)

## What is happyml?

happyml is a scripting language and framework to make machine learning tasks easier to create and usable by everybody. 

## Current State
This project is pre-alpha.

[Recent Notable additions](docs/NOTABLE_CHANGES.md)

[Current Roadmap](docs/ROADMAP.md)

## What can it do?

**For the moment, you still need to use C++ instead of the scripting language**, and it does
require a deeper understanding of how machine learning works, but happyml still tries to make
it as simple as possible. It won't be long before this isn't necessary.

Here are two C++ examples:
* [C++ MNIST example](src/example/example_mnist_model_convolution.cpp).
* [C++ BPE example](src/test/test_byte_pair_encoding.cpp).

The happyml scripting language is a work in progress, but parts of it are complete.

You can define tasks backed by a machine learning model and train it to solve the task by giving 
it a set of example inputs and the corresponding outputs. Task execution from the scripting language
is a work in progress, but nearly done.

See the happyml scripting language example: 
*  [Titanic Survivor example](docs/examples/TITANIC.md)

Or read the docs for more information: [Scripting Docs](docs/README.md).


## What makes happyml special?

1. EDUCATIONAL: I've made an effort to document what I've learned throughout, so that you can learn from it as well.
2. SMALL MEMORY FOOTPRINT: It supports 8-bit quantization for training, not just inference.
3. OPTIMIZED FOR CHEAP HARDWARE: The entire architecture is optimized for the CPU rather than the GPU, which allows the use of cheaper hardware with acceptable performance.
4. DECLARATIVE INTERFACE: (Still a work in progress, but the basics of it exist.)

So, I used a fancy term "quantization", which more or less is another way of saying that I let you use less memory by using approximate values. You are trading accuracy for lower memory consumption.

What I've found is that, if you are careful about which layers of the model are quantized and you have a mechanism for compressing the series of operations applied to the data, there is very little impact on the quality of the end results. That said, supporting models like this means that the performance is worse because of the techniques I used to reduce the amount of estimation.

For a small model, you don't have to use 8-bit or 16-bit features and can use 32-bit features, but for models that wouldn't fit on commodity hardware? This makes it possible to train models that would otherwise require a considerable amount of expensive, specialized hardware.

Largely, what makes this framework special is how I made tensors immutable and stacked "views" on top of them to effectively compress the tensors in memory without losing much precision.

## Using

There are options as to how you use happyml. 

1. C++ CODE: (Current) You could use the code directly, like in the examples: [MNIST example](src/example/example_mnist_model_convolution.cpp)
2. SCRIPTING: (In Progress) The happyml scripting language is still a work in progress: [Scripting Docs](docs/README.md) 
3. COMPILED LIBRARY: (Future) You could compile it to a library. I didn't provide an example of this, yet, it's on my todo list.

We have a working lexer and parser to make scripting a reality, but there is still work to be done on filling out functionality. The work remaining on the interpreter is not so much hard as plentiful and a little boring.


## Why did you build this?

Three reasons I built happyml:
1. LEARNING: I want to understand every aspect of what goes into the science and algorithms fueling the future, and share what I learn along the way. The next generation can pick up this torch and build their own dreams.
2. DEMOCRATIZING POWER: The most powerful machine learning done today is done on millions of dollars of hardware by a relatively few people. My hope is that this framework can be used by anybody on commodity hardware to build their own dreams. 
3. STUBBORNNESS: There are a million reasons not to build this, which is enough to harden my resolve to build it.

# MIT License

See the [LICENSE](LICENSE) file for details.
