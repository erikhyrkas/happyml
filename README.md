![happyml](happyml.png)

# What is happyml?

happyml is a small machine learning library with a simple interface intended for everybody. 

# MIT License

See the [LICENSE](LICENSE) file for details.

### What can it do?

The components are present to build a number of different types of deep learning models.

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
* Dataset Shuffle is now in-place and the shuffler can be shared between datasets to keep them in sync. (4/23)
* SGDM Optimizer with demon (4/23)
* Adam Optimizer with demon (4/23)
* Rotary Positional Embedding (4/23)
* One Hot Encoding (4/23)
* New Logo (4/23)
* Byte Pair Encoding (3/23)
* Basic Binary File Support (more to come for datasets) (3/23)
* New Style Guide (that I'm occasionally following--I'll try to do better) (3/23)
* Working Parser/Interpreter (but integration with framework still needs most features) (3/23)
* Documentation (2/23)
* Working Lexer (1/23)

Nice-to-haves for alpha:
* Need to finish the half float and test. It currently doesn't handle any edge conditions and could produce incorrect results in some situations.
* Finish interpreter commands to handle interfacing with happyml through a dsl. (Lexer and parser now have a working foundation.)

Back-of-the-mind considerations:
* The save format could be more efficient and compact. (The file support utilities exist as of 3/23, but the models aren't using it.)
* I need to create an example that uses multiple inputs. The train() function doesn't take in multiple data sets and there will need to be some other small improvements.

At that point, the code will be in an alpha state, but I still won't have even tackled encoder-decoder and decoder-only requirements. For beta, I'd like to see at least decoder-only support. There's also Reinforcement Learning from Human Feedback (RLHF) that I'll ponder, but I doubt I can work that into the beta let along the alpha. That journey will continue.

This project is still a long way off from democratizing the power of ml. Right now, it's mostly focused on making large models run on commodity hardware. Next steps will be to make the interface to using it easier for people without a data science background. And finally, I hope that everything I build is educational.

# Why did you build this?

Three reasons I built happyml:
1. LEARNING: I want to understand every aspect of what goes into the science and algorithms fueling the future, and share what I learn along the way. The next generation can pick up this torch and build their own dreams.
2. DEMOCRATIZING POWER: The most powerful machine learning done today is done on millions of dollars of hardware by a relatively few people. While I may not invent a way of democratizing the power of machine learning, I hope this is a small step toward inspiring somebody smarter than me to bring it about. See the section on democratizing power.
3. STUBBORNNESS: There are a million reasons not to build this, which is enough to harden my resolve to build it.
