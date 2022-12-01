# Compiling
I use CLion with Visual Studio 2022 community for the c++ runtime libraries.
* https://visualstudio.microsoft.com/downloads/

Look at the [clion_settings.png](clion_settings.png) for an example on configuring CLion. Pay special attention to the x64 setting. I had to type that value, since it was not in the dropdown options.

I didn't include the idea project settings files, so you'll have to make a project from source, but everything should work
based on the [CMakeLists.txt](CMakeLists.txt).

# microml

Why did you build this? You shouldn't have. Seriously. Why?

Three reasons: 
1. LEARNING: I want to understand every aspect of what goes into the science and algorithms fueling the future, and share what I learn along the way. The next generation can pick up this torch and build their own dreams.
2. DEMOCRATIZING POWER: The most powerful machine learning done today is done on millions of dollars of hardware by a relatively few people. While I may not invent a way of democratizing the power of machine learning, I hope this is a small step toward inspiring somebody smarter than me to bring it about.
3. STUBBORNNESS: There are a million reasons not to build this, which is enough to harden my resolve to build it. 

If you want to use the latest and coolest algorithms, you can use them relatively cheaply if you have some rudimentary programming skills to build some amazing things:
* https://openai.com/api/
* https://www.midjourney.com/
* https://developer.grammarly.com/

Looking for nearly as good but cheaper (DISCLAIMER: I haven't used these, but aware they exist):
* https://playground.forefront.ai/models/free-gpt-j-playground
* https://nlpcloud.com/home/playground/
* https://replicate.com/docs/reference/http

Maybe you want to build your own because you want to avoid the restrictive licensing agreements or because you want to learn, the first place (and probably last place) you need to go is:
* https://huggingface.co/

Hugging Face has the latest and greatest open source machine learning models and documentation. They have a hosting service, but if you are learning or doing your own projects, you could likely do it cheaper:
* https://colab.research.google.com/ (Useful if you want to use thousands of dollars of hardware with very low cost)
* https://www.paperspace.com/ (Useful if you want to use tens of thousands of dollars of hardware with low-ish cost)

What about SageMaker or Databricks or Dataiku or xyz? If you have the money and the know-how, there are many options that are industrial strength. Chances are, you are a data scientist, and you have a budget. 

Why are you still here? The above are clearly the very best options available. Get out of here! Build! Have fun! Learn!

No? You still want to see microml? You are that one mythical person that read this far?

I document pretty much everything that I learn as I write the code. There might be gaps where it didn't occur to me to document them, or maybe I was in the flow and didn't take time to explain what I was doing. I'll eventually try to circle back and fix these things.

My goal is for microml to run on commodity hardware for both making predictions and for training. I want it to work for the average person that doesn't own tens of thousands or millions of dollars in hardware, but still wants to experiment with training their own model, or taking what I've built and expanding on it to design their own algorithms. For a lot of people it will be hard to grasp that the goal is to make this possible at all rather than to make it fast. There have been advances in quantization (the process of using less memory to achieve similar results through approximations) for using models to make predictions, but not so much for training new models.

I'm not delusional. What I've built won't be as flexible or as fast as frameworks made by Google or Meta, and there's a very high chance that I will make mistakes along the way that render this code useless. However, this code is meant to run huge models on very small amounts of hardware. Not necessarily faster, but to possibly do it. And let's be real: I doubt anybody besides me will ever use this code.

The biggest issue, as I see it, with modern machine learning models is that they aren't accessible for most people. Either you work for a company that can afford to experiment on expensive hardware, or you leverage somebody else's work and live with the restrictive licensing agreements.

The second-biggest issue, as I see it, is that machine learning is largely only done by data scientists. The world needs data scientists, however machine learning is a tool that should be available to everybody.

You probably couldn't build a top of the line car all by yourself. You might understand the basics of how an engine works. Maybe you know how seat belts work. However, there are thousands of parts in a modern car that very few people on the planet have the knowledge and skill to not only design but craft. 

That doesn't mean you shouldn't drive cars. Machine learning is a lot like that in my mind. Right now, most machine learning is built by data scientists and used by data scientists. But, even in that field, it is growing so rapidly and expanding to the point where I'm not convinced the average data scientist could or even would want to build everything from scratch. And there will be data scientists who object to my analogy. Machine learning models today require a lot of knowledge to "drive", but who's fault is that? We're so busy trying to build new models we aren't taking the time to make sure they are usable. 

Making machine learning models accessible is still a journey. The big companies exposing REST APIs is probably a decent start, but we have a ways to go yet before the average software developer feels comfortable training their own model or even incorporating a pre-trained model into their code.

Building a framework like this requires knowing some calculus and linear algebra. That's going to be off-putting for a lot of folks who haven't had a math class since junior high. That doesn't mean you should avoid machine learning though. Whether you learn the math or not, the basic ideas behind it are something that anybody can understand if given the right teacher. Software developers use technologies that they don't fully understand the math behind all the time. Most people have no idea how a database optimizes queries or manages internal structure, but they can get data out of it.  

I wrote this in C++, which is going to be off-putting for a lot of people. C++ is not a dead language, but it's not a popular language in the general population either. I chose it because I wanted this to work well with very little hardware and I needed some very low-level control over the data structures to do this. I could rebuild this with Rust and possibly C# or Java. I don't see myself ever building this in R, Go, Swift, Python, or JavaScript -- I'm not even sure what sort of black magic it would take to build this in JavaScript... maybe that is something that I should be thinking about if I want this to be accessible. I would like to roll this work into my most recent programming language Dog -- which is something I haven't finished and may never finish. Don't worry too much about the language. Read the comments and learn.

I want to help the process of making machine learning accessible. Maybe this is a small step, possibly in the wrong direction, but it's the best I can do right now. It is my deepest hope that this helps you and is a small ripple that eventually helps others as well.


# High-level Design

Here's a collection of random notes and thoughts... I'll come back and clean this up later... I hope. I wrote these down to focus my initial plan, but I'm not one to stick to a plan that isn't working, so there will be lots of changes as I realize where I was being dumb.

Quarter:
* 8-bit float with bias and offset
* Bias controls how granular the number is.
* A small bias lets us make big numbers, but the numbers are spread apart.
* A large bias lets us make accurate small numbers, but the total range we can represent is very small
* Offset is a 32-bit float we add to the quarter's value, allowing us to shift the range of numbers we can repsent.
* For example, if you only need to represent positive numbers, and your bias lets you represent -0.5 to 0.5 accurately,
  shifting by +0.5 will let you accurately represent 0 to 1.0.
* The Quarter is a storage mechanism for the floating point representation, but we convert it to a float for math
  because we'd lose accuracy when doing many mathematical operations in a row.
* Offset is flawed since bias always gives the greatest accuracy around 0 and shifting 0 means that you are shifting
  where the greatest accuracy is to be at that offset.

Tensor:
* Currently, 3d: Channels x Rows x Columns
* Row Vector is 1 row with N columns
* Column Vector is 1 column with N rows
* Matrix is N Rows and M Columns
* Channels let us have multiple vectors or matrices
* Fairly sure that 3 dimensions is correct for my purposes, but:
  * Might add 4th dimension: batch (which is currently external to tensor)
  * Might remove 3rd dimension, channels, which might be wasteful to have internal for single vectors/matrices
* Quarter Tensors are currently supported, but Bit Tensors would be useful. Bit Tensors would have to be immutable

Tensor View:
* Shows the result of a transformation on the tensor without modifying the tensor
* Can be stacked to allow us to apply many transformations without using the memory to hold many copies of the
  original tensor during transformation

Data Set:
* Raw data from a single source (file, network location)
* Aware of how to parse the file into records

Data Encoder:
* Converts raw data into a Tensor

Data Decoder:
* Converts a tensor into usable data (similar to original raw data)

Data Source:
* Zero or more Data Sets (could generate data)
* Data Encoder -- used at this point so that we don't have to constantly re-encode input each epoch
  and saves memory by not needing raw data beyond the initial moment to encode.

Optimizer:
* Use a loss function to compute the accuracy of a prediction
* Manage state tensors on layer used for optimizing

Neuron:
* Weights
* Bias (optional)
* Optimizer state tensors
* Activation function
* Regularizer
* takes in Loss function
* Specific input and output shape (shape = tensor dimensions)

Neuron Block:
* Specialized layer
* Represent common blocks of layers used to build different popular models
* While the size of these blocks may vary, the pattern is consistent
* Neuron definitions
* Abstracts the intricacies of the layer details

Model:
* represents a DAG (directed acyclic graph) of layer/layer blocks and checks for cycles when adding new layers
* Layers
* Optimizer
* Decoder -- convert raw tensor outputs to data similar to raw data
* Trained using a Data Source, which configures the input shape
* Produces one or more outputs

Neuron Architecture:
* Factory to add many layers/layer blocks to a model and configure it
* Represents common combinations of layer blocks to solve common problems
* Simple configuration to cover core decision points -- like input and output shape, number of neurons or layers, etc.

# Why this project is important to me -- an opinion piece

There's a difference between data science and machine learning. Data Science is a very technical discipline that requires a deep understanding of math. Machine learning is the fruits of their labor and a powerful tool that everybody can use. I want to make machine learning more accessible.

There is a lot of gate keeping in machine learning.

Data scientists study for years, working hard to obtain the knowledge necessary to create new algorithms -- machine learning models. It requires huge amounts of effort to gain that understanding, and even more effort to apply it in real life situations.

The idea that the average developer could use a machine learning algorithm is laughable from the perspective of somebody who has a deep understanding of the math and how to build and configure models. It shouldn't be. The reason it is so hard is that the state of the technology is not accessible.

In addition, a lot of corporate data science has a low tolerance for risk and is both hard and rather boring -- there's a couple of popular techniques around regression and classification that require a tremendous amount of knowledge to apply, and a lot of research into the data to even attempt, but the end results are something that follows a fairly predictable pattern.

This is rather unfortunate, since there are many situations where machine learning could integrate into workflow tools and products.

Data science and machine learning are expensive. The high barrier to entry means there are few qualified people that can use machine learning algorithms, let alone build new ones, and they are paid well. The hardware and software that are part of the ecosystem is specialized and comes with a hefty price tag. In my opinion, if more people and companies could use machine learning, the price of machine learning would go down. This wouldn't change that data scientists are still valuable, but they won't spend their time on trivial tasks.

Most of the fun machine learning models that do cool things that everybody would like to do are expensive to build and run and there's a risk they won't be useful, or are something you can pay to use from a big company like OpenAI as long as you are okay with the licensing agreement.

Machine learning frameworks are built by data scientists for data scientists and don't make any effort to be accessible to people without years of formal education. You could argue that there are people trying to make it more accessible, but not by making the frameworks easier to use for ordinary developers but by trying to teach people the core of data science. Not everybody wants to be a data scientist, they just want to have cool toys.

What's more, terminology in data science is highly technical and most people aren't familiar with it. This is intimidating and keeps people away.

Even worse, data science continues to advance at a dizzying speed, and the existing frameworks are littered with the dead and discarded parts that nobody should ever need again. Sifting through huge amounts of code for the bits that are actually still relevant is challenging, and once you do, those bits are probably no longer relevant because somebody came up with a new, better approach.

Data science is 95% math and 5% mediocre code. Most of the software engineers that I've met barely know algebra, let alone how to do linear regression. Most of the data scientists I've met could only technicality be labeled as software engineers--and they'd probably prefer you not call them that.

I don't think machine learning should only be for data scientists. It should be for everybody, and I think everybody wins. Data scientist continue to build new, cutting edge technologies and more people and more situations get to use it. This raises the visibility and value of these models and helps software developers succeed.

What I'm building here is a first baby step and is a long way from accessibility. I do try to document what I know as I go, which addresses part of the issue, but as I said earlier, most people just want to use machine learning as a tool, they don't want to know the math. This is akin to driving a car rather than building one. I'd like to get to the point where this is usable for everybody without the data science background. This may be an unobtainable goal.