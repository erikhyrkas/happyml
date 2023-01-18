# MIT License

See the [LICENSE](LICENSE) file for details.

# Using

There are options as to how you use happyml. 

1. You could use the code directly, like in the examples.
2. You could compile it to a library. I didn't provide an example of this, yet, but I can add that to my todo list.
3. And, eventually, you'll be able to use it through the happyml scripting language, which is still a work in progress. See [my proposed language specification](src/lang/spec.md). This is something that I'm working on. 

# Compiling
I use CLion with Visual Studio 2022 community for the C++ runtime libraries.
* https://visualstudio.microsoft.com/downloads/

Look at the [clion_settings.png](clion_settings.png) for an example on configuring CLion. Pay special attention to the x64 setting. I had to type that value, since it was not in the dropdown options.

I didn't include the idea project settings files, so you'll have to make a project from source, but everything should work
based on the [CMakeLists.txt](CMakeLists.txt).

# Current State
This project isn't even in a complete alpha stage, yet.

Nice-to-haves for alpha:
* A test() function that could take a test data set and return a loss. This could be used for early stopping, but also for tests.
* Need to fix and check-in Adam optimizer. I'm not even going to check it in until it seems plausibly right and I need to refactor the model object's training to support it correctly. I built the mini-batch gradient decent optimizer first because it was easier to make (even though I still had issues building it correctly -- that is part of the learning process), and it let me test all the other code.
* Need to finish the half float and test. It currently doesn't handle any edge conditions and could produce incorrect results in some situations.
* Would like to create a lexer-parser to handle interfacing with happyml through a dsl.

Back-of-the-mind considerations:
* _I think I've updated most of the formatting at this point._ ~~Need to format the code to C++ standards, since I've been doing so many languages that I have clearly forgotten what is standard.~~
* _Convolutional Layers are training slow._ ~~Need to fix convolutional layers.~~
* _The save format could be more efficient and compact._ ~~Need to be able to save and restore state, even if in a simple format.~~
* There is a bug with multiple-inputs where the input shuffles aren't synchronized. This can be addressed, but I need to create an example that uses multiple inputs. Maybe I can get an image and label data set. I'm thinking I'll create a new type of data set that combined multiple other sources to keep them in sync when shuffling and offer "channels" to split those input into different neurons at the start of training. This will also be a chance to implement and test Convolutional 2d Same layers.

Stretch goals for alpha:
* ~~Early stopping policy~~
* Lexer/parser to take in an input stream and create, train, load, and use models through a standard input stream


At that point, the code will be in an alpha state, but I still won't have even tackled encoder-decoder and decoder-only requirements. For beta, I'd like to see at least decoder-only support. There's also Reinforcement Learning from Human Feedback (RLHF) that I'll ponder, but I doubt I can work that into the beta let along the alpha. That journey will continue.

This project is still a long way off from democratizing the power of ml. Right now, it's mostly focused on making large models run on commodity hardware. Next steps will be to make the interface to using it easier for people without a data science background. And finally, I hope that everything I build is educational.


# happyml

Why did you build this? You shouldn't have. 

Seriously, you shouldn't have. 

Three reasons I built happyml:
1. LEARNING: I want to understand every aspect of what goes into the science and algorithms fueling the future, and share what I learn along the way. The next generation can pick up this torch and build their own dreams.
2. DEMOCRATIZING POWER: The most powerful machine learning done today is done on millions of dollars of hardware by a relatively few people. While I may not invent a way of democratizing the power of machine learning, I hope this is a small step toward inspiring somebody smarter than me to bring it about. See the section on democratizing power.
3. STUBBORNNESS: There are a million reasons not to build this, which is enough to harden my resolve to build it.

# Democratizing the Power of ML

Here are the three ways this project aims to democratize the power of ML:
1. Support building large models that train on inexpensive, commodity hardware.
2. Provide an interface to using common machine learning approaches to software and data engineers without needing to be a data scientist.
3. Create a place for people to learn about ML that doesn't assume you have formal training or the math background to understand it.

The first goal reduces the expense of building models. Right now, the cutting edge models of the world can only be built on very expensive hardware. There's no chance this framework will ever be as fast as what can be done by throwing money at the problem, but making large models possible for less money, even if they are slower to run is a huge win for democratizing the power of ML. There might come a time where the price of hardware goes down dramatically and this goal will seem ridiculous. I'm counting on the complexity of ML to keep pace with the advancements in hardware.

The second goal is about providing an interface to ML that lets software developers communicate their end goal in a way they are familiar with and then building the model for them. This is similar to how SQL works with databases. You specify what you want, not how you want it retrieved. You leave it to the database to calculate the optimal path to get that data. I think ML needs to get to this point so that it usable by a larger audience who might not be mathematicians and data scientists. There are already tools available and are growing to make ML available through simple web interfaces, but these tools aren't for software engineers, and they aren't free. I see a future where happyml may be something directly used by code or deployed as a webservice and accessed through a REST API.

The last goal is an area that there is already a huge push for, which is creating means for people to learn the math and science of ML. I want to support this goal, but at the same time, this is already the area that is the strongest.

# Model Risk Management

Whether you use happyml or another framework, I've decided that it's important to take a moment to mention model risk management.

With the explosion of popular ml models, like ChatGPT, I've seen examples of tremendously poor model risk management, and want to remind people that ML Models aren't infallible. I believe that I read somewhere that a good model is 70% accurate, and that an accuracy of 90% was the best you could hope for.

People are using ML models without any thought to the risk that the output might be wrong. Models are great and with the proper processes around using them can be more accurate than humans, but they are still wrong often enough that it is important to insulate yourself from those risks. Don't blindly use model outputs as if they must be right. Use the model outputs along with your own knowledge and other information to make the best decisions you can.

Using a model to generate text and then treating that text like fact is a terrible plan. A tool like ChatGPT can help you with a draft, but it is a tool that the user uses to make a draft. It is on the user to then take that draft and make a finished result. 

Models are only as good as the data and processes used to make them, and nobody has perfect data or perfect processes. Be careful out there.

# Learning more
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

No? You still want to see happyml? You are that one mythical person that read this far?

I document pretty much everything that I learn as I write the code. There might be gaps where it didn't occur to me to document them, or maybe I was in the flow and didn't take time to explain what I was doing. I'll eventually try to circle back and fix these things.

My goal is for happyml to run on commodity hardware for both making predictions and for training. I want it to work for the average person that doesn't own tens of thousands or millions of dollars in hardware, but still wants to experiment with training their own model, or taking what I've built and expanding on it to design their own algorithms. For a lot of people it will be hard to grasp that the goal is to make this possible at all rather than to make it fast. There have been advances in quantization (the process of using less memory to achieve similar results through approximations) for using models to make predictions, but not so much for training new models.

I'm not delusional. What I've built won't be as flexible or as fast as frameworks made by Google or Meta, and there's a very high chance that I will make mistakes along the way that render this code useless. However, this code is meant to run huge models on very small amounts of hardware. Not necessarily faster, but to possibly do it. And let's be real: I doubt anybody besides me will ever use this code.

The biggest issue, as I see it, with modern machine learning models is that they aren't accessible for most people. Either you work for a company that can afford to experiment on expensive hardware, or you leverage somebody else's work and live with the restrictive licensing agreements.

The second-biggest issue, as I see it, is that machine learning is largely only done by data scientists. The world needs data scientists, however machine learning is a tool that should be available to everybody.

You probably couldn't build a top of the line car all by yourself. You might understand the basics of how an engine works. Maybe you know how seat belts work. However, there are thousands of parts in a modern car that very few people on the planet have the knowledge and skill to not only design but craft. 

That doesn't mean you shouldn't drive cars. Machine learning is a lot like that in my mind. Right now, most machine learning is built by data scientists and used by data scientists. But, even in that field, it is growing so rapidly and expanding to the point where I'm not convinced the average data scientist could or even would want to build everything from scratch. And there will be data scientists who object to my analogy. Machine learning models today require a lot of knowledge to "drive", but who's fault is that? We're so busy trying to build new models we aren't taking the time to make sure they are usable. 

Making machine learning models accessible is still a journey. The big companies exposing REST APIs is probably a decent start, but we have a ways to go yet before the average software developer feels comfortable training their own model or even incorporating a pre-trained model into their code.

Building a framework like this requires knowing some calculus and linear algebra. That's going to be off-putting for a lot of folks who haven't had a math class since junior high. That doesn't mean you should avoid machine learning though. Whether you learn the math or not, the basic ideas behind it are something that anybody can understand if given the right teacher. Software developers use technologies that they don't fully understand the math behind all the time. Most people have no idea how a database optimizes queries or manages internal structure, but they can get data out of it.  

I wrote this in C++, which is going to be off-putting for a lot of people. C++ is not a dead language, but it's not a popular language in the general population either. I chose it because I wanted this to work well with very little hardware and I needed some very low-level control over the data structures to do this. I could rebuild this with Rust and possibly C# or Java. I don't see myself ever building this in R, Go, Swift, Python, or JavaScript -- I'm not even sure what sort of black magic it would take to build this in JavaScript... maybe that is something that I should be thinking about if I want this to be accessible. I would like to roll this work into my most recent programming language Dog -- which is something I haven't finished and may never finish. Don't worry too much about the language. Read the comments and learn.

I want to help the process of making machine learning accessible. Maybe this is a small step, possibly in the wrong direction, but it's the best I can do right now. It is my deepest hope that this helps you and is a small ripple that eventually helps others as well.


HappymlDSL:
* factory that builds a model with easy-ish to understand syntax.
* Simple configuration to cover core decision points -- like input and output shape, number of neurons or layers, etc.

# Why this project is important to me -- an opinion piece

There's a difference between data science and machine learning. Data Science is a very technical discipline that requires a deep understanding of math. Machine learning is the fruits of their labor and a powerful tool that everybody can use. I want to make machine learning more accessible.

There is a lot of gate keeping in machine learning. I'm not blaming data scientists for this. It's a combination of the expense of hardware and people to do data science, the amount of knowledge required to use existing technologies, and risk of failure that leads companies minimizing the amount of machine learning they use and maximizing their chances of success with the people they choose to use machine learning. What's more, there's a huge risk for companies that make important decisions on the results of machine learning and they want to minimize the risk of the outputs being wrong.

Data scientists study for years, working hard to obtain the knowledge necessary to create new algorithms -- machine learning models. It requires huge amounts of effort to gain that understanding, and even more effort to apply it in real life situations.

The idea that the average software developer could use a machine learning algorithm is laughable from the perspective of somebody who has a deep understanding of the math and how to build and configure models. It shouldn't be. The reason it is so hard is that the state of the technology is not accessible.

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


