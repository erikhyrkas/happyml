# Code Style Standards
For this project, you'll notice the code style is all over the place. As somebody who programs in many 
languages throughout the same day, I fall into patterns of using CamelCase, snake_case, and pascalCase. 
This is not a good habit. 

I was tempted to follow the Intellij default style, but it used pascalCase in places that felt unnatural 
to me, like Class names. And if I broke away from its standard there, then the rules started getting messy.

I'm a big fan of simple rules that are easy to remember and follow, so here's my
attempt at a brief style guide.

Style Guide:

Case:
* `Constants` are `SCREAMING_SNAKE_CASE`.
* `Classes` are `CamelCase`.
* `Everything else` is lowercase `snake_case`.

Other rules:
* `Classes` are `nouns`.
* `Functions` start with `verbs`.
* Member variables are suffixed with underscore (`_`).

Final note: That last rule of suffixing member variables caused me a lot of consternation. I HATE the `m_` prefix
that is often used on member variables and the `t_` prefix on function parameters. I didn't want to have any prefix
or suffix at all, but I also hate having to come up with weird names for setter parameters and constructor parameters 
to ensure clarity on what is happening. 