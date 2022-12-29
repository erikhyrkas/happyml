
A 32-bit float has the general structure of: sign bit + 8 exponent bits + 23 mantissa bits

There is an extra hidden constant called bias that is 127, which we'll cover below.

The sign bit is really straight forward:
sign 0 = positive, sign 1 = negative

The exponent bits have a small twist:
The exponent bits can be converted to an unsigned int, but we have to subtract the bias to get the actual exponent.
The bias for a standard float is 127. We need that bias so that we can represent negative exponents, which
in turn lets us have fractions. In our future calculations, we are going to raise 2 to the power of this
exponent.

For our 8-bit float's exponent, we'll make the bias dynamic. A bias of 8 allows for reasonable accuracy between 0 and 1
where a bias of 0 allows larger possible numbers, but with lower accuracy.

The mantissa bits may require you to read this twice:
You've probably seen scientific notation before. A number like 6.23e10. That's 6.23 x 2^10.
We have en exponent already from the previous exponent bits, so we know what is going on
the right side of the 'e', but that number on the left of the e? That's the mantissa.

We don't have the mantissa itself yet, just some bits that we can use to calculate it.

The mantissa bits don't just hold a number we use like the 6.23 in our example of 6.23e10. These bits
hold a number written in base two that we have to decode and do math on to get to a number that you'd recognize.

What's might throw you is that the bits are on the right side of a decimal point, so the exponents we use
are going to take a moment to adjust to.

This has to do with having the most significant bits (the values that have the greatest impact) be the closest
to zero, which in our case is the left. We do this when we write 6.23. The two (2) is more significant than the 3.
We're doing the same thing in base 2.

Say we have a three bit mantissa, which we will.

You might think that you know how to read a binary number:
110 = 2^2 + 2^1 + 0   = 4+2+0 = 6
100 = 2^2 +   0 + 0   = 4+0+0 = 4
010 =   0 + 2^1 + 0   = 0+2+0 = 2
001 =   0 +   0 + 2^0 = 0+0+1 = 1
111 = 2^2 + 2^1 + 2^0 = 4+2+1 = 7

THIS IS WRONG! Unfortunately, the mantissa bits represent one plus the faction of a number.

Let's start with 100. This is actually a mantissa of 1.5. So, how do we get there?
The first thing to know is that the mantissa always has a constant of one added to it. The hidden bit.
The other thing we do is read each bit from left to right, like we are looking in a mirror, using
a negative number for the power.

So, if we have the bits 100, we have a single bit to the right of our constant whole number 1 (we could have
written the mantissa bits as 1.100) we do: 1 + 2^-1 = 1 + 0.5 = 1.5.

How about 001? We write in our constant leading number, so we have 1.001, then we do the math:
1 + 0 + 0 + 2^-3 = 1 + 0.125 = 1.125.

Let's do some more for practice:
bits: base 2 conversion:       decimal addition:        result:
100 = 1 + 2^-1 +    0 +    0 = 1 + 0.5                = 1.5
010 = 1 +    0 + 2^-2 +    0 = 1       + 0.25         = 1.25
001 = 1 +    0 +    0 + 2^-3 = 1              + 0.125 = 1.125
110 = 1 + 2^-1 + 2^-2 +    0 = 1 + 0.5 + 0.25         = 1.75
111 = 1 + 2^-1 + 2^-2 + 2^-3 = 1 + 0.5 + 0.25 + 0.125 = 1.875

Now that we have the mantissa, we can use it in a formula to build an actual number:
(-1^sign) * (2^exponent) * (mantissa)

My examples will focus on an 8-bit float, since that's easier to show the work for and
what we are building. A 32-bit float works the same, with just a bigger bias and more
bits.

Bits:        Math:                                         Show work:       Result:
1 1111 111 = special case                                                     =  NAN
0 1111 000 = special case                                                     =  Positive Infinity
1 1111 000 = special case                                                     =  Negative Infinity
1 0000 000 = super special case see below
For a bias of 0:
0 1110 111 = (-1^0) * (2^14) * (1 + 2^-1 + 2^-2 + 2^-3)  =  1 * 16384 * 1.875 =   30720
0 0001 001 = (-1^0) * (2^1)  * (1 + 2^-3)                =  1 * 2 * 1.125     =   2.25
0 0000 010 = (-1^0) * (2^0)  * (1 + 2^-2)                =  1 * 1 * 1.25      =   1.25
0 0000 001 = (-1^0) * (2^0)  * (1 + 2^-3)                =  1 * 1 * 1.125     =   1.125
0 0000 000 = (-1^0) * (2^0)  * 0                         =  1 * 1 * 0         =   0
1 1110 111 = (-1^1) * (2^14) * (1 + 2^-1 + 2^-2 + 2^-3)  = -1 * 16384 * 1.875 =  -30720

Super special note on 10000000.
When we try to represent numbers less than our bias can handle, something
interesting happens: For bias 0, let's take the floating point 1.0 and convert it. 1.0f
becomes 0 01111111 00000000000000000000000, well sometimes it becomes 0 01111111 00000000000000000000001, but
you get the point. It's 2^0 power + ~0. At bias 0 though, we can only make 0 0111 000, which is 1.25. So,
what happens if we round down? We get 0 0000 000, which is zero.

By default, 10000000 represents -0. Negative zero is mathematically equivalent to positive zero. It feels
terrible to waste this one permutation on something that is largely unused (there are some weird cases
where people have used -0 for their own evil purposes in some libraries, but I'm not inclined to support it.)

I'm inclined to use 10000000 to support something like 1/3 (0.33333...) or pi (3.14...) or euler's number (2.718...).

I don't expect people to use the 8-bit numbers to hold constants. I expect them to hold the results of math done
with 32-bit numbers. So, if they do 2*pi, I might need to hold that. Or if they raise e to the fourth power,
I might need to represent that. This means that I'm not just picking between a couple different common constants
but between values that I'll need to represent.

NOTE: Coming back to this. Bias 0 has no representation of 1 or -1.
SECOND NOTE: Only for Bias 0, I'm going to use 1 1110 110 to represent 1 and 1 0000 000 to represent -1.
We lose our second most negative number, but we can represent 1 which is highly useful.
THIRD NOTE: I've observed that representing the space between the smallest representable value and zero is
larger than other gaps and is problematic. For bias other than 0, I'm going to use 10000000 to represent half
of the current smallest value of that bias.


Offset
Let's talk for a moment about offset, which is my way of shifting the range of numbers that the 8-bit float will
represent. Bias impacts our granularity at the cost of it's maximum range. If we want to represent big-ish numbers
we can't have a high granularity (large bias) without doing something. By using an offset, we might be able
to achieve a decently high bias to allow good granularity, but only around that offset.

In general purpose computing, we may never know what a decent offset is, but in machine learning, we make many
passes trying to find the optimal numbers, honing our numbers as we go. If we find all of our numbers are at the
top end of our range that the bias we picked is using, by switching the offset on the next pass we can slowly
hone in on the area that we want to actually look at.

Clearly, this solution doesn't let us have our cake and eat it too. A high bias means a small range. We can
never have a large range and great granularity with this solution. My hope instinct is that this should be
fine for ML even if it isn't great for general computing.


UPDATE: I've decided to remove offset. For ML, the bias within the last layer already deals with offset
at a model level, and we don't need to account for it everywhere. It just makes things more complicated
without making them truly better.