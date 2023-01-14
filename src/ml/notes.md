# Random Notes

I doubt there is much accurate in this file.

Just tracking some of my thoughts. Some of these are from day one and some are not, and they need to be cleaned up at some point to be consumable by a larger audience as anything useful. Right now, it's likely only meaningful to me and useful to pretty much nobody (including me.)

I probably could delete this file and nobody would care.

## High-level Design

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

Data Decoder (doesn't exist yet):
* Converts a tensor into usable data (similar to original raw data)

Data Source (doesn't exist yet):
* Zero or more Data Sets (could generate data)
* Data Encoder -- used at this point so that we don't have to constantly re-encode input each epoch
  and saves memory by not needing raw data beyond the initial moment to encode.

Optimizer:
* Use a loss function to compute the accuracy of a prediction
* Manage state tensors on layer used for optimizing

Neural Network Node:
* Weights
* Bias (optional)
* Optimizer state tensors
* Activation function
* Regularizer
* takes in Loss function
* Specific input and output shape (shape = tensor dimensions)

* Model:
* represents a DAG (directed acyclic graph) of layer/layer blocks and checks for cycles when adding new layers
* Layers
* Optimizer
* Decoder -- convert raw tensor outputs to data similar to raw data
* Trained using a Data Source, which configures the input shape
* Produces one or more outputs