# FsTsetlin

Implements a **Tsetlin machine** learning system in F#. The key difference between this and other Testlin machine implementations is that this library uses *tensor operations* to parallelize learning and prediction. FsTsetlin utilizes the tensor library underpinning TorchSharp/PyTorch. The libary has been tested to work on both CPU and GPU (although extensive performance testing has not been performed as of yet)

## Tsetline Machine 
Tsetlin machine (TM) is a recently developed machine learning system based on automata (finite state machines / propositional logic) learning. Please see [this paper](https://arxiv.org/abs/1804.01508) for details. 

TM is said to be competitive on many tasks with other ML methods (both DL & classical). However, the main draw is that TM-based models are faster to train and more energy-efficient when used for inferencing than models based on other ML methods - while providing similar accuracy. As ML becomes pervasive, the compute and power costs of deployed models becomes non-trivial. TM may help to reign in runtime and training costs associated with ML at scale.

Although this implementation is in F#, the goal is to define a largely language-agnostic computation approach that can be easily ported to other languages (e.g. Python, Java, etc.), as long as the language has a binding to libtorch - the C++ tensor library underlying PyTorch.

There are other GPU implementations available (see [github repo](https://github.com/cair/TsetlinMachine)). None of these use tensor operations from a standard tensor library. By using PyTorch as an established standard, the desire is to gain wider deployment portability. For example, the TM may be trained on a GPU but later deployed to a CPU-based environment for inferencing, to save costs.

## Examples

- [NoisyXor.fsx](/FsTsetlin/models/NoisyXor.fsx) (binary)
- [BinaryIris.fsx](/FsTsetlin/models/BinaryIris.fsx) (multiclass)

Datasets courtesy of https://github.com/cair/TsetlinMachine
