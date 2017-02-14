# dfa cnn

Direct feedback alignment learning algorithm for multilayer feed forward network and multilayer convolutional neural network in golang. The learning algorithm is based on the paper from 
"Direct Feedback Alignment Provides Learning in
Deep Neural Networks" (https://arxiv.org/abs/1609.01596) and "http://www.breloff.com/no-backprop-part2/".


To install:

1. go get github.com/jonysugianto/dfa_cnn

2. a) download minist dataset from http://yann.lecun.com/exdb/mnist/
   b) unzip all zip files and put in to the folder: /tmp/minist

3. a) cd minist_example/minist_mlnn (minist recognition using multi layer feed forward network only)

   b) go build 

   c) ./minist_mlnn

4. a) cd minist_example/minist_cnn (minist recognition using multi layer convolutional neural network
                                   and multi layer feed forward network)

   b) go build 

   c) ./minist_cnn