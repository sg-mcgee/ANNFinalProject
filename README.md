# ANNFinalProject
The final project for my artificial neural networks class
The file must contain the images and labels files.
The images are contained in the "Project Divided Data.zip"

This is an attempt to build a Convolutional-Residual Neural Network to recognize my own personal handwriting. 
Initially, the CNN builds feature maps of the provided handwriting samples. This is followed by the RNN
interpreting the data based on the previous and future values at each column. This is decoded using a linear
layer. Finally, the CRC loss provided by pytorch allows for entire sentences to be decoded at once, rather than
individual letters.
