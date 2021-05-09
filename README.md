
# Projected-Hamming-Dissimilarity

Christian Hansen\*, Casper Hansen\*, Jakob Grue Simonsen, Christina Lioma. Projected Hamming Dissimilarity for Bit-Level Importance Coding in Collaborative Filtering. in Proceedings of the Web Conference 2021 (WWW'21). <br>
\* denotes equal contribution.
<br>

The structure of the code is as described below. The model is implemented in tensorflow 1.12 using python 3.6.
<br>
## data directory
- This contains the txt2mat.py script that converts the raw data (provided via the links within the paper) to .mat files.
- Use this link to download the datasets and splits used exactly in our paper: https://www.dropbox.com/s/v69u3l9e897egqp/data.zip?dl=0

## code directory
- This contains the implementation of our model.
- main.py is the main file, which runs and evaluates the model. Check the argparse arguments on line 97 for how to run on different datasets and different configurations.
- model.py contains the model implementation.
- tf_dataset_maker.py transforms .mat files into tfrecords used by the model.


