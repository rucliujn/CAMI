# Overview #
This is an implementation of the Category-Aware Multi-Interest model (CAMI) for personalized product search. Please cite the following paper if you plan to use it for your project：

*	Jiongnan Liu, Zhicheng Dou, Qiannan Zhu, Ji-rong Wen. A Category-aware Multi-interest Model for Personalized Product Search. WWW 2022
    
### Requirements: ###
    1. To run the cami model in ./ProductSearch/ and the python scripts in ./utils/, python 3.0+ and Tensorflow v1.3+ are needed. (In the paper, we used python 3.6 and Tensorflow v1.4.0)
    2. To run the jar package in ./utils/AmazonDataset/jar/, JDK 1.7 is needed
    3. To compile the java code in ./utils/AmazonDataset/java/, galago from lemur project (https://sourceforge.net/p/lemur/wiki/Galago%20Installation/) is needed. 

### Data Preparation ###
Download the code and follow the ''Data Preparation'' section Download the code and follow the ''Data Preparation'' section in this [link](https://github.com/QingyaoAi/Explainable-Product-Search-with-a-Dynamic-Relation-Embedding-Model)

### Example Parameter Settings ###
We provide running bashes for all the four datasets we used in our paper.

### Reproduce Results ###
We provide the data partition used in our paper and the trainied model parameters for CAMI-h in .