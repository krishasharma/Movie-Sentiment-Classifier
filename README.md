# Movie Review Sentiment Classification

## Classifiers

Skeletons for the Naive Bayes and Logistic Regression classifiers you need to implement are in `classifiers.py`.
`main.py` wraps these and reports accuracy.

To run a baseline which always predicts `0` as the label, run:

```bash
python main.py --model "AlwaysPredictZero"
```

To run your NaiveBayes classifier, run:

```bash
python main.py --model "NaiveBayes"
```
To run your LogisticRegression classifier, run:

```bash
python main.py --model "LogisticRegression"
```


## Data

Data has been pre-split into training, dev, and test splits in `data/`, with a CSV file for each split.

This assignment uses the "Large Movie Review Dataset", containing 50,000 polar movie reviews (either positive or negative). We sub-sample these into train, dev, and test splits. This was dataset was introduced in the following paper:

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). [Learning Word Vectors for Sentiment Analysis](https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf). The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

More information about the dataset is available here: [https://ai.stanford.edu/~amaas/data/sentiment/](https://ai.stanford.edu/~amaas/data/sentiment/)

`download_and_split_data.py` Downloads the complete dataset and creates the splits used in this assignment. This has already been run for you and is not needed for your implementation.
