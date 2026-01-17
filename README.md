# RNN DIY

I will use Sentimantic Classification on IMDb to test the RNN made by my own.

## Sentimantic Classification

### Task

Binary sentiment classification on movie reviews.

Given a review text as a time-ordered sequence of tokens:
- Input: a sequence of tokens (word indices) `X = (w1, w2, ... , wT)`
- Output: a binary label `y ∈ {0, 1}`
  - 0 = negative
  - 1 = positive

We train a recurrent model that reads tokens in order and predicts the sentiment
from the final hidden state (many-to-one sequence classification).

## Dataset
IMDb Large Movie Review Dataset (Large Movie Review Dataset v1.0):
- 25,000 training reviews
- 25,000 test reviews
- labels are balanced (pos/neg)
- raw text reviews with sentiment polarity labels

We load it via Hugging Face Datasets (`stanfordnlp/imdb`) for convenience.
(Original dataset is hosted by Stanford.) 


