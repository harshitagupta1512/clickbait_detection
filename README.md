## Objective
Implement a deep similarity-aware attentive model to capture and represent similarities between the misleading titles and the target content with better expressiveness.

Evaluate the effectiveness of the model on two real world datasets and compare it with series of baseline and state-of-the-arts methods.

## Methodology

The paper implements this task by following the framework of three parts: 
1) Learning latent representations, 
2) Learning the similarities,
3) Using the similarity for further predictions.

## Dataset

### Clickbait Challange
Full dataset can be found [here](https://zenodo.org/record/5530410)

### Fake News Challenge
Full dataset can be found [here](https://github.com/FakeNewsChallenge/fnc-1.git)

## Loading and Preprocessing

### Clickbait Challange
The dataset is in json Lines format, which contains the following fields:
```python
['id', 'postMedia', 'targetCaptions', 'postText', 'postTimestamp', 'targetTitle', 'targetDescription', 'targetKeywords', 'targetParagraphs', 'appendedTargetParagraphs','truthClass']
```

out of which we are interested in the following fields:
```python
['targetTitle', 'targetParagraphs', 'truthClass']
```

during preprocessing we remove the punctuations and stopwords from the text and convert the text to lower case next we perform stemming on the text.

This stemmed data is then used to create the word embeddings using the sentence_transformers library.

### Fake News Challenge
The dataset is in CSV format, which contains the following fields:
```python
['Body ID', 'articleBody','Headline', 'Body ID', 'Stance']
```
out of wich we are interested in the following fields:
```python
['articleBody', 'Headline', 'Stance']
```

during preprocessing we remove the punctuations and stopwords from the text and convert the text to lower case next we perform stemming on the text similar to the clickbait challange.

This stemmed data is then used to create the word embeddings using the sentence_transformers library.

## Work Distribution
- Preprocess FNC data - Siddharth
- Preprocess Clickbait dataset - Siddharth
- Implement vectorization - Harshita
- Encodings for FNC data - Dhruv, Siddharth
- Encodings for Clickbait data - Naval, Siddharth
- Map encodings for stances and bodies - Harshita
- Implement Bidirectional GRU layer - Naval, Dhruv
- Implement Attention with context layer - Naval, Harshita
- Global Similarity - Harshita
- Local Similarity - Harshita, Siddharth
- Train model (Learning the similarities) - Naval, Dhruv
- Preprocessing and encoding test data - Dhruv
- Testing the model - Naval, Dhruv
- Comparison with baseline methods - Harshita, Siddharth
- Report - Harshita, Siddharth, Dhruv
- Presentation - Harshita, Siddharth, Dhruv
