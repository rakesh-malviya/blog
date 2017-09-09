---
title: '1. Word2vec Part 1: Basics'
layout: post
crosspost_to_medium: true
tags:
- NLP
---

For NLP with Deep learning there is need to represent text into data that can be understood by Neural networks.

## One Hot encoding

In this encoding we represent word vectors by zeros and ones. 

For e.g.  assume we have following text (or corpus): **"I like Deep learning and NLP"**


Than our vocabulary is as follows: 

| Word | Word |
| Index|      |
|:---|:-------|
| 0   | I   |
| 1   | like   |
| 2   | Deep  |
| 3   | learning   |
| 4   |  and  |
| 5   | NLP   |


Than **one-hot representation** of each word will be as follows:

| Word | One-hot|
|:---| :-------|
| I   | 100000 |
| like   | 010000 |
| Deep  | 001000 |
| learning   | 000100 |
|  and  | 000010 |
| NLP   | 000001 |

Notice the position of word in vocabulary and position of the bit set in its one hot representation

Under this representation, each word is Independent. It hard find its relationship with other words in Corpus. Because of this one hot is also called **Local representation**

## Word2Vec Intuition

**"You shall know a word by the company it keeps" - By J.R. Firth 1957"**   

You can get lot of value by representing a word by means of its neighbors. This way of representing a word given the distributions of its neighbors is called **Distributional similarity based representation**

Word2Vec is also called **Distributed representation** because of how it is different from "One Hot" representation which is local.

1. **government debt problems turning into** banking **crises as has happened in**  
2. **saying that Europe needs unified** banking **regulation to replace the hodgepodge**  
In above two sentences words in **bold** describe the word "banking"


**Warning Note:** Below example gives only a partial picture of how word2vec algorithm can understands similarity between words but this is not how this algorithm is implemented:

	Let us assume we have the following sentences in our text.

	1. Sam is **good** boy
	2. Sam is **fine** boy
	3. Sam is **great** boy

	Given the neighbor words "Sam", "is", and "boy" algorithms understands that "good", "fine", and "great" are similar words

Word2Vec algorithm can be implemented in 2 ways:

1. Skip Gram model
2. CBOW (Continuous Bag-of-word) Model

## Skip Gram model

Let us assume we have following text (or corpus): **"I like Deep learning and NLP"**

|Word vec|Word|
|----------|-------|
|$$w_0$$|**I**|
|$$w_1$$|**like**|
|$$w_2$$|**deep**|
|$$w_3$$|**learning**|
|$$w_4$$|**and**|
|$$w_5$$|**NLP**|

Skip gram defines a model that predicts context words given center word $$w_t$$. So the skip gram model trains to maximize probability of context word (neighbor words) given center words.   
i.e. $$ P(w_{c}  \vert  w_t) $$

For  t = 0  
center word, $$w_t$$ =$$w_0$$ = ("I")  
let window for neighbors is 5 so,  
context words = $$w_{c}$$ = ("like","deep")  

For t= 2:  
center word, $$w_t$$ = $$w_2$$ = ("deep")  
let window for neighbors is 5 so,  
context words = $$w_{c}$$ = ("I", "like", "learning", "and")  

So we can start by initialize random words vectors than adjust there values while training to maximize probability  $$ P(w_c \vert w_t) $$ or minimize loss function $$J =1- P(w_c \vert w_t) $$

#### Algorithm steps:
1. Look at different positions of $$t$$. 
2. Calculate loss function $$J =1- P(w_c \vert w_t) $$
3.  Adjust values of word vectors $$w_t$$ 

## CBOW (Continuous Bag-of-word) Model
CBOW defines a model that predicts center word $$w_t$$ given context words. So CBOW model trains to maximizeprobability of context word (neighbor words) given center words.   
i.e. $$ P(w_t \vert w_c) $$

    For us humans it is very easy to predict fill in the blanks like below:
    The Leopard ____ very fast
		
## Next ?
In next posts we will dive deep into word2vec derivations and algorithms

## References

1. Mikolov, Tomas, et al. "Distributed representations of words and phrases and their compositionality." Advances in neural information processing systems. 2013.

2. Stanford CS224n: Natural Language Processing with Deep Learning