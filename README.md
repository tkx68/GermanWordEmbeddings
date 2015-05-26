# GermanWordEmbeddings
There has been a lot of research into training of word embeddings on English corpora. This toolkit applies deep learning via [word2vec](https://radimrehurek.com/gensim/models/word2vec.html) on German corpora to train and evaluate German models.

1. [Obtaining corpora](#obtention)
2. [Preprocessing](#preprocessing)
3. [Training models](#training)
3. [Vocabulary](#vocabulary)
3. [Evaluation](#evaluation)

## Obtaining corpora <a name="obtention"></a>
There are multiple possibilities for an obtention of huge German corpora that are public and free to use. For example the German Wikipedia:
```shell
wget http://download.wikimedia.org/dewiki/latest/dewiki-latest-pages-articles.xml.bz2
```
Or German news in 2007 to 2013:
```shell
for i in 2007 2008 2009 2010 2011 2012 2013; do
	wget http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.$i.de.shuffled.gz
done
```

## Preprocessing <a name="preprocessing"></a>
## Training models <a name="training"></a>
## Vocabulary <a name="vocabulary"></a>
## Evaluation <a name="evaluation"></a>