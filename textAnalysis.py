from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from scribeFiles import summary_text

def visualize_sentiment(text, custom_stopwords=None, figure_names=False):
    # Create a SentimentIntensityAnalyzer object
    analyzer = SentimentIntensityAnalyzer()

    # Create lists for sentences and their corresponding sentiment scores
    sentences = []
    compound_scores = []
    sentiment_classes = {"Positive": 0, "Neutral": 0, "Negative": 0}
    words = []

    # Get stop words
    stopwords_list = set(stopwords.words('english'))
    if custom_stopwords:
        stopwords_list.update(custom_stopwords)

    # Iterate over each sentence in the text
    for sentence in text.split('.'):
        sentences.append(sentence)
        sentiment = analyzer.polarity_scores(sentence)
        compound_scores.append(sentiment['compound'])
        

        if sentiment['compound'] > 0.2:
            sentiment_classes["Positive"] += 1
        elif sentiment['compound'] < 0.1:
            sentiment_classes["Negative"] += 1
        else:
            sentiment_classes["Neutral"] += 1

        # Collect words for word cloud
        words.extend([word for word in sentence.split() if word not in stopwords_list])

    # Normalize the sentiment class counts
    normalized_sentiment_classes = normalize_sentiment(sentiment_classes)

    # Define colors for sentiment classes
    sentiment_colors = {"Positive": "green", "Neutral": "gray", "Negative": "red"}

    # Plot sentiment polarity
    plt.figure(figsize=(10, 6))
    plt.plot(compound_scores)
    plt.xlabel('Sentence index')
    plt.ylabel('Sentiment Polarity')
    plt.title(f'{figure_names} Summary Sentiment Analysis')
    

    # Plot normalized sentiment classes
    plt.figure(figsize=(10, 6))
    plt.bar(normalized_sentiment_classes.keys(), normalized_sentiment_classes.values(),
            color=[sentiment_colors[key] for key in normalized_sentiment_classes.keys()])
    plt.title(f'Normalized {figure_names} Summary Sentiment Classification')
    if figure_names:
        plt.savefig(f'{figure_names}summary_sentiment_classification.png')
    plt.show()

    # Generate and plot word cloud
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stopwords_list,
                          min_font_size=10).generate_from_frequencies(Counter(words))

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.title(f'{figure_names} Key Phrases')
    if figure_names:
        plt.savefig(f'{figure_names}_word_cloud.png')
    plt.show()
    
    return compound_scores



custom_stopwords =['The', 'Dr', 'This', 'The', '.', 'I']

analyzed_text = visualize_sentiment_vader(summary_text, custom_stopwords, figure_names='Summary text')