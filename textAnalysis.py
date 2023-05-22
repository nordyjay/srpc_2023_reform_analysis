from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from scribeFiles import group1_text, group2_text, group3_text, group4_text, group5_text, summary_text

def normalize_sentiment(sentiment_classes):
    # Calculate the total count of sentiment classes
    total_count = sum(sentiment_classes.values())

    # Normalize the sentiment class counts
    normalized_sentiment_classes = {
        label: count / total_count
        for label, count in sentiment_classes.items()
    }

    return normalized_sentiment_classes

def visualize_sentiment(text, custom_stopwords=None, figure_names=False):
    # Create a TextBlob object
    blob = TextBlob(text)

    # Create lists for sentences and their corresponding sentiment polarity
    sentences = []
    polarities = []
    sentiment_classes = {"Positive": 0, "Neutral": 0, "Negative": 0}
    words = []

    # Get stop words
    stopwords_list = set(stopwords.words('english'))
    if custom_stopwords:
        stopwords_list.update(custom_stopwords)
        print(stopwords_list)

    # Iterate over each sentence in the text
    for sentence in blob.sentences:
        sentences.append(str(sentence))
        sentiment = sentence.sentiment.polarity
        polarities.append(sentiment)

        if sentiment > 0:
            sentiment_classes["Positive"] += 1
        elif sentiment < 0:
            sentiment_classes["Negative"] += 1
        else:
            sentiment_classes["Neutral"] += 1

        # Collect words for word cloud
        words.extend([word for word in sentence.words if word not in stopwords_list])

    # Normalize the sentiment class counts
    normalized_sentiment_classes = normalize_sentiment(sentiment_classes)

    # Define colors for sentiment classes
    sentiment_colors = {"Positive": "green", "Neutral": "gray", "Negative": "red"}

    # Plot sentiment polarity
    plt.figure(figsize=(10, 6))
    plt.plot(polarities)
    plt.xlabel('Sentence index')
    plt.ylabel('Sentiment Polarity')
    plt.title(f'{figure_names} Raw Note Sentiment Analysis')
    #if figure_names:
    #    plt.savefig(f'{figure_names}summary_sentiment_polarity.png')
    #plt.show()

    # Plot normalized sentiment classes
    plt.figure(figsize=(10, 6))
    plt.bar(normalized_sentiment_classes.keys(), normalized_sentiment_classes.values(),
            color=[sentiment_colors[key] for key in normalized_sentiment_classes.keys()])
    plt.title(f'Normalized {figure_names} Raw Note Sentiment Classification')
    if figure_names:
        plt.savefig(f'{figure_names}raw_note_sentiment_classification.png')
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
    
    return polarities

group_texts = {'Group1 text': group1_text, 
               'Group2 text': group2_text, 
               'Group3 text': group3_text, 
               'Group4 text': group4_text, 
               'Group5 text': group5_text}

custom_stopwords =['The', 'Dr', 'This', 'The', '.', 'witt', 'I']
sent_list = []
for key, value in group_texts.items():
    analyzed_text = visualize_sentiment(value, custom_stopwords, figure_names=key)
    sent_list.append(analyzed_text)