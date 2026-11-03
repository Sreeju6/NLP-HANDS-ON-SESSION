import nltk
from nltk.corpus import inaugural
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download resources
nltk.download('inaugural')
nltk.download('stopwords')

# Obama speeches (2009 and 2013)
obama_files = ['2009-Obama.txt', '2013-Obama.txt']

# Trump speeches (2017 and 2025)
trump_files = ['2017-Trump.txt', '2025-Trump.txt']

# Combine speeches
obama_text = ' '.join(inaugural.words(fileids=obama_files))
trump_text = ' '.join(inaugural.words(fileids=trump_files))

# Convert to lowercase
obama_text = obama_text.lower()
trump_text = trump_text.lower()

# Stopwords
stop_words = set(stopwords.words('english'))

# Generate word clouds
obama_wc = WordCloud(width=800, height=400, background_color='white',
                     stopwords=stop_words).generate(obama_text)

trump_wc = WordCloud(width=800, height=400, background_color='white',
                     stopwords=stop_words).generate(trump_text)

# Display
plt.figure(figsize=(16,6))

plt.subplot(1,2,1)
plt.imshow(obama_wc, interpolation='bilinear')
plt.axis('off')
plt.title('Obama Inaugural Speeches (2009 & 2013)')

plt.subplot(1,2,2)
plt.imshow(trump_wc, interpolation='bilinear')
plt.axis('off')
plt.title('Trump Inaugural Speeches (2017 & 2025)')

plt.tight_layout()
plt.show()