import math
import string
from collections import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

# Download NLTK data if not already downloaded
nltk.download('punkt')

# Function to create a frequency matrix of words in each sentence
def create_frequency_matrix(sentences):
    freq_matrix = defaultdict(dict)
    for idx, sentence in enumerate(sentences):
        words = word_tokenize(sentence.lower())
        for word in words:
            word = word.strip(string.punctuation)
            if word:
                freq_matrix[idx][word] = freq_matrix[idx].get(word, 0) + 1
    return freq_matrix

# Function to calculate Term Frequency (TF)
def calculate_tf(freq_matrix):
    tf_matrix = {}
    for sentence, word_freqs in freq_matrix.items():
        total_words = sum(word_freqs.values())
        tf_matrix[sentence] = {word: count / total_words for word, count in word_freqs.items()}
    return tf_matrix

# Function to create a table for documents per word
def create_documents_per_word(freq_matrix):
    doc_word_count = defaultdict(int)
    for word_freqs in freq_matrix.values():
        for word in word_freqs:
            doc_word_count[word] += 1
    return doc_word_count

# Function to calculate Inverse Document Frequency (IDF)
def calculate_idf(documents_per_word, total_sentences):
    idf_matrix = {}
    for word, count in documents_per_word.items():
        idf_matrix[word] = math.log(total_sentences / (1 + count))
    return idf_matrix

# Function to calculate TF-IDF
def calculate_tf_idf(tf_matrix, idf_matrix):
    tf_idf_matrix = {}
    for sentence, word_freqs in tf_matrix.items():
        tf_idf_matrix[sentence] = {word: tf * idf_matrix[word] for word, tf in word_freqs.items()}
    return tf_idf_matrix

# Function to score sentences based on their TF-IDF values
def score_sentences(tf_idf_matrix):
    sentence_scores = {}
    for sentence, word_scores in tf_idf_matrix.items():
        sentence_scores[sentence] = sum(word_scores.values())
    return sentence_scores

# Function to find the threshold for sentence selection
def find_threshold(sentence_scores):
    return sum(sentence_scores.values()) / len(sentence_scores)

# Function to generate the summary
def generate_summary(sentences, sentence_scores, threshold):
    summary = []
    for idx, score in sentence_scores.items():
        if score >= threshold:
            summary.append(sentences[idx])
    return " ".join(summary)

# Input text
text = "In the last two years, the adoption of AI-driven Large Language Models (LLMs) has dramatically surged in various industries and applications. It mainly started with OpenAI debuting their state-of-the-art models like GPT-3, GTP-3.5, GTP-4, ChatGPT. This is followed by other major players such as Google introducing LaMDA, Bard, Gemini, Meta (Facebook) launching LLaMa, Anthropic releases Claude, Microsoft releases Co-pilot and Amazon rolls out Bedrock and Titan. These LLMs are pushing the boundaries of Natural Language Processing (NLP), helping industries in areas like automating customer service, enhancing content creation, improving data analysis with more efficiency."

print("Input Text: ")
print(text)
# Step 1: Tokenize sentences
sentences = sent_tokenize(text)

# Step 2: Create a frequency matrix of words in each sentence
frequency_matrix = create_frequency_matrix(sentences)

# Step 3: Calculate Term Frequency (TF)
tf_matrix = calculate_tf(frequency_matrix)

# Step 4: Create a table for documents per word
documents_per_word = create_documents_per_word(frequency_matrix)

# Step 5: Calculate Inverse Document Frequency (IDF)
idf_matrix = calculate_idf(documents_per_word, len(sentences))

# Step 6: Calculate TF-IDF
tf_idf_matrix = calculate_tf_idf(tf_matrix, idf_matrix)

# Step 7: Score the sentences
sentence_scores = score_sentences(tf_idf_matrix)

# Step 8: Find the threshold
threshold = find_threshold(sentence_scores)

# Step 9: Generate the summary
summary = generate_summary(sentences, sentence_scores, threshold)

# Output the summary
print("\nSummary: ")
print(summary)