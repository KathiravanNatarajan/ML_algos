import random

# Sample list of words (you can replace this with your own list)
word_list = # Enter your dataframe here

# Function to generate a random sentence with 1000 words
def generate_random_sentence(word_list):
    words_in_sentence = 1000
    selected_words = random.sample(word_list, words_in_sentence)
    random_sentence = ' '.join(selected_words)
    return random_sentence.capitalize() + '.'

# Generating a random sentence
random_sentence = generate_random_sentence(word_list)
print(random_sentence)
