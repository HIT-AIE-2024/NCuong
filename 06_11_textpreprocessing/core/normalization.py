import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary packages
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

class TextNormalizer:
    '''
        A class provides methods to stem and lemamatize a text, converting it to its original form. 
    '''
    def __init__(self):
        '''
            Inittial the PorterStemmer and WordNetLemmatizer object
        '''
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    
    def stemming(self, sentence: str) -> str:
        '''
            Stemming is a technique that reduces words to their roots by only removing suffixes, without considering the meaning of the words

            Parameters: 
                sentence: a paragraph or a text input
            
            Return:
                a stemming sentence
        '''
        # Separate sentences to text
        words = word_tokenize(sentence)
        
        # Apply normalize for each word 
        stemmed_words = [self.stemmer.stem(word) for word in words]

        return ' '.join(stemmed_words)
    

    def lemmatization(self, sentence: str, pos: str ='v') -> str:
        '''
            Lemmatization is a technique that reduces words to their roots by basing on the meaning of the words

            Parameters:
                sentence: a paragraph or a text input
                pos: part of speech tags. Valid options are `"n"` for nouns,
                `"v"` for verbs, `"a"` for adjectives, `"r"` for adverbs and `"s"`
                for satellite adjectives.

            Return:
                a lemmatizing sentence
        '''
        # Separate sentences to text
        words = word_tokenize(sentence)
        
        # Apply normalize for each word
        lemmatized_words = [self.lemmatizer.lemmatize(word, pos) for word in words]

        return ' '.join(lemmatized_words)



if __name__ == '__main__':
    #Example
    normalizer = TextNormalizer()
    sentence = "The cats are running quickly towards the building while the dogs were barking loudly."

    # Perform stemming and lemmatization for sentence
    stemmed_sentence = normalizer.stemming(sentence=sentence)
    lemmatized_sentence = normalizer.lemmatization(sentence=sentence)

    # Display results
    print(f'Original sentence: {sentence}')
    print(f'Stemming result: {stemmed_sentence}')
    print(f'Lemmatization result: {lemmatized_sentence}')