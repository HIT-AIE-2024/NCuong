from nltk.tokenize import WordPunctTokenizer, TreebankWordTokenizer


class TextTokenizer:
    '''
        A class provides methods to whitespace, wordpunct, and treebank a text, separate it to smaller chunk
    '''
    def __init__(self):
        '''
            Initital WordPunctTokenizer and TreebankWordDetokenzier object
        '''
        self.wordpunct = WordPunctTokenizer()
        self.treebank = TreebankWordTokenizer()


    def whitespaceTokenizer(self, text: str) -> list:
        '''
            Tokenizes the text based on whitespace.

            Parameters:
                text: a paragraph or text input
            
            Returns:
                a list of tokens split by whitespace.
        '''
        return text.split()
    

    def wordpunctTokenizer(self, text: str) -> list:
        '''
            Tokenizes the text using WordPunctTokenizer

            Parameters:
                text: a paragraph or text input

            Returns:
                a list of tokens that splitted by all punctuation
        '''
        return self.wordpunct.tokenize(text)
    

    def treebankwordTokenizer(self, text: str) -> list:
        '''
            Tokenizes the text using treebankwordTokenizer

            Parameters:
                text: a paragraph or text input

            Returns:
                a list of tokens that splitted by contractions and punctuation
        '''
        return self.treebank.tokenize(text)
    

if __name__ == '__main__':
    #Example

    text = "This is an example sentence, for tokenization! Let's test it."
    tokenizer = TextTokenizer()

    #Perform the methods of tokenizer for sentence
    whitespaced_text = tokenizer.whitespaceTokenizer(text)
    wordPuncted_text = tokenizer.wordpunctTokenizer(text)
    treebanked_text = tokenizer.treebankwordTokenizer(text)

    #Display results
    print(f'Original text: {text}')
    print(f'Whitespace result: {whitespaced_text}')
    print(f'Wordpunct result: {wordPuncted_text}')      
    print(f'Treebank result: {treebanked_text}')
