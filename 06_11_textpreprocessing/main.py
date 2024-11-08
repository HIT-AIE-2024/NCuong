from core.normalization import TextNormalizer
from core.tokenization import TextTokenizer


if __name__ == '__main__':
    #Example
    text = "This is an example sentence, for tokenization! Let's test it."

    #Tokenization
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


    # Normalization
    normalizer = TextNormalizer()

    # Perform stemming and lemmatization for sentence
    stemmed_sentence = normalizer.stemming(sentence=text)
    lemmatized_sentence = normalizer.lemmatization(sentence=text)

    # Display results
    print(f'Original sentence: {text}')
    print(f'Stemming result: {stemmed_sentence}')
    print(f'Lemmatization result: {lemmatized_sentence}')