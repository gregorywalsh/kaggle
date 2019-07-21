import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def get_wordnet_pos(self, treebank_tag):
    """
    return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v)
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        # As default pos in lemmatization is Noun
        return wordnet.NOUN

def pos_tag(self, tokens):
    # find the pos tagginf for each tokens [('What', 'WP'), ('can', 'MD'), ('I', 'PRP') ....
    # convert into feature set of [('What', 'What', ['WP']), ('can', 'can', ['MD']), ... ie [original WORD, Lemmatized word, POS tag]
    pos_tokens = [nltk.pos_tag(token) for token in tokens]
    lemmatizer = WordNetLemmatizer()
    pos_tokens = [[(word, lemmatizer.lemmatize(word, self.get_wordnet_pos(pos_tag)), [pos_tag]) for (word, pos_tag)
                   in pos] for pos in pos_tokens]
    return pos_tokens
