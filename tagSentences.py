from flair.models import *
from flair.data import Sentence




class Tagger:
    def __init__(self, tagger):
        self.tagger = tagger

    def get_POSTAGS(self, sent):

        words = []
        pos_tags = []
        for i, label in enumerate(sent.get_labels('pos')):
            pos_tags.append(label.value)
            words.append(label.data_point.text)
        
        assert len(words) == len(pos_tags)
        return words, pos_tags, len(words)
    
    def get_NERTAGS(self, sent, N):
        ner_tags = N*['<UNK>']
        for label in sent.get_labels('ner'):
            extract = str(label)[:14]
            start = int(extract.split(']')[0].split('[')[1].split(':')[0])
            end = int(extract.split(']')[0].split('[')[1].split(':')[1]) - 1
            
            if start == end:
                ner_tags[start] = label.value
            else:
                ner_tags[start] = label.value
                ner_tags[end] = label.value

        return ner_tags

    def run(self, sentence):
        sent = Sentence(sentence)
        self.tagger.predict(sent)

        words, pos_tags, N = self.get_POSTAGS(sent)
        ner_tags = self.get_NERTAGS(sent, N)
        return words, pos_tags, ner_tags



if __name__ == '__main__':
    tagger = MultiTagger.load(['pos', 'ner'])
    tags = Tagger(tagger)
    words, pos_tags, ner_tags = tags.run("architecturally, the school has a catholic character. atop the main building\'s gold dome is a golden statue of the virgin mary.")
    print('Words: {}'.format(words))
    print('POS Tags: {}'.format(pos_tags))
    print('NER Tags: {}'.format(ner_tags))
    print(len(words), len(pos_tags), len(ner_tags))

