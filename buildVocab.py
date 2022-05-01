from collections import Counter
from flair.data import Sentence
import pandas as pd
from more_itertools import locate

def context_to_ids(text, word2idx):
      
        context_ids = [word2idx[word] for word in text]
        assert len(context_ids) == len(text)
        return context_ids

def makeLower(text):
    text = [t.lower() for t in text]
    return text

def getAnswer(a, b, N):
    for v_1 in b:
        for v_2 in a:
            if ((v_1 - v_2)+1) == N:
                return [v_2, v_1]
    return 0


class Vocab:
    def __init__(self, train_df, valid_df):
        self.train_df = train_df
        self.valid_df = valid_df

    def build_word_vocab(self,):
        list_df = [self.train_df, self.valid_df]
        words = []
        for df in list_df:
            for i, f in df.iterrows():
                w = f.loc['Small_Words']
                q = f.loc['Small_Questions']
                words += w
                words += q

        word_counter = Counter(words)
        word_vocab = sorted(word_counter, key=word_counter.get, reverse=True)
        word_vocab.insert(0, '<unk>')
        word_vocab.insert(1, '<pad>')
        word2idx = {word:idx for idx, word in enumerate(word_vocab)}
        idx2word = {v:k for k,v in word2idx.items()}
    
    
        return word2idx, idx2word, word_vocab
    

    def getErrors(self, df, idx2word):
        # Making sure tokenization is fine with our experiment
        error_indices = []
        for index, row in df.iterrows():
                answer_tokens = [w.text.lower() for w in Sentence(row.loc['answer'])]
                answer_start, answer_end = row.loc['label']
                answer_span = row.loc['context'][answer_start:answer_end].lower()
                new_answer_tokens = [w.text for w in Sentence(answer_span)]
                if (answer_tokens == new_answer_tokens):
                    continue
                else:
                    error_indices.append(index)

        #  Making sure answer indices are correct
        error_indices2 = []
        for index, row in df.iterrows():
            answer_tokens = [w.text.lower() for w in Sentence(row.loc['answer'])]
            words = row.loc['Small_Words']
            context_ids = row.loc['context_ids']
            try:
                indices = [words.index(a) for a in answer_tokens]
                ids4token = [context_ids[i] for i in indices]
                correct_tokens = [idx2word[i] for i in ids4token]
                if (correct_tokens == answer_tokens):
                    continue
                else:
                    error_indices2.append(index)
            except:
                error_indices2.append(index)
        
        xerr = error_indices + error_indices2
        err = list(set(xerr))
        return err
    def getAnsLabels(self, df):
        label_ids = []
        for index, row in df.iterrows():
            answer_tokens = [w.text.lower() for w in Sentence(row.loc['answer'])]
            words = row.loc['Small_Words']
            indices_start = list(locate(words, lambda x: x == answer_tokens[0]))
            indices_end = list(locate(words, lambda x: x == answer_tokens[-1]))
            label_id = getAnswer(indices_start, indices_end, len(answer_tokens))
            label_ids.append(label_id)
        return label_ids

    def run(self,):

        # Didnt do eval on Words column

        print(' 2.1 Converting Arrays from Strings to Arrays')
        self.train_df['Small_Words'] = self.train_df["Small_Words"].apply(eval)
        self.train_df['Pos_words'] = self.train_df["Pos_words"].apply(eval)
        self.train_df['Ner_words'] = self.train_df["Ner_words"].apply(eval)

        self.train_df['Questions'] = self.train_df["Questions"].apply(eval)
        self.train_df['Pos_qsts'] = self.train_df["Pos_qsts"].apply(eval)
        self.train_df['Ner_qsts'] = self.train_df["Ner_qsts"].apply(eval)

        self.valid_df['Small_Words'] = self.valid_df["Small_Words"].apply(eval)
        self.valid_df['Pos_words'] = self.valid_df["Pos_words"].apply(eval)
        self.valid_df['Ner_words'] = self.valid_df["Ner_words"].apply(eval)

        self.valid_df['Questions'] = self.valid_df["Questions"].apply(eval)
        self.valid_df['Pos_qsts'] = self.valid_df["Pos_qsts"].apply(eval)
        self.valid_df['Ner_qsts'] = self.valid_df["Ner_qsts"].apply(eval)

        self.train_df['Small_Questions'] = self.train_df["Questions"].apply(makeLower)
        self.valid_df['Small_Questions'] = self.valid_df["Questions"].apply(makeLower)


        print(' 2.2 Building Vocabulary')
        word2idx, idx2word, word_vocab = self.build_word_vocab()

        print(' 2.3 Converting to Tokens to Ids')
        self.train_df['context_ids'] = self.train_df['Small_Words'].apply(context_to_ids, word2idx=word2idx)
        self.train_df['question_ids'] = self.train_df['Small_Questions'].apply(context_to_ids, word2idx=word2idx)

        self.valid_df['context_ids'] = self.valid_df['Small_Words'].apply(context_to_ids, word2idx=word2idx)
        self.valid_df['question_ids'] = self.valid_df['Small_Questions'].apply(context_to_ids, word2idx=word2idx) 

        print(' 2.4 Removing Error Examples')
        err_ind = self.getErrors(self.train_df, idx2word)
        self.train_df.drop(err_ind, inplace=True)
        self.train_df.reset_index(drop=True, inplace=True)
        err_ind = self.getErrors( self.valid_df, idx2word)
        self.valid_df.drop(err_ind, inplace=True)
        self.valid_df.reset_index(drop=True, inplace=True)

        print(' 2.5 Getting Answer Indices')
        labels = self.getAnsLabels(self.train_df)
        self.train_df['label_idx'] = pd.Series(labels)

        labels = self.getAnsLabels(self.valid_df)
        self.valid_df['label_idx'] = pd.Series(labels)

        return self.train_df, self.valid_df, word2idx, idx2word, word_vocab

        # Also dropped the label_idx with 0 value seperately


if __name__ == '__main__':
    vocab = Vocab()

