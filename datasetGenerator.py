import torch
# from flair.data import Sentence


class SquadDataset:
    def __init__(self, data, batch_size): 
        self.batch_size = batch_size
        data = [data[i:i+self.batch_size] for i in range(0, len(data), self.batch_size)]
        self.data = data
    
    def __len__(self,):
        return len(self.data)
    
    def cols(self,):
        print(self.data[0].columns)

        
    # def get_span(self, s):
    #     ss = Sentence(s)
    #     pos = 0
    #     span = []
    #     for v in ss:
    #         idx = s.find(v.text, pos)
    #         span.append((idx, idx + len(v.text)))
    #         pos = idx + len(v.text)
    #     return span
    
    def __iter__(self,):
          
        for batch in self.data:
            
            context_text = []
            answer_text = []
            pos_words, pos_qsts, ner_words, ner_qsts = [], [], [], []
            small_words, small_questions = [], []
            # spans = []
            
            # Get words texts 
            for w in batch.Small_Words:
                small_words.append(w)
            # batch_small_words = []
            # max_pos_len = max([len(p) for p in small_words])

            # for i in range(len(small_words)):
            #     batch_small_words.append(['<unk>']*max_pos_len)
            # for index, sent in enumerate(small_words):
            #     batch_small_words[index][:len(sent)] = sent[:]

            # Get small questions 
            for w in batch.Small_Questions:
                small_questions.append(w)
            # batch_small_questions = []
            # max_pos_len = max([len(p) for p in small_questions])

            # for i in range(len(small_questions)):
            #     batch_small_questions.append(['<unk>']*max_pos_len)
            # for index, sent in enumerate(small_questions):
            #     batch_small_questions[index][:len(sent)] = sent[:]

            # Get context and position spans 
            for ctx in batch.context:
                context_text.append(ctx)
                # spans.append(self.get_span(ctx))

            # Get answer texts 
            for ans in batch.answer:
                answer_text.append(ans)

            # Padding for Pos Tags
            batch_pos_words = []
            for a in batch.Pos_words:
                pos_words.append(a)
            max_pos_len = max([len(p) for p in pos_words])

            for i in range(len(pos_words)):
                batch_pos_words.append([100]*max_pos_len)
            for index, sent in enumerate(pos_words):
                batch_pos_words[index][:len(sent)] = sent[:]


            batch_pos_qsts = []
            for a in batch.Pos_qsts:
                pos_qsts.append(a)
            max_pos_len = max([len(p) for p in pos_qsts])

            for i in range(len(pos_qsts)):
                batch_pos_qsts.append([100]*max_pos_len)
            for index, sent in enumerate(pos_qsts):
                batch_pos_qsts[index][:len(sent)] = sent[:]


            # Padding for Ner Tags
            batch_ner_words = []
            for a in batch.Ner_words:
                ner_words.append(a)
            max_ner_len = max([len(p) for p in ner_words])

            for i in range(len(ner_words)):
                batch_ner_words.append([100]*max_ner_len)
            for index, sent in enumerate(ner_words):
                batch_ner_words[index][:len(sent)] = sent[:]
            

            batch_ner_qsts = []
            for a in batch.Ner_qsts:
                ner_qsts.append(a)
            max_ner_len = max([len(p) for p in ner_qsts])
            for i in range(len(ner_qsts)):
                batch_ner_qsts.append([100]*max_ner_len)
            for index, sent in enumerate(ner_qsts):
                batch_ner_qsts[index][:len(sent)] = sent[:]


            # Get padding to length of larges context
            max_context_len = max([len(ctx) for ctx in batch.context_ids])
            padded_context = torch.LongTensor(len(batch), max_context_len).fill_(1)
            
            # Fill padded context with context ids
            for i, ctx in enumerate(batch.context_ids):
                padded_context[i, :len(ctx)] = torch.LongTensor(ctx)
                
            # Pad questions with argest length
            max_question_len = max([len(ques) for ques in batch.question_ids])
            padded_question = torch.LongTensor(len(batch), max_question_len).fill_(1)
            
            # Fill padded question with question ids
            for i, ques in enumerate(batch.question_ids):
                padded_question[i, :len(ques)] = torch.LongTensor(ques)
            
            ids = list(batch.id)  
            label = torch.LongTensor(list(batch.label_idx))
                             
            yield (padded_context, padded_question, batch_pos_words, batch_pos_qsts, batch_ner_words, batch_ner_qsts, label, context_text, answer_text, ids, small_words, small_questions)
            
if __name__ == '__main__': 
    train_df = None
    train_dataset = SquadDataset(train_df, 2)
    next(iter(train_dataset))