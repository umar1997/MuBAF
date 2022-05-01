import pandas as pd
import json


def getTagDF(file_name):
    Tags = pd.read_csv(file_name)
    Tags.drop(columns=Tags.columns[0], axis=1, inplace=True)
    Tags. rename(columns = {'Ids':'id'}, inplace = True)
    return Tags



class DataPreparation:
    def __init__(self, path):
        self.path = path

    def load_json(self,):

        with open(self.path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return data

    def describe_info(self, data):

        print("Length of data: ", len(data['data']))
        print("Data Keys: ", data['data'][0].keys())
        print("Title: ", data['data'][0]['title'])


    def parse_data(self, data):

        data = data['data']
        qa_list = []

        for paragraphs in data:

            for para in paragraphs['paragraphs']:
                context = para['context']

                for qa in para['qas']:
                    
                    id = qa['id']
                    question = qa['question']
                    
                    for ans in qa['answers']:
                        answer = ans['text']
                        ans_start = ans['answer_start']
                        ans_end = ans_start + len(answer)
                        
                        qa_dict = {}
                        qa_dict['id'] = id
                        qa_dict['context'] = context
                        qa_dict['question'] = question
                        qa_dict['label'] = [ans_start, ans_end]

                        qa_dict['answer'] = answer
                        qa_list.append(qa_dict)    

        return qa_list

    def preprocess_df(self, df):
        
        def to_lower(text):
            return text.lower()

        df.context = df.context.apply(to_lower)
        df.question = df.question.apply(to_lower)
        df.answer = df.answer.apply(to_lower)
    
        return df

    def run(self,):

        data = self.load_json()
        data_list = self.parse_data(data)
        data_df = pd.DataFrame(data_list)
        data_df = self.preprocess_df(data_df)

        return data_df


if __name__ == '__main__':
    dataPrep = DataPreparation('./data/Squad/train-v1.1.json')
    df = dataPrep.run()
    print('Length of data: {}'.format(len(df)))
