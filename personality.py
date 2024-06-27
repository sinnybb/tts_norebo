import time
import pandas as pd

from transformers import AutoTokenizer, pipeline, BertTokenizer, BertForSequenceClassification
from nltk.corpus import stopwords

from concurrent.futures import ThreadPoolExecutor

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import re

from sklearn.preprocessing import StandardScaler
import torch

import multiprocessing as mp
print(mp.cpu_count(), mp.current_process().name)

class PersonalityModel:
    def __init__(self):
        # GPU 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 모델 및 토그나이져 설정
        self.tokenizer = BertTokenizer.from_pretrained("Minej/bert-base-personality")
        self.bert_model = BertForSequenceClassification.from_pretrained("Minej/bert-base-personality").to(self.device)
        self.scaler = StandardScaler()

        # 주요/비주요 crt df
        self.character_contents = {} 
        self.character_contents_n = {}
        self.n_character = []
        self.script_num = {}

    def pre_character_content(self,db_data):
        db_data.rename(columns = {'character' : 'crt_name'}, inplace = True)
        db_data.rename(columns = {'contents' : 'content'}, inplace = True)

        characters = db_data['crt_name'].unique()
        
        # 특수문자가 포함된 등장인물 이름에 대한 이스케이프 처리
        self.characters = [character.replace("'", "\'") if isinstance(character, str) else character for character in characters]
        self.novel = db_data[['crt_name', 'content']]

        # 전처리는 수행
        self.novel['content'] = self.novel['content'].apply(self.preprocess_text)

    @staticmethod
    def preprocess_text(text):
        if not isinstance(text, str):
            text = ''
        else:
            text = text.lower()
            text = re.sub(r'\W', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_character_contents(self, character):
        # start_time = time.time()

        crt_contents = self.novel[self.novel['crt_name'] == character]
        if character != 'narrator':
            character_dialogues = crt_contents['content'].tolist()
            character_dialogues = [self.preprocess_text(dialogue) for dialogue in character_dialogues]
            self.character_contents[character] = character_dialogues
            self.script_num[character]=len(character_dialogues)# 대사 수를 추가

        else :
            self.character_contents[character] = None
            self.n_character.append(character)
            self.script_num[character]=0  #narrator: 대사 수 0
    
        # end_time = time.time()
        # print(f"Extracting dialogues for {character}: {end_time - start_time} seconds")

    def personality_detection(self, texts):
        # start_time = time.time()

        # 배치 처리를 위해 토큰화 및 패딩
        inputs = self.tokenizer(texts, truncation=True, padding=True, return_tensors="pt", max_length=512).to(self.device)
        
        with torch.no_grad():
            # 모델로부터 예측값 계산
            outputs = self.bert_model(**inputs)
        predictions = outputs.logits.detach().cpu().numpy()  # 각 텍스트에 대한 예측값 가져오기

        # end_time = time.time()
        # print(f"Personality detection time: {end_time - start_time} seconds")
        return predictions

    def loop_dialogue(self, dialogues):
        # start_time = time.time()

        # 배치 사이즈 설정
        batch_size = 8
        num_batches = (len(dialogues) + batch_size - 1) // batch_size
        predictions = []

        for i in range(num_batches):
            # 배치 생성
            batch_texts = dialogues[i * batch_size: (i + 1) * batch_size]
            batch_predictions = self.personality_detection(batch_texts)
            predictions.append(batch_predictions)

        # 예측 결과를 하나로 병합
        predictions = np.concatenate(predictions, axis=0)
        label_name = ['Var_E', 'Var_N', 'Var_A', 'Var_C', 'Var_O']
        df = pd.DataFrame(predictions, columns=label_name)

        end_time = time.time()
        # print(f"Loop dialogue time: {end_time - start_time} seconds")
        return df

    def content_by_character(self):
        characters = [character for character in self.characters if character != 'narrator']
        with ThreadPoolExecutor() as executor:
            list(executor.map(self.extract_character_contents, characters))
    
    def feature_scaled_mode(self):
        # start_time = time.time()

        # 캐릭터 대사 추출
        with ThreadPoolExecutor() as executor:
            executor.map(self.extract_character_contents, self.characters)

        character_ocean = {}  # script num 계산하기
        for character, dialogues in self.character_contents.items():
            if dialogues:
                ocean_df = self.loop_dialogue(dialogues)
                std_df = np.std(ocean_df, axis=0).round(2)

                character_ocean[character] = std_df.tolist()

        character_ocean_df = pd.DataFrame.from_dict(character_ocean, orient='index', columns=['Var_E', 'Var_N', 'Var_A', 'Var_C', 'Var_O'])
        ocean_scaled_array = self.scaler.fit_transform(character_ocean_df)
        ocean_scaled = np.round(ocean_scaled_array, 2)
        character_ocean_scaled = pd.DataFrame(ocean_scaled, index=character_ocean_df.index, columns=character_ocean_df.columns)

        for n_character in self.n_character:
            character_ocean_scaled.loc[n_character] = [-1,-1,-1,-1,-1]

        # script_num을 DataFrame으로 변환
        script_num_df = pd.DataFrame(self.script_num,index=['script_num']).T
        character_ocean_scaled = pd.concat([character_ocean_scaled, script_num_df],axis=1)
        # end_time = time.time()
        # print(f"Feature scaled mode time: {end_time - start_time} seconds")             

        return character_ocean_scaled
    
    @staticmethod
    def filtering(row, bk_num):
        ocean_values = ['Var_E', 'Var_N', 'Var_A', 'Var_C', 'Var_O']
        speed = [] 
        pitch = [] 

        for ocean_value in ocean_values:
            if row.get(ocean_value, -1) != -1:  
                value = row[ocean_value]
                if ocean_value in ['Var_E', 'Var_O']:  
                    if value >= 0.75:
                        speed.append(3) 
                        pitch.append(3)
                    elif value >= 0.50:
                        speed.append(2)
                        pitch.append(2) 
                    elif value >= 0.25:
                        speed.append(1) 
                        pitch.append(1)
                    else:
                        speed.append(0)
                        pitch.append(0)
                elif ocean_value == 'Var_N':  
                    if value >= 0.70:
                        speed.append(2) 
                        pitch.append(-2)
                    elif value >= 0.40:
                        speed.append(1)
                        pitch.append(-1)
                    else:
                        speed.append(0) 
                        pitch.append(0)
                elif ocean_value == 'Var_A':  
                    if value >= 0.70:
                        speed.append(2) 
                        pitch.append(-2)
                    elif value >= 0.40:
                        speed.append(1)
                        pitch.append(-1)
                    else:
                        speed.append(0) 
                        pitch.append(0)
                elif ocean_value == 'Var_C':  
                    if value >= 0.50:
                        speed.append(-2) 
                        pitch.append(0)
                    elif value >= 0.25:
                        speed.append(-1) 
                        pitch.append(0)
                    else:
                        speed.append(0)
                        pitch.append(0)
            else:
                speed.append(0)
                pitch.append(0)

        result = pd.Series({'bk_num': str(bk_num), 'speed': np.mean(speed), 'pitch': np.mean(pitch),'script_num':row['script_num']})
        return result

    @staticmethod
    def append_gender(ocean_df, p_model_db, characters):
        for character in characters:
            if character in ocean_df.index and character in p_model_db.index:
                if ocean_df.loc[character, 'crt_name'] == p_model_db.loc[character, 'crt_name']:
                    ocean_df.loc[character, 'gender'] = p_model_db.loc[character, 'gender']
                else:
                    ocean_df.loc[character, 'gender'] = None  # 캐릭터 이름이 다를 때도 추가하지 않음
            else:
                ocean_df.loc[character, 'gender'] = None  # 캐릭터가 없을 때도 추가하지 않음
        return ocean_df


# if __name__ == '__main__':
    # start = time.time()
    # data = pd.read_excel('novel.xlsx')
    # p_model = PersonalityModel()   
    # p_model.pre_character_content(data)

    # # 등장인물 변수 생성  # 'narrator'인 경우에 제외
    # characters = [character for character in p_model.characters if character != 'narrator']

    # # `extract_character_contents` 메서드 호출을 통해 등장인물별 대화 내용 추출
    # with ThreadPoolExecutor() as executor:
    #     result = list(executor.map(p_model.extract_character_contents, characters))
    # # 등장인물별 성격 특성 분석 수행
    # character_ocean = pd.DataFrame(p_model.feature_scaled_mode())

    # end = time.time()
    # print(character_ocean)
    # print(f'** Personality model total time ',round((end-start)/60, 2),' minutes **')
    
    # character_ocean.to_excel('character_personality_final.xlsx')



# if __name__ == '__main__':
#     data = pd.read_excel('novel.xlsx')
#     data.columns=['chapter', 'crt_name', 'content']

#     e_model = EmotionModel()
#     pre_data = e_model.preprocess_data(data)
#     match_emotion_voice = e_model.match_emotion_voice(e_model.emotion_analysis_loop(pre_data))

#     data['emotion_voice']=match_emotion_voice['emotion_voice']
#     print(data)