## Norebo
: 소설 등장인물의 감정과 성격을 분석하여, 그에 맞는 목소리를 부여하는 서비스 by Microsoft Azure

<p align="center">** 사이트 화면 및 프로젝트 소개 PPT **</p>
<p align="center">
  <img src="https://github.com/sinnybb/tts_norebo/assets/153700515/f085aa6e-107b-454a-8a58-12cb9bc54f4e" width="500" height="300" style="display:inline-block;"/>
  <img src="https://github.com/sinnybb/tts_norebo/assets/153700515/af158c11-7135-4426-a999-27d83c599bbb" width="500" height="300" style="display:inline-block;"/>
</p>

<hr>

## 목차
1. [소개](#소개)
2. [라이선스](#라이선스)
3. [출처](#출처)


## 소개
Norebo는 소설 등장인물의 감정과 성격을 분석하여, Microsoft Azure의 음성 합성 기술을 이용해 그에 맞는 목소리를 부여하는 서비스입니다. 
이 서비스는 소설을 더욱 생동감 있게 만들고, 독자들에게 새로운 경험을 제공하는 것을 목표로 합니다. 
본 프로젝트의 내용에서는 그 중 등장인물의 personality를 분석하고 이를 html로 보여주는 과정을 정리했습니다.
Ocean big5 BERT model을 통해 성격을 분류하고, 각 값들을 StandardScaler()로 스케일링한 후 정제하여 대표값을 찾아 인물의 성격을 추측합니다. 


## 라이선스
이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.


## 출처
이 프로젝트에서는 Hugging Face의 모델을 사용하였습니다. 아래는 사용한 모델의 출처입니다:

사용 모델: [Minej/bert-base-personality](https://huggingface.co/Minej/bert-base-personality)
내용 : 사람의 성격을 OCEAN big5로 구분하여 분석을 하게 됩니다. 아래는 구분되는 5가지 성격유형입니다.
- "Extroversion": 외향성
- "Neuroticism": 신경성
- "Agreeableness": 우호성
- "Conscientiousness": 성실성
- "Openness": 개방성

Hugging Face의 모델을 제공해주신 모든 분들께 감사드립니다.
