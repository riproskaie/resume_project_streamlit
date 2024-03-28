import streamlit as st
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertModel,
    BertConfig,
    Trainer,
    TrainingArguments,
    AdamW,
)
from transformers.optimization import get_cosine_schedule_with_warmup
from kobert_tokenizer import KoBERTTokenizer
from streamlit_option_menu import option_menu  # 따로 설치 필요함
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 모델 정의
class CustomBertModel(nn.Module):
    def __init__(self, bert_pretrained, dropout_rate=0.5):
        super(CustomBertModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_pretrained)
        self.dr = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        last_hidden_state = output["last_hidden_state"]
        x = self.dr(last_hidden_state[:, 0, :])
        x = self.fc(x)
        return x


# 지정한 모델을 가져옴
bert1_tokenizer = KoBERTTokenizer.from_pretrained(
    "skt/kobert-base-v1", last_hidden_state=True
)
bert2_tokenizer = AutoTokenizer.from_pretrained(
    "WhitePeak/bert-base-cased-Korean-sentiment"
)

bert1_model = BertModel.from_pretrained("skt/kobert-base-v1", return_dict=False)
bert2_model = CustomBertModel("WhitePeak/bert-base-cased-Korean-sentiment")

bert1_model = torch.load("bert-base-v3.pth")  # 학습된 모델 자체를 불러온다
# state_dict_old = ckpt['state_dict']
# state_dict = {}
# for key, value in state_dict_old.items():
#    key = key[7:]
#     state_dict[key] = value
# bert1_model.load_state_dict(state_dict, strict=False)

bert2_model.load_state_dict(
    torch.load("Personal_Statement_bert_base-v3_weights.pth"), strict=False
)
# 학습된 모델의 가중치만을 불러와서 모델에 넣는다

bert1_model.to(device)
bert2_model.to(device)

with st.sidebar:
    choice = option_menu(
        "모델",
        [
            "skt/bert-base-v1",
            "WhitePeak/bert-base-cased-Korean-sentiment",
        ],
        icons=["camera fill", "kanban"],
        menu_icon="app-indicator",
        default_index=0,
        styles={
            "container": {"padding": "4!important", "background-color": "#fafafa"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#fafafa",
            },
            "nav-link-selected": {"background-color": "#08c7b4"},
        },
    )


if choice == "skt/bert-base-v1":
    tokenizer = bert1_tokenizer
    model = bert1_model
elif choice == "WhitePeak/bert-base-cased-Korean-sentiment":
    tokenizer = bert2_tokenizer
    model = bert2_model

st.sidebar.markdown("개발자 GitHub - [Team Résumé](https://github.com/ru2zi/Resume)")

st.title("자소서 본인 확인 서비스")
user_input = st.text_area("여기에 자소서를 입력하세요", height=500)
button = st.button("분석")

d = {1: ":blue[적합한 자소서입니다.]", 0: ":red[부적합한 자소서입니다!]"}

if user_input and button:
    with torch.no_grad():
        inputs = tokenizer(
            [user_input],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        token_type_ids = torch.zeros_like(attention_mask).to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        _, preds = torch.max(outputs, dim=1)
        # y_pred = np.argmax(outputs.logits.detach().numpy(), axis=1)

    st.write(f"분석 결과: {d[preds.item()]}")
