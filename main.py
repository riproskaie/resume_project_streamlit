import streamlit as st
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from streamlit_option_menu import option_menu  # 따로 설치 필요함
import torch

device = torch.device("cpu")


# 지정한 모델을 가져오는 함수
@st.cache_data
def get_model(mod):
    tokenizer = AutoTokenizer.from_pretrained(mod)
    model = AutoModelForSequenceClassification.from_pretrained(mod)
    m2w = {
        "jaehyeong/koelectra-base-v3-generalized-sentiment-analysis": "koelectra-base-v3.pth",
        "j5ng/kcbert-formal-classifier": "kcbert-formal-classifier.pth",
    }
    model.to(device)
    model.load_state_dict(torch.load(m2w[mod]))

    return tokenizer, model


with st.sidebar:
    choice = option_menu(
        "모델",
        [
            "jaehyeong/koelectra-base-v3-generalized-sentiment-analysis",
            "j5ng/kcbert-formal-classifier",
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

st.sidebar.markdown("개발자 GitHub - [Team Résumé](https://github.com/ru2zi/Resume)")

tokenizer, model = get_model(choice)

st.title("이력서 본인 확인 서비스")
user_input = st.text_area("여기에 자소서를 입력하세요", height=500)
button = st.button("분석")

d = {1: "Your work", 0: "Busted!"}

if user_input and button:
    test_sample = tokenizer(
        [user_input], padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    output = model(**test_sample)
    # st.write("Logits: ", output.logits)
    y_pred = np.argmax(output.logits.detach().numpy(), axis=1)
    st.write(f"Prediction: :blue[{d[y_pred[0]]}]")
