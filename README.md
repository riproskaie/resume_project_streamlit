# 자소서 본인 확인 서비스

BERT 계열 언어모델을 fine-tuning하여 구현한 자소서 본인확인 서비스입니다.

## 사용법

1. GPU 자원을 사용 가능한 Google Cloud Platform 인스턴스를 생성합니다. (모델과 가중치 파일은 폴더 내에 미리 저장되어 있어야 합니다)
2. 본 repo를 clone합니다.

```
git config --global credential.helper store
git clone https://github.com/riproskaie/resume_project_streamlit
```

3. GCP 환경에서 프로젝트 폴더 내로 이동 후 파이썬 가상환경을 설치합니다.

```
sudo apt-get update
sudo apt-get install python3.8-venv -y
cd resume_project_streamlit
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

4. `main.py` 파일을 `runOnSave` 옵션으로 실행합니다. 이때 GCP 방화벽 설정에서 streamlit 태그가 설정되어 있고 TCP 포트 8501이 활성화되어 있어야 합니다.

```
nohup streamlit run main.py --server.runOnSave true &
```

5. vscode 환경에서 ssh config 설정 파일을 엽니다. `Host`는 임의로 설정합니다. `HostName` 부분을 GCP 인스턴스의 외부 IP로 수정합니다. `IdentityFile`에는 `gcp_rsa_4096` 보안 키가 저장된 경로를 입력합니다. `User`는 GCP 계정에서 사용하는 아이디와 동일하게 설정해야 합니다.

```
Host GCP_T4
    HostName 104.199.238.124
    IdentityFile C:\\Users\r2com\.ssh\gcp_rsa_4096
    User sarasu.i.moo
```

6. 다음을 실행하여 streamlit을 활성화시킵니다.

```
cat nohup.out
```

7. `:8501`로 끝나는 외부 접속 주소가 "External URL"이라는 이름으로 생성되면, 이를 통해 접속합니다.
