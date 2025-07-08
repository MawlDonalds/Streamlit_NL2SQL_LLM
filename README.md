# Streamlit_NL2SQL_LLM
Aplication Convertion Natural Language To Query SQL With streamlit and langchain.

## Instalasi dan setup
1. Clone Repository
```
  git clone https://github.com/MawlDonalds/Streamlit_NL2SQL_LLM
```
2. buka terminal dan masukan sintax dibawah untuk menginstall requirement yang diperlukan
```
pip install -r requirementes.txt
```
3. ubah bagian google_api_key dengan menggunakan api gemini kalian
```
  google_api_key = "AIzaSyCg9s6LVrv5_z14QwAaGJnpYMIDU0lIjXc"
  if not google_api_key:
      raise ValueError("GOOGLE_API_KEY belum diatur")
```
4. sesuaikan dbname, user, password, host, dan port
**pastikan anda sudah memiliki database dan memiliki data didalamnya**
```
  db_config = {
      "dbname": "test_db",
      "user": "postgres",
      "password": "admin123",
      "host": "localhost",
      "port": "5432"
  }
```
### Cara RUN
```
  streamlit run app.py
```

--- 
*TERIMAKASIH SELAMAT MENCOBA*

