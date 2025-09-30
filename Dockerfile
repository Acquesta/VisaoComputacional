# Use a imagem oficial do Python 3.9 como base.
# A tag 'slim' é uma versão mais leve, ideal para produção.
FROM python:3.9-slim

# Define o diretório de trabalho dentro do container.
WORKDIR /app

COPY requirements.txt .

# Instala as bibliotecas Python listadas no requirements.txt.
# A opção --no-cache-dir reduz o tamanho da imagem final.
RUN pip install --no-cache-dir -r requirements.txt

# Copia todos os arquivos do diretório atual (onde está o Dockerfile)
# para o diretório de trabalho dentro do container.
COPY . .

# Expõe a porta 5000 para permitir a comunicação com a aplicação Flask.
EXPOSE 5000

# O comando que será executado quando o container iniciar.
CMD ["python", "app.py"]
