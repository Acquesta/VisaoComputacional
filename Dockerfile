#Use uma imagem base oficial do PyTorch com suporte a CUDA 11.8
#Isso garante que todos os drivers e bibliotecas da NVIDIA estejam presentes
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

#Define o diretório de trabalho dentro do container
WORKDIR /app

#Copia o arquivo de dependências para o diretório de trabalho
COPY requirements.txt .

COPY . .

#Instala as dependências do Python
#O --no-cache-dir reduz o tamanho da imagem final
RUN pip install --no-cache-dir -r requirements.txt

#Copia o restante dos arquivos da aplicação (código, modelos, etc.)
COPY . .

#Expõe a porta em que o Flask/SocketIO irá rodar
EXPOSE 5000

#Comando para iniciar a aplicação quando o container for executado
CMD ["python", "app.py"]