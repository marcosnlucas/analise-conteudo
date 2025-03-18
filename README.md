# Análise de Conteúdo SEO

Ferramenta de análise semântica de conteúdo para SEO usando embeddings e OpenAI.

## Deploy no Render

1. Crie uma conta no [Render](https://render.com)
2. Conecte seu repositório GitHub
3. Clique em "New Web Service"
4. Selecione o repositório
5. Configure as variáveis de ambiente:
   - `OPENAI_API_KEY`: Sua chave da API da OpenAI

O Render vai detectar automaticamente o projeto Python e usar o `requirements.txt` e `render.yaml` para configurar o ambiente.

## Desenvolvimento Local

1. Clone o repositório
2. Crie um ambiente virtual: `python -m venv venv`
3. Ative o ambiente: 
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Instale as dependências: `pip install -r requirements.txt`
5. Crie um arquivo `.env` com sua `OPENAI_API_KEY`
6. Execute: `python embedding.py`

## Estrutura do Projeto

- `embedding.py`: Aplicação principal Flask
- `requirements.txt`: Dependências Python
- `render.yaml`: Configuração do Render
- `gunicorn_config.py`: Configuração do servidor Gunicorn

## Limitações do Plano Gratuito do Render

- 512 MB RAM
- Spin down após 15 minutos de inatividade
- 750 horas de uso por mês
- Largura de banda limitada
