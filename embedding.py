import os
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
import openai
from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap
import unidecode
import re
from sklearn.cluster import KMeans
import scipy.stats as stats
from dotenv import load_dotenv
import seaborn as sns

# Carrega variáveis de ambiente
load_dotenv()

# Configuração do OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

# Configuração do Flask
app = Flask(__name__)
CORS(app)

# Configuração do Matplotlib
plt.style.use('default')  # Mudando para o estilo default em vez de seaborn
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 100

# Encoder personalizado para numpy
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app.json_encoder = NumpyEncoder

def get_embedding(text):
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-3-small"
        )
        
        if not response or not response.data or len(response.data) == 0:
            return None
            
        return response.data[0].embedding
    except Exception as e:
        print(f"Erro ao gerar embedding: {str(e)}")
        return None

def analyze_text(text):
    try:
        # Obtém embedding
        embedding = get_embedding(text)
        if embedding is None:
            return None
            
        embedding_array = np.array(embedding)
        
        # Métricas básicas
        metrics = {
            'dimension_count': len(embedding),
            'mean_value': float(np.mean(embedding_array)),
            'std_dev': float(np.std(embedding_array)),
            'min_value': float(np.min(embedding_array)),
            'max_value': float(np.max(embedding_array)),
            'median_value': float(np.median(embedding_array)),
            'abs_mean': float(np.mean(np.abs(embedding_array))),
            'positive_count': int(np.sum(embedding_array > 0)),
            'negative_count': int(np.sum(embedding_array < 0)),
            'zero_count': int(np.sum(embedding_array == 0)),
            'significant_dims': int(np.sum(np.abs(embedding_array) > 0.1)),
            'skewness': float(stats.skew(embedding_array)),
            'kurtosis': float(stats.kurtosis(embedding_array)),
            'entropy': float(stats.entropy(np.abs(embedding_array))),
            'variance': float(np.var(embedding_array)),
            'percentile_25': float(np.percentile(embedding_array, 25)),
            'percentile_75': float(np.percentile(embedding_array, 75)),
            'iqr': float(np.percentile(embedding_array, 75) - np.percentile(embedding_array, 25)),
            'coeficiente_variacao': float(np.std(embedding_array) / np.mean(np.abs(embedding_array))),
            'densidade_semantica': float(np.sum(np.abs(embedding_array)) / len(embedding_array)),
            'complexidade_lexical': float(np.std(np.abs(embedding_array))),
            'equilibrio_semantico': float(np.abs(np.mean(embedding_array))),
            'distribuicao_tematica': float(stats.entropy(np.abs(embedding_array) + 1e-10))
        }
        
        # Análise de clusters
        n_clusters = min(5, len(embedding) // 20)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embedding_array.reshape(-1, 1))
        
        # Organiza clusters mantendo a ordem original
        clusters = []
        cluster_strengths = []  # Lista para armazenar forças dos clusters
        
        for i in range(n_clusters):
            mask = cluster_labels == i
            cluster_values = embedding_array[mask]
            cluster_indices = np.where(mask)[0]
            
            # Calcula força relativa
            força_relativa = float(np.abs(np.mean(cluster_values)) / np.mean(np.abs(embedding_array)))
            cluster_strengths.append((i, força_relativa))
            
            # Extrai trecho do texto relacionado ao cluster
            start_pos = int(len(text) * (min(cluster_indices) / len(embedding)))
            end_pos = int(len(text) * (max(cluster_indices) / len(embedding)))
            text_chunk = text[max(0, start_pos-100):min(len(text), end_pos+100)]
            
            # Encontra a frase mais próxima do centro do cluster
            center_pos = (start_pos + end_pos) // 2
            sentences = re.split('[.!?]+', text)
            closest_sentence = ""
            min_distance = float('inf')
            current_pos = 0
            
            for sentence in sentences:
                sentence_len = len(sentence)
                sentence_center = current_pos + sentence_len // 2
                distance = abs(sentence_center - center_pos)
                if distance < min_distance:
                    min_distance = distance
                    closest_sentence = sentence.strip()
                current_pos += sentence_len
            
            cluster = {
                'original_id': i,  # Mantém o ID original para referência
                'size': int(np.sum(mask)),
                'avg_value': float(np.mean(cluster_values)),
                'std_dev': float(np.std(cluster_values)),
                'min_value': float(np.min(cluster_values)),
                'max_value': float(np.max(cluster_values)),
                'start_dim': int(min(cluster_indices)),
                'end_dim': int(max(cluster_indices)),
                'força_relativa': força_relativa,
                'entropia': float(stats.entropy(np.abs(cluster_values))),
                'densidade': float(np.sum(np.abs(cluster_values)) / len(cluster_values)),
                'trecho_texto': text_chunk,
                'frase_central': closest_sentence,
                'palavras_chave': [],
                'tema': None,
                'dimensoes': cluster_indices.tolist()
            }
            
            # Classifica o cluster
            if np.mean(cluster_values) > 0:
                cluster['tipo'] = "Presença Forte"
            else:
                cluster['tipo'] = "Ausência Notável"
                
            if cluster['força_relativa'] > 1.5:
                cluster['relevância'] = "Alta"
            elif cluster['força_relativa'] > 0.75:
                cluster['relevância'] = "Média"
            else:
                cluster['relevância'] = "Baixa"
            
            clusters.append(cluster)
        
        # Ordena cluster_strengths por força relativa
        cluster_strengths.sort(key=lambda x: x[1], reverse=True)
        
        # Cria nova lista de clusters ordenada
        ordered_clusters = []
        for rank, (original_id, _) in enumerate(cluster_strengths):
            cluster = next(c for c in clusters if c['original_id'] == original_id)
            cluster['id'] = rank + 1  # Agora o mais forte é #1, segundo mais forte #2, etc.
            ordered_clusters.append(cluster)
        
        return {
            'metrics': metrics,
            'clusters': ordered_clusters,
            'embedding': embedding_array.tolist(),
            'cluster_labels': cluster_labels.tolist()
        }
        
    except Exception as e:
        print(f"Erro na análise: {str(e)}")
        return None

def analyze_with_openai(clusters):
    try:
        # Prepara os dados dos clusters para a análise
        clusters_text = []
        for cluster in clusters:
            cluster_text = (
                f"Grupo {cluster['id']}: {cluster['tipo']} {cluster['relevância']}\n"
                f"- Força: {cluster['força_relativa']:.2f}x média\n"
                f"- Tamanho: {cluster['size']} características\n"
                f"- Valor médio: {cluster['avg_value']:.4f}\n"
                f"- Valor máximo: {cluster['max_value']:.4f}\n"
                f"- Entropia: {cluster['entropia']:.4f}\n"
                f"- Densidade: {cluster['densidade']:.4f}\n"
                f"- Trecho relevante: '{cluster['trecho_texto']}'\n"
                f"- Frase central: '{cluster['frase_central']}'"
            )
            clusters_text.append(cluster_text)

        # Junta todos os clusters em um único texto
        all_clusters = "\n\n".join(clusters_text)

        # Prompt para a OpenAI analisar os clusters
        prompt = f"""Analise os seguintes grupos de características semânticas encontrados no texto:

{all_clusters}

Forneça uma análise detalhada em português do Brasil, incluindo:

1. Uma análise detalhada de cada grupo, explicando:
   - O que cada grupo representa
   - A importância relativa do grupo
   - Como ele se relaciona com o texto
   - Insights específicos baseados nos valores numéricos

2. Uma análise geral que:
   - Explique como os grupos se relacionam entre si
   - Identifique padrões importantes
   - Destaque insights relevantes para SEO

Por favor, estruture sua resposta em duas seções:
1. "Análise dos Grupos" - com a análise individual de cada grupo
2. "Análise Geral" - com a análise do conjunto e recomendações

Use uma linguagem clara e profissional, mantendo o foco em insights acionáveis."""

        # Faz a chamada para a API
        response = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "Você é um especialista em análise semântica de texto e SEO."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )

        # Extrai a resposta
        analysis = response.choices[0].message.content

        # Formata a resposta em seções
        formatted_analysis = {
            'grupos': analysis.split('Análise Geral')[0].strip(),
            'recomendacoes': 'Análise Geral' + analysis.split('Análise Geral')[1].strip()
        }

        return formatted_analysis

    except Exception as e:
        print(f"Erro na análise OpenAI: {str(e)}")
        return None

def save_plot_to_base64():
    """Salva o gráfico atual em formato base64"""
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_plots(analysis, text):
    try:
        plots = {}
        embedding = np.array(analysis['embedding'])
        print("Gerando gráficos...")
        
        # 1. Overview plot
        print("1. Gerando visão geral...")
        plt.figure(figsize=(12, 6))
        plt.plot(embedding, alpha=0.5)
        plt.title('Visão Geral do Embedding')
        plt.xlabel('Dimensão')
        plt.ylabel('Valor')
        plt.grid(True, alpha=0.3)
        plots['overview'] = save_plot_to_base64()
        plt.close()
        
        # 2. Top dimensions plot
        print("2. Gerando dimensões principais...")
        plt.figure(figsize=(12, 6))
        abs_values = np.abs(embedding)
        top_indices = np.argsort(abs_values)[-20:]
        top_values = embedding[top_indices]
        
        colors = ['#2ca02c' if v > 0 else '#d62728' for v in top_values]
        plt.bar(range(len(top_values)), top_values, color=colors)
        plt.title('Top 20 Dimensões mais Significativas')
        plt.xlabel('Número da Característica')
        plt.ylabel('Intensidade')
        plt.xticks(range(len(top_values)), top_indices)
        
        # Correlaciona com palavras do texto
        print("2.1 Gerando correlações...")
        words = text.split()
        correlations = []
        for idx in top_indices:
            pos = int(len(words) * (idx / len(embedding)))
            start = max(0, pos - 5)
            end = min(len(words), pos + 6)
            context = ' '.join(words[start:end])
            correlations.append(f"Dim {idx}: '{context}'")
            
        plots['top_dimensions'] = save_plot_to_base64()
        plt.close()
        
        # 3. Heatmap plot
        print("3. Gerando mapa de calor...")
        plt.figure(figsize=(12, 8))
        try:
            matrix_size = (32, 48)  # 32 * 48 = 1536 (tamanho do embedding)
            reshaped_embedding = embedding.reshape(matrix_size)
            
            # Criar o heatmap
            sns.set_style("whitegrid")
            cmap = sns.diverging_palette(220, 20, as_cmap=True)  # Do azul ao vermelho
            
            sns.kdeplot(
                x=reshaped_embedding[:, 0],
                y=reshaped_embedding[:, 1],
                cmap=cmap,
                fill=True,
                levels=20
            )
            
            plt.scatter(
                reshaped_embedding[:, 0],
                reshaped_embedding[:, 1],
                c='black',
                alpha=0.5,
                s=30
            )
            
            plt.colorbar()
            
            plt.title('Mapa de Calor das Dimensões')
            plt.xlabel('Grupo de Dimensões')
            plt.ylabel('Grupo de Dimensões')
            plots['heatmap'] = save_plot_to_base64()
            print("Mapa de calor gerado com sucesso")
        except Exception as e:
            print(f"Erro ao gerar mapa de calor: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            plt.close()
        
        # 4. Clusters plot
        print("4. Gerando grupos temáticos...")
        plt.figure(figsize=(12, 6))
        cluster_labels = np.array(analysis['cluster_labels'])
        unique_clusters = np.unique(cluster_labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
        
        for cluster, color in zip(unique_clusters, colors):
            mask = cluster_labels == cluster
            values = embedding[mask]
            cluster_info = analysis['clusters'][int(cluster)]
            label = f"Grupo {cluster+1}: {cluster_info['tipo']} ({cluster_info['relevância']})"
            plt.hist(values, bins=30, alpha=0.5, color=color, label=label)
        
        plt.axvline(x=0, color='red', linestyle='-', alpha=0.3)
        plt.title('Distribuição dos Grupos Temáticos')
        plt.xlabel('Intensidade das Características')
        plt.ylabel('Frequência')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        plots['clusters'] = save_plot_to_base64()
        plt.close()
        
        # 5. Histogram plot
        print("5. Gerando histograma...")
        plt.figure(figsize=(12, 6))
        plt.hist(embedding, bins=50, color='#3498db', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='-', alpha=0.3)
        plt.title('Distribuição das Características do Texto')
        plt.xlabel('Intensidade')
        plt.ylabel('Frequência')
        plt.grid(True, alpha=0.3)
        plots['histogram'] = save_plot_to_base64()
        plt.close()
        
        print("Gráficos gerados:", list(plots.keys()))
        
        dimension_contexts = {
            'title': 'Contextos das Dimensões mais Significativas',
            'correlations': correlations
        }
        
        return plots, dimension_contexts
        
    except Exception as e:
        print(f"Erro ao gerar gráficos: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def extract_relevant_text(content, dimensions):
    """Extrai o trecho de texto mais relevante para um conjunto de dimensões"""
    # Por enquanto, retorna um trecho do início do texto
    # No futuro, podemos melhorar isso para encontrar trechos mais relevantes
    return content[:500]

def plot_embedding_overview(embedding):
    """Cria visualização geral dos valores de embedding"""
    plt.figure(figsize=(12, 6))
    plt.plot(embedding)
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.3)
    plt.title('Visão Geral dos Valores de Embedding')
    plt.xlabel('Dimensão')
    plt.ylabel('Valor')
    plt.grid(True)
    
    return save_plot_to_base64()

def plot_top_dimensions(embedding):
    """Plota as dimensões mais importantes"""
    embedding = np.array(embedding)
    magnitudes = np.abs(embedding)
    top_indices = np.argsort(magnitudes)[-10:]
    top_values = embedding[top_indices]
    
    plt.figure(figsize=(12, 6))
    colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in top_values]
    plt.bar(range(10), top_values, color=colors)
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.3)
    plt.title('Top 10 Características Mais Importantes')
    plt.xlabel('Número da Característica')
    plt.ylabel('Intensidade')
    plt.xticks(range(10), top_indices)
    
    return save_plot_to_base64()

def plot_dimension_clusters(embedding, clusters):
    """Plota mapa de calor dos grupos de características"""
    plt.figure(figsize=(12, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))
    
    for i, (cluster, color) in enumerate(zip(clusters, colors)):
        # Pega os índices das dimensões deste cluster
        dims = [j for j, x in enumerate(cluster['cluster_labels']) if x == i]
        values = embedding[dims]
        
        # Usa o tema identificado ou o tipo como fallback
        tema = cluster.get('tema') or cluster.get('tipo', f'Grupo {i+1}')
        relevancia = cluster.get('relevância', 'Média')
        
        # Plota os valores deste cluster
        plt.plot(dims, values, label=f"{tema} ({relevancia})", color=color, alpha=0.7)
    
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.3)
    plt.title('Grupos Temáticos Identificados')
    plt.xlabel('Dimensão')
    plt.ylabel('Valor')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True)
    
    return save_plot_to_base64()

def plot_activation_histogram(embedding):
    """Plota histograma da distribuição de valores"""
    plt.figure(figsize=(12, 6))
    plt.hist(embedding, bins=50, color='#3498db', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='-', alpha=0.3)
    plt.title('Distribuição das Características do Texto')
    plt.xlabel('Intensidade')
    plt.ylabel('Frequência')
    plt.grid(True, alpha=0.3)
    
    return save_plot_to_base64()

def plot_pca(embedding):
    """Visualiza grupos de características usando PCA"""
    segment_size = 256
    num_segments = len(embedding) // segment_size
    data_matrix = np.zeros((num_segments, segment_size))
    
    for i in range(num_segments):
        start = i * segment_size
        end = start + segment_size
        data_matrix[i] = embedding[start:end]
    
    if num_segments > 1:
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(data_matrix)
        
        plt.figure(figsize=(12, 8))
        plt.scatter(pca_results[:, 0], pca_results[:, 1], c='#3498db')
        
        for i in range(num_segments):
            start = i * segment_size
            end = start + segment_size - 1
            plt.annotate(f"Parte {i+1}", 
                         (pca_results[i, 0], pca_results[i, 1]),
                         fontsize=8)
        
        plt.title('Visualização PCA das Partes do Texto')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.grid(True, alpha=0.3)
    else:
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, "Texto muito curto para análise PCA", 
                 ha='center', va='center', fontsize=12)
        plt.axis('off')
    
    return save_plot_to_base64()

def analyze_embedding(embedding):
    """Analisa o embedding para extrair métricas importantes"""
    embedding = np.array(embedding)
    abs_embedding = np.abs(embedding)
    
    # Calcula métricas principais
    metrics = {
        "dimension_count": int(len(embedding)),
        "mean_value": float(np.mean(embedding)),
        "std_dev": float(np.std(embedding)),
        "min_value": float(np.min(embedding)),
        "min_dimension": int(np.argmin(embedding)),
        "max_value": float(np.max(embedding)),
        "max_dimension": int(np.argmax(embedding)),
        "median_value": float(np.median(embedding)),
        "positive_count": int(np.sum(embedding > 0)),
        "negative_count": int(np.sum(embedding < 0)),
        "zero_count": int(np.sum(embedding == 0)),
        "abs_mean": float(np.mean(abs_embedding)),
        "significant_dims": int(np.sum(abs_embedding > 0.1))
    }
    
    # Encontra clusters de ativação com limiar adaptativo
    threshold = np.percentile(abs_embedding, 90)  # Usa os 10% maiores valores como significativos
    significant_dims = np.where(abs_embedding > threshold)[0]
    
    clusters = []
    if len(significant_dims) > 0:
        current_cluster = [int(significant_dims[0])]
        
        for i in range(1, len(significant_dims)):
            if significant_dims[i] - significant_dims[i-1] <= 10:  # Aumentando a janela de agrupamento
                current_cluster.append(int(significant_dims[i]))
            else:
                if len(current_cluster) > 0:
                    clusters.append(current_cluster)
                current_cluster = [int(significant_dims[i])]
        
        if len(current_cluster) > 0:
            clusters.append(current_cluster)
    
    # Filtra e caracteriza clusters
    cluster_info = []
    for i, cluster in enumerate(clusters):
        if len(cluster) >= 2:  # Clusters precisam ter pelo menos 2 dimensões
            values = embedding[cluster]
            avg_value = float(np.mean(values))
            max_value = float(np.max(np.abs(values)))
            
            # Determina o tipo do cluster baseado nos valores
            if avg_value > 0:
                cluster_type = "Presença Forte"
            else:
                cluster_type = "Ausência Notável"
            
            # Calcula a força relativa do cluster
            relative_strength = max_value / threshold
            
            cluster_info.append({
                "id": i+1,
                "dimensions": [int(d) for d in cluster],
                "start_dim": int(min(cluster)),
                "end_dim": int(max(cluster)),
                "size": int(len(cluster)),
                "avg_value": avg_value,
                "max_value": max_value,
                "tipo": cluster_type,
                "força_relativa": float(relative_strength),
                "relevância": "Alta" if relative_strength > 1.5 else "Média"
            })
    
    # Ordena clusters por força relativa
    cluster_info = sorted(cluster_info, key=lambda x: abs(x['força_relativa']), reverse=True)
    
    # Top dimensões por magnitude
    top_indices = sorted(range(len(embedding)), key=lambda i: abs(embedding[i]), reverse=True)[:10]
    top_dimensions = [{"dimension": int(idx), "value": float(embedding[idx])} for idx in top_indices]
    
    return {
        "metrics": metrics,
        "clusters": cluster_info,
        "top_dimensions": top_dimensions
    }

# Template HTML
html_head = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Análise Semântica de Conteúdos</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .dashboard-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 2rem;
            background-color: #001B3D;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1000;
        }

        .logo-link {
            display: flex;
            align-items: center;
            text-decoration: none;
        }

        .dashboard-logo {
            height: 40px;
            width: auto;
        }

        .header-button {
            padding: 8px 16px;
            border-radius: 4px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            background-color: transparent;
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .header-button:hover {
            background-color: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.3);
        }

        .header-button.outline {
            background-color: transparent;
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: white;
        }

        .header-button.outline:hover {
            background-color: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.3);
        }

        /* Ajuste para o conteúdo não ficar embaixo do header fixo */
        body {
            padding-top: 80px;
        }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <header class="dashboard-header">
        <a href="https://interno.webestrategica.com.br" class="logo-link">
            <img 
                src="https://webestrategica.com.br/wp-content/uploads/2023/12/logo-webestrategica-v2.webp" 
                alt="Web Estratégica Logo" 
                class="dashboard-logo"
            />
        </a>
        <div style="display: flex; gap: 8px; align-items: center">
            <button 
                onclick="window.open('https://sites.google.com/webestrategica.com.br/oraculo/home', '_blank')"
                class="header-button"
            >
                Oráculo
            </button>
            <button 
                onclick="window.open('https://drive.google.com/drive/folders/1ynQGqFUDmBgl6ghE6iW8BTCf9zJ8RiJS?usp=sharing', '_blank')"
                class="header-button"
            >
                Projetos
            </button>
            <button 
                onclick="console.log('Logout clicked')" 
                class="header-button outline"
            >
                Sair
            </button>
        </div>
    </header>

    <div class="container mx-auto px-4 py-8">
        <h1 class="text-3xl font-bold mb-8">Análise Semântica de Conteúdos</h1>
        
        <!-- Formulário -->
        <form id="analyzeForm" class="mb-8">
            <div class="mb-4">
                <label for="content" class="block text-gray-700 text-sm font-bold mb-2">
                    Digite ou cole o texto para análise:
                </label>
                <textarea
                    id="content"
                    name="content"
                    rows="10"
                    class="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                    placeholder="Digite ou cole seu texto aqui..."
                ></textarea>
            </div>
            <button
                type="submit"
                class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
            >
                Analisar
            </button>
        </form>
        
        <!-- Loading -->
        <div id="loading" class="hidden">
            <div class="flex items-center justify-center">
                <div class="animate-spin rounded-full h-32 w-32 border-t-2 border-b-2 border-blue-500"></div>
            </div>
            <p class="text-center mt-4 text-gray-600">Analisando texto, por favor aguarde...</p>
        </div>
        
        <!-- Resultados -->
        <div id="results" class="hidden space-y-6">
            <!-- Métricas -->
            <div class="bg-white p-6 rounded-lg shadow">
                <h2 class="text-xl font-bold mb-4">Métricas</h2>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <!-- Distribuição -->
                    <div>
                        <h3 class="font-semibold text-gray-800 mb-2">Distribuição</h3>
                        <div class="space-y-2">
                            <p class="text-sm text-gray-600">Características Positivas: <span id="posCount">-</span></p>
                            <p class="text-xs text-gray-500">Conceitos presentes no texto</p>
                            <p class="text-sm text-gray-600">Características Negativas: <span id="negCount">-</span></p>
                            <p class="text-xs text-gray-500">Conceitos ausentes ou contrastantes</p>
                        </div>
                    </div>
                    
                    <!-- Força -->
                    <div>
                        <h3 class="font-semibold text-gray-800 mb-2">Força</h3>
                        <div class="space-y-2">
                            <p class="text-sm text-gray-600">Características Significativas: <span id="sigDims">-</span></p>
                            <p class="text-xs text-gray-500">Dimensões com valor > 0.1, indica profundidade do conteúdo</p>
                            <p class="text-sm text-gray-600">Força Média: <span id="meanStr">-</span></p>
                            <p class="text-xs text-gray-500">Intensidade média das características, indica especificidade</p>
                        </div>
                    </div>
                    
                    <!-- Equilíbrio -->
                    <div>
                        <h3 class="font-semibold text-gray-800 mb-2">Equilíbrio</h3>
                        <div class="space-y-2">
                            <p class="text-sm text-gray-600">Desvio Padrão: <span id="stdDev">-</span></p>
                            <p class="text-xs text-gray-500">Variação nas características, indica consistência</p>
                            <p class="text-sm text-gray-600">Mediana: <span id="median">-</span></p>
                            <p class="text-xs text-gray-500">Tendência central, indica viés do conteúdo</p>
                        </div>
                    </div>
                </div>
                
                <!-- Métricas Avançadas -->
                <div class="mt-6 border-t pt-4">
                    <h3 class="font-semibold text-gray-800 mb-2">Métricas Avançadas</h3>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                            <p class="text-sm text-gray-600">Densidade Semântica: <span id="density">-</span></p>
                            <p class="text-xs text-gray-500">Concentração de significado por dimensão</p>
                        </div>
                        <div>
                            <p class="text-sm text-gray-600">Complexidade Lexical: <span id="complexity">-</span></p>
                            <p class="text-xs text-gray-500">Variação e riqueza do vocabulário</p>
                        </div>
                        <div>
                            <p class="text-sm text-gray-600">Equilíbrio Semântico: <span id="balance">-</span></p>
                            <p class="text-xs text-gray-500">Harmonia entre conceitos presentes e ausentes</p>
                        </div>
                        <div>
                            <p class="text-sm text-gray-600">Distribuição Temática: <span id="distribution">-</span></p>
                            <p class="text-xs text-gray-500">Variedade e equilíbrio dos temas</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Grupos de Temas -->
            <div class="bg-white p-6 rounded-lg shadow">
                <h2 class="text-xl font-bold mb-4">Grupos de Temas</h2>
                
                <!-- Explicação -->
                <div class="mb-6 text-sm text-gray-600 bg-gray-50 p-4 rounded">
                    <h3 class="font-medium mb-2">Como interpretar:</h3>
                    <ul class="list-disc pl-5 space-y-2">
                        <li><span class="font-medium">Presença Forte</span>: Temas bem desenvolvidos e explícitos no texto</li>
                        <li><span class="font-medium">Ausência Notável</span>: Temas que contrastam ou estão faltando</li>
                        <li><span class="font-medium">Relevância</span>: Indica a importância do tema (Alta, Média, Baixa)</li>
                        <li><span class="font-medium">Força</span>: Quanto maior, mais distintivo é o tema</li>
                        <li><span class="font-medium">Valor Médio</span>: Positivo indica presença, negativo indica ausência</li>
                    </ul>
                </div>

                <div id="clustersContainer" class="space-y-4">
                    <!-- Os grupos serão inseridos aqui -->
                </div>
            </div>
            
            <!-- Análise OpenAI -->
            <div class="bg-white p-6 rounded-lg shadow">
                <h2 class="text-xl font-bold mb-4">Análise Semântica</h2>
                <div class="space-y-4">
                    <div>
                        <h3 class="font-medium text-gray-800 mb-2">Análise dos Grupos</h3>
                        <div id="clustersAnalysis" class="text-gray-600"></div>
                    </div>
                    <div>
                        <h3 class="font-medium text-gray-800 mb-2">Análise Geral</h3>
                        <div id="generalAnalysis" class="text-gray-600"></div>
                    </div>
                </div>
            </div>
            
            <!-- Visualizações -->
            <div class="bg-white p-6 rounded-lg shadow">
                <h2 class="text-xl font-bold mb-4">Visualizações</h2>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <h3 class="font-medium text-gray-800 mb-2">Visão Geral</h3>
                        <img id="overview-chart" class="w-full h-auto" alt="Visão geral do texto" />
                        <div class="mt-4 text-sm text-gray-600">
                            <strong>Como interpretar:</strong>
                            <ul class="list-disc ml-4">
                                <li>Cada ponto representa uma característica do texto</li>
                                <li>Picos para cima indicam presença forte de um conceito</li>
                                <li>Picos para baixo indicam ausência de um conceito</li>
                                <li>Quanto mais variação no gráfico, mais rico é o conteúdo</li>
                            </ul>
                        </div>
                    </div>
                    
                    <div>
                        <h3 class="font-medium text-gray-800 mb-2">Principais Características</h3>
                        <img id="top-dimensions-chart" class="w-full h-auto" alt="Principais características" />
                        <div class="mt-4 text-sm text-gray-600">
                            <strong>Como interpretar:</strong>
                            <ul class="list-disc ml-4">
                                <li>Barras verdes mostram conceitos fortemente presentes no texto</li>
                                <li>Barras vermelhas mostram conceitos ausentes ou opostos</li>
                                <li>Quanto maior a barra, mais importante é aquela característica</li>
                            </ul>
                        </div>
                    </div>
                    
                    <div>
                        <h3 class="font-medium text-gray-800 mb-2">Mapa de Calor das Dimensões</h3>
                        <img id="heatmap-chart" class="w-full h-auto" alt="Mapa de calor das dimensões" />
                        <div class="mt-4 text-sm text-gray-600">
                            <strong>Como interpretar:</strong>
                            <ul class="list-disc ml-4">
                                <li><strong>Pontos Pretos:</strong> Cada ponto representa uma frase do texto. Frases próximas são semanticamente similares.</li>
                                <li><strong>Cores e Intensidade:</strong>
                                    <ul class="list-disc ml-4">
                                        <li>Vermelho (centro): Alta densidade de frases similares</li>
                                        <li>Azul (bordas): Baixa densidade de frases</li>
                                        <li>Branco (transição): Densidade média</li>
                                    </ul>
                                </li>
                                <li><strong>Interpretação:</strong>
                                    <ul class="list-disc ml-4">
                                        <li>Centro vermelho: Tema principal ou mensagem central</li>
                                        <li>Pontos próximos: Frases com assuntos relacionados</li>
                                        <li>Pontos distantes: Aspectos diferentes ou complementares</li>
                                        <li>Gradiente suave: Indica boa coesão textual</li>
                                    </ul>
                                </li>
                            </ul>
                        </div>
                    </div>
                    
                    <div>
                        <h3 class="font-medium text-gray-800 mb-2">Grupos Temáticos</h3>
                        <img id="clusters-chart" class="w-full h-auto" alt="Grupos temáticos identificados" />
                        <div class="mt-4 text-sm text-gray-600">
                            <strong>Como interpretar:</strong>
                            <ul class="list-disc ml-4">
                                <li>Cada cor representa um grupo temático diferente</li>
                                <li>O eixo horizontal mostra a intensidade das características</li>
                                <li>A altura das barras mostra quantas características têm aquela intensidade</li>
                                <li>Valores positivos indicam presença do tema, negativos indicam ausência</li>
                            </ul>
                        </div>
                    </div>
                    
                    <div>
                        <h3 class="font-medium text-gray-800 mb-2">Distribuição</h3>
                        <img id="histogram-chart" class="w-full h-auto" alt="Distribuição de características" />
                        <div class="mt-4 text-sm text-gray-600">
                            <strong>Como interpretar:</strong>
                            <ul class="list-disc ml-4">
                                <li>Distribuição ampla: texto rico em conteúdo</li>
                                <li>Concentração no zero: texto genérico</li>
                                <li>Linha vermelha: ponto zero</li>
                                <li>Valores bem distribuídos indicam texto rico e específico</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        window.addEventListener('DOMContentLoaded', function() {
            const form = document.getElementById('analyzeForm');
            if (!form) return;
            
            form.addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const contentEl = document.getElementById('content');
                if (!contentEl) return;
                
                const content = contentEl.value;
                if (!content || content.trim().length === 0) {
                    alert('Por favor, insira um texto para análise.');
                    return;
                }
                
                const loadingEl = document.getElementById('loading');
                const resultsEl = document.getElementById('results');
                
                if (loadingEl) loadingEl.classList.remove('hidden');
                if (resultsEl) resultsEl.classList.add('hidden');
                
                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            content: content.trim()
                        })
                    });
                    
                    if (!response.ok) {
                        throw new Error('Erro na resposta do servidor');
                    }
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    
                    // Atualiza gráficos
                    if (data.plots) {
                        const charts = {
                            'overview-chart': data.plots.overview || '',
                            'top-dimensions-chart': data.plots.top_dimensions || '',
                            'clusters-chart': data.plots.clusters || '',
                            'histogram-chart': data.plots.histogram || '',
                            'heatmap-chart': data.plots.heatmap || ''
                        };
                        
                        Object.entries(charts).forEach(([id, base64]) => {
                            const img = document.getElementById(id);
                            if (img && base64) {
                                img.src = 'data:image/png;base64,' + base64;
                                img.style.display = 'block';
                            } else if (img) {
                                img.style.display = 'none';
                            }
                        });
                    }
                    
                    // Atualiza métricas
                    if (data.analysis && data.analysis.metrics) {
                        const metrics = data.analysis.metrics;
                        
                        // Distribuição
                        const positiveCount = document.getElementById('posCount');
                        const negativeCount = document.getElementById('negCount');
                        
                        if (positiveCount) positiveCount.textContent = metrics.positive_count || '0';
                        if (negativeCount) negativeCount.textContent = metrics.negative_count || '0';
                        
                        // Força
                        const significantDims = document.getElementById('sigDims');
                        const meanStrength = document.getElementById('meanStr');
                        
                        if (significantDims) {
                            const dims = metrics.significant_dims || 0;
                            const total = metrics.dimension_count || 0;
                            significantDims.textContent = dims + ' de ' + total;
                        }
                        if (meanStrength) {
                            const mean = metrics.abs_mean || 0;
                            meanStrength.textContent = mean.toFixed(4);
                        }
                        
                        // Equilíbrio
                        const stdDev = document.getElementById('stdDev');
                        const median = document.getElementById('median');
                        
                        if (stdDev) {
                            const std = metrics.std_dev || 0;
                            stdDev.textContent = std.toFixed(4);
                        }
                        if (median) {
                            const med = metrics.median_value || 0;
                            median.textContent = med.toFixed(4);
                        }
                        
                        // Métricas Avançadas
                        const density = document.getElementById('density');
                        const complexity = document.getElementById('complexity');
                        const balance = document.getElementById('balance');
                        const distribution = document.getElementById('distribution');
                        
                        if (density) {
                            const dens = metrics.densidade_semantica || 0;
                            density.textContent = dens.toFixed(4);
                        }
                        if (complexity) {
                            const comp = metrics.complexidade_lexical || 0;
                            complexity.textContent = comp.toFixed(4);
                        }
                        if (balance) {
                            const bal = metrics.equilibrio_semantico || 0;
                            balance.textContent = bal.toFixed(4);
                        }
                        if (distribution) {
                            const dist = metrics.distribuicao_tematica || 0;
                            distribution.textContent = dist.toFixed(4);
                        }
                    }
                    
                    // Atualiza clusters
                    const clustersContainer = document.getElementById('clustersContainer');
                    if (clustersContainer && data.analysis && data.analysis.clusters) {
                        clustersContainer.innerHTML = '';
                        
                        const clusters = data.analysis.clusters;
                        if (!clusters || clusters.length === 0) {
                            clustersContainer.innerHTML = 
                                '<div class="bg-gray-50 p-4 rounded">' +
                                '<p class="text-gray-500">' +
                                'Nenhum grupo de temas significativo detectado. ' +
                                'Tente um texto mais longo ou com temas mais distintos.' +
                                '</p>' +
                                '</div>';
                        } else {
                            clusters.forEach(cluster => {
                                const clusterEl = document.createElement('div');
                                
                                // Define cor baseada no tipo e relevância
                                const borderColor = cluster.tipo === "Presença Forte" ? 
                                    (cluster.relevância === "Alta" ? "border-green-500" : "border-green-300") :
                                    (cluster.relevância === "Alta" ? "border-red-500" : "border-red-300");
                                
                                clusterEl.className = 'bg-gray-50 p-4 rounded border-l-4 ' + borderColor + ' mb-4';
                                
                                const id = cluster.id || '?';
                                const tipo = cluster.tipo || 'Desconhecido';
                                const relevancia = cluster.relevância || 'Média';
                                const forca = (cluster.força_relativa || 0).toFixed(2);
                                const size = cluster.size || 0;
                                const avgValue = (cluster.avg_value || 0).toFixed(4);
                                const maxValue = (cluster.max_value || 0).toFixed(4);
                                const startDim = cluster.start_dim || '?';
                                const endDim = cluster.end_dim || '?';
                                
                                const relevanciaClass = relevancia === "Alta" ? 
                                    "bg-blue-100 text-blue-800" : 
                                    "bg-gray-100 text-gray-800";
                                    
                                const avgValueClass = (cluster.avg_value || 0) > 0 ? 
                                    "text-green-600" : 
                                    "text-red-600";
                                
                                clusterEl.innerHTML = 
                                    '<div class="flex items-center justify-between">' +
                                        '<h3 class="font-medium text-gray-800">' +
                                            'Grupo ' + id + ': ' + tipo +
                                            '<span class="ml-2 px-2 py-1 text-sm rounded ' + relevanciaClass + '">' +
                                                'Relevância ' + relevancia +
                                            '</span>' +
                                        '</h3>' +
                                        '<span class="text-sm text-gray-500">' +
                                            'Força: ' + forca + 'x média' +
                                        '</span>' +
                                    '</div>' +
                                    '<div class="grid grid-cols-1 md:grid-cols-3 gap-2 mt-3">' +
                                        '<div>' +
                                            '<span class="text-gray-600 text-sm">Tamanho:</span>' +
                                            '<span class="font-medium">' + size + ' características</span>' +
                                        '</div>' +
                                        '<div>' +
                                            '<span class="text-gray-600 text-sm">Valor Médio:</span>' +
                                            '<span class="font-medium ' + avgValueClass + '">' + avgValue + '</span>' +
                                        '</div>' +
                                        '<div>' +
                                            '<span class="text-gray-600 text-sm">Valor Máximo:</span>' +
                                            '<span class="font-medium">' + maxValue + '</span>' +
                                        '</div>' +
                                    '</div>' +
                                    '<div class="mt-2 text-sm text-gray-600">' +
                                        'Características ' + startDim + '-' + endDim +
                                    '</div>';
                                
                                clustersContainer.appendChild(clusterEl);
                            });
                        }
                    }
                    
                    // Atualiza análises
                    if (data.openai_analysis) {
                        const clustersAnalysis = document.getElementById('clustersAnalysis');
                        const generalAnalysis = document.getElementById('generalAnalysis');
                        
                        if (clustersAnalysis && data.openai_analysis.grupos) {
                            clustersAnalysis.innerHTML = data.openai_analysis.grupos.replace(/\\n/g, '<br>');
                        }
                        
                        if (generalAnalysis && data.openai_analysis.recomendacoes) {
                            generalAnalysis.innerHTML = data.openai_analysis.recomendacoes.replace(/\\n/g, '<br>');
                        }
                    }
                    
                    // Mostra resultados
                    if (loadingEl) loadingEl.classList.add('hidden');
                    if (resultsEl) resultsEl.classList.remove('hidden');
                    
                    // Rola para os resultados
                    if (resultsEl) {
                        resultsEl.scrollIntoView({ behavior: 'smooth' });
                    }
                    
                } catch (error) {
                    console.error('Erro:', error);
                    alert('Erro ao analisar o texto: ' + error.message);
                    if (loadingEl) loadingEl.classList.add('hidden');
                }
            });
        });
        
        function updateCharts(data) {
            console.log("Atualizando gráficos...");
            console.log("Gráficos disponíveis:", Object.keys(data.plots));
            
            // Mapeia os IDs dos gráficos para suas URLs base64
            const charts = {
                'overview-chart': data.plots.overview || '',
                'top-dimensions-chart': data.plots.top_dimensions || '',
                'heatmap-chart': data.plots.heatmap || '',
                'clusters-chart': data.plots.clusters || '',
                'histogram-chart': data.plots.histogram || ''
            };
            
            // Atualiza cada gráfico
            Object.entries(charts).forEach(([id, base64]) => {
                console.log(`Atualizando ${id}...`);
                const img = document.getElementById(id);
                if (img && base64) {
                    img.src = `data:image/png;base64,${base64}`;
                    img.style.display = 'block';
                } else {
                    console.log(`Elemento não encontrado ou sem dados: ${id}`);
                    if (img) img.style.display = 'none';
                }
            });

            // Atualiza contextos das dimensões
            if (data.dimension_contexts) {
                const contextDiv = document.getElementById('dimension-contexts');
                if (contextDiv) {
                    let html = `<h3 class="font-medium mb-2">${data.dimension_contexts.title}</h3><ul class="list-disc pl-5">`;
                    data.dimension_contexts.correlations.forEach(correlation => {
                        html += `<li class="mb-1">${correlation}</li>`;
                    });
                    html += '</ul>';
                    contextDiv.innerHTML = html;
                    contextDiv.style.display = 'block';
                }
            }
        }
    </script>
</body>
</html>
"""

html_body = """
</html>
"""

@app.route('/')
def home():
    return html_head + html_body

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        
        if not data or 'content' not in data:
            print("Erro: Dados inválidos")
            return jsonify({'error': 'Dados inválidos'}), 400
            
        content = data['content']
        if not content or len(content.strip()) == 0:
            print("Erro: Texto vazio")
            return jsonify({'error': 'Texto vazio'}), 400

        print("Normalizando texto...")
        # Remove caracteres especiais e normaliza o texto
        content = unidecode.unidecode(content)
        content = re.sub(r'[^\w\s]', ' ', content)
        content = ' '.join(content.split())

        print("Analisando texto...")
        analysis = analyze_text(content)
        if analysis is None:
            print("Erro: Não foi possível analisar o texto")
            return jsonify({'error': 'Não foi possível analisar o texto'}), 500

        print("Gerando gráficos...")
        # Gera gráficos
        plots, dimension_contexts = generate_plots(analysis, content)
        if plots is None:
            print("Erro: Não foi possível gerar os gráficos")
            return jsonify({'error': 'Não foi possível gerar os gráficos'}), 500
            
        print("Analisando com OpenAI...")
        # Analisa com OpenAI
        openai_analysis = analyze_with_openai(analysis['clusters'])
        if openai_analysis is None:
            print("Erro: Não foi possível obter análise da OpenAI")
            return jsonify({'error': 'Não foi possível obter análise da OpenAI'}), 500

        print("Gráficos gerados com sucesso:", list(plots.keys()))
        return jsonify({
            'analysis': analysis,
            'plots': plots,
            'dimension_contexts': dimension_contexts,
            'openai_analysis': openai_analysis
        })

    except Exception as e:
        print(f"Erro inesperado: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Iniciando Ferramenta de Análise de Texto...")
    print("Visite http://127.0.0.1:5001 em seu navegador para usar a ferramenta.")
    
    # Configura backend não-interativo do matplotlib
    import matplotlib
    matplotlib.use('Agg')
    
    app.run(debug=True, port=5001)
