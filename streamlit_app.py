import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.base import BaseEstimator, TransformerMixin
import unidecode

# -----------------------
# Funções e classes usadas no pipeline
# -----------------------
def padronizar_nomes(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(' ', '_')
        .str.replace('-', '_')
        .map(lambda x: unidecode.unidecode(x))
    )
    return df

def padronizar_categoricas(df):
    for col in df.select_dtypes(include='object').columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(' ', '_')
            .str.replace('-', '_')
            .map(lambda x: unidecode.unidecode(x))
        )
    return df

class ValoresNulosTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = X.copy()
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
        return df

class CriarDummyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = X.copy()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
        return df

# -----------------------
# Função para exportar Excel
# -----------------------
@st.cache_data
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Resultados')
    writer.close()
    return output.getvalue()

# -----------------------
# Funções para gráficos
# -----------------------
@st.cache_data
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Previsto')
    ax.set_ylabel('Real')
    ax.set_title('Matriz de Confusão')
    st.pyplot(fig)

@st.cache_data
def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='blue', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('Falso Positivo')
    ax.set_ylabel('Verdadeiro Positivo')
    ax.set_title('Curva ROC')
    ax.legend()
    st.pyplot(fig)

@st.cache_data
def plot_score_distribution(y_prob, y_true):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.histplot(y_prob[y_true==0], color='red', label='Classe 0', kde=True, stat="density")
    sns.histplot(y_prob[y_true==1], color='blue', label='Classe 1', kde=True, stat="density")
    ax.set_xlabel('Score previsto')
    ax.set_ylabel('Densidade')
    ax.set_title('Distribuição dos Scores por Classe')
    ax.legend()
    st.pyplot(fig)

@st.cache_data
def plot_classes_histogram(y_true, y_pred):
    df_plot = pd.DataFrame({'Real': y_true, 'Previsto': y_pred})
    fig, ax = plt.subplots(figsize=(8,6))
    sns.countplot(data=df_plot.melt(value_vars=['Real','Previsto']), x='value', hue='variable', ax=ax)
    ax.set_xlabel('Classe')
    ax.set_ylabel('Contagem')
    ax.set_title('Comparação: Classes Reais vs Classes Previstas')
    st.pyplot(fig)

# -----------------------
# CSS sidebar
# -----------------------
st.markdown("""
    <style>
    /* ======== LAYOUT GERAL ======== */
    [data-testid="stSidebar"] {
        background-color: #C8E6C9; /* verde claro */
        color: #1B5E20; /* verde escuro */
    }

    .stApp {
        background-color: #BBDEFB; /* azul claro */
        color: #0D47A1;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #0D47A1;
    }

    /* ======== 1) UPLOADER: caixa e textos ======== */
    [data-testid="stFileUploader"] section {
        background-color: #FFFFFF !important; /* fundo branco */
        border-radius: 8px;
        border: 1px solid #A5D6A7;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: #388E3C;
    }

    /* Textos brancos dentro do uploader (ficarão pretos) */
    [data-testid="stFileUploader"] div,
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] p {
        color: #000000 !important; /* preto */
    }

    /* ======== 2) SELECTBOX (Escolha do gráfico) ======== */
    [data-baseweb="select"] > div {
        background-color: #C8E6C9 !important; /* verde claro */
        color: #1B5E20 !important;
        border-radius: 6px;
        border: 1px solid #81C784;
    }

    [data-baseweb="select"]:hover > div {
        background-color: #A5D6A7 !important;
    }

    /* ======== 3) BOTÕES ACIMA DO DATAFRAME ======== */
    [data-testid="StyledFullScreenButton"],
    [data-testid="StyledDownloadButton"],
    [data-testid="StyledCopyToClipboardButton"],
    [data-testid="StyledViewportButton"] {
        background-color: #C8E6C9 !important; /* verde claro */
        color: #1B5E20 !important;
        border-radius: 6px;
    }

    [data-testid="StyledFullScreenButton"]:hover,
    [data-testid="StyledDownloadButton"]:hover,
    [data-testid="StyledCopyToClipboardButton"]:hover,
    [data-testid="StyledViewportButton"]:hover {
        background-color: #81C784 !important;
        color: #FFF !important;
    }

    /* ======== 4) MENSAGENS (sucesso e instruções) ======== */
    /* Modelo "carregado ✅" e "Escolha o gráfico" */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stSuccess"] div,
    [data-testid="stSelectboxLabel"] {
        color: #1B5E20 !important; /* verde escuro */
        font-weight: 600;
    }

    /* Ajuste geral para textos dentro da sidebar */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #1B5E20 !important; /* verde escuro para consistência */
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# Função principal do app
# -----------------------
def main():
    st.set_page_config(page_title='EBAC - José Eduardo', layout='wide')
    st.title('Projeto Final - EBAC')
    st.markdown("---")
    st.markdown("""
    ## **Descrição do Projeto – Escoragem de Clientes**

    Este projeto consiste na criação de um **pipeline de pré-processamento e modelagem preditiva** aplicado a dados de clientes, com o objetivo de **avaliar o risco de inadimplência**.  

    O aplicativo foi desenvolvido em **Streamlit**, permitindo:

    - **Carregar arquivos CSV** com dados de clientes  
    - **Pré-processar automaticamente** os dados (padronização de nomes, tratamento de valores nulos e codificação de variáveis categóricas)  
    - **Realizar previsões** utilizando um modelo treinado previamente (`modelo_final.pkl`)  
    - **Visualizar resultados** de forma interativa com tabelas, gráficos e métricas de desempenho  
    - **Analisar o modelo** por meio de:
    - Matriz de Confusão  
    - Curva ROC  
    - Distribuição dos Scores  
    - Histograma de Classes  
    - 📥 **Baixar os resultados** em formato Excel para documentação ou análise posterior  

    O aplicativo também conta com uma **sidebar interativa**, com links para o **LinkedIn** e **GitHub**, proporcionando acesso rápido à documentação e ao código-fonte do projeto.  

    ---

    **Tecnologias utilizadas:**  
    `Python`, `Pandas`, `Scikit-learn`, `Seaborn`, `Matplotlib`, `Streamlit`
    """)

    # -----------------------
    # Sidebar: logo, links e upload
    # -----------------------
    with st.sidebar:
        st.markdown('<div class="sidebar-heading">Projeto Final - Ciência de Dados</div>', unsafe_allow_html=True)
        st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRwrZNbO5oYRyX9vLAO78p-EwuLy0omoWU2_Q&s", width=150)
        st.markdown(
            '<div class="sidebar-text">Bem-vindo ao app projetado para carregar seu CSV, gerar previsões e analisar gráficos interativos do modelo.</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="sidebar-link">'
            '<a href="https://www.linkedin.com/in/jos%C3%A9-eduardo-souza-leite/" target="_blank">🔗 LinkedIn</a>'
            '<a href="https://github.com/orgs/ebac-data-science/repositories" target="_blank">🐙 GitHub</a>'
            '</div>',
            unsafe_allow_html=True
        )
        st.header("📂 Carregue seu arquivo CSV")
        data_file = st.file_uploader("Selecione o arquivo CSV", type=['csv'])

    # -----------------------
    # Processamento e outputs principais (fora da sidebar)
    # -----------------------
    if data_file is not None:
        # Ler CSV
        df = pd.read_csv(data_file)
        st.write("### 🧾 Dados originais")
        st.dataframe(df.head())

        # Pré-processamento
        df = padronizar_nomes(df)
        df = padronizar_categoricas(df)

        # Separar target se existir
        y_true = None
        if 'mau' in df.columns:
            y_true = df['mau']
            df = df.drop(columns=['mau'])

        # Carregar modelo
        nome_arquivo = 'modelo_final.pkl'
        try:
            model = joblib.load(nome_arquivo)
            st.success(f'Modelo "{nome_arquivo}" carregado ✅')
        except Exception as e:
            st.error(f'Erro ao carregar o modelo: {e}')
            return

        # Barra de progresso
        progress_bar = st.progress(0)

        # Previsões
        try:
            y_prob = model.predict_proba(df)[:, 1]
            y_pred = model.predict(df)
            progress_bar.progress(30)
        except Exception as e:
            st.error(f'Erro ao gerar previsões: {e}')
            return

        # DataFrame final com score e classe prevista
        df_saida = df.copy()
        df_saida['score'] = y_prob
        df_saida['classe_prevista'] = y_pred
        progress_bar.progress(60)

        st.markdown("### ✅ Escoragem concluída com sucesso, os gráficos estão sendo processados.")
        st.dataframe(df_saida.head())

        # Download Excel
        df_xlsx = to_excel(df_saida)
        st.download_button(
            label='📥 Baixar resultados em Excel',
            data=df_xlsx,
            file_name='escoragem_resultados.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        progress_bar.progress(70)

        # -----------------------
        # Gráficos interativos
        # -----------------------
        if y_true is not None:
            st.markdown("## 📊 Análises do modelo")
            graph_option = st.selectbox(
                "Escolha o gráfico para exibir:",
                ['Matriz de Confusão', 'Curva ROC', 'Distribuição dos Scores', 'Histograma de Classes']
            )

            if graph_option == 'Matriz de Confusão':
                plot_confusion_matrix(y_true, y_pred)
                st.markdown("""
                    <div style="background-color:#f5f5f5; padding:15px; border-radius:10px; color:black;">
                        <b>Matriz de Confusão:</b><br>
                        • Mostra a contagem de previsões corretas e incorretas do modelo.<br>
                        • Linhas: valores reais.<br>
                        • Colunas: valores previstos.<br>
                        • A diagonal principal indica acertos, fora da diagonal indica erros.
                    </div>
                """, unsafe_allow_html=True)
                progress_bar.progress(100)

            elif graph_option == 'Curva ROC':
                plot_roc_curve(y_true, y_prob)
                st.markdown("""
                    <div style="background-color:#f5f5f5; padding:15px; border-radius:10px; color:black;">
                         <b>Curva ROC:</b><br>
                         • Mede a capacidade do modelo em diferenciar entre classes.<br>
                         • AUC próximo de 1 indica bom desempenho.
                     </div>
                """, unsafe_allow_html=True)
                progress_bar.progress(100)

            elif graph_option == 'Distribuição dos Scores':
                plot_score_distribution(y_prob, y_true)
                st.markdown("""
                    <div style="background-color:#f5f5f5; padding:15px; border-radius:10px; color:black;">
                        <b>Distribuição dos Scores:</b><br>
                        • Mostra como os scores previstos se distribuem para cada classe.<br>
                        • Permite identificar sobreposição entre classes.
                    </div>
                """, unsafe_allow_html=True)
                progress_bar.progress(100)

            elif graph_option == 'Histograma de Classes':
                plot_classes_histogram(y_true, y_pred)
                st.markdown("""
                <div style="background-color:#f5f5f5; padding:15px; border-radius:10px; color:black;">
                    <b>Histograma de Classes:</b><br>
                    • Mostra contagem de cada classe real e prevista.<br>
                    • Permite comparar distribuição de classes.
                </div>
            """, unsafe_allow_html=True)
                progress_bar.progress(100)

            st.success("✅ Análise gráfica concluída!")

    else:
        st.info("Carregue um arquivo CSV na barra lateral para começar.")

# -----------------------
# Executa o app
# -----------------------
if __name__ == '__main__':
    main()
