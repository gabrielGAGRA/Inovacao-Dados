# streamlit_app.py

import streamlit as st
import openai
import time
import os
from typing import Dict, List
from pydantic import BaseModel

# --- Modelos de Configuração (de config.py) ---


class AssistantConfig(BaseModel):
    id: str
    name: str
    description: str


AVAILABLE_ASSISTANTS: Dict[str, AssistantConfig] = {
    "organizador_atas": AssistantConfig(
        id="asst_gl4svzGMPxoDMYskRHzK62Fk",
        name="Organizador de Atas",
        description="Especialista em organizar e estruturar atas de reunião",
    ),
    "criador_propostas": AssistantConfig(
        id="asst_gqDpEGoOpRvpUai7fdVxgg4d",
        name="Criador de Propostas Comerciais",
        description="Especialista em criar propostas comerciais persuasivas",
    ),
}

DEFAULT_ASSISTANT = "organizador_atas"

# --- Configuração da Página e Estilos ---

# Define a configuração da página
st.set_page_config(
    page_title="ChatGPT Interface",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# CSS customizado para um visual mais próximo do original
st.markdown(
    """
<style>
/* Estilos globais */
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    line-height: 1.6;
}

/* Personalização do chat */
.stChatMessage {
    padding: 0.5rem 0;
}

.stChatMessage .avatar {
    width: 40px !important;
    height: 40px !important;
}

/* Botões e elementos de interface */
.stButton > button {
    border-radius: 6px;
    font-weight: 500;
    transition: all 0.2s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

/* Estilização da sidebar */
section[data-testid="stSidebar"] {
    background-color: #f7f7f8;
    border-right: 1px solid #e5e5e5;
}

section[data-testid="stSidebar"] button {
    background-color: #007acc;
    color: white;
    border: none;
    margin-bottom: 1rem;
}

section[data-testid="stSidebar"] button:hover {
    background-color: #005999;
}

/* Estilização do input de chat */
div[data-testid="stChatInput"] textarea {
    border-radius: 8px;
    border: 1px solid #e5e5e5;
}

div[data-testid="stChatInput"] textarea:focus {
    border-color: #007acc;
    box-shadow: 0 0 0 1px #007acc;
}

/* Para modo escuro */
@media (prefers-color-scheme: dark) {
    section[data-testid="stSidebar"] {
        background-color: #1e1e1e;
        border-right: 1px solid #333;
    }
    
    div[data-testid="stChatInput"] textarea {
        border: 1px solid #333;
        background-color: #2d2d2d;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

# --- Lógica de Backend (Integrada do backend.py) ---

# Importações para o streaming de resposta
from openai import OpenAI, AssistantEventHandler

try:
    # Inicializa o cliente OpenAI usando as secrets do Streamlit
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        # Fallback para variável de ambiente
        api_key = os.getenv("OPENAI_API_KEY", "")

    if not api_key:
        st.error(
            "Chave da API da OpenAI não encontrada. Por favor, configure seus secrets no Streamlit Cloud.",
            icon="🚨",
        )
        st.stop()

    client = OpenAI(api_key=api_key)
except Exception as e:
    st.error(
        f"Erro ao inicializar o cliente OpenAI: {str(e)}",
        icon="🚨",
    )
    st.stop()


# --- Classe para Streaming de Resposta (Melhoria do backend.py) ---
# Usa o EventHandler para um streaming real e eficiente


class StreamingEventHandler(AssistantEventHandler):
    def __init__(self, text_placeholder):
        super().__init__()
        self.text_placeholder = text_placeholder
        self.full_response = ""

    def on_text_delta(self, delta, snapshot):
        # Adiciona o novo trecho de texto ao placeholder e atualiza o conteúdo
        self.full_response += delta.value
        self.text_placeholder.markdown(self.full_response + "▌")

    def on_end(self):
        # Exibe a resposta final sem o cursor
        self.text_placeholder.markdown(self.full_response)

    def get_full_response(self):
        return self.full_response


# --- Funções de Utilidade ---


def get_avatar_svg(role: str) -> str:
    """Retorna o SVG para o avatar do usuário ou assistente"""
    if role == "user":
        # SVG para o avatar do usuário
        return """
        <svg width="36" height="36" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="18" cy="18" r="18" fill="#007acc"/>
            <circle cx="18" cy="14" r="6" fill="white"/>
            <path d="M7 32c0-6.5 5-11 11-11s11 4.5 11 11" fill="white"/>
        </svg>
        """
    else:
        # SVG para o avatar do assistente (ChatGPT)
        return """
        <svg width="36" height="36" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="18" cy="18" r="18" fill="#19c37d"/>
            <path d="M11 14.5C11 10.91 13.91 8 17.5 8C21.09 8 24 10.91 24 14.5C24 18.09 21.09 21 17.5 21C13.91 21 11 18.09 11 14.5Z" fill="white"/>
            <path d="M7 29.5C7 25.21 11.21 21 15.5 21H19.5C23.79 21 28 25.21 28 29.5" fill="white"/>
        </svg>
        """


def get_avatar_html(role: str) -> str:
    """Retorna o HTML para exibir o avatar como base64"""
    import base64
    from io import BytesIO

    svg = get_avatar_svg(role)
    svg_bytes = svg.encode()
    b64 = base64.b64encode(svg_bytes).decode()
    return f"data:image/svg+xml;base64,{b64}"


# --- Inicialização do Estado da Sessão ---
# O st.session_state é o equivalente do Streamlit ao localStorage ou variáveis de classe do JS

if "session_id" not in st.session_state:
    st.session_state.session_id = (
        f"session_{int(time.time())}"  # ID único para a sessão
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if "assistant_key" not in st.session_state:
    st.session_state.assistant_key = DEFAULT_ASSISTANT


# --- Interface da Sidebar (de index.html) ---
with st.sidebar:
    st.title("ChatGPT Interface")

    if st.button("＋ Nova Conversa", use_container_width=True):
        # Reseta o estado da conversa para iniciar um novo chat
        st.session_state.messages = []
        st.session_state.thread_id = None
        st.rerun()

    st.header("Configurações")

    # Seletor de Assistente
    assistant_options = {
        key: assistant.name for key, assistant in AVAILABLE_ASSISTANTS.items()
    }

    selected_assistant_key = st.selectbox(
        label="Assistente:",
        options=assistant_options.keys(),
        format_func=lambda key: assistant_options[key],
        key="selected_assistant",
    )

    # Lógica para confirmar mudança de assistente no meio da conversa
    if selected_assistant_key != st.session_state.assistant_key:
        if st.session_state.messages:  # Se já houver mensagens
            st.warning("Mudar de assistente irá iniciar uma nova conversa.", icon="⚠️")
            if st.button("Confirmar e iniciar nova conversa", use_container_width=True):
                st.session_state.assistant_key = selected_assistant_key
                st.session_state.messages = []
                st.session_state.thread_id = None
                st.rerun()
            else:
                # Reverte a seleção se o usuário não confirmar
                st.session_state.selected_assistant = st.session_state.assistant_key
        else:
            # Se não houver mensagens, apenas muda o assistente
            st.session_state.assistant_key = selected_assistant_key
            st.rerun()

    # Exibe a descrição do assistente selecionado
    assistant_info = AVAILABLE_ASSISTANTS[st.session_state.assistant_key]
    st.markdown(
        f"<small>*{assistant_info.description}*</small>", unsafe_allow_html=True
    )

    st.markdown("---")
    st.info(
        "O tema (Claro/Escuro) pode ser alterado no menu 'Settings' do próprio Streamlit (canto superior direito).",
        icon="🎨",
    )
    st.markdown(
        "<small>ChatGPT Interface v2.0 (Streamlit Edition)</small>",
        unsafe_allow_html=True,
    )


# --- Interface Principal do Chat ---

# Header do Chat
st.markdown(f"### {assistant_info.name}")

# Exibe mensagens do histórico
if not st.session_state.messages:
    st.info(f"Como posso te ajudar hoje como {assistant_info.name}?", icon="👋")

for msg in st.session_state.messages:
    # Usa avatares SVG gerados dinamicamente
    avatar_url = get_avatar_html(msg["role"])
    with st.chat_message(msg["role"], avatar=avatar_url):
        st.markdown(msg["content"])


# Input do usuário (substitui o <textarea> e <button>)
if prompt := st.chat_input("Digite sua mensagem aqui..."):
    # Adiciona e exibe a mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=get_avatar_html("user")):
        st.markdown(prompt)

    # Prepara para receber a resposta do assistente
    with st.chat_message("assistant", avatar=get_avatar_html("assistant")):
        # Se não houver um thread, cria um novo
        if not st.session_state.thread_id:
            try:
                thread = client.beta.threads.create()
                st.session_state.thread_id = thread.id
            except Exception as e:
                st.error(f"Erro ao criar a thread: {e}", icon="🚨")
                st.stop()

        # Adiciona a mensagem do usuário à thread
        try:
            client.beta.threads.messages.create(
                thread_id=st.session_state.thread_id, role="user", content=prompt
            )

            # Cria o placeholder para a resposta em streaming
            response_placeholder = st.empty()

            # Inicializa o handler de streaming
            handler = StreamingEventHandler(response_placeholder)

            # Cria e faz o streaming da run
            with client.beta.threads.runs.stream(
                thread_id=st.session_state.thread_id,
                assistant_id=assistant_info.id,
                event_handler=handler,
            ) as stream:
                # O stream é processado pelo handler em tempo real
                stream.until_done()

            # Adiciona a resposta completa do assistente ao histórico
            assistant_response = handler.get_full_response()
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_response}
            )

        except Exception as e:
            st.error(
                f"Ocorreu um erro ao se comunicar com a API da OpenAI: {e}", icon="🚨"
            )
            # Remove a última mensagem do usuário para que ele possa tentar novamente
            st.session_state.messages.pop()
