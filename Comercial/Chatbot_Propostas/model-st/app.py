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
    "ata_para_proposta": AssistantConfig(
        id="workflow",  # ID especial para o workflow
        name="Ata Desorganizada para Proposta",
        description="Automatiza a organização da ata e criação da proposta comercial",
    ),
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

DEFAULT_ASSISTANT = "ata_para_proposta"

# --- Classe para Streaming de Resposta (Melhoria do backend.py) ---
# Usa o EventHandler para um streaming real e eficiente
from openai import AssistantEventHandler


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


# --- Função para processar o workflow de ata para proposta ---
def process_ata_to_proposal_workflow(user_prompt):
    """
    Processa o workflow completo: ata desorganizada -> ata organizada -> proposta
    """
    try:
        # Etapa 1: Organizar a ata
        st.info("🔄 Organizando a ata...", icon="📝")

        # Criar thread para o organizador de atas
        thread_ata = client.beta.threads.create()

        # Adicionar mensagem do usuário
        client.beta.threads.messages.create(
            thread_id=thread_ata.id, role="user", content=user_prompt
        )

        # Executar o organizador de atas
        run_ata = client.beta.threads.runs.create_and_poll(
            thread_id=thread_ata.id,
            assistant_id=AVAILABLE_ASSISTANTS["organizador_atas"].id,
        )

        # Obter a resposta organizada
        messages_ata = client.beta.threads.messages.list(thread_id=thread_ata.id)
        ata_organizada = messages_ata.data[0].content[0].text.value

        # Mostrar a ata organizada
        with st.chat_message(
            "assistant", avatar=os.path.join(SCRIPT_DIR, "assets", "img", "gpt.png")
        ):
            st.markdown("### 📋 Ata Organizada")
            st.markdown(ata_organizada)

        # Adicionar ao histórico
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"### 📋 Ata Organizada\n\n{ata_organizada}",
            }
        )

        # Etapa 2: Criar proposta
        st.info("🔄 Construindo proposta comercial...", icon="💼")

        # Criar thread para o criador de propostas
        thread_proposta = client.beta.threads.create()

        # Adicionar a ata organizada como input para a proposta
        client.beta.threads.messages.create(
            thread_id=thread_proposta.id,
            role="user",
            content=f"Com base na seguinte ata organizada, crie uma proposta comercial:\n\n{ata_organizada}",
        )

        # Executar o criador de propostas
        run_proposta = client.beta.threads.runs.create_and_poll(
            thread_id=thread_proposta.id,
            assistant_id=AVAILABLE_ASSISTANTS["criador_propostas"].id,
        )

        # Obter a proposta criada
        messages_proposta = client.beta.threads.messages.list(
            thread_id=thread_proposta.id
        )
        proposta_criada = messages_proposta.data[0].content[0].text.value

        # Mostrar a proposta
        with st.chat_message(
            "assistant", avatar=os.path.join(SCRIPT_DIR, "assets", "img", "gpt.png")
        ):
            st.markdown("### 💼 Proposta Comercial")
            st.markdown(proposta_criada)

        # Adicionar ao histórico
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"### 💼 Proposta Comercial\n\n{proposta_criada}",
            }
        )

        return True

    except Exception as e:
        st.error(f"Erro durante o processamento do workflow: {e}", icon="🚨")
        return False


# --- Configuração da Página e Estilos ---

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define a configuração da página
st.set_page_config(
    page_title="ChatGPT Interface",
    page_icon=os.path.join(SCRIPT_DIR, "assets", "img", "favicon-32x32.png"),
    layout="centered",
    initial_sidebar_state="expanded",
)

# Aplica o CSS customizado diretamente
st.markdown(
    """
<style>
/* Remove a borda padrão do topo do header */
header[data-testid="stHeader"] {
    border-top: none;
}

/* Estiliza os containers das mensagens para se parecerem com os balões de chat */
[data-testid="stChatMessage"] {
    background-color: #f5f5f5;
    /* secondaryBackgroundColor */
    border-radius: 8px;
    /* --border-radius */
    padding: 1rem;
    margin-bottom: 1rem;
}

/* Remove o fundo da mensagem do assistente para ter controle total */
[data-testid="stChatMessage"]:has(img[alt="assistant-avatar"]) {
    background-color: transparent;
}

/* Estilo para avatares */
img[data-testid="stAvatar"] {
    width: 32px;
    height: 32px;
}
</style>
""",
    unsafe_allow_html=True,
)

# --- Lógica de Backend (Integrada do backend.py) ---

try:
    # Inicializa o cliente OpenAI usando as secrets do Streamlit
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception as e:
    st.error(
        "Chave da API da OpenAI não encontrada. Por favor, configure seus secrets no Streamlit Cloud.",
        icon="🚨",
    )
    st.stop()


# --- Classe para Streaming de Resposta (Melhoria do backend.py) ---
# Usa o EventHandler para um streaming real e eficiente
from openai import AssistantEventHandler


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
            # Remove a linha que tentava reverter a seleção - isso causava o erro
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
    # Usa avatares customizados para replicar o visual
    avatar_img = (
        os.path.join(SCRIPT_DIR, "assets", "img", "user.png")
        if msg["role"] == "user"
        else os.path.join(SCRIPT_DIR, "assets", "img", "gpt.png")
    )
    with st.chat_message(msg["role"], avatar=avatar_img):
        st.markdown(msg["content"])


# Input do usuário (substitui o <textarea> e <button>)
if prompt := st.chat_input("Digite sua mensagem aqui..."):
    # Adiciona e exibe a mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message(
        "user", avatar=os.path.join(SCRIPT_DIR, "assets", "img", "user.png")
    ):
        st.markdown(prompt)

    # Verifica se é o workflow de ata para proposta
    if st.session_state.assistant_key == "ata_para_proposta":
        # Processa o workflow automatizado
        success = process_ata_to_proposal_workflow(prompt)
        if not success:
            # Remove a última mensagem do usuário se houve erro
            st.session_state.messages.pop()
    else:
        # Comportamento normal para outros assistentes
        # Prepara para receber a resposta do assistente
        with st.chat_message(
            "assistant", avatar=os.path.join(SCRIPT_DIR, "assets", "img", "gpt.png")
        ):
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
                    f"Ocorreu um erro ao se comunicar com a API da OpenAI: {e}",
                    icon="🚨",
                )
                # Remove a última mensagem do usuário para que ele possa tentar novamente
                st.session_state.messages.pop()
