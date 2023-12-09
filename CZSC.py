import os
import streamlit as st

st.set_page_config(page_title="知识问答", page_icon="🤖", layout="wide")


@st.cache_resource
def initialize_model(model_name='qwen-max-1201', api_key=None, api_base=None):
    from langchain.vectorstores import DeepLake
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import RetrievalQA
    from langchain.embeddings.openai import OpenAIEmbeddings

    os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY'] if api_key is None else api_key
    os.environ['OPENAI_API_BASE'] = st.secrets['OPENAI_API_BASE'] if api_base is None else api_base
    os.environ['ACTIVELOOP_TOKEN'] = st.secrets['ACTIVELOOP_TOKEN']

    embeddings = OpenAIEmbeddings()
    db = DeepLake(dataset_path='hub://zengbin93/czsc', read_only=True, embedding_function=embeddings)

    # db = DeepLake(dataset_path='hub://zengbin93/czsc', read_only=True)
    retriever = db.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['k'] = 20

    # model_names = ['gpt-3.5-turbo', 'qwen-max-1201', 'gpt-4', 'Baichuan2']
    model = ChatOpenAI(model=model_name)           # 'gpt-3.5-turbo', 'qwen-max-1201', 'gpt-4', 'Baichuan2'
    qa = RetrievalQA.from_llm(model, retriever=retriever)
    return qa


def show_kbqa():
    """

    参考资料：

    1. https://github.com/chatchat-space/Langchain-Chatchat
    2. https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps

    """
    st.subheader("CZSC代码库QA", divider="rainbow")
    c1, c2, c3, c4, c5 = st.columns([3, 3, 3, 1, 1])
    api_help = "任何支持 OpenAI API 的模型都可以使用。API_KEY 和 API_BASE 必须同时提供，否则使用默认值。"
    api_key = c1.text_input(label="请输入API_KEY", value="使用默认值", help=api_help)
    api_base = c2.text_input(label="请输入API_BASE", value="使用默认值", help=api_help)
    # , 'gpt-4', 'Baichuan2'
    model_name = c3.selectbox(label="请选择模型", options=['gpt-3.5-turbo', 'qwen-max-1201', 'Baichuan2'], index=1)
    if c4.button("清空历史消息"):
        st.session_state.messages = []
    if c5.button("微信捐赠支持，接GPT4"):
        st.success("感谢支持，请加微信：zengbin93，备注：捐赠")
    st.divider()

    api_key = None if api_key == "使用默认值" else api_key
    api_base = None if api_base == "使用默认值" else api_base
    qa = initialize_model(model_name=model_name, api_key=api_key, api_base=api_base)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = qa.run(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == '__main__':
    show_kbqa()
