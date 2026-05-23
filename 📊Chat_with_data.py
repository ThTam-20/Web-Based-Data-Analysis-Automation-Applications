import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel, model_validator
from dotenv import load_dotenv

from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

from src.logger.base import BaseLogger
from src.models.llms import load_llm
from src.utils import execute_plt_code

# load environment variables
load_dotenv()
logger = BaseLogger()

MODEL_NAME = "openai/gpt-oss-120b:free"

AGENT_PREFIX = """You are a data analysis assistant. You have access to a pandas DataFrame called `df`.

CRITICAL RULES:
1. You MUST use the python_repl_ast tool to execute ALL code. NEVER output code as text in your response.
2. Before plotting, ALWAYS run df.head(), df.columns, df.dtypes first to understand the data.
3. Create ONLY ONE chart per query. Keep it focused and relevant to the user's question.
4. Use matplotlib (plt) and seaborn (sns) for plotting. Both are available.
5. Do NOT call plt.show() — the chart will be captured automatically.
6. Always set proper title, xlabel, ylabel in Vietnamese if the user asks in Vietnamese.
7. Use plt.figure(figsize=(10, 6)) for a clean chart size.
8. Use plt.tight_layout() at the end to avoid overlapping labels.

The tool parameter name is "query" (not "code"). Pass your Python code as the "query" parameter.
"""



# --- Fix PythonInputs ValidationError ---
# Một số model miễn phí (OpenRouter) gửi tham số 'code' thay vì 'query'.
class PythonInputsCompat(BaseModel):
    """Schema tương thích cho PythonAstREPLTool — chấp nhận cả 'query' và 'code'."""
    query: str

    @model_validator(mode="before")
    @classmethod
    def accept_code_as_query(cls, values):
        if isinstance(values, dict) and "query" not in values and "code" in values:
            values["query"] = values.pop("code")
        return values


def _patch_agent_tools(agent):
    """Patch tất cả PythonAstREPLTool trong agent để dùng schema tương thích."""
    for tool in agent.tools:
        if tool.name == "python_repl_ast":
            tool.args_schema = PythonInputsCompat
    return agent


def _extract_action_code(response):
    """Trích xuất code đã thực thi từ intermediate_steps một cách an toàn."""
    steps = response.get("intermediate_steps", [])
    if not steps:
        return None

    last_action = steps[-1][0]
    tool_input = last_action.tool_input

    if isinstance(tool_input, dict):
        return tool_input.get("query") or tool_input.get("code")
    elif isinstance(tool_input, str):
        return tool_input
    return None


def _extract_plt_code_from_text(text):
    """Trích xuất code matplotlib từ output text khi agent trả code dạng text
    thay vì thực thi qua tool.

    Tìm code blocks (```python ... ```) hoặc code có chứa plt/sns.
    """
    # Tìm trong code blocks markdown
    code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    for block in code_blocks:
        if "plt" in block or "sns" in block:
            return block.strip()

    return None


def process_query(da_agent, query):
    try:
        response = da_agent.invoke({"input": query})
    except Exception as e:
        st.error(f"Lỗi khi gọi agent: {e}")
        logger.info(f"Agent error: {e}")
        return

    output = response.get("output", "")

    # Cố gắng trích xuất code từ intermediate_steps (agent thực thi tool)
    action = _extract_action_code(response)

    # Nếu không có intermediate_steps nhưng output chứa plt code dạng text,
    # trích xuất và thực thi
    if not action:
        action = _extract_plt_code_from_text(output)

    if action and ("plt" in action or "sns" in action):
        st.write(output)

        fig = execute_plt_code(action, df=st.session_state.df)
        if fig:
            st.pyplot(fig)

        st.write("**Executed code:**")
        st.code(action)

        to_display_string = output + "\n" + f"```python\n{action}\n```"
        st.session_state.history.append((query, to_display_string))

    else:
        st.write(output)
        st.session_state.history.append((query, output))



def display_chat_history():
    st.markdown("## Chat History: ")
    for i, (q, r) in enumerate(st.session_state.history):
        st.markdown(f"**Query: {i+1}:** {q}")
        st.markdown(f"**Response: {i+1}:** {r}")
        st.markdown("---")



def main():
    
    #set up streamlit interface
    st.set_page_config(page_title="📊 Smart Data Analysis Tool", page_icon="📊", layout="centered")
    st.header("📊 Smart Data Analysis Tool")
    st.write(
        "### Welcome to our data analysis tool. This tools can assist your daily data analysis tasks."
    )

    #load llm model
    llm = load_llm(model_name=MODEL_NAME)
    logger.info(f"### Successfully loaded {MODEL_NAME} !###")


    #upload csv file
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload your csv file here", type="csv") 


    #initial chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    
    # --- Fix: Lưu df vào session_state khi upload file mới ---
    # Chỉ đọc CSV khi có file mới upload, không phụ thuộc vào widget state
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)

    # --- Fix: Dùng session_state.df làm nguồn dữ liệu chính ---
    # Cho phép tiếp tục hỏi sau khi chuyển trang rồi quay lại
    if st.session_state.get("df") is not None:
        st.write("### Your uploaded data: ", st.session_state.df.head())

    #create data analysis agent to query with our data
        da_agent = create_pandas_dataframe_agent(
            llm=llm,
            df=st.session_state.df,
            agent_type="tool-calling",
            allow_dangerous_code=True,
            verbose=True,
            return_intermediate_steps=True,
            prefix=AGENT_PREFIX,
        )

        # Patch tool schema để tương thích với các model miễn phí
        _patch_agent_tools(da_agent)
        logger.info("### Sucessfully loaded data analysis agent !###")

    # --- Fix: Dùng st.form để Enter cũng submit query ---
        with st.form(key="query_form"):
            query = st.text_input("Enter your questions: ")
            submit_button = st.form_submit_button("Run query")

        if submit_button and query:
            with st.spinner("Processing..."):
                process_query(da_agent, query) 

    else:
        st.info("Vui lòng upload file CSV ở sidebar để bắt đầu phân tích dữ liệu.")


    #display chat history
    st.divider()
    display_chat_history()



if __name__ == "__main__":
    main()
