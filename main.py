from dotenv import load_dotenv
from langchain.agents import tool
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackManager
from langchain.chat_models import ChatOpenAI
from langchain.llms.gpt4all import GPT4All
from langchain.prompts import PromptTemplate
from langchain.tools.render import render_text_description
import os

load_dotenv()


### the tool decorator from langchain helps us define a function as a langchain agent tool
### Creates an instance from the tool class in langchain and populates the propeties for you
@tool
def get__text_length(text: str) -> int:
    """
    :returns the length of a text by characters
    :param text: a string of text
    :return: an integer that represents the number of characters in the string text
    """
    return len(text)


if __name__ == "__main__":

    tools = [get__text_length]
    gpt4all_path = os.getenv("GPT4ALL_PATH")

    # Concatenate with another path or filename
    llm_path = os.path.join(gpt4all_path, "nous-hermes-llama2-13b.Q4_0.gguf")

    callback_manager = BaseCallbackManager([StreamingStdOutCallbackHandler()])

    template = """Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:"""

    prompt = PromptTemplate.from_template(template).partial(
        tools=render_text_description(tools),
        tool_names="".join([tool.name for tool in tools]),
    )
    llm = GPT4All(
        temp=0,
        model=llm_path,
        callback_manager=callback_manager,
        verbose=True,
        stop=["\nObservation"],
    )
    agent = {"input": lambda x: x["input"]}|prompt | llm
    res = agent.invoke({"input": "How long is the word DOG"})
    #print(res)


