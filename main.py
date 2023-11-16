from typing import Union, List

from dotenv import load_dotenv
from langchain.agents import tool
from langchain.callbacks import StreamingStdOutCallbackHandler, StdOutCallbackHandler
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.callbacks.base import BaseCallbackManager
from langchain.chat_models import ChatOpenAI
from langchain.llms.gpt4all import GPT4All
from langchain.prompts import PromptTemplate
from langchain.schema import AgentFinish, AgentAction
from langchain.schema.runnable import RunnableConfig
from langchain.tools.render import render_text_description
from langchain.agents.format_scratchpad.log import format_log_to_str
import os

load_dotenv()


### the tool decorator from langchain helps us define a function as a langchain agent tool
### Creates an instance from the tool class in langchain and populates the propeties for you
@tool
def get_text_length(text: str) -> int:
    """
    returns the length of a text by characters
    :param text: a string of text
    :return: an integer that represents the number of characters in the string text
    """
    text_cleaned = text.replace('"', "").replace("\n", "")
    return len(text_cleaned)


##


def get_tool_from_name(tools: List[str], tool_name: str) -> tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Could not find tool with name{tool_name}")


if __name__ == "__main__":
    tools = [get_text_length]
    gpt4all_path = os.getenv("GPT4ALL_PATH")

    # Concatenate with another path or filename
    llm_path = os.path.join(gpt4all_path, "nous-hermes-llama2-13b.Q4_0.gguf")

    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the name of action to take, should be one of [{tool_names}]
Action Input: the input to the action with quotations
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template).partial(
        tools=render_text_description(tools),
        tool_names="".join([tool.name for tool in tools]),
    )
    llm = GPT4All(temp=0, model=llm_path, verbose=True, streaming=True, max_tokens=1000)
    intermediate_steps = []
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm.bind(stop=["Observation"])
        | ReActSingleInputOutputParser()
    )

    agent_step = ""
    while not isinstance(agent_step, AgentFinish):
        ##we defined agent_step to be one of two types
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "what is the length in character of the text: Dog? ",
                "agent_scratchpad": intermediate_steps,
            }
        )

        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool = get_tool_from_name(tools, tool_name)
            tool_input = agent_step.tool_input

            observation = tool.func(tool_input)
            intermediate_steps.append((agent_step, str(observation)))

    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)
