import operator
from typing import Annotated, Any, TypedDict, cast

from langchain.agents import create_agent  # create_react_agent is now deprecated
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from src.prompts import (
    MEMORY_SUMMARY_PROMPT,
    get_chat_prompt_template,
    get_intent_classification_prompt,
)
from src.schemas import (
    ALLOWED_GRAPH_SCHEMAS,
    AnswerResponse,
    CalculationResponse,
    SummarizationResponse,
    UpdateMemoryResponse,
    UserIntent,
)


# TODO: The AgentState class is already implemented for you.  Study the
# structure to understand how state flows through the LangGraph
# workflow.  See README.md Task 2.1 for detailed explanations of
# each property.
class AgentState(TypedDict):
    """
    The agent state object
    """

    # Current conversation
    user_input: str | None
    messages: Annotated[list[BaseMessage], add_messages]

    # Intent and routing
    intent: UserIntent | None
    next_step: str

    # Memory and context
    conversation_summary: str
    active_documents: list[str] | None

    # Current task state
    current_response: dict[str, Any] | None
    tools_used: list[str]

    # Session management
    session_id: str | None
    user_id: str | None

    # TODO: Modify actions_taken to use an operator.add reducer
    actions_taken: Annotated[list[str], operator.add]


def invoke_react_agent(
    response_schema: type[BaseModel], messages: list[BaseMessage], llm, tools
) -> tuple[dict[str, Any], list[str]]:
    llm_with_tools = llm.bind_tools(tools)

    agent = create_agent(
        model=llm_with_tools,  # Use the bound model
        tools=tools,
        response_format=response_schema,
    )

    result = agent.invoke({"messages": messages})
    tools_used = [
        t.name for t in result.get("messages", []) if isinstance(t, ToolMessage)
    ]

    return (result, tools_used)


# TODO: Implement the classify_intent function.
# This function should classify the user's intent and set the next step in the workflow.
# Refer to README.md Task 2.2
def classify_intent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Classify user intent and update next_step. Also records that this
    function executed by appending "classify_intent" to actions_taken.
    """

    llm = config.get("configurable", {}).get("llm")
    history = state.get("messages", [])

    # TODO Configure the llm chat model for structured output
    structured_llm = llm.with_structured_output(UserIntent)

    # TODO Create a formatted prompt with conversation history and user input
    prompt = get_intent_classification_prompt()

    chain = prompt | structured_llm

    intent_type = chain.invoke(
        {
            "user_input": state.get("user_input"),
            "conversation_history": state.get("messages"),
        }
    ).intent_type

    match intent_type:
        case "qa":
            next_step = "qa_agent"
        case "summarization":
            next_step = "summarization_agent"
        case "calculation":
            next_step = "calculation_agent"
        case _:
            next_step = "qa_agent"

    # TODO: Add conditional logic to set next_step based on intent

    return cast(
        AgentState,
        {
            "actions_taken": ["classify_intent"],
            "intent": intent_type,
            "next_step": next_step,
            # TODO: Update state intent and next_step
        },
    )


def qa_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle Q&A tasks and record the action.
    """
    llm = config.get("configurable", {}).get("llm")
    tools = config.get("configurable", {}).get("tools")

    prompt_template = get_chat_prompt_template("qa")

    messages = prompt_template.invoke(
        {
            "input": state["user_input"],
            "chat_history": state.get("messages", []),
        }
    ).to_messages()

    result, tools_used = invoke_react_agent(AnswerResponse, messages, llm, tools)

    return cast(
        AgentState,
        {
            "messages": result.get("messages", []),
            "actions_taken": ["qa_agent"],
            "current_response": result,
            "tools_used": tools_used,
            "next_step": "update_memory",
        },
    )


# TODO: Implement the summarization_agent function. Refer to README.md Task 2.3
def summarization_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle summarization tasks and record the action.
    """
    llm = config.get("configurable", {}).get("llm")
    tools = config.get("configurable", {}).get("tools")

    prompt_template = get_chat_prompt_template("summarization")

    messages = prompt_template.invoke(
        {
            "input": state["user_input"],
            "chat_history": state.get("messages", []),
        }
    ).to_messages()

    result, tools_used = invoke_react_agent(SummarizationResponse, messages, llm, tools)

    return cast(
        AgentState,
        {
            "messages": result.get("messages", []),
            "actions_taken": ["summarization_agent"],
            "current_response": result,
            "tools_used": tools_used,
            "next_step": "update_memory",
        },
    )


# TODO: Implement the calculation_agent function. Refer to README.md Task 2.3
def calculation_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle calculation tasks and record the action.
    """
    llm = config.get("configurable", {}).get("llm")
    tools = config.get("configurable", {}).get("tools")

    prompt_template = get_chat_prompt_template("calculation")

    messages = prompt_template.invoke(
        {
            "input": state["user_input"],
            "chat_history": state.get("messages", []),
        }
    ).to_messages()

    result, tools_used = invoke_react_agent(CalculationResponse, messages, llm, tools)

    return cast(
        AgentState,
        {
            "messages": result.get("messages", []),
            "actions_taken": ["calculation_agent"],
            "current_response": result,
            "tools_used": tools_used,
            "next_step": "update_memory",
        },
    )


# TODO: Finish implementing the update_memory function. Refer to README.md Task 2.4
def update_memory(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Update conversation memory and record the action.
    """

    # TODO: Retrieve the LLM from config

    llm = config.get("configurable", {}).get("llm")

    prompt_with_history = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(MEMORY_SUMMARY_PROMPT),
            MessagesPlaceholder("chat_history"),
        ]
    ).invoke(
        {
            "chat_history": state.get("messages", []),
        }
    )

    structured_llm = llm.with_structured_output(
        UpdateMemoryResponse
        # TODO Pass in the correct schema from scheams.py to extract conversation summary, active documents
    )

    response = structured_llm.invoke(prompt_with_history)
    return cast(
        AgentState,
        {
            "conversation_summary": response.summary,  # TODO: Extract summary from response
            "active_documents": response.document_ids,  # TODO: Update with the current active documents
            "next_step": END,  # TODO: Update the next step to end
        },
    )


def should_continue(state: AgentState) -> str:
    """Router function"""
    return state.get("next_step", "end")


# TODO: Complete the create_workflow function. Refer to README.md Task 2.5
def create_workflow(llm, tools):
    """
    Creates the LangGraph agents.
    Compiles the workflow with an InMemorySaver checkpointer to persist state.
    """
    workflow = StateGraph(AgentState)

    # TODO: Add all the nodes to the workflow by calling workflow.add_node(...)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("qa_agent", qa_agent)
    workflow.add_node("summarization_agent", summarization_agent)
    workflow.add_node("calculation_agent", calculation_agent)
    workflow.add_node("update_memory", update_memory)

    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        should_continue,
        {
            # TODO: Map the intent strings to the correct node names
            "qa_agent": "qa_agent",
            "summarization_agent": "summarization_agent",
            "calculation_agent": "calculation_agent",
            "end": END,
        },
    )

    # TODO: For each node add an edge that connects it to the update_memory node
    # qa_agent -> update_memory
    # summarization_agent -> update_memory
    # calculation_agent -> update_memory
    workflow.add_edge("qa_agent", "update_memory")
    workflow.add_edge("summarization_agent", "update_memory")
    workflow.add_edge("calculation_agent", "update_memory")
    workflow.add_edge("update_memory", END)

    # Initialise a serialiser for use with the schemas that we have mainlisted
    custom_serialiser = JsonPlusSerializer(
        allowed_msgpack_modules=ALLOWED_GRAPH_SCHEMAS
    )

    # TODO Modify the return values below by adding a checkpointer with InMemorySaver
    checkpointer = InMemorySaver(serde=custom_serialiser)
    return workflow.compile(checkpointer=checkpointer)
