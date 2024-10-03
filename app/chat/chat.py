import random

from langchain_community.chat_models import ChatOpenAI

from app.chat.chains.retrieval import StreamingConversationalRetrievalChain
from app.chat.models import ChatArgs
from app.chat.vector_stores import retriever_map
from app.chat.llm import llm_map
from app.chat.memory import memory_map

from app.web.api import (
    get_conversation_components,
    set_conversation_components
)

component_type_map = {
    "retriever": retriever_map,
    "memory": memory_map,
    "llm": llm_map
}


def select_component(
        component_type: str,
        component_name: str
):
    component_map = component_type_map[component_type]

    if not component_name:
        component_name = random.choice(list(component_map.keys()))

    return component_name, component_map[component_name]


def select_components(types, components):
    selected_components = {}
    for component_type in types:
        component_name = components[component_type]
        selected_components[component_type] = select_component(component_type, component_name)

    return selected_components


def build_components(components, chat_args):
    built_components = {}
    for component_type in list(components.keys()):
        built_components[component_type] = components[component_type][1](chat_args)

    return built_components


def get_components(conversation_id):
    return get_conversation_components(conversation_id)


def save_components(conversation_id, components):
    set_conversation_components(
        conversation_id=conversation_id,
        llm=components["llm"][0],
        memory=components["memory"][0],
        retriever=components["retriever"][0]
    )


def build_chat(chat_args: ChatArgs):
    components = get_components(chat_args.conversation_id)

    selected_components = select_components(
        list(component_type_map.keys()),
        components
    )

    print(selected_components)

    save_components(chat_args.conversation_id, selected_components)

    built_components = build_components(selected_components, chat_args)

    llm = built_components["llm"]
    condense_question_llm = ChatOpenAI(streaming=False)
    memory = built_components["memory"]
    retriever = built_components["retriever"]

    return StreamingConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        condense_question_llm=condense_question_llm,
        retriever=retriever
    )

