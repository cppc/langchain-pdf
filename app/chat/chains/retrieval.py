from langchain.chains import ConversationalRetrievalChain
from .streamable import StreamableChain


class StreamingConversationalRetrievalChain(
    StreamableChain,
    ConversationalRetrievalChain
):
    pass
