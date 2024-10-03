from flask import current_app
from queue import Queue
from threading import Thread

from app.chat.callbacks import StreamingHandler


class StreamableChain:
    def stream(self, input_prompt):
        queue = Queue()
        handler = StreamingHandler(queue)

        def task(app_context):
            app_context.push()
            self(input_prompt, callbacks=[handler])

        Thread(target=task, args=[
            current_app.app_context()
        ]).start()

        token = ""
        while token is not None:
            token = queue.get()
            if token is not None:
                yield token

