import asyncio
import unittest

from agent import register_session_handlers


class FakeSession:
    def __init__(self):
        self.handlers = {}
        self.started = False

    def on(self, event_name, handler):
        self.handlers[event_name] = handler

    async def start(self, *args, **kwargs):
        self.started = True
        if "user_speech_committed" not in self.handlers:
            raise AssertionError("user speech handler was not registered before session start")


class SessionHandlerTests(unittest.TestCase):
    def test_registers_user_speech_handler_before_session_start(self):
        session = FakeSession()

        register_session_handlers(
            session=session,
            on_agent_speech_started=lambda ev: None,
            on_agent_speech_finished=lambda ev: None,
            on_agent_speech_interrupted=lambda ev: None,
            on_user_speech_committed=lambda ev: None,
        )

        asyncio.run(session.start())


if __name__ == "__main__":
    unittest.main()
