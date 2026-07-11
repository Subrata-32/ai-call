import asyncio
import logging
import unittest

from agent import speak_initial_greeting


class FakeSession:
    def __init__(self):
        self.say_calls = []
        self.reply_calls = []

    async def say(self, text, **kwargs):
        self.say_calls.append((text, kwargs))

    async def generate_reply(self, **kwargs):
        self.reply_calls.append(kwargs)


class InitialGreetingTests(unittest.TestCase):
    def test_sends_greeting_with_session_say(self):
        session = FakeSession()
        asyncio.run(speak_initial_greeting(session, "Hello there", logging.getLogger("test")))
        self.assertEqual(session.say_calls[0][0], "Hello there")


if __name__ == "__main__":
    unittest.main()
