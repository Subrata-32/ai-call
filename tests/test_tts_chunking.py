import unittest

from agent import prepare_tts_text


class TtsChunkingTests(unittest.TestCase):
    def test_keeps_two_sentences_for_intro(self):
        text = "Hello, this is Aryan from RapidX AI. I help businesses with AI voice agents and automation. What kind of business are you running?"
        self.assertEqual(
            prepare_tts_text(text),
            "Hello, this is Aryan from RapidX AI. I help businesses with AI voice agents and automation.",
        )


if __name__ == "__main__":
    unittest.main()
