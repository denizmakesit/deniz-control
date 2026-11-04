"""Offline unit tests for the English voice command parser.

    python -m unittest discover -s tests -p "test_commands.py" -v
"""

import unittest

from deniz_control.voice.commands import CommandParser, normalize
from deniz_control.voice.recognizer import clean_text

APPS = ["Safari", "Notes", "Google Chrome", "Visual Studio Code", "Spotify", "Finder",
        "System Settings", "Calendar"]


class ParserTests(unittest.TestCase):
    def setUp(self):
        self.p = CommandParser(APPS)

    def a(self, text):
        c = self.p.parse(text)
        return c.action, c.arg

    def test_whisper_style_punctuation_and_case(self):
        self.assertEqual(self.a("Double click."), ("double_click", None))
        self.assertEqual(self.a("Right click!"), ("right_click", None))
        self.assertEqual(self.a("Click."), ("click", None))

    def test_fillers(self):
        self.assertEqual(self.a("Hey Deniz, click please."), ("click", None))
        self.assertEqual(self.a("Okay, new tab."), ("hotkey", ("command", "t")))

    def test_control(self):
        self.assertEqual(self.a("Enable control."), ("enable", None))
        self.assertEqual(self.a("disable control"), ("disable", None))
        self.assertEqual(self.a("Stop."), ("disable", None))

    def test_scroll(self):
        self.assertEqual(self.a("Scroll down."), ("scroll", ("down", 1.0)))
        self.assertEqual(self.a("scroll up a lot"), ("scroll", ("up", 2.0)))
        self.assertEqual(self.a("Scroll down a little."), ("scroll", ("down", 0.5)))
        self.assertEqual(self.a("Scroll to the top."), ("hotkey", ("command", "up")))

    def test_type_keeps_original_casing(self):
        self.assertEqual(self.a("Type Hello World!"), ("type", "Hello World"))
        self.assertEqual(self.a("write: Meeting at 5 PM, see you there."),
                         ("type", "Meeting at 5 PM, see you there"))

    def test_real_session_transcripts(self):
        # exact Whisper outputs from a live test session
        self.assertEqual(self.a("-type youtube"), ("type", "youtube"))
        self.assertEqual(self.a("-type youtube to Safari."), ("type", ("youtube", "Safari")))
        self.assertEqual(self.a("Type YouTube."), ("type", "YouTube"))

    def test_open_site_in_app(self):
        self.assertEqual(self.a("Open YouTube in Safari."),
                         ("open_url", ("https://www.youtube.com", "Safari")))
        self.assertEqual(self.a("open github.com in chrome"),
                         ("open_url", ("https://github.com", "Google Chrome")))

    def test_type_with_preposition_but_no_app(self):
        # "in" followed by a non-app must stay part of the text
        self.assertEqual(self.a("type meet me in the lobby"), ("type", "meet me in the lobby"))

    def test_search(self):
        self.assertEqual(self.a("Search for weather in Istanbul."), ("search", "weather in Istanbul"))
        self.assertEqual(self.a("Google best pizza"), ("search", "best pizza"))

    def test_open_apps_and_sites(self):
        self.assertEqual(self.a("Open Safari."), ("open_app", "Safari"))
        self.assertEqual(self.a("open chrome"), ("open_app", "Google Chrome"))
        self.assertEqual(self.a("Launch VS Code."), ("open_app", "Visual Studio Code"))
        self.assertEqual(self.a("Open YouTube."), ("open_url", "https://www.youtube.com"))
        self.assertEqual(self.a("Go to github.com"), ("open_url", "https://github.com"))
        self.assertEqual(self.a("open settings"), ("open_app", "System Settings"))

    def test_misheard_open(self):
        # real Whisper output for a synthesized "open Safari"
        self.assertEqual(self.a("Oh Safari!"), ("open_app", "Safari"))
        self.assertEqual(self.a("Spotify."), ("open_app", "Spotify"))

    def test_quit_app(self):
        self.assertEqual(self.a("Quit Spotify."), ("quit_app", "Spotify"))
        # "close tab" must stay a hotkey, not "quit app 'tab'"
        self.assertEqual(self.a("Close tab."), ("hotkey", ("command", "w")))

    def test_fuzzy_fixed_phrase(self):
        self.assertEqual(self.a("Close tap."), ("hotkey", ("command", "w")))
        self.assertEqual(self.a("right clic"), ("right_click", None))

    def test_hotkeys_and_volume(self):
        self.assertEqual(self.a("Copy."), ("hotkey", ("command", "c")))
        self.assertEqual(self.a("Press enter."), ("hotkey", ("enter",)))
        self.assertEqual(self.a("Volume up!"), ("volume", "up"))
        self.assertEqual(self.a("mute"), ("volume", "mute"))

    def test_unknown_and_empty(self):
        self.assertEqual(self.a("The weather is nice today."), ("unknown", None))
        self.assertEqual(self.a(""), ("unknown", None))
        self.assertEqual(self.a("Uh."), ("unknown", None))

    def test_normalize_keeps_domains(self):
        self.assertEqual(normalize("Open YouTube.com."), "open youtube.com")

    def test_clean_text_drops_non_speech(self):
        self.assertEqual(clean_text(" [BLANK_AUDIO] "), "")
        self.assertEqual(clean_text("(music) Scroll down."), "Scroll down.")


if __name__ == "__main__":
    unittest.main()
