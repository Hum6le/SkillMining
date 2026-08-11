from __future__ import annotations

import unittest

from asi_offline import build_asi_workflow, render_asi_library


def _function(name: str, episode_id: str) -> dict:
    return {
        "episode_id": episode_id,
        "name": name,
        "parameters": ["username"],
        "action_start_index": 0,
        "action_end_index": 2,
        "function_source": "def ignored_source(username): pass",
        "action_template": [
            {"action": "pull-up-account", "arguments": ["username"]},
            {"action": "enter-details", "arguments": ["username"]},
            {"action": "make-password", "arguments": []},
        ],
    }


class ASIOfflineLibraryTest(unittest.TestCase):
    def test_freezes_first_definition_and_renders_parameterized_procedure(self) -> None:
        library = render_asi_library([_function("recover_password", "one"), _function("recover_password", "two")])
        self.assertEqual(len(library.functions), 1)
        self.assertEqual(library.functions[0]["episode_id"], "one")
        self.assertEqual(library.duplicate_names[0]["episode_id"], "two")
        self.assertIn("take_action('pull-up-account', [username])", library.rendered_text)
        self.assertNotIn("ignored_source", library.rendered_text)

    def test_runtime_is_label_hidden_and_requires_primitive_predictions(self) -> None:
        workflow = build_asi_workflow(render_asi_library([_function("recover_password", "one")]).rendered_text)
        prompt = workflow.format_prompt()
        self.assertIn("no scenario or subflow label is available", prompt.replace("\n", " "))
        self.assertIn("primitive ABCD action", prompt.replace("\n", " "))
        self.assertIn("recover_password", prompt)


if __name__ == "__main__":
    unittest.main()
