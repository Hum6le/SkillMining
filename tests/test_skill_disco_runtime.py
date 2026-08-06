from __future__ import annotations

import unittest

from skill_disco.runtime import build_skill_disco_workflow


class SkillDiscoRuntimeTest(unittest.TestCase):
    def test_library_is_injected_as_label_free_procedural_guidance(self) -> None:
        workflow = build_skill_disco_workflow("## Skill: recover_account_password\nProcedure:\n1. make-password()")
        prompt = workflow.format_prompt()
        self.assertIn("Infer the relevant skill from the current dialogue only", prompt)
        self.assertIn("recover_account_password", prompt)
        self.assertNotIn("subflow:", prompt.lower())


if __name__ == "__main__":
    unittest.main()
