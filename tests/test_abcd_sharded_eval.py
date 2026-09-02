import unittest

from eval_tod.abcd.sharded_eval import merge_turn_results, parse_workflow_ids, shard_conversations


class ShardedEvaluationTest(unittest.TestCase):
    def test_parse_and_shard(self):
        self.assertEqual(parse_workflow_ids("a,b,a, c"), ["a", "b", "c"])
        conversations = [{"convo_id": str(i)} for i in range(5)]
        self.assertEqual([r["convo_id"] for r in shard_conversations(conversations, 0, 2)], ["0", "2", "4"])
        self.assertEqual([r["convo_id"] for r in shard_conversations(conversations, 1, 2)], ["1", "3"])

    def test_merge(self):
        conversations = [{"convo_id": "a"}, {"convo_id": "b"}]
        merged = merge_turn_results(conversations, [[
            {"convo_id": "b", "turn_index": 2},
            {"convo_id": "a", "turn_index": 1},
        ]])
        self.assertEqual([row["convo_id"] for row in merged], ["a", "b"])
        with self.assertRaises(ValueError):
            merge_turn_results(conversations, [[{"convo_id": "a", "turn_index": 1}]])


if __name__ == "__main__":
    unittest.main()
