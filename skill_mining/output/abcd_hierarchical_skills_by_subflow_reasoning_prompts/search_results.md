# Semantic Motif Interpretation

You are given a mined workflow motif. Infer only semantic properties supported by the structural evidence and representative traces. Do not invent business rules. If a property is not identifiable, return `unknown`.

Return JSON with this schema:
```json
{
  "semantic_name": "...",
  "goal": "...",
  "parameters": [],
  "preconditions": [],
  "postconditions": [],
  "branches": [
    {"guard": "...", "meaning": "...", "evidence_session_ids": []}
  ],
  "confidence": 0.0,
  "unknowns": []
}
```

## Mined evidence

```json
{
  "instruction": "Infer semantic names, preconditions, effects and branch guards only from the structural motif and its representative traces. Return an abstention when evidence is insufficient.",
  "backbone_motif": {
    "path": [
      "search_results:search-faq",
      "search_results:log-out-in",
      "search_results:instructions"
    ],
    "supporting_sessions": [
      "5177",
      "399"
    ]
  },
  "branch_motifs": [],
  "reference_motifs": [
    {
      "attachment": [
        "search_results:log-out-in",
        "search_results:instructions"
      ],
      "path": [
        "search_results:try-again"
      ],
      "support": 1,
      "support_sessions": [
        "5177"
      ],
      "representative_traces": [
        {
          "session_id": "5177",
          "dialogue": "acmebrands, how may i help you? the search bar isn't working let me help you with that. are the search results showing no results found? yes agent is looking for solutions ... could you log out and then log back in, and try to search again? i just tried that, it still didn't work. ok agent is looking for solutions ... what product are you searching for? jacket can you put a different product in the search engine, like boots? i put boots, nothing showing up. i am sorry that nothing so far is work",
          "local_path": [
            "search_results:try-again"
          ],
          "start_step": 1,
          "end_step": 1
        }
      ],
      "semantic_status": "evidence_pending",
      "layer": "reference"
    }
  ]
}
```
