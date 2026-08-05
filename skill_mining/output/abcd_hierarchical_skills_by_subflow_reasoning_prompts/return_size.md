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
      "return_size:pull-up-account",
      "return_size:validate-purchase",
      "return_size:membership",
      "return_size:enter-details"
    ],
    "supporting_sessions": [
      "8141",
      "4162",
      "3342"
    ]
  },
  "branch_motifs": [
    {
      "attachment": [
        "return_size:enter-details",
        "END"
      ],
      "path": [
        "return_size:update-order"
      ],
      "support": 2,
      "support_sessions": [
        "4162",
        "3342"
      ],
      "representative_traces": [
        {
          "session_id": "4162",
          "dialogue": "hello, how can i help you hi i want to start a return, the item i received was the wrong size. my name is albert sanders and my order id is <order_id> id happily help you with that. account has been pulled up for albert sanders . thanks, i bought these boots in size 9 but they really hurt my feet okay i found your account, albert. now i need your email address and username <email> username: <username> purchase validation in progress ... thank you i was able to validate your purhcase. now what le",
          "local_path": [
            "return_size:update-order"
          ],
          "start_step": 4,
          "end_step": 4
        },
        {
          "session_id": "3342",
          "dialogue": "hello! thank you for contacting us today. how can i help you? i would like to return an item because it is the wrong size perfect. can i get your name please? norman bouchard account has been pulled up for norman bouchard. thank you, norman. can you please provide me with your username, email address, and the order id also? username: <username> and email address: <email> order id: <order_id> purchase validation in progress ... thank you! do you know what your membership level is? silver membersh",
          "local_path": [
            "return_size:update-order"
          ],
          "start_step": 4,
          "end_step": 4
        }
      ],
      "semantic_status": "evidence_pending",
      "layer": "branch"
    }
  ],
  "reference_motifs": []
}
```
