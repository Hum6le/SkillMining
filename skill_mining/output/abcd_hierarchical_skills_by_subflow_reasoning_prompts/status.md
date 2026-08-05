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
      "status:pull-up-account",
      "status:verify-identity",
      "status:validate-purchase",
      "status:ask-the-oracle"
    ],
    "supporting_sessions": [
      "5735",
      "8904"
    ]
  },
  "branch_motifs": [],
  "reference_motifs": [
    {
      "attachment": [
        "status:ask-the-oracle",
        "END"
      ],
      "path": [
        "status:update-order"
      ],
      "support": 1,
      "support_sessions": [
        "5735"
      ],
      "representative_traces": [
        {
          "session_id": "5735",
          "dialogue": "hello! thank you for choosing acmebrands. how may i assist you? i need to make sure i am getting my package tomorrow i got an email stating it would be delivered some other time i'll be glad to help you with that. would you provide your full name, please? joseph banter account has been pulled up for joseph banter. thank you, joseph! to continue assisting you, may i please get your account id and order id? account id: <account_id> <order_id> identity verification in progress ... thank you! to con",
          "local_path": [
            "status:update-order"
          ],
          "start_step": 4,
          "end_step": 4
        }
      ],
      "semantic_status": "evidence_pending",
      "layer": "reference"
    }
  ]
}
```
