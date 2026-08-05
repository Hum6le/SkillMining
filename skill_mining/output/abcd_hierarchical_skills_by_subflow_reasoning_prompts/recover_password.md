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
      "recover_password:verify-identity",
      "recover_password:enter-details",
      "recover_password:make-password"
    ],
    "supporting_sessions": [
      "5231",
      "6601"
    ]
  },
  "branch_motifs": [],
  "reference_motifs": [
    {
      "attachment": [
        "START",
        "recover_password:enter-details"
      ],
      "path": [
        "recover_password:pull-up-account"
      ],
      "support": 1,
      "support_sessions": [
        "6601"
      ],
      "representative_traces": [
        {
          "session_id": "6601",
          "dialogue": "hello, how can i help you today hi i forgot my password to my account. my name is crystal minh. account has been pulled up for  crystal minh.. okay, could i get your username please <username> details of <username> have been entered. a password has been generated. okay, here is your new password 3mihalbfbem you can log in and change it again if you want to. is there anything else i can help you with great. thanks that's all okay, have a nice day",
          "local_path": [
            "recover_password:pull-up-account"
          ],
          "start_step": 0,
          "end_step": 0
        }
      ],
      "semantic_status": "evidence_pending",
      "layer": "reference"
    },
    {
      "attachment": [
        "START",
        "recover_password:verify-identity"
      ],
      "path": [
        "recover_password:pull-up-account"
      ],
      "support": 1,
      "support_sessions": [
        "5231"
      ],
      "representative_traces": [
        {
          "session_id": "5231",
          "dialogue": "hello and thank you for contacting acmebrand. how may i help you? hello! i'd like to check my order status, but i forgot my password. oh alright, that shou;dn't be too much of an issue. let's try to get your password reset. thank you could you give me you full name so i can pull up your account please? yes, it is alessandro phoenix account has been pulled up for alessandro phoenix. alright the next bit of information i'll need is your username. do you have that available? i don't, sorry. would m",
          "local_path": [
            "recover_password:pull-up-account"
          ],
          "start_step": 0,
          "end_step": 0
        }
      ],
      "semantic_status": "evidence_pending",
      "layer": "reference"
    }
  ]
}
```
