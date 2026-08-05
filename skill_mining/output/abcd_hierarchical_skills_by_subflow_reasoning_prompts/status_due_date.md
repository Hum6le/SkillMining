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
      "status_due_date:pull-up-account",
      "status_due_date:verify-identity",
      "status_due_date:subscription-status",
      "status_due_date:enter-details"
    ],
    "supporting_sessions": [
      "10365",
      "6156"
    ]
  },
  "branch_motifs": [],
  "reference_motifs": [
    {
      "attachment": [
        "status_due_date:enter-details",
        "END"
      ],
      "path": [
        "status_due_date:send-link"
      ],
      "support": 1,
      "support_sessions": [
        "10365"
      ],
      "representative_traces": [
        {
          "session_id": "10365",
          "dialogue": "hello, thanks for contacting acmebrands. how can i help today? hi, this is albert sanders, gold member i have a premium subscription with acme brands, but i'm having a bit of trouble finding my account details i see, are you needing help accessing your account? yes, well, really i just want to know when my annual fee is due.  i can pay it today if there's anything outstanding account has been pulled up for albert sanders. do you have your account id and order id albert? yes, i do: <account_id> <",
          "local_path": [
            "status_due_date:send-link"
          ],
          "start_step": 4,
          "end_step": 4
        }
      ],
      "semantic_status": "evidence_pending",
      "layer": "reference"
    },
    {
      "attachment": [
        "status_due_date:enter-details",
        "END"
      ],
      "path": [
        "status_due_date:update-account"
      ],
      "support": 1,
      "support_sessions": [
        "6156"
      ],
      "representative_traces": [
        {
          "session_id": "6156",
          "dialogue": "hi! how can i help you? yes i would like  to check the status of my subscription. i would like to keep. trying to see when the annual fee is due and if its today i would like to pay it, if i owe anything sure, i can look into that for you. what is your full name? alessandro phoenix and my account id <account_id> also my order id is  <order_id> account has been pulled up for alessandro phoenix. identity verification in progress ... querying the system for subscription status ... thank you for all",
          "local_path": [
            "status_due_date:update-account"
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
