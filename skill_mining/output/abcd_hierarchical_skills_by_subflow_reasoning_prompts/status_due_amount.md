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
      "status_due_amount:verify-identity",
      "status_due_amount:subscription-status",
      "status_due_amount:send-link"
    ],
    "supporting_sessions": [
      "960",
      "4019"
    ]
  },
  "branch_motifs": [],
  "reference_motifs": [
    {
      "attachment": [
        "START",
        "status_due_amount:subscription-status"
      ],
      "path": [
        "status_due_amount:pull-up-account"
      ],
      "support": 1,
      "support_sessions": [
        "960"
      ],
      "representative_traces": [
        {
          "session_id": "960",
          "dialogue": "hi, how can i help you? i'm needing to check on the status of my subscription. let me check on that for you. one moment please. can i have your full name or account id? crystal minh account has been pulled up for crystal minh. my account id is <account_id> and what is your question about your subscription? i would like to check on the status to make sure i still have it. do you have an order id? <order_id> querying the system for subscription status ... you do have an active subscription. howeve",
          "local_path": [
            "status_due_amount:pull-up-account"
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
        "status_due_amount:verify-identity"
      ],
      "path": [
        "status_due_amount:pull-up-account"
      ],
      "support": 1,
      "support_sessions": [
        "4019"
      ],
      "representative_traces": [
        {
          "session_id": "4019",
          "dialogue": "hi, how can i help you? what is the status of my subscription i want to keep it sure, let me check up on that for you. what is your full name? norman bouchard account has been pulled up for norman bouchard. thank you, can i have your account id and order id? <order_id> <account_id> identity verification in progress ... querying the system for subscription status ... your subscription is inactive as your bill was due yesterday. a link will be sent. i've sent you a link where you can check on your",
          "local_path": [
            "status_due_amount:pull-up-account"
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
        "status_due_amount:send-link",
        "END"
      ],
      "path": [
        "status_due_amount:enter-details"
      ],
      "support": 1,
      "support_sessions": [
        "4019"
      ],
      "representative_traces": [
        {
          "session_id": "4019",
          "dialogue": "hi, how can i help you? what is the status of my subscription i want to keep it sure, let me check up on that for you. what is your full name? norman bouchard account has been pulled up for norman bouchard. thank you, can i have your account id and order id? <order_id> <account_id> identity verification in progress ... querying the system for subscription status ... your subscription is inactive as your bill was due yesterday. a link will be sent. i've sent you a link where you can check on your",
          "local_path": [
            "status_due_amount:enter-details"
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
