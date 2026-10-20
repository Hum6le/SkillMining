"""Bounded subprocess target: stop a compiled skill at its first unknown step."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skill_disco.synthesis import validate_skill_source
from skill_disco.callable_runtime import _SAFE_BUILTINS


class Pending(Exception):
    def __init__(self, action, slots):
        self.action, self.slots = action, slots


class Prefix:
    def __init__(self, previous, vocabulary):
        self.previous, self.vocabulary, self.position = previous, vocabulary, 0

    def step(self, action, slots):
        slots = [str(v) for v in slots]
        if self.position == len(self.previous):
            if action not in self.vocabulary:
                raise ValueError('Unknown policy label')
            raise Pending(action, slots)
        expected = self.previous[self.position]
        if action != expected['name'] or slots != expected['slots']:
            raise ValueError('Predicted prefix mismatch')
        self.position += 1
        return expected['observation'], sorted(self.vocabulary)


def main():
    p = json.load(sys.stdin)
    validate_skill_source(p['source'], p['skill_name'])
    candidates = []
    for start in range(len(p['previous'])+1):
        env = Prefix(p['previous'][start:], p['vocabulary'])
        scope = {'__builtins__': _SAFE_BUILTINS, 'env': env}
        try:
            exec(compile(p['source'], '<sgd-skill>', 'exec'), scope)
            scope[p['skill_name']](**p['arguments'])
        except Pending as pending:
            candidates.append((env.position, start, {'heads': p['vocabulary'][pending.action],
                'serialized_payload': pending.slots, 'matched_prior_policies': env.position}))
        except Exception:
            continue
    result = max(candidates, key=lambda row: (row[0], row[1]))[2] if candidates else None
    print(json.dumps({'projection': result}))


if __name__ == '__main__':
    main()
