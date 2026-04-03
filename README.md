# --- patch: save gold_index for multiple_choice tasks ---

# place it in lm_eval.evaluator.evaluate() right after:


if log_samples:
    target = task.doc_to_target(doc)
    example = {
        "doc_id": doc_id_true,
        "doc": doc,
        "target": target,
        "arguments": [req.args for req in requests],
        "resps": [req.resps for req in requests],
        "filtered_resps": [
            req.filtered_resps[filter_key] for req in requests
        ],
        "filter": filter_key,
        "metrics": list(metrics.keys()),
        "doc_hash": hash_string(
            json.dumps(
                requests[0].doc,
                indent=2,
                default=handle_non_serializable,
                ensure_ascii=False,
            )
        ),
        "prompt_hash": hash_string(requests[0].arguments[0]),
        "target_hash": hash_string(str(target)),
    }


## patch code: 

    # --- start patch ---
    if task.OUTPUT_TYPE == "multiple_choice":
        _gold = (
            task.doc_to_text(doc)
            if task.multiple_input
            else target
        )
        _choices = task.doc_to_choice(doc)
        if isinstance(_gold, str):
            _gold = _choices.index(_gold) if _gold in _choices else -100
        elif isinstance(_gold, list):
            _gold = [
                _choices.index(g) if isinstance(g, str) and g in _choices else g
                for g in _gold
            ]
            
        extra_fields = dict(
            gold_index = _gold,
            choices=_choices,
            choice_lengths=[float(len(i)) for i in _choices],
            doc_index=doc_id_true,
        )
            
        example["extra_fields"] = extra_fields
    # --- end patch ---

## --- end patch ---