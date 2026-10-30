"""Grade every sampled response with the math-verification protocol."""


def identity_filter(responses, docs):
    """Preserve the full sample list through upstream's custom filter."""
    return responses


def process_results(doc, results):
    from math_verify import parse, verify

    responses = results[0]
    if not isinstance(responses, list) or not responses:
        raise ValueError("Use the identity filter to retain all repeated responses")
    answer_key = next(key for key in doc if key.lower() == "answer")
    target = parse(f"${doc[answer_key]}$")
    scores = [int(verify(target, parse(response))) for response in responses]
    return {"accuracy": sum(scores) / len(scores)}
