import random
import string

from rlm import RLM
from rlm.logger import RLMLogger

secret_number = random.randint(100_000_000, 999_999_999)
filler_lines = ["".join(random.choices(string.ascii_lowercase + " ", k=120)) for _ in range(50_000)]
insert_at = random.randint(len(filler_lines) // 3, 2 * len(filler_lines) // 3)
filler_lines.insert(insert_at, f"SECRET_NUMBER={secret_number}")
haystack = "\n".join(filler_lines)

rlm = RLM(
    backend="vllm",
    backend_kwargs={
        "model_name": "qwen3.6:35b-a3b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "sampling_args": {"max_tokens": 2000},
        "timeout": 600,
    },
    environment="local",
    max_iterations=10,
    logger=RLMLogger(log_dir="./logs"),
    verbose=True,
)

result = rlm.completion(
    "The context contains ~50k lines of random text with a single line "
    "matching the pattern SECRET_NUMBER=<digits>. Find and return ONLY the "
    f"numeric value.\n\n{haystack}"
)

print(f"\nModel found: {result.response}")
print(f"Actual number: {secret_number}")
print(f"Correct: {str(secret_number) in result.response}")
