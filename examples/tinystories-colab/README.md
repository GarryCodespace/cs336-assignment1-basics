# TinyStories Colab training result

A 1,901,696-parameter Transformer trained on a Tesla T4 on 5 November 2026.
The model has four layers, a hidden dimension of 128, a context length of 128,
and a 4,096-token BPE vocabulary.

| Metric | Result |
| --- | ---: |
| Completed updates | 6,000 |
| Model training time (excludes data preparation) | 157.9 seconds |
| Tokens processed during training | 12,288,000 |
| Final validation loss | 2.3963 |
| Final validation perplexity | 10.98 |

BPE was trained on the full training split. Preparation produced 558,691,196
training tokens and 5,641,010 validation tokens. The short model run sampled
12.29 million tokens from the prepared training corpus; it did not process a
full epoch.

See [metrics.jsonl](./metrics.jsonl), [config.json](./config.json), and
[summary.json](./summary.json) for the recorded run data. Reproduce the pipeline
with [the Colab notebook](../../notebooks/CS336_TinyStories_GPU.ipynb).
Corpora, encoded token files, and the model checkpoint remain local and are
excluded from Git.

## Generated stories

These are unedited model outputs sampled at temperature 0.8 and top-p 0.95,
with up to 200 new tokens. The model produces simple story-like text, with
repetition and inconsistent details at this small training budget. The JSON
outputs are in [samples.json](./samples.json).

### Once upon a time

Once upon a time, there was a little dog named Max. Max loved to play in the park with his friends. One day, Max went to the park with his friends.
While playing, Max saw a big tree. He was curious and wanted to find out what was inside. Max did not want to run fast. He asked his friend, a kind boy named Tim, "I can help you get a new toy to play with."
Max thought for a moment and said, "I will help you. I will help you!" Max was very happy. They played with the toy all day long. The toy was not a toy, but he was glad to be friends.
<|endoftext|>

### One day, a little girl named Lily

One day, a little girl named Lily found a bottle of sugar. She wanted to make her small mess. She wanted to fix it, but her mom said, "No, Tim. I am too high for you. Let's get some of the other toys together."
Lily and her mom walked to the kitchen. They saw the Buddy and picked it up. They went inside and saw many things. They were in the kitchen. But then, something unexpected happened. A big dog jumped from the tree and started to play. Lily was sad and said, "I'm sorry, Max. I just wanted to play with the dog. Let's play together again and make a better friend."
Lily nodded and knew that her time, Lily could help Tim. They flew down together to the park and show them all. They sang a happy song about the unexpected shapes that could not go away. Lily and Max learned that they should not go back to play together and not hurt them anymore.
<|endoftext|>

### Tom and his dog

Tom and his dog, Max. He wanted to play with Lily.
"Can I have someCan I play with you?" Lily asked.
"Sure, but I can be lonely. Thank you for fixing my car." Max said. He liked his ice cream too.
"No, Lily. Thank you, Max," Lily said.
They ran to the store and put the car away. They got the car back and forth. They were not happy. They felt very proud and safe. They played with the car and the car all day. They had so much fun.
<|endoftext|>
