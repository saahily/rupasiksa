# रूपशिक्षा (rūpaśikṣā)

Can LLMs learn Pāṇinian generativity? रूपशिक्षा (_rūpaśikṣā_: learning proper forms) is an RL environment that asks a model to generate the correct Sanskrit surface form for a morphological specification — root, verb class, tense/mood, voice, person, number — and verifies the answer against Pāṇini's Aṣṭādhyāyī using [vidyut-prakriya](https://github.com/ambuda-org/vidyut).

I first built this in May 2025 as [a PR to prime-rl](https://github.com/KhoomeiK/prime-rl/pull/3), when reward functions lived in the genesys registry. The ecosystem has since standardized on [verifiers](https://github.com/PrimeIntellect-ai/verifiers), so it now lives here as a standalone environment: you can eval any API model against it, or train on it with [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl).

## Why Sanskrit morphology

The Aṣṭādhyāyī is a complete generative grammar, written ~2,400 years before Chomsky: about 4,000 rules that derive every valid Sanskrit word form. That makes correctness decidable — no LLM judge, no reference answers, just the rules.

The space is also too large to memorize. Roughly 2,300 _dhātus_ (roots) crossed with 10 _lakāras_ (tense/moods), 2 _prayogas_ (voices), 3 _puruṣas_ (persons), and 3 _vacanas_ (numbers) yields hundreds of thousands of finite verb forms, most of which appear in no corpus at all. Sanskrit is thin in pretraining data on top of that, so a model that scores well here is applying the derivation system, not recalling text.

## Scoring

For each specification, the reward function derives all valid surface forms with vidyut-prakriya and checks the model's answer against them:

- An exact match with any valid form scores 1.0. Ubhayapadī roots have both _parasmaipada_ and _ātmanepada_ forms; either counts.
- An answer matching an intermediate form in the _prakriyā_ (derivation) earns partial credit, scaled by how far along the derivation that form appears. A model that stops deriving too early gets rewarded for the distance it covered.
- Answers are accepted in any script [vidyut-lipi](https://github.com/ambuda-org/vidyut/tree/main/vidyut-lipi) can detect — Devanagari, IAST, SLP1, Harvard-Kyoto, Telugu, Brahmi, and a dozen others.

A task looks like this:

```
Generate the correct Sanskrit surface form for the following morphological specification:

Dhātu: BU
Gaṇa: BvAdi
Lakāra: la~w
Prayoga: kartari
Puruṣa: praTama
Vacana: eka

Please provide the correct surface form following Pāṇinian grammar rules, and in the format: [[surface_form]]
```

The correct answer is `[[Bavati]]` (भवति), for a reward of 1.0. The test suite in [tests/test_reward.py](tests/test_reward.py) has full derivation traces for the partial-credit cases if you want to see how the prakriyā scoring works step by step.

## Usage

```bash
# eval an API model
uv tool install prime
prime env install saahily/rupasiksa
prime eval run rupasiksa -m openai/gpt-5-mini

# or from source
git clone https://github.com/saahily/rupasiksa && cd rupasiksa
uv venv && uv pip install -e ".[dev]"
uv run pytest
```

```python
from rupasiksa import load_environment

env = load_environment(max_eval_examples=200)
```

## Dataset

[saahily/sanskrit-morphology](https://huggingface.co/datasets/saahily/sanskrit-morphology): 200,000 unique _tiṅanta_ (finite verb) specifications sampled from the Dhātupāṭha — about 30% of all possible combinations without _sanādi pratyayas_ (derivational suffixes) or _upasargas_ (prefixes) — split 90/10 train/test. `scripts/build_dataset.py` regenerates it.

## Future improvements

- Held-out splits over dhātu × lakāra combinations, to separate rule generalization from interpolation
- Support for _subantas_ (nominals), _kṛdantas_ (participles), and _taddhitāntas_ (secondary derivatives)
- More verbal complexity via sanādi pratyayas and upasargas

Grammar engine by [Vidyut](https://github.com/ambuda-org/vidyut), from the Ambuda project.
