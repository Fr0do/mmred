#!/usr/bin/env python
"""Upload MMReD dataset to HuggingFace in MERA-compatible format.

Usage:
    python scripts/upload_hf_dataset.py --push
    python scripts/upload_hf_dataset.py --dry-run  # generate locally without pushing
    python scripts/upload_hf_dataset.py --push --private --n-questions 100
"""

import argparse
import json
import sys
from pathlib import Path

import datasets

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_mera_dataset import (
    generate_mera_dataset,
    DC_TASKS,
    MERA_SEQ_LENGTHS,
    QTYPE_TO_TASK,
    format_sequence_as_text,
)

REPO_ID = "dondosss/mmred_mera"

# 10 Russian SAP-formatted prompts for MERA evaluation (must match raw_dataset_meta.json).
# Semantic blocks per MERA docs/dataset_formatting.md: Задача / Контекст / Формат ответа / Вопрос / Ответ.
PROMPTS = [
    "Задача:\nПроанализируй перемещения и расположение персонажей по шагам.\n\nКонтекст:\nПоследовательность шагов показывает, какие персонажи находятся в каких комнатах.\n{context}\n\nФормат ответа:\nОтветь одним словом или одним числом в формате: Ответ: X\n\nВопрос:\n{question}\n\nОтвет:",
    "Задача:\nОпредели ответ на вопрос по последовательности состояний комнат.\n\nКонтекст:\nНа каждом шаге перечислены комнаты и находящиеся в них персонажи.\n{context}\n\nФормат ответа:\nУкажи один ответ: имя персонажа, название комнаты или число. Последняя строка ответа должна иметь формат: Ответ: X\n\nВопрос:\n{question}\n\nОтвет:",
    "Задача:\nРеши задачу о том, где и сколько времени находились персонажи.\n\nКонтекст:\nДанные заданы как упорядоченная последовательность шагов с комнатами и персонажами.\n{context}\n\nФормат ответа:\nНапиши ровно один ответ после префикса Ответ:. Пример: Ответ: Кухня\n\nВопрос:\n{question}\n\nОтвет:",
    "Задача:\nВычисли нужный факт о присутствии персонажей в комнатах.\n\nКонтекст:\nИспользуй все шаги последовательности как временной ряд расположений.\n{context}\n\nФормат ответа:\nОтвет должен состоять из префикса Ответ: и одного слова или числа.\n\nВопрос:\n{question}\n\nОтвет:",
    "Задача:\nНайди краткий ответ по истории перемещений персонажей.\n\nКонтекст:\nНиже приведены шаги, где для каждой комнаты указаны присутствующие персонажи.\n{context}\n\nФормат ответа:\nВерни только один результат в формате: Ответ: X\n\nВопрос:\n{question}\n\nОтвет:",
    "Задача:\nОтветь на вопрос о перемещениях персонажей между комнатами.\n\nКонтекст:\nДана последовательность шагов; на каждом шаге указано, кто находится в каждой комнате.\n{context}\n\nФормат ответа:\nПоследняя строка должна быть в формате: Ответ: X, где X — одно имя, одна комната или одно число.\n\nВопрос:\n{question}\n\nОтвет:",
    "Задача:\nОпредели требуемый факт по временной последовательности расположений.\n\nКонтекст:\nКаждый шаг описывает состояние всех комнат и присутствующих персонажей.\n{context}\n\nФормат ответа:\nДай краткий ответ после префикса Ответ:. Пример: Ответ: Иван\n\nВопрос:\n{question}\n\nОтвет:",
    "Задача:\nПроанализируй все шаги и вычисли ответ на вопрос.\n\nКонтекст:\nНиже приведены состояния комнат во времени; порядок шагов важен.\n{context}\n\nФормат ответа:\nИспользуй ровно один результат в форме: Ответ: X\n\nВопрос:\n{question}\n\nОтвет:",
    "Задача:\nНайди персонажа, комнату или число, которые требуются в вопросе.\n\nКонтекст:\nПоследовательность показывает распределение персонажей по комнатам на каждом шаге.\n{context}\n\nФормат ответа:\nЗапиши один ответ с префиксом Ответ:, например: Ответ: 12\n\nВопрос:\n{question}\n\nОтвет:",
    "Задача:\nСопоставь вопрос со всеми состояниями комнат и выведи краткий результат.\n\nКонтекст:\nИспользуй полную последовательность шагов, чтобы учесть все посещения и совпадения.\n{context}\n\nФормат ответа:\nОтвет должен иметь вид: Ответ: X. X — одно слово или одно число.\n\nВопрос:\n{question}\n\nОтвет:",
]

# Offset to avoid ID collision between shots and test examples
SHOTS_ID_OFFSET = 10_000


def build_hf_features() -> datasets.Features:
    """Define HuggingFace dataset features matching MERA format."""
    return datasets.Features(
        {
            "instruction": datasets.Value("string"),
            "inputs": {
                "context": datasets.Value("string"),
                "question": datasets.Value("string"),
            },
            "outputs": datasets.Value("string"),
            "meta": {
                "id": datasets.Value("int32"),
                "categories": {
                    "task_type": datasets.Value("string"),
                    "seq_len": datasets.Value("int32"),
                    "atype": datasets.Value("string"),
                },
            },
        }
    )


def _normalize_sample(sample: dict, prompt_idx: int, sample_idx: int) -> dict:
    """Reshape a generate_mera_dataset sample to the final HF schema.

    Args:
        sample: Raw sample produced by generate_mera_dataset.
        prompt_idx: Index into PROMPTS for the instruction slot.
        sample_idx: Fallback index used when sample has no "meta.id".

    Returns:
        Sample dict conforming to build_hf_features().
    """
    meta = sample.get("meta", {})
    return {
        "instruction": PROMPTS[prompt_idx % len(PROMPTS)],
        "inputs": sample["inputs"],
        "outputs": sample["outputs"],
        "meta": {
            "id": int(meta.get("id", sample_idx + 1)),
            "categories": {
                "task_type": meta.get("task", "unknown"),
                "seq_len": int(meta.get("seq_len", 0)),
                "atype": meta.get("atype", "person"),
            },
        },
    }


def _load_shots_from_file(shots_file: Path) -> list[dict]:
    """Load raw few-shot examples from the in_context JSON file.

    Args:
        shots_file: Path to the in_context_examples.json file.

    Returns:
        List of raw example dicts.  Empty list if the file is missing.
    """
    if not shots_file.exists():
        print(f"  Warning: shots file not found at {shots_file}; shots splits will be empty.")
        return []

    with open(shots_file, encoding="utf-8") as fh:
        return json.load(fh)


def _build_shots_records(raw_shots: list[dict]) -> list[dict]:
    """Convert raw in-context examples to HF schema records.

    Args:
        raw_shots: List of dicts from in_context_examples.json.

    Returns:
        List of schema-conforming dicts with IDs starting at SHOTS_ID_OFFSET + 1.
    """
    records = []
    for i, ex in enumerate(raw_shots):
        prompt_idx = i % len(PROMPTS)
        qtype = ex.get("qtype", "")
        records.append(
            {
                "instruction": PROMPTS[prompt_idx],
                "inputs": {
                    "context": format_sequence_as_text(ex["sequence"]),
                    "question": ex["question"],
                },
                "outputs": str(ex["answer"]),
                "meta": {
                    "id": SHOTS_ID_OFFSET + i + 1,
                    "categories": {
                        "task_type": QTYPE_TO_TASK.get(qtype, qtype),
                        "seq_len": int(ex.get("seq_len", 0)),
                        "atype": ex.get("atype", "person"),
                    },
                },
            }
        )
    return records


def _config_task_code(config_name: str) -> str:
    """Derive the MERA task code from a config name.

    Example: "mmred_dc_sa_c_64" -> "DC-SA-C"

    Args:
        config_name: Dataset config name (e.g. "mmred_dc_sa_c_64").

    Returns:
        Upper-cased, hyphen-separated task code.
    """
    # Strip leading "mmred_" and trailing "_<seq_len>" suffix
    without_prefix = config_name.removeprefix("mmred_")
    task_slug = without_prefix.rsplit("_", 1)[0]
    return task_slug.upper().replace("_", "-")


def build_config_datasets(
    task_datasets: dict[str, list[dict]],
    features: datasets.Features,
    all_shots: list[dict],
    n_shots: int = 5,
) -> dict[str, datasets.DatasetDict]:
    """Assemble one DatasetDict per config with "test" and "shots" splits.

    Args:
        task_datasets: Mapping of config name -> list of raw samples from
            generate_mera_dataset.
        features: HuggingFace Features schema.
        all_shots: All shots records (schema-conforming dicts).
        n_shots: Maximum number of shots to include per config.

    Returns:
        Mapping of config name -> DatasetDict{"shots": ..., "test": ...}.
    """
    config_dicts: dict[str, datasets.DatasetDict] = {}

    for config_name, samples in task_datasets.items():
        # Normalize test samples
        normalized = [
            _normalize_sample(s, prompt_idx=i, sample_idx=i)
            for i, s in enumerate(samples)
        ]
        test_ds = datasets.Dataset.from_list(normalized, features=features)

        # Select shots relevant to this config's task code
        task_code = _config_task_code(config_name)
        config_shots = [
            s
            for s in all_shots
            if s["meta"]["categories"]["task_type"] == task_code
        ]
        if not config_shots:
            # Fallback: use the first n_shots from the full pool
            print(
                f"  Warning: no shots found for task_code={task_code!r}; "
                "falling back to first available shots."
            )
            config_shots = all_shots[:n_shots]

        shots_ds = datasets.Dataset.from_list(config_shots[:n_shots], features=features)

        config_dicts[config_name] = datasets.DatasetDict(
            {"shots": shots_ds, "test": test_ds}
        )
        print(
            f"  {config_name}: test={len(test_ds)}, shots={len(shots_ds)}"
        )

    return config_dicts


def push_or_save(
    config_dicts: dict[str, datasets.DatasetDict],
    output_dir: Path,
    push: bool,
    private: bool,
) -> None:
    """Push configs to HuggingFace Hub or save to disk.

    Args:
        config_dicts: Mapping of config name -> DatasetDict.
        output_dir: Base directory for local saves (used when push=False).
        push: If True, push to REPO_ID on HuggingFace Hub.
        private: Whether to create the HF repository as private.
    """
    for config_name, dd in config_dicts.items():
        if push:
            print(f"  Pushing {config_name} -> {REPO_ID} ...")
            dd.push_to_hub(
                REPO_ID,
                config_name=config_name,
                private=private,
            )
        else:
            local_path = output_dir / "hf" / config_name
            dd.save_to_disk(str(local_path))
            print(f"  Saved {config_name} -> {local_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload MMReD dataset to HuggingFace in MERA-compatible format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--push",
        action="store_true",
        help="Push generated dataset to HuggingFace Hub.",
    )
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate dataset locally without uploading.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/update the HF repository as private (only relevant with --push).",
    )
    parser.add_argument(
        "--n-questions",
        type=int,
        default=50,
        metavar="N",
        help="Number of questions per task type per sequence length. Default: 50.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0xBADFACE,
        help="Random seed for reproducible generation. Default: 0xBADFACE.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/mmred_mera_upload",
        metavar="DIR",
        help="Working directory for intermediate files. Default: /tmp/mmred_mera_upload.",
    )
    parser.add_argument(
        "--n-few-shot",
        type=int,
        default=5,
        metavar="N",
        help="Number of few-shot examples per task type. Default: 5.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # ------------------------------------------------------------------ #
    # 1. Generate MERA-formatted dataset                                  #
    # ------------------------------------------------------------------ #
    print(f"Generating MERA-format dataset (n_questions={args.n_questions}, seed={args.seed:#x})...")
    task_datasets = generate_mera_dataset(
        output_dir=output_dir,
        mode="text",
        language="ru",  # MERA prompts are in Russian
        n_questions=args.n_questions,
        seed=args.seed,
        n_few_shot=args.n_few_shot,
    )

    # ------------------------------------------------------------------ #
    # 2. Load few-shot examples                                           #
    # ------------------------------------------------------------------ #
    shots_file = output_dir / "in_context_examples.json"
    raw_shots = _load_shots_from_file(shots_file)
    all_shots = _build_shots_records(raw_shots)
    print(f"Loaded {len(all_shots)} shots records from {shots_file}.")

    # ------------------------------------------------------------------ #
    # 3. Build per-config DatasetDicts                                    #
    # ------------------------------------------------------------------ #
    features = build_hf_features()
    print(f"\nAssembling {len(task_datasets)} dataset configs...")
    config_dicts = build_config_datasets(
        task_datasets=task_datasets,
        features=features,
        all_shots=all_shots,
        n_shots=args.n_few_shot,
    )

    # ------------------------------------------------------------------ #
    # 4. Push to Hub or save locally                                      #
    # ------------------------------------------------------------------ #
    destination = REPO_ID if args.push else str(output_dir / "hf")
    print(f"\n{'Pushing to' if args.push else 'Saving locally at'} {destination}...")
    push_or_save(
        config_dicts=config_dicts,
        output_dir=output_dir,
        push=args.push,
        private=args.private,
    )

    total_test = sum(len(dd["test"]) for dd in config_dicts.values())
    print(
        f"\nDone. {len(config_dicts)} configs, {total_test} test samples total. "
        f"{'Pushed to' if args.push else 'Saved at'} {destination}."
    )


if __name__ == "__main__":
    main()
