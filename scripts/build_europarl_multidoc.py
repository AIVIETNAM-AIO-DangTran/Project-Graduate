"""Build a fresh, document-stratified Europarl pilot without changing old data.

Requires lxml and sacremoses from the project's regular virtual environment.
Uses the existing en/de/fr ZIP archives and OPUS alignment files.
"""

import argparse
import csv
import gzip
import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from zipfile import ZipFile

from lxml import etree
from sacremoses import MosesDetokenizer

FIELDNAMES = ["pair", "split", "doc_id", "context_group_id", "src_sent_id",
              "tgt_sent_id", "previous_src", "src", "tgt"]
PAIR_FILES = {"de": "de-en.xml.gz", "fr": "en-fr.xml.gz"}
PLAN = {
    "train": {"documents": 40, "de_per_document": 20, "fr_per_document": 10},
    "validation": {"documents": 10, "de_per_document": 10, "fr_per_document": 5},
    "test": {"documents": 10, "de_per_document": 10, "fr_per_document": 5},
}
DETOKENIZERS = {lang: MosesDetokenizer(lang=lang) for lang in ("en", "de", "fr")}


def tag(element):
    return etree.QName(element).localname if isinstance(element.tag, str) else ""


def normalize(reference):
    reference = reference.removesuffix(".gz")
    while reference.startswith("./"):
        reference = reference[2:]
    return reference.lstrip("/")


def stable_rank(seed, namespace, value):
    return hashlib.sha256(f"{seed}:{namespace}:{value}".encode("utf-8")).hexdigest()


def split_for_document(doc, seed):
    bucket = int(stable_rank(seed, "split", doc)[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def oriented_group(group, target_language):
    left, right = group.get("fromDoc"), group.get("toDoc")
    if not left or not right:
        return None
    left, right = normalize(left), normalize(right)
    left_lang, right_lang = left.split("/", 1)[0], right.split("/", 1)[0]
    if (left_lang, right_lang) == ("en", target_language):
        return left, right, False
    if (left_lang, right_lang) == (target_language, "en"):
        return right, left, True
    return None


def one_to_one(link, reverse):
    parts = link.get("xtargets", "").split(";")
    if len(parts) != 2:
        return None
    left, right = parts[0].split(), parts[1].split()
    if len(left) != 1 or len(right) != 1:
        return None
    return (right[0], left[0]) if reverse else (left[0], right[0])


def link_groups(path):
    """Keep links until their group ends, then release completed XML groups."""
    with gzip.open(path, "rb") as stream:
        for _, element in etree.iterparse(stream, events=("end",)):
            if tag(element) != "linkGrp":
                continue
            yield element
            element.clear()
            while element.getprevious() is not None:
                del element.getparent()[0]


def legacy_documents(directory):
    excluded, hashes = set(), {}
    for split in PLAN:
        path = directory / f"{split}.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if "doc_id" not in (reader.fieldnames or []):
                raise ValueError(f"Missing doc_id column: {path}")
            for row in reader:
                excluded.add(normalize(row["doc_id"]))
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not excluded:
        raise ValueError("No legacy document IDs found. Check --legacy-dir.")
    return excluded, hashes


def alignment_index(path, language):
    """First pass retains document references and link counts, not all links."""
    index = {}
    group_count = 0
    for group in link_groups(path):
        orientation = oriented_group(group, language)
        if orientation is None:
            continue
        english_doc, target_doc, reverse = orientation
        info = index.setdefault(english_doc, {"targets": set(), "links": 0})
        info["targets"].add(target_doc)
        info["links"] += sum(tag(link) == "link" and one_to_one(link, reverse) is not None
                             for link in group)
        group_count += 1
        if group_count % 1000 == 0:
            print(f"  {language}: indexed {group_count} document groups", flush=True)
    # Ambiguous document-level associations are excluded rather than guessed.
    return {doc: {"target": next(iter(info["targets"])), "links": info["links"]}
            for doc, info in index.items() if len(info["targets"]) == 1}


def candidate_documents(indexes, excluded, seed, multiplier, plan):
    available = (set(indexes["de"]) & set(indexes["fr"])) - excluded
    candidates = {}
    for split, settings in plan.items():
        required = settings["de_per_document"]
        docs = [doc for doc in available if split_for_document(doc, seed) == split
                and min(indexes[lang][doc]["links"] for lang in ("de", "fr")) >= required]
        docs.sort(key=lambda doc: (stable_rank(seed, "document-order", doc), doc))
        if len(docs) < settings["documents"]:
            raise ValueError(f"Only {len(docs)} candidate documents for {split}; "
                             f"need {settings['documents']}.")
        candidates[split] = docs[:settings["documents"] * multiplier]
        print(f"{split}: {len(docs)} indexed candidates; inspecting {len(candidates[split])}")
    return candidates


def selected_alignment_maps(path, language, wanted):
    """Second pass retains links only for the bounded candidate set."""
    values = defaultdict(lambda: defaultdict(set))
    for group in link_groups(path):
        orientation = oriented_group(group, language)
        if orientation is None:
            continue
        doc, _, reverse = orientation
        if doc not in wanted:
            continue
        for link in group:
            if tag(link) != "link":
                continue
            ids = one_to_one(link, reverse)
            if ids is not None:
                values[doc][ids[0]].add(ids[1])
    clean = {}
    for doc, source_targets in values.items():
        mappings = {source: next(iter(targets)) for source, targets in source_targets.items()
                    if len(targets) == 1}
        target_counts = Counter(mappings.values())
        clean[doc] = {source: target for source, target in mappings.items()
                      if target_counts[target] == 1}
    return clean


def sentence_text(sentence, language):
    tokens = [element.text.strip() for element in sentence.iter()
              if tag(element) == "w" and element.text and element.text.strip()]
    if tokens:
        return DETOKENIZERS[language].detokenize(tokens).strip()
    return " ".join(" ".join(sentence.itertext()).split())


def parse_document(archive, reference, language):
    with archive.open(f"Europarl/xml/{normalize(reference)}") as stream:
        root = etree.parse(stream).getroot()
    paragraphs = [element for element in root.iter() if tag(element).lower() == "p"]
    if not paragraphs:
        raise ValueError("No explicit paragraph boundaries")
    paragraph_numbers = {element: i for i, element in enumerate(paragraphs)}
    sentences, previous, groups = {}, {}, {}
    previous_in_group = {}
    for sentence in root.iter():
        if tag(sentence) != "s":
            continue
        paragraph = next((parent for parent in sentence.iterancestors()
                          if tag(parent).lower() == "p"), None)
        if paragraph is None:
            continue
        sid = sentence.get("id")
        text = sentence_text(sentence, language)
        if not sid or not text:
            previous_in_group[paragraph] = ""
            continue
        if sid in sentences:
            raise ValueError(f"Duplicate sentence ID: {sid}")
        sentences[sid] = text
        previous[sid] = previous_in_group.get(paragraph, "")
        groups[sid] = f"{normalize(reference)}::paragraph-{paragraph_numbers[paragraph]}"
        previous_in_group[paragraph] = text
    return sentences, previous, groups


def text_key(text):
    return " ".join(text.split())


def sample_rows(archives, indexes, maps, candidates, seed, plan):
    rows, manifest, skipped = [], [], []
    # Avoid repeated English current sentences even across separate documents.
    used_sources = set()
    used_translations = set()
    for split, settings in plan.items():
        selected = 0
        for doc in candidates[split]:
            if selected == settings["documents"]:
                break
            try:
                source, previous, groups = parse_document(archives["en"], doc, "en")
                targets = {lang: parse_document(archives[lang], indexes[lang][doc]["target"], lang)[0]
                           for lang in ("de", "fr")}
            except (KeyError, etree.XMLSyntaxError, ValueError) as error:
                skipped.append({"split": split, "doc_id": doc, "reason": str(error)})
                continue
            common = set(maps["de"].get(doc, {})) & set(maps["fr"].get(doc, {}))
            good = []
            local_sources = set()
            local_translations = set()
            for sid in sorted(common, key=lambda sid: (stable_rank(seed, f"sentences:{doc}", sid), sid)):
                src, prev = source.get(sid, ""), previous.get(sid, "")
                if not src or not prev or text_key(src) == text_key(prev):
                    continue
                if not 2 <= len(src.split()) <= 100:
                    continue
                source_key = text_key(src)
                if source_key in used_sources or source_key in local_sources:
                    continue
                translations = {lang: targets[lang].get(maps[lang][doc][sid], "")
                                for lang in ("de", "fr")}
                if any(not 2 <= len(text.split()) <= 100 for text in translations.values()):
                    continue
                translation_keys = {(lang, source_key, text_key(text))
                                    for lang, text in translations.items()}
                if translation_keys & used_translations or translation_keys & local_translations:
                    continue
                good.append(sid)
                local_sources.add(source_key)
                local_translations.update(translation_keys)
            needed = settings["de_per_document"]
            if len(good) < needed:
                skipped.append({"split": split, "doc_id": doc,
                                "reason": f"Only {len(good)} eligible shared source sentences; need {needed}"})
                continue
            chosen_de = good[:needed]
            chosen_fr = sorted(chosen_de, key=lambda sid: (stable_rank(seed, f"fr-subset:{doc}", sid), sid))[:settings["fr_per_document"]]
            for language, ids in (("de", chosen_de), ("fr", chosen_fr)):
                for sid in ids:
                    target_id = maps[language][doc][sid]
                    row = dict(pair=f"en-{language}", split=split, doc_id=doc,
                               context_group_id=groups[sid], src_sent_id=sid,
                               tgt_sent_id=target_id, previous_src=previous[sid],
                               src=source[sid], tgt=targets[language][target_id])
                    rows.append(row)
                    used_translations.add((language, text_key(row["src"]), text_key(row["tgt"])))
            used_sources.update(text_key(source[sid]) for sid in chosen_de)
            manifest.append({"split": split, "doc_id": doc,
                             "target_documents": {lang: indexes[lang][doc]["target"] for lang in ("de", "fr")},
                             "source_ids_en_de": chosen_de, "source_ids_en_fr": chosen_fr,
                             "eligible_shared_sentences": len(good)})
            selected += 1
            if selected % 10 == 0 or selected == settings["documents"]:
                print(f"{split}: selected {selected}/{settings['documents']} documents", flush=True)
        if selected != settings["documents"]:
            raise ValueError(f"Only {selected}/{settings['documents']} eligible documents for {split}. "
                             "No dataset written. Retry with --candidate-multiplier 8.")
    # Shuffle deterministically without relying on input file or dict order.
    rows.sort(key=lambda row: (stable_rank(seed, "row-order", json.dumps(row, sort_keys=True)),
                               row["pair"], row["doc_id"], row["src_sent_id"]))
    return rows, manifest, skipped


def validate(rows, excluded, plan):
    if any(row["split"] not in plan or row["pair"] not in ("en-de", "en-fr") for row in rows):
        raise ValueError("Unknown split or language pair")
    docs_by_split, sources_by_split, translations_by_split = {}, {}, {}
    statistics = {}
    for split, settings in plan.items():
        part = [row for row in rows if row["split"] == split]
        if any(not str(row[field]).strip() for row in part for field in FIELDNAMES):
            raise ValueError(f"Empty value in {split}")
        docs = {row["doc_id"] for row in part}
        if len(docs) != settings["documents"] or docs & excluded:
            raise ValueError(f"Document coverage or legacy exclusion failed: {split}")
        if len({tuple(row[field] for field in ("pair", "doc_id", "src_sent_id", "tgt_sent_id"))
                for row in part}) != len(part):
            raise ValueError(f"Duplicate row keys: {split}")
        translations = {(row["pair"], text_key(row["src"]), text_key(row["tgt"])) for row in part}
        if len(translations) != len(part):
            raise ValueError(f"Duplicate translations: {split}")
        pair_counts = Counter(row["pair"] for row in part)
        for doc in docs:
            by_language = {lang: [row for row in part if row["doc_id"] == doc and row["pair"] == f"en-{lang}"]
                           for lang in ("de", "fr")}
            for lang in ("de", "fr"):
                if len(by_language[lang]) != settings[f"{lang}_per_document"]:
                    raise ValueError(f"Per-document quota failed: {split}, {doc}, {lang}")
            de = {row["src_sent_id"]: row for row in by_language["de"]}
            for row in by_language["fr"]:
                twin = de.get(row["src_sent_id"])
                if twin is None or any(row[field] != twin[field]
                                       for field in ("src", "previous_src", "context_group_id")):
                    raise ValueError(f"Shared bilingual source consistency failed: {doc}")
        if any(text_key(row["src"]) == text_key(row["previous_src"]) for row in part):
            raise ValueError(f"Current source equals context: {split}")
        docs_by_split[split] = docs
        sources_by_split[split] = {text_key(row["src"]) for row in part}
        translations_by_split[split] = translations
        statistics[split] = {"num_examples": len(part), "num_documents": len(docs),
                             "pair_counts": dict(pair_counts),
                             "documents": dict(sorted(Counter(row["doc_id"] for row in part).items()))}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if (docs_by_split[a] & docs_by_split[b] or sources_by_split[a] & sources_by_split[b]
                or translations_by_split[a] & translations_by_split[b]):
            raise ValueError(f"Cross-split overlap: {a} / {b}")
    return statistics


def write_dataset(output, rows, report, plan):
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new --output-dir.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="europarl-build-", dir=output.parent) as temp:
        staging = Path(temp) / "dataset"
        staging.mkdir()
        hashes = {}
        for split in plan:
            path = staging / f"{split}.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(row for row in rows if row["split"] == split)
            hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        report["csv_sha256"] = hashes
        (staging / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"Output appeared during build: {output}")
        staging.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/europarl"))
    parser.add_argument("--legacy-dir", type=Path, default=Path("data/processed/europarl_pilot"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/europarl_multidoc_v2"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-multiplier", type=int, default=4)
    parser.add_argument("--train-documents", type=int, default=PLAN["train"]["documents"])
    parser.add_argument("--validation-documents", type=int, default=PLAN["validation"]["documents"])
    parser.add_argument("--test-documents", type=int, default=PLAN["test"]["documents"])
    parser.add_argument(
        "--exclude-dir", type=Path, action="append", default=[],
        help="Additional processed dataset directory whose documents must not enter this dataset. Repeat as needed.",
    )
    args = parser.parse_args()
    if args.candidate_multiplier < 1:
        parser.error("--candidate-multiplier must be at least 1")
    if min(args.train_documents, args.validation_documents, args.test_documents) < 1:
        parser.error("Each document count must be positive")
    plan = {
        split: {**settings, "documents": getattr(args, f"{split}_documents")}
        for split, settings in PLAN.items()
    }
    root = args.project_root.resolve()
    raw, legacy, output = [(root / path).resolve() for path in (args.raw_dir, args.legacy_dir, args.output_dir)]
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new --output-dir.")
    excluded, legacy_hashes = legacy_documents(legacy)
    additional_exclusions = {}
    for directory in args.exclude_dir:
        directory = (root / directory).resolve()
        if directory == output:
            parser.error("An exclusion directory cannot be the output directory")
        documents, hashes = legacy_documents(directory)
        excluded.update(documents)
        additional_exclusions[str(directory)] = hashes
    for path in [*(raw / f"{lang}.zip" for lang in ("en", "de", "fr")),
                 *(raw / filename for filename in PAIR_FILES.values())]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing corpus file: {path}")
    print(f"Seed: {args.seed}; excluded legacy documents: {len(excluded)}", flush=True)
    print("Indexing both alignment files; no model or GPU is used.", flush=True)
    indexes = {lang: alignment_index(raw / filename, lang) for lang, filename in PAIR_FILES.items()}
    candidates = candidate_documents(indexes, excluded, args.seed, args.candidate_multiplier, plan)
    wanted = {doc for docs in candidates.values() for doc in docs}
    print(f"Reading sentence links for {len(wanted)} candidate documents...", flush=True)
    maps = {lang: selected_alignment_maps(raw / filename, lang, wanted)
            for lang, filename in PAIR_FILES.items()}
    with ExitStack() as stack:
        archives = {lang: stack.enter_context(ZipFile(raw / f"{lang}.zip")) for lang in ("en", "de", "fr")}
        rows, manifest, skipped = sample_rows(archives, indexes, maps, candidates, args.seed, plan)
    statistics = validate(rows, excluded, plan)
    report = {"dataset": "Europarl multidoc document-stratified", "seed": args.seed, "plan": plan,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "candidate_multiplier": args.candidate_multiplier,
              "excluded_legacy_documents": sorted(excluded), "legacy_csv_sha256": legacy_hashes,
              "additional_exclusion_csv_sha256": additional_exclusions,
              "raw_directory": str(raw), "alignment_files": PAIR_FILES,
              "candidate_counts": {split: len(docs) for split, docs in candidates.items()},
              "splits": statistics, "selected_documents": manifest, "skipped_candidates": skipped,
              "checks": {"exact_quotas": "PASS", "shared_bilingual_documents": "PASS",
                         "shared_source_context_consistency": "PASS", "duplicate_keys": 0,
                         "duplicate_translation_pairs": 0, "cross_split_document_overlap": 0,
                         "cross_split_current_source_overlap": 0, "legacy_document_overlap": 0},
              "limitations": "Structural validation does not verify semantic equivalence of all OPUS alignments. "
                             "The bilingual-intersection sampling restricts the population to sentences available in both target languages."}
    write_dataset(output, rows, report, plan)
    print("\n=== MULTIDOCUMENT DATASET: CHECKS PASS ===")
    for split, stats in statistics.items():
        print(f"{split}: {stats['num_examples']} pairs; {stats['num_documents']} documents; {stats['pair_counts']}")
    print("Semantic equivalence still needs sample review before training.")
    print(f"Dataset: {output}")
    print(f"Summary: {output / 'summary.json'}")


if __name__ == "__main__":
    main()
