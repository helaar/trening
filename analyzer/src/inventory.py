#!/usr/bin/env python3
"""
Lag en YAML-oversikt over hvilke meldinger/felt som finnes i en FIT-fil,
samt hvor mange forekomster av hvert felt som er observert.

Bruk:
    python fit_field_inventory.py --fitfile minøkt.fit
eller:
    python fit_field_inventory.py --fitfile minøkt.fit --output oversikt.yaml
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from fitparse import FitFile


def collect_field_counts(fit_path: str) -> dict:
    """
    Les FIT-filen og tell felter pr. meldings-type.
    Returnerer en nested dict: message_name -> Counter(field_name -> count)
    """
    fitfile = FitFile(fit_path)
    counts = defaultdict(Counter)

    for message in fitfile.get_messages():
        msg_name = message.name or message.def_mesg.name or "unknown"

        # Standardfelter
        for field in message:
            field_name = field.name or f"field_{field.field_def_num}"
            counts[msg_name][field_name] += 1

        # Evt. developer fields
        developer_fields = getattr(message, "developer_fields", None)
        if developer_fields:
            for field in developer_fields:
                # developer-felt kan mangle navn, bruk felt-ID som fallback
                field_name = field.name or f"developer_{field.field_def_num}"
                counts[msg_name][f"developer:{field_name}"] += 1

    return counts


def counters_to_yaml_structure(counts: dict) -> dict:
    """
    Gjør om counts-structuren til en YAML-vennlig representasjon:
    message_name:
      - field: count
      - field2: count
    """
    yaml_data = {}
    for msg_name in sorted(counts):
        field_counter = counts[msg_name]
        yaml_data[msg_name] = [
            {field: int(field_counter[field])}
            for field in sorted(field_counter)
        ]
    return yaml_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lag en YAML-oversikt over hvilke felter som finnes i en FIT-fil."
    )
    parser.add_argument("--fitfile", required=True, help="Path til FIT-fil.")
    parser.add_argument(
        "--output",
        help="Path til YAML-utfil. Default: <fitfilnavn>-fields.yaml.",
    )
    args = parser.parse_args()

    fit_path = Path(args.fitfile).expanduser().resolve()
    if not fit_path.exists():
        raise FileNotFoundError(f"Finner ikke FIT-fil: {fit_path}")

    # Finn output-path
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = fit_path.with_name(f"{fit_path.stem}-fields.yaml")

    counts = collect_field_counts(str(fit_path))
    yaml_struct = counters_to_yaml_structure(counts)

    with output_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(yaml_struct, fh, allow_unicode=True, sort_keys=False)

    print(f"Ferdig! Skrev felt-oversikt til: {output_path}")


if __name__ == "__main__":
    main()