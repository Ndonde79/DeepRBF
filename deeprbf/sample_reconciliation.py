# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Traceable, fail-closed matching of sample identifiers across exports."""
import pandas as pd


def normalize_sample_id(values):
    """Preserve leading zeros and alphanumeric IDs; remove numeric .0 export suffixes."""
    return values.astype('string').str.strip().str.upper().str.replace(r'^(\d+)\.0+$', r'\1', regex=True)


def unique_sample_rows(frame, hole='hole_id', sample='sample_id'):
    """Keep only populated, unique hole/sample keys; quarantine ambiguous keys."""
    data = frame.copy()
    data[hole] = data[hole].astype('string').str.strip().str.upper()
    data[sample] = normalize_sample_id(data[sample])
    populated = data[hole].notna() & data[sample].notna() & data[hole].ne('') & data[sample].ne('')
    return data.loc[populated & ~data.duplicated([hole,sample],keep=False)].copy()
