import re
from typing import Dict, Iterable, List, Optional


ONE_TO_THREE = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}
THREE_TO_ONE = {value: key for key, value in ONE_TO_THREE.items()}
MUTATION_PATTERN = re.compile(r"^\s*([A-Za-z]{1,3})(\d+)([A-Za-z]{1,3})\s*$")


def normalize_residue_code(code: str) -> str:
    normalized = str(code).strip().upper()
    if not normalized:
        raise ValueError("Empty residue code in mutation spec.")
    if len(normalized) == 1:
        if normalized not in ONE_TO_THREE:
            raise ValueError(f"Unsupported residue code: {code}")
        return ONE_TO_THREE[normalized]
    if len(normalized) == 3 and normalized in THREE_TO_ONE:
        return normalized
    raise ValueError(f"Unsupported residue code: {code}")


def normalize_mutation_specs(raw_mutations) -> List[dict]:
    if raw_mutations is None:
        return []
    if isinstance(raw_mutations, str):
        items = [part.strip() for part in raw_mutations.split(",") if part.strip()]
    elif isinstance(raw_mutations, Iterable):
        items = [str(part).strip() for part in raw_mutations if str(part).strip()]
    else:
        raise TypeError("mutations must be a string, list, tuple, or set.")

    normalized_specs = []
    seen_residue_ids = {}
    for item in items:
        match = MUTATION_PATTERN.match(item)
        if not match:
            raise ValueError(f"Invalid mutation spec: {item}. Expected format like I202L.")

        from_code, residue_id_text, to_code = match.groups()
        residue_id = int(residue_id_text)
        from_resname = normalize_residue_code(from_code)
        to_resname = normalize_residue_code(to_code)

        if residue_id in seen_residue_ids:
            previous = seen_residue_ids[residue_id]
            raise ValueError(
                f"Duplicate mutation residue id {residue_id}: {previous} and {item}"
            )

        normalized = {
            "spec": item,
            "residue_id": residue_id,
            "from_resname": from_resname,
            "to_resname": to_resname,
        }
        normalized_specs.append(normalized)
        seen_residue_ids[residue_id] = item

    return normalized_specs


def build_mutation_lookup(raw_mutations) -> Dict[int, dict]:
    return {
        item["residue_id"]: item
        for item in normalize_mutation_specs(raw_mutations)
    }


def get_mutated_resname(residue_id: int, original_resname: str, mutation_lookup: Dict[int, dict]) -> str:
    mutation = mutation_lookup.get(int(residue_id))
    if not mutation:
        return original_resname
    return mutation["to_resname"]


def validate_mutation_observations(
    observed_resnames: Dict[int, set],
    pdb_path: str,
    mutation_lookup: Dict[int, dict],
) -> None:
    for residue_id, mutation in mutation_lookup.items():
        observed = observed_resnames.get(int(residue_id))
        if not observed:
            raise ValueError(
                f"Mutation {mutation['spec']} cannot be applied: residue {residue_id} not found in {pdb_path}"
            )

        from_resname = mutation["from_resname"]
        to_resname = mutation["to_resname"]
        if from_resname not in observed and to_resname not in observed:
            observed_text = ", ".join(sorted(observed))
            raise ValueError(
                f"Mutation {mutation['spec']} expected residue {from_resname} at {residue_id}, "
                f"but found [{observed_text}] in {pdb_path}"
            )


def mutate_pdb_atom_line(line: str, mutation_lookup: Dict[int, dict]) -> str:
    if not line.startswith(("ATOM", "HETATM")):
        return line
    try:
        residue_id = int(line[22:26])
    except ValueError:
        return line

    mutation = mutation_lookup.get(residue_id)
    if not mutation:
        return line

    mutated_resname = mutation["to_resname"]
    return f"{line[:17]}{mutated_resname:>3s}{line[20:]}"


def read_mutated_pdb_text(pdb_path: str, mutation_lookup: Optional[Dict[int, dict]] = None) -> str:
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        if not mutation_lookup:
            return handle.read()
        return "".join(mutate_pdb_atom_line(line, mutation_lookup) for line in handle)
