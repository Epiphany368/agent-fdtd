"""Read a leg's pre-declared criteria out of ``plan.md`` or the criteria index.

``describe_run`` must never claim a criterion it did not read.  This module
returns the exact block it found and where it came from, and returns nothing
when the leg cannot be located.

Two sources, in order:

1. ``docs/leg-criteria.json`` — the criteria index, extracted from ``plan.md``
   by :func:`write_criteria_index` at export time.  A published tree keeps
   ``plan.md`` private, so the index is the only source there.
2. ``plan.md`` itself — the working repository keeps working with no index.

When neither source is readable the result carries ``criteria: []`` and a note
naming what is missing; it never raises.

The extraction rule (used by :func:`extract_index`, and stated so it can be
audited): a *registration* is a heading matching ``LEG <n>`` that contains the
marker ``REGISTERED`` or ``(registered`` — measured outcomes and written
amendments are history, not registrations, and are skipped.  From each
registration the parser takes, verbatim from ``plan.md``:

* the one-line goal: the heading text after the em dash, minus the
  ``(registered ...)`` tail;
* the resource tag: the ``Resource tag: ``<tag>```` line, or ``null`` for the
  older registrations that declared Fork/GPU in prose instead;
* the criteria block: the ``Pre-declared ...`` / ``Criteria.`` block, cut at
  the next ``Stop conditions`` / ``Files ...`` / ``Fork`` / ``GPU`` marker;
  bullet items and table rows become one item each, a prose block is split at
  its own ``C<n> —``/``A<n> —`` markers when those exist;
* acceptance commands and their pre-declared exit codes: backticked spans that
  look like commands, paired with backticked integers that follow an
  ``exit(s)`` word, or — inside an acceptance table — the first backticked
  integer in the ``exit`` column.

The published index carries four fields per registered leg — ``leg``, ``goal``,
``resource``, ``exits`` — and nothing else.  A leg with no stated exit code gets
``exits: []``, which is lawful; the extractor fails only when *no* registered
leg yields an index entry, so an index that silently came out empty can never
be published.
"""
import json
import re
from pathlib import Path
import sys


_LEG_IN_NAME = re.compile(r"leg[_-]?(\d+)", re.IGNORECASE)
_LEG_HEADING = re.compile(
    r"^(?P<hashes>#{1,6})\s*(?:`?v3\.0`?\s*)?LEG\s+(?P<num>\d+)\b(?P<rest>.*)$",
    re.IGNORECASE,
)
_CRITERIA_START = re.compile(
    r"^\s*(?:#{3,}\s*pre-declared|(?:\*\*)?pre-declared\s+(?:acceptance|criteria|exits?)"
    r"|\*\*criteria\b)",
    re.IGNORECASE,
)
_STOP = re.compile(
    r"^\s*(?:#{1,3}\s|\*\*(?:Stop conditions|Files this leg|Files the leg|Fork\b|GPU\b|"
    r"Resource tag|Employee|Deliverable|Cost\b|Dependencies|Why\b|Registration|Pre-declared)\b)",
    re.IGNORECASE,
)
_RESOURCE_TAG = re.compile(r"Resource tag[:\s]*`?([A-Za-z0-9_-]+)`?")
_CMD_HINT = re.compile(r"(?:\.sh\b|unittest|pytest|python|sha256sum|\bgit\b|-m )")
_BACKTICKED_INT = re.compile(r"`(\d{1,3})`")
_BACKTICKED_SPAN = re.compile(r"`([^`\n]+)`")
_TABLE_ROW = re.compile(r"^\s*\|")
_TABLE_SEP = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_PROSE_ITEM = re.compile(r"(?=\b[A-C]\d{1,2}\s*[—–]\s)")


def leg_from_config(config_path):
    match = _LEG_IN_NAME.search(Path(config_path).name)
    return int(match.group(1)) if match else None


def _section_end(lines, start, level, limit):
    """End of a section: the next heading of the same or higher markdown level.

    A later h3 outcome/amendment for another leg therefore cannot cut an h2
    registration's section off before its criteria block.
    """
    for index in range(start + 1, limit):
        match = re.match(r"^(#{1,6})\s", lines[index])
        if match and len(match.group(1)) <= level:
            return index
    return limit


def _sections(lines):
    """Map leg number -> (start, end) line indexes of its section.

    A registration heading wins over any later outcome/amendment heading for
    the same leg number; a leg with no registration keeps the old behaviour
    (last heading wins).
    """
    headings = []
    for index, line in enumerate(lines):
        match = _LEG_HEADING.match(line)
        if match:
            headings.append((int(match.group("num")), len(match.group("hashes")), index, match.group("rest")))
    sections = {}
    for position, (leg, level, start, rest) in enumerate(headings):
        limit = len(lines)
        for _, _, next_start, _ in headings[position + 1:]:
            limit = next_start
            break
        end = _section_end(lines, start, level, limit)
        if re.search(r"\(registered\b", rest, re.IGNORECASE) or re.search(r"\bREGISTERED\b", rest):
            sections[leg] = (start, end)
        else:
            sections.setdefault(leg, (start, end))
    return sections


def _criteria_blocks(lines, start, end):
    """Every criteria block inside a section, in order.

    A registration may state its criteria and, later, the acceptance commands
    and exit codes that discharge them; both blocks are criteria, so both are
    read rather than only the first.
    """
    blocks = []
    index = start + 1
    while index < end:
        if _CRITERIA_START.match(lines[index]):
            block_end = end
            for later in range(index + 1, end):
                if _STOP.match(lines[later]):
                    block_end = later
                    break
            blocks.append((index, block_end))
            index = block_end
        else:
            index += 1
    return blocks


def _block_content(lines, block_start, block_end):
    """The block's lines without its heading, including a wrapped bold heading."""
    first = lines[block_start]
    index = block_start + 1
    if not re.match(r"^\s*(?:\*\*)?pre-declared\b", first, re.IGNORECASE):
        return lines[block_start:block_end]
    if first.count("**") % 2 == 1:
        while index < block_end and "**" not in lines[index]:
            index += 1
        index += 1
    return lines[index:block_end]


def _marker_prefix(line):
    """Strip a leading ``**Pre-declared acceptance.**``-style marker."""
    return re.sub(
        r"^\s*(?:#{3,}\s*)?(?:\*\*)?(?:pre-declared\s+(?:acceptance|criteria|exits?)|criteria)\b[.:*]*\s*",
        "",
        line,
        count=1,
        flags=re.IGNORECASE,
    )


def _items(block):
    """Turn a criteria block into a list of criterion strings.

    Bullets and acceptance-table rows become one item each; a prose block is
    split at its own ``C<n> —``/``A<n> —`` markers when the text declares
    them, and is otherwise kept as one item.
    """
    items = []
    text = []
    for line in block:
        stripped = line.strip()
        if _TABLE_ROW.match(line):
            if _TABLE_SEP.match(line):
                continue
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if cells and re.fullmatch(r"[-\s]*", cells[0]) is None and not _is_table_header(cells):
                items.append(stripped)
            continue
        numbered = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if stripped.startswith(("- ", "* ")):
            if text:
                items.extend(_split_prose(" ".join(text)))
                text = []
            items.append(stripped[2:].strip())
        elif numbered:
            if text:
                items.extend(_split_prose(" ".join(text)))
                text = []
            items.append(numbered.group(2).strip())
        elif line[:1].isspace() and stripped and items and not text:
            items[-1] = f"{items[-1]} {stripped}"
        elif stripped:
            text.append(stripped)
    if text:
        items.extend(_split_prose(" ".join(text)))
    return [item for item in items if item]


def _is_table_header(cells):
    return any(re.fullmatch(r"[-\s]*", cell) for cell in cells) or all(
        cell and not cell.startswith("`") for cell in cells
    )


def _split_prose(text):
    text = _marker_prefix(text)
    if not text:
        return []
    markers = list(_PROSE_ITEM.finditer(text))
    if not markers:
        return [text]
    parts = []
    for position, match in enumerate(markers):
        end = markers[position + 1].start() if position + 1 < len(markers) else len(text)
        part = text[match.start():end].strip()
        if part:
            parts.append(part)
    return parts


_EXIT_WORD = re.compile(r"\bexits?\b", re.IGNORECASE)
_LEADING_EXIT = re.compile(r"^\s*(?:\*\*)?`\d{1,3}`(?:\*\*)?(?=\s|$)")


def _commands_and_exits(items):
    """Pair acceptance commands with the exit codes pre-declared beside them.

    Within one criterion item: every backticked span that looks like a command
    is recorded; when the item speaks of ``exit(s)``, every backticked bare
    integer in it is a pre-declared exit code, paired with the command span
    nearest to it in the item (the one before it wins a tie), or with ``null``
    when the item names no command.  An acceptance table's rows pair the
    ``command`` column with the first backticked integer of the ``exit``
    column.  Commands with no stated exit code are kept, with an empty list.
    """
    commands = []
    for item in items:
        if _TABLE_ROW.match(item):
            commands.extend(_commands_from_table_row(item))
            continue
        if _EXIT_WORD.search(item) or _LEADING_EXIT.match(item):
            spans = [
                (m.start(), m.group(1))
                for m in _BACKTICKED_SPAN.finditer(item)
                if _CMD_HINT.search(m.group(1))
            ]
            grouped = {}
            for match in _BACKTICKED_INT.finditer(item):
                code = int(match.group(1))
                position = _nearest_span(spans, match.start())
                command = spans[position][1] if position is not None else None
                if command not in grouped:
                    grouped[command] = []
                if code not in grouped[command]:
                    grouped[command].append(code)
            for command, codes in grouped.items():
                commands.append({"command": command, "exits": codes})
        else:
            seen = set()
            for match in _BACKTICKED_SPAN.finditer(item):
                span = match.group(1)
                if _CMD_HINT.search(span) and span not in seen:
                    seen.add(span)
                    commands.append({"command": span, "exits": []})
    collapsed = []
    for entry in commands:
        if collapsed and collapsed[-1]["command"] == entry["command"]:
            for code in entry["exits"]:
                if code not in collapsed[-1]["exits"]:
                    collapsed[-1]["exits"].append(code)
        else:
            collapsed.append({"command": entry["command"], "exits": list(entry["exits"])})
    return collapsed


def _nearest_span(spans, at):
    """Index into ``spans`` of the command span nearest ``at``; before wins ties."""
    if not spans:
        return None
    before = [i for i, (pos, _span) in enumerate(spans) if pos <= at]
    after = [i for i, (pos, _span) in enumerate(spans) if pos > at]
    if before and after:
        return before[-1] if at - spans[before[-1]][0] <= spans[after[0]][0] - at else after[0]
    return before[-1] if before else after[0]


def _commands_from_table_row(item):
    cells = [cell.strip() for cell in item.strip().strip("|").split("|")]
    command = None
    for cell in cells:
        spans = [span for span in _BACKTICKED_SPAN.findall(cell) if _CMD_HINT.search(span)]
        if spans:
            command = spans[0]
            break
    exits = []
    for cell in cells:
        ints = _BACKTICKED_INT.findall(cell)
        if ints:
            exits = [int(ints[0])]
            break
    if command is None and not exits:
        return []
    return [{"command": command, "exits": exits}]


def extract_index(plan_path):
    """Parse every registered leg in ``plan.md`` into index entries.

    One entry per registration, carrying ``leg``, ``goal``, ``resource`` and
    ``exits``.  A registration with no stated exit code still gets an entry,
    with ``exits: []``.  Raises :class:`CriteriaExtractionError` only when the
    result would cover no leg at all.
    """
    lines = Path(plan_path).read_text(encoding="utf-8").splitlines()
    headings = []
    for index, line in enumerate(lines):
        match = _LEG_HEADING.match(line)
        if match:
            headings.append((int(match.group("num")), len(match.group("hashes")), index, match.group("rest")))
    registrations = {}
    for leg, _, start, rest in headings:
        if re.search(r"\(registered\b", rest, re.IGNORECASE) or re.search(r"\bREGISTERED\b", rest):
            registrations.setdefault(leg, start)
    entries = []
    for leg, start in sorted(registrations.items()):
        level = len(_LEG_HEADING.match(lines[start]).group("hashes"))
        end = len(lines)
        for index in range(start + 1, len(lines)):
            match = re.match(r"^(#{1,6})\s", lines[index])
            if match and len(match.group(1)) <= level:
                end = index
                break
        rest = _LEG_HEADING.match(lines[start]).group("rest")
        goal = re.split(r"\s+—\s+", rest, maxsplit=1)
        goal = goal[1] if len(goal) > 1 else goal[0]
        goal = re.sub(r"\s*\((?:registered|recorded)\b[^)]*\)\s*$", "", goal).strip()
        tag = None
        for line in lines[start:end]:
            match = _RESOURCE_TAG.search(line)
            if match:
                tag = match.group(1)
                break
        criteria = []
        for block_start, block_end in _criteria_blocks(lines, start, end):
            criteria.extend(_items(_block_content(lines, block_start, block_end)))
        entries.append(
            {
                "leg": leg,
                "goal": goal,
                "resource": tag,
                "exits": _commands_and_exits(criteria),
            }
        )
    if not entries:
        raise CriteriaExtractionError("plan.md registers no leg, so the criteria index would be empty")
    return entries


class CriteriaExtractionError(Exception):
    pass


def write_criteria_index(plan_path, out_path):
    """Extract the index and write it; returns the leg count."""
    entries = extract_index(plan_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "leg-criteria-index v1",
        "extracted_from": "plan.md",
        "legs": entries,
    }
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return len(entries)


def _index_entry(index_path, leg):
    try:
        data = json.loads(Path(index_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None, None
    if not isinstance(data, dict):
        return None, None
    for entry in data.get("legs", []):
        if isinstance(entry, dict) and entry.get("leg") == leg:
            return entry, None
    return None, data


def _index_criteria(entry):
    """Render an index entry's ``exits`` as the criteria ``describe_run`` quotes."""
    criteria = []
    for pair in entry.get("exits") or []:
        if not isinstance(pair, dict):
            continue
        command = pair.get("command")
        codes = [code for code in (pair.get("exits") or []) if isinstance(code, int)]
        if command and codes:
            criteria.append(f"`{command}` exits {', '.join(str(code) for code in codes)}")
        elif command:
            criteria.append(f"`{command}`")
        elif codes:
            criteria.append(f"exits {', '.join(str(code) for code in codes)}")
    return criteria


def criteria_for(config_path, plan_path=None):
    """Return ``{"criteria": [...], "criteria_text": str, "criteria_source": str}``."""
    if plan_path:
        plan_path = Path(plan_path)
        index_path = plan_path.parent / "docs" / "leg-criteria.json"
    else:
        root = Path(config_path).resolve().parents[1]
        plan_path = root / "plan.md"
        index_path = root / "docs" / "leg-criteria.json"
    leg = leg_from_config(config_path)
    if leg is None:
        return {
            "criteria": [],
            "criteria_text": "",
            "criteria_source": None,
            "criteria_note": "the config name carries no leg number, so no pre-declared criteria were read",
        }
    entry, data = _index_entry(index_path, leg)
    if entry is not None:
        criteria = _index_criteria(entry)
        return {
            "criteria": criteria,
            "criteria_text": "\n".join(criteria),
            "criteria_source": f"docs/leg-criteria.json#leg-{leg}",
            "criteria_note": None,
        }
    try:
        lines = plan_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        if data is not None:
            note = (
                f"LEG {leg} is not in {index_path.name} and plan.md is not readable: {exc}"
            )
        else:
            note = f"no readable criteria index ({index_path.name}) and plan.md is not readable: {exc}"
        return {
            "criteria": [],
            "criteria_text": "",
            "criteria_source": None,
            "criteria_note": note,
        }
    section = _sections(lines).get(leg)
    if section is None:
        return {
            "criteria": [],
            "criteria_text": "",
            "criteria_source": None,
            "criteria_note": f"no LEG {leg} section was found in plan.md; no criteria were read",
        }
    start, end = section
    blocks = _criteria_blocks(lines, start, end)
    if not blocks:
        return {
            "criteria": [],
            "criteria_text": "",
            "criteria_source": None,
            "criteria_note": f"LEG {leg} has no pre-declared criteria block; nothing was claimed",
        }
    criteria = []
    block_lines = []
    for block_start, block_end in blocks:
        criteria.extend(_items(_block_content(lines, block_start, block_end)))
        block_lines.extend(_block_content(lines, block_start, block_end))
    return {
        "criteria": criteria,
        "criteria_text": "\n".join(block_lines).strip(),
        "criteria_source": f"plan.md:{blocks[0][0] + 1}",
        "criteria_note": None,
    }


def _main(argv):
    if len(argv) != 3:
        print("usage: plan.py PLAN.md OUT.json", file=sys.stderr)
        return 64
    try:
        count = write_criteria_index(argv[1], argv[2])
    except (CriteriaExtractionError, OSError) as exc:
        print(f"leg-criteria: {exc}", file=sys.stderr)
        return 65
    print(f"leg-criteria: {count} registered leg(s) extracted -> {argv[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
