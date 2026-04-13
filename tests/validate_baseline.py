"""validate_baseline.py — verifica os pipelines contra um gabarito de teste.

Carrega um arquivo `expected_baseline.json` (default: o vizinho deste script) e,
para cada imagem listada, roda os pipelines `pipeline.py` (grade) e/ou
`pipeline_free.py` (livre) como subprocessos. Compara o `Counter` dos símbolos
decodificados com `expected_multiset`. Imprime o resultado por imagem e devolve
exit code 0 se tudo bater, 1 se houver pelo menos uma divergência.

Uso típico:

    python tests/validate_baseline.py                       # roda os dois
    python tests/validate_baseline.py --pipeline free       # só pipeline_free
    python tests/validate_baseline.py --pipeline grid       # só pipeline (grade)
    python tests/validate_baseline.py --baseline tests/outro.json
    python tests/validate_baseline.py --images-root /caminho/para/fotos

A ideia de invocar via subprocess é deliberada: o script reproduz exatamente o
caminho de uso real dos pipelines, sem importar nenhum módulo interno —
mudanças nos pipelines não exigem mudanças aqui.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


# Diretório que contém pipeline.py e pipeline_free.py.
ECC200DECODE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = Path(__file__).resolve().parent / "expected_baseline.json"
DEFAULT_TEMPLATE = ECC200DECODE_ROOT / "template.png"
DEFAULT_IMAGES = Path(__file__).resolve().parent / "images"


def parse_args():
    p = argparse.ArgumentParser(
        description="Roda os pipelines contra um gabarito e reporta divergências."
    )
    p.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help="Arquivo JSON com expected_multiset e lista de imagens.",
    )
    p.add_argument(
        "--pipeline",
        choices=("grid", "free", "pyapparq", "both", "all"),
        default="both",
        help=(
            "Qual pipeline rodar. 'grid' = pipeline.py, 'free' = pipeline_free.py, "
            "'pyapparq' = PyAppArq/pipeline.py via API, 'both' = grid + free, "
            "'all' = grid + free + pyapparq."
        ),
    )
    p.add_argument(
        "--images-root",
        type=Path,
        default=DEFAULT_IMAGES,
        help="Diretório onde as imagens listadas no baseline estão (default: tests/images/).",
    )
    p.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help="Caminho do template usado pela ortorretificação.",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Interpretador Python usado para invocar os pipelines (default: o atual).",
    )
    return p.parse_args()


def load_baseline(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    images = raw.get("images")
    expected_multiset = raw.get("expected_multiset")
    if not images:
        raise ValueError(f"{path}: campo 'images' vazio ou ausente")
    if not expected_multiset:
        raise ValueError(f"{path}: campo 'expected_multiset' vazio ou ausente")
    return {
        "images": list(images),
        "expected_multiset": Counter(expected_multiset),
        "expected_count": int(raw.get("expected_count") or sum(expected_multiset.values())),
    }


def run_pipeline_grid(python, image_path, template_path):
    """Roda pipeline.py e devolve o Counter dos símbolos decodificados."""
    with tempfile.TemporaryDirectory() as tmp:
        ortho = Path(tmp) / "ortho.png"
        grid_out = Path(tmp) / "grid.txt"
        cmd = [
            str(python),
            str(ECC200DECODE_ROOT / "pipeline.py"),
            "--input", str(image_path),
            "--template", str(template_path),
            "--output", str(ortho),
            "--grid-output", str(grid_out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"pipeline.py falhou para {image_path}: {proc.stderr.strip() or proc.stdout.strip()}"
            )
        text = grid_out.read_text(encoding="utf-8")
    return Counter(s for s in text.split() if s != "?")


def run_pipeline_free(python, image_path, template_path):
    """Roda pipeline_free.py e devolve o Counter dos símbolos decodificados."""
    with tempfile.TemporaryDirectory() as tmp:
        ortho = Path(tmp) / "ortho.png"
        results = Path(tmp) / "symbols.json"
        annotated = Path(tmp) / "annotated.png"
        cmd = [
            str(python),
            str(ECC200DECODE_ROOT / "pipeline_free.py"),
            "--input", str(image_path),
            "--template", str(template_path),
            "--output", str(ortho),
            "--results-json", str(results),
            "--annotated-output", str(annotated),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"pipeline_free.py falhou para {image_path}: {proc.stderr.strip() or proc.stdout.strip()}"
            )
        data = json.loads(results.read_text(encoding="utf-8"))
    return Counter(item["text"] for item in data.get("symbols", []))


def run_pipeline_pyapparq(python, image_path, template_path):
    """Roda PyAppArq/pipeline.py via API (em subprocesso para isolar imports).

    PyAppArq não tem CLI — é módulo de biblioteca usado pela GUI. Aqui chamamos
    process_image() em um subprocesso para garantir o mesmo isolamento que os
    outros runners e para reusar o venv ativo do --python.
    """
    with tempfile.TemporaryDirectory() as tmp:
        snippet = (
            "import sys, json\n"
            f"sys.path.insert(0, {str(ECC200DECODE_ROOT / 'PyAppArq')!r})\n"
            "import pipeline as pl\n"
            f"_, grid, _ = pl.process_image({str(image_path)!r}, {str(template_path)!r})\n"
            "syms = [s for row in grid for s in row if s != '?']\n"
            "print(json.dumps(syms))\n"
        )
        cmd = [str(python), "-c", snippet]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"PyAppArq pipeline falhou para {image_path}: "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )
        # A última linha do stdout deve ser o JSON; outras linhas podem ser
        # progress callbacks impressas pelo módulo.
        last_line = proc.stdout.strip().split("\n")[-1]
        syms = json.loads(last_line)
    return Counter(syms)


def report_image(label, image_name, decoded, expected):
    missing = expected - decoded
    extra = decoded - expected
    status = "OK" if not missing and not extra else "FAIL"
    print(
        f"  {label:5s} {image_name:24s}"
        f" count={sum(decoded.values()):3d}"
        f" missing={dict(missing)} extra={dict(extra)}"
        f" [{status}]"
    )
    return status == "OK"


def main():
    args = parse_args()
    baseline = load_baseline(args.baseline)

    print(f"Baseline: {args.baseline}")
    print(
        f"Esperado: {baseline['expected_count']} símbolos por imagem,"
        f" multiset={dict(baseline['expected_multiset'])}"
    )
    print()

    runners = []
    if args.pipeline in ("grid", "both", "all"):
        runners.append(("grid", run_pipeline_grid))
    if args.pipeline in ("free", "both", "all"):
        runners.append(("free", run_pipeline_free))
    if args.pipeline in ("pyapparq", "all"):
        runners.append(("pyapparq", run_pipeline_pyapparq))

    expected = baseline["expected_multiset"]
    all_ok = True

    for label, runner in runners:
        print(f"=== {label} ===")
        for image_name in baseline["images"]:
            image_path = args.images_root / image_name
            if not image_path.exists():
                print(f"  {label:5s} {image_name:24s} [MISSING-FILE]")
                all_ok = False
                continue
            try:
                decoded = runner(args.python, image_path, args.template)
            except RuntimeError as exc:
                print(f"  {label:5s} {image_name:24s} [ERROR] {exc}")
                all_ok = False
                continue
            ok = report_image(label, image_name, decoded, expected)
            all_ok = all_ok and ok
        print()

    print("RESULT:", "OK" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
