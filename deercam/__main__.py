"""python -m deercam <command>

  sync      download new photos from the shared Reveal feed
  detect    run MegaDetector over new photos and crop animals
  analyze   classify crops (buck/doe/fawn) and describe racks with the vision LLM
  verify    double-check buck calls (runs automatically after analyze; use for older crops)
  identify  match buck crops against named bucks
  run       sync + detect + analyze + identify
  serve     start the review UI on http://localhost:8600
  export-demo  write a read-only static copy of the UI and data (e.g. for GitHub Pages)
  ranches   list the ranches (each is one place with cameras, with its own photos and bucks)

Commands work on one ranch: --ranch <id>, else the first one.
"""
import argparse
import logging


def main():
    parser = argparse.ArgumentParser(prog="deercam")
    parser.add_argument("--ranch", help="ranch id (see `ranches`); default: the first ranch")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_sync = sub.add_parser("sync")
    p_sync.add_argument("--full", action="store_true", help="walk the whole feed instead of stopping at known photos")
    p_sync.add_argument("--days", type=int, help="only photos from the last N days")
    sub.add_parser("detect")
    p_an = sub.add_parser("analyze")
    p_an.add_argument("--limit", type=int, help="analyze at most N crops (handy for trying a provider)")
    p_an.add_argument("--workers", type=int, default=6, help="parallel LLM requests")
    sub.add_parser("verify")
    p_id = sub.add_parser("identify")
    p_id.add_argument("--recheck", action="store_true", help="also re-match everything in the review queue")
    p_id.add_argument("--limit", type=int, help="match at most N crops")
    p_run = sub.add_parser("run")
    p_run.add_argument("--days", type=int)
    p_srv = sub.add_parser("serve")
    p_srv.add_argument("--port", type=int, default=8600)
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--reload", action="store_true", help="restart automatically when code changes (for development)")
    p_demo = sub.add_parser("export-demo")
    p_demo.add_argument("out", help="output folder (replaced)")
    p_demo.add_argument("--no-map", action="store_true", help="leave camera locations out")
    sub.add_parser("ranches")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "urllib3", "botocore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.cmd == "serve":
        import uvicorn
        from pathlib import Path
        uvicorn.run("deercam.web.app:app", host=args.host, port=args.port, reload=args.reload,
                    reload_dirs=[str(Path(__file__).parent)] if args.reload else None,
                    reload_includes=["*.py", "*.html"] if args.reload else None)
        return

    from . import ranches
    if args.cmd == "ranches":
        for r in ranches.all_ranches():
            print(f"{r['id']:24} {r['name']}")
        return
    rid = args.ranch or ranches.default_id()
    if not ranches.exists(rid):
        parser.error(f"no ranch {rid!r}; see `python -m deercam ranches`")
    with ranches.use(rid):
        command(args)


def command(args):
    if args.cmd == "run":
        from .pipeline import run
        print(run(step=lambda label: logging.info("== %s", label), days=args.days))
        return
    if args.cmd == "sync":
        from .sync import sync
        sync(full=args.full, days=args.days)
    if args.cmd == "detect":
        from .detect import detect
        detect()
    if args.cmd == "analyze":
        from .analyze import analyze
        analyze(limit=args.limit, workers=args.workers)
    if args.cmd == "verify":
        from .analyze import verify_bucks
        verify_bucks()
    if args.cmd == "identify":
        from .identify import identify
        identify(recheck=args.recheck, limit=args.limit)
    if args.cmd == "export-demo":
        from pathlib import Path

        from .demo import export
        print(f"wrote {export(Path(args.out), include_map=not args.no_map)} API answers to {args.out}")


if __name__ == "__main__":
    main()
