"""Single entrypoint for the pipeline. Each subcommand is idempotent."""

from __future__ import annotations

import numpy as np
import typer

from pipeline import config, curate, export, glitch, score, simulate

app = typer.Typer(add_completion=False, help="Spot the Glitch — local LeWM pipeline.")


@app.command()
def simulate_cmd(
    n: int = typer.Option(config.DEFAULT_N_ROLLOUTS, "--n", help="Number of normal rollouts."),
    force: bool = typer.Option(False, "--force", help="Regenerate even if outputs exist."),
) -> None:
    """Rollout N normal Push-T trajectories to data/trajectories/."""
    out_dir = config.TRAJ_DIR
    if out_dir.exists() and any(out_dir.glob("*.npz")) and not force:
        typer.echo(f"[simulate] {out_dir} already populated; skipping (use --force to override)")
        return
    paths = simulate.simulate_many(n=n, out_dir=out_dir)
    typer.echo(f"[simulate] wrote {len(paths)} trajectories to {out_dir}")


app.command(name="simulate")(simulate_cmd)


@app.command()
def glitch_cmd(
    seed: int = typer.Option(config.BASE_SEED, "--seed", help="Determinism seed."),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Produce one glitched variant per normal trajectory using planned family mix."""
    out_dir = config.GLITCHED_DIR
    if out_dir.exists() and any(out_dir.glob("*.npz")) and not force:
        typer.echo(f"[glitch] {out_dir} already populated; skipping (use --force)")
        return
    sources = sorted(config.TRAJ_DIR.glob("*.npz"))
    if not sources:
        raise typer.Exit(code=1)
    mix = glitch.plan_family_mix(total=len(sources), seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (src_path, family) in enumerate(zip(sources, mix, strict=True)):
        src = simulate.load_trajectory(src_path)
        base = glitch.BaseTrajectory(
            frames=src.frames, states=src.states, actions=src.actions, seed=src.seed
        )
        g = glitch.apply_family(base, family=family, seed=seed + i)
        glitch.save_glitched(g, out_dir / f"g_{i:04d}.npz")
    typer.echo(f"[glitch] wrote {len(sources)} glitched trajectories to {out_dir}")


app.command(name="glitch")(glitch_cmd)


@app.command()
def score_cmd(force: bool = typer.Option(False, "--force")) -> None:
    """Run LeWM inference to produce surprise arrays."""
    surprise_base = config.SURPRISE_DIR
    if surprise_base.exists() and any(surprise_base.rglob("*.npy")) and not force:
        typer.echo(f"[score] {surprise_base} already populated; skipping (use --force)")
        return
    result = score.score_all()
    typer.echo(
        f"[score] scored {len(result['normal'])} normal + {len(result['glitched'])} glitched"
    )


app.command(name="score")(score_cmd)


def _load_curate_inputs() -> tuple[list[curate.GlitchedEntry], list[curate.NormalEntry]]:
    glitched_entries: list[curate.GlitchedEntry] = []
    for npz_path in sorted(config.GLITCHED_DIR.glob("*.npz")):
        g = glitch.load_glitched(npz_path)
        curve = np.load(config.SURPRISE_DIR / "glitched" / f"{npz_path.stem}.npy")
        glitched_entries.append(
            curate.GlitchedEntry(
                id=npz_path.stem,
                curve=curve,
                tier=curate.assign_tier(curve),
                glitch_family=g.glitch_family,
                source_id=f"traj_{g.source_seed - config.BASE_SEED:04d}",
            )
        )
    normal_entries: list[curate.NormalEntry] = []
    for npz_path in sorted(config.TRAJ_DIR.glob("*.npz")):
        curve = np.load(config.SURPRISE_DIR / "normal" / f"{npz_path.stem}.npy")
        normal_entries.append(curate.NormalEntry(id=npz_path.stem, curve=curve))
    return glitched_entries, normal_entries


@app.command()
def curate_cmd(
    seed: int = typer.Option(config.BASE_SEED, "--seed"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Select 30 pairs and write selection.json."""
    out_path = config.DATA_DIR / "selection.json"
    if out_path.exists() and not force:
        typer.echo(f"[curate] {out_path} already exists; skipping (use --force)")
        return
    glitched_entries, normal_entries = _load_curate_inputs()
    selection = curate.select_pairs(glitched=glitched_entries, normal=normal_entries, seed=seed)
    curate.save_selection(selection, out_path)
    typer.echo(f"[curate] wrote {len(selection.pairs)} pairs to {out_path}")


app.command(name="curate")(curate_cmd)


@app.command()
def export_cmd() -> None:
    """Encode MP4s + build quiz-data.json + run acceptance checks."""
    selection = curate.load_selection(config.DATA_DIR / "selection.json")
    curves: dict[str, np.ndarray] = {}
    for p in selection.pairs:
        for cid in (p.clip_a_id, p.clip_b_id):
            sub = "glitched" if cid.startswith("g_") else "normal"
            curves[cid] = np.load(config.SURPRISE_DIR / sub / f"{cid}.npy")
    export.export_all(selection, curves)


app.command(name="export")(export_cmd)


@app.command()
def quick() -> None:
    """Dev fast path: small N, writes to data/debug/ for hand inspection."""
    config.DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    typer.echo(f"[quick] running n={config.QUICK_N_ROLLOUTS} end-to-end into {config.DEBUG_DIR}")
    # Implementation intentionally minimal: run simulate + glitch + score into the main dirs
    # with a smaller N. Curate/export expect the main dirs.
    simulate.simulate_many(n=config.QUICK_N_ROLLOUTS, out_dir=config.TRAJ_DIR)
    mix = glitch.plan_family_mix(total=config.QUICK_N_ROLLOUTS, seed=config.BASE_SEED)
    for i, (src_path, family) in enumerate(
        zip(sorted(config.TRAJ_DIR.glob("*.npz")), mix, strict=True)
    ):
        src = simulate.load_trajectory(src_path)
        base = glitch.BaseTrajectory(
            frames=src.frames, states=src.states, actions=src.actions, seed=src.seed
        )
        g = glitch.apply_family(base, family=family, seed=config.BASE_SEED + i)
        glitch.save_glitched(g, config.GLITCHED_DIR / f"g_{i:04d}.npz")
    score.score_all()
    typer.echo("[quick] done. Inspect data/surprise/ for curves.")


app.command(name="quick")(quick)


if __name__ == "__main__":
    app()
