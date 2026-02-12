# AnimateDiff V4 — Development Guidelines

## Workflow Rules

### Commit Early, Commit Often
- Every completed module/feature MUST be committed before moving to the next
- Never accumulate more than ~3 files of uncommitted changes
- Use descriptive commit messages in English

### Task Tracking
- For multi-step work, always create a TaskList at the start
- Update task status as work progresses (pending → in_progress → completed)
- Before `/clear` or when context gets long, run `/compact` first

### Context Preservation
- When starting a large task, write the plan to `memory/WIP.md`
- Update `WIP.md` after each milestone (commit)
- Clear `WIP.md` when the task is fully complete
- Keep `MEMORY.md` for stable, verified knowledge only

### Code Conventions
- Python 3.10+ (f-strings, type hints, walrus operator OK)
- All new modules: include module docstring + `__all__` exports
- Backend modules: subclass `BackendBase` from `core/base_pipeline.py`
- PostProcess modules: follow existing pattern in `postprocess/`
- Config: JSON storyboard format (see `examples/`)

## Architecture

```
animatediff/
├── backends/     # Video generation backends (wan, hunyuan, cogvideo, ltx, etc.)
├── core/         # Pipeline infrastructure (base_pipeline, story_engine, vram, etc.)
├── postprocess/  # Post-processing (upscale, interpolation, audio, deflicker, etc.)
└── data/         # Static data (camera presets, LUTs, etc.)
scripts/          # CLI entry points (animate.py, story.py, produce_trailer_*.py)
examples/         # Storyboard JSON configs
```

## Key Constraints
- MPS (Apple Silicon): float32 only, no CPU offload, no FP8/MoE models
- RTX 4090 (24GB): Wan 2.2 5B at 832x480 with model_cpu_offload
- Always test imports before committing new modules
