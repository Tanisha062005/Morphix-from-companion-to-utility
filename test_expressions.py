"""
Morphix — OLED Eye Animator Test Bench  (37 States)
====================================================
Navigation
----------
  ← / →      Cycle through all 37 states one at a time
  ↑ / ↓      Jump to the start of the previous / next category
  1 – 9      Direct-access states 0 – 8
  0          State 9 (HEART_EYES)
  A – Z      States 10 – 35 (A=BLUSHING …)
  SPACE      Manual blink
  ESC / Q    Quit
"""

import math
import sys
import time

try:
    import pygame
except ImportError:
    print("ERROR: pygame not installed.  Run:  pip install pygame")
    sys.exit(1)

# Make sure the project root is on the path when run from the repo root
sys.path.insert(0, ".")

from src.display.eye_animator import (
    EyeAnimator,
    EyeState,
    CATEGORIES,
    _ALL_STATES,
    _get_font,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FACE_W    = EyeAnimator.WINDOW_W   # 480
HUD_H     = 96                     # height of the info panel below the face
TOTAL_H   = EyeAnimator.WINDOW_H + HUD_H  # 336
FPS       = EyeAnimator.FPS

C_BG_HUD  = ( 10,  10,  20)
C_TEXT_HI = (  0, 229, 255)   # neon cyan
C_TEXT_LO = ( 90, 140, 160)
C_TEXT_DIM= ( 50,  80,  95)
C_CAT_CLR = (255, 200,  60)
C_BORDER  = ( 30,  50,  70)

# ---------------------------------------------------------------------------
# Key → state-index mapping
# ---------------------------------------------------------------------------
_NUMKEY_MAP: dict[int, int] = {
    pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2,
    pygame.K_4: 3, pygame.K_5: 4, pygame.K_6: 5,
    pygame.K_7: 6, pygame.K_8: 7, pygame.K_9: 8,
    pygame.K_0: 9,
}
_ALPHA_MAP: dict[int, int] = {}
for _i, _k in enumerate([
    pygame.K_a, pygame.K_b, pygame.K_c, pygame.K_d, pygame.K_e,
    pygame.K_f, pygame.K_g, pygame.K_h, pygame.K_i, pygame.K_j,
    pygame.K_k, pygame.K_l, pygame.K_m, pygame.K_n, pygame.K_o,
    pygame.K_p, pygame.K_q, pygame.K_r, pygame.K_s, pygame.K_t,
    pygame.K_u, pygame.K_v, pygame.K_w, pygame.K_x, pygame.K_y,
    pygame.K_z
]):
    _ALPHA_MAP[_k] = 10 + _i

# Category boundary list (for ↑/↓ jumping)
_CAT_STARTS = [start for _, start, _ in CATEGORIES]


def _state_category(idx: int) -> str:
    for name, start, end in CATEGORIES:
        if start <= idx < end:
            return name
    return "Unknown"


def _jump_category(current_idx: int, direction: int) -> int:
    """Return the start index of the next (+1) or previous (−1) category."""
    cat_starts = _CAT_STARTS
    if direction > 0:
        for s in cat_starts:
            if s > current_idx:
                return s
        return cat_starts[0]  # wrap around
    else:
        for s in reversed(cat_starts):
            if s < current_idx:
                return s
        return cat_starts[-1]  # wrap around


# ---------------------------------------------------------------------------
# HUD rendering
# ---------------------------------------------------------------------------
def _render_hud(surface: pygame.Surface, idx: int, total: int,
                font_lg: pygame.font.Font,
                font_md: pygame.font.Font,
                font_sm: pygame.font.Font) -> None:
    state    = _ALL_STATES[idx]
    cat_name = _state_category(idx)

    # Background
    hud_rect = pygame.Rect(0, EyeAnimator.WINDOW_H, FACE_W, HUD_H)
    pygame.draw.rect(surface, C_BG_HUD, hud_rect)
    pygame.draw.line(surface, C_BORDER, (0, EyeAnimator.WINDOW_H),
                     (FACE_W, EyeAnimator.WINDOW_H), 2)

    cx  = FACE_W // 2
    y0  = EyeAnimator.WINDOW_H + 8

    # State name (index / total)
    name_str = f"[{idx+1:02d}/{total}]  {state.name}"
    name_txt = font_lg.render(name_str, True, C_TEXT_HI)
    surface.blit(name_txt, (cx - name_txt.get_width()//2, y0))

    # Category
    cat_txt = font_md.render(cat_name, True, C_CAT_CLR)
    surface.blit(cat_txt, (cx - cat_txt.get_width()//2, y0 + 28))

    # Navigation hints
    nav1 = "[<] [>] cycle     [^] [v] jump category     SPACE blink"
    nav2 = "1-9 0 A-Z direct keys     ESC / Q quit"
    n1   = font_sm.render(nav1, True, C_TEXT_LO)
    n2   = font_sm.render(nav2, True, C_TEXT_DIM)
    surface.blit(n1, (cx - n1.get_width()//2, y0 + 52))
    surface.blit(n2, (cx - n2.get_width()//2, y0 + 70))

    # Mini category progress dots
    dot_total = len(CATEGORIES)
    dot_r     = 5
    dot_gap   = 18
    dot_start = cx - (dot_total * dot_gap) // 2 + dot_gap // 2
    cur_cat   = 0
    for i, (_, s, e) in enumerate(CATEGORIES):
        if s <= idx < e:
            cur_cat = i
            break

    for di in range(dot_total):
        colour = C_TEXT_HI if di == cur_cat else C_TEXT_DIM
        pygame.draw.circle(surface, colour,
                           (dot_start + di * dot_gap, y0 + 44), dot_r)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((FACE_W, TOTAL_H))
    pygame.display.set_caption(f"Morphix — OLED Eye Animator Test Bench ({len(_ALL_STATES)} States)")
    clock  = pygame.time.Clock()

    font_lg = _get_font("consolas", 18, bold=True) or pygame.font.Font(None, 18)
    font_md = _get_font("consolas", 14, bold=True) or pygame.font.Font(None, 14)
    font_sm = _get_font("consolas", 13) or pygame.font.Font(None, 13)

    # Animator — render face into a sub-surface
    animator   = EyeAnimator(FACE_W, EyeAnimator.WINDOW_H)
    face_surf  = pygame.Surface((FACE_W, EyeAnimator.WINDOW_H))

    current_idx = 0
    animator.set_state(_ALL_STATES[current_idx])

    import io
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    print("\n" + "=" * 54)
    print(f"  Morphix OLED Eye Animator - Test Bench ({len(_ALL_STATES)} States)")
    print("=" * 54)
    print("  [<] [>] : cycle states     [^] [v] : jump category")
    print("  1-9     : states 0-8       0       : state 9")
    print("  A-Z     : states 10-35     SPACE   : manual blink")
    print("  ESC / Q : quit")
    print("=" * 54 + "\n")

    prev_time = time.time()

    while True:
        # ── Events ───────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

            if event.type == pygame.KEYDOWN:
                # Quit
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    sys.exit(0)

                # Manual blink
                elif event.key == pygame.K_SPACE:
                    animator.trigger_blink()

                # Arrow left / right — cycle
                elif event.key == pygame.K_LEFT:
                    current_idx = (current_idx - 1) % len(_ALL_STATES)
                    animator.set_state(_ALL_STATES[current_idx])
                    print(f"[TEST BENCH] -> {_ALL_STATES[current_idx].name}")

                elif event.key == pygame.K_RIGHT:
                    current_idx = (current_idx + 1) % len(_ALL_STATES)
                    animator.set_state(_ALL_STATES[current_idx])
                    print(f"[TEST BENCH] -> {_ALL_STATES[current_idx].name}")

                # Arrow up / down — jump category
                elif event.key == pygame.K_UP:
                    current_idx = _jump_category(current_idx, -1)
                    animator.set_state(_ALL_STATES[current_idx])
                    print(f"[TEST BENCH] -> {_ALL_STATES[current_idx].name}"
                          f"  ({_state_category(current_idx)})")

                elif event.key == pygame.K_DOWN:
                    current_idx = _jump_category(current_idx, +1)
                    animator.set_state(_ALL_STATES[current_idx])
                    print(f"[TEST BENCH] -> {_ALL_STATES[current_idx].name}"
                          f"  ({_state_category(current_idx)})")

                # Number keys 1–9 / 0
                elif event.key in _NUMKEY_MAP:
                    target = _NUMKEY_MAP[event.key]
                    if target < len(_ALL_STATES):
                        current_idx = target
                        animator.set_state(_ALL_STATES[current_idx])
                        print(f"[TEST BENCH] -> {_ALL_STATES[current_idx].name}")

                # Alpha keys A–Z
                elif event.key in _ALPHA_MAP:
                    target = _ALPHA_MAP[event.key]
                    if target < len(_ALL_STATES):
                        current_idx = target
                        animator.set_state(_ALL_STATES[current_idx])
                        print(f"[TEST BENCH] -> {_ALL_STATES[current_idx].name}")

        # ── Update & draw ────────────────────────────────────────────
        now       = time.time()
        dt        = min(now - prev_time, 0.05)   # clamp to 50 ms max
        prev_time = now

        animator.update(dt)
        animator.draw(face_surf)

        screen.blit(face_surf, (0, 0))
        _render_hud(screen, current_idx, len(_ALL_STATES), font_lg, font_md, font_sm)

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
