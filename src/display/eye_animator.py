"""
Morphix Desktop Companion Robot — OLED Eye Animator (Chibi Edition v3)
=======================================================================
File: src/display/eye_animator.py

30-state Chibi robot face with:
  • Vertical glowing pill eyes, arc-happy eyes, heart/wink/X/circle eyes
  • Mini mouth (smile/frown/flat/O/dots/squiggle/?/kiss/wavy-anxious)
  • Full particle system  (tears · confetti · hearts · bang-stars · flowers)
  • 13 state-specific overlay renderers
  • Smooth lerp transitions  +  natural micro-blink loop
  • Non-blocking daemon thread  (EyeAnimatorThread)

State categories
----------------
  A. Core Lifecycle  (0–5) : IDLE_NEUTRAL  LISTENING  THINKING  SPEAKING  SLEEPY  DEEP_SLEEP
  B. Emotions       (6–19) : HAPPY  EXCITED  WINKING  HEART_EYES  BLUSHING
                             SAD  CRYING  WORRIED  SURPRISED  CONFUSED
                             POUTY  ANGRY  DIZZY  SICK_COLD
  C. System / HUD  (20–22) : BATTERY_FULL  BATTERY_LOW  GREET_HELLO
  D. Interactive   (23–29) : MEAL_BURGER  MEAL_NOODLES  AFFECTION_PET
                             INTERACTION_KISS_HUG  INTERACTION_BANG_SHOOT
                             INTERACTION_RPS  BIRTHDAY_PARTY
"""

from __future__ import annotations

import dataclasses
import math
import random
import threading
import time
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("Morphix.EyeAnimator")

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_BG        = (  0,   0,   0)
C_EYE       = (  0, 229, 255)   # #00e5ff  neon cyan
C_MOUTH     = (  0, 200, 230)   # slightly dimmer
C_HEART     = (255, 100, 180)   # #ff64b4  pink
C_HILITE    = (200, 255, 255)   # ice-white highlight
C_ZZZ       = (  0, 229, 255)
C_BLUSH     = (255, 120, 160)   # cheek pink
C_TEAR      = ( 80, 170, 255)   # teardrop blue
C_STAR      = (255, 220,  80)   # sparkle yellow
C_BATTERY_G = ( 80, 255, 100)   # battery green
C_BATTERY_R = (255,  80,  60)   # battery red
C_SICK      = (150, 220,  80)   # sick green
C_PARTY_HAT = (255, 100,  50)   # party hat orange
C_WHITE     = (240, 255, 255)

_CONFETTI_COLS = [
    (255, 80, 80), (255, 220, 80), (80, 255, 120),
    (80, 200, 255), (255, 80, 200), (255, 160, 60),
    (180, 80, 255),
]

_FONT_CACHE: dict = {}

def _get_font(name: str = "consolas", size: int = 14, bold: bool = False):
    """Retrieves or caches a Pygame font to eliminate per-frame instantiation overhead."""
    key = (name, size, bold)
    if key not in _FONT_CACHE:
        try:
            import pygame
            if not pygame.font.get_init():
                pygame.font.init()
            _FONT_CACHE[key] = pygame.font.SysFont(name, size, bold=bold)
        except Exception:
            try:
                import pygame
                _FONT_CACHE[key] = pygame.font.Font(None, size)
            except Exception:
                return None
    return _FONT_CACHE.get(key)

# ---------------------------------------------------------------------------
# EyeState — 37 states  (order defines cycling order in test bench)
# ---------------------------------------------------------------------------
class EyeState(Enum):
    # A. Core Lifecycle (6)
    IDLE_NEUTRAL           = auto()
    LISTENING              = auto()
    THINKING               = auto()
    SPEAKING               = auto()
    SLEEPY                 = auto()
    DEEP_SLEEP             = auto()
    # B. Emotions & Moods (14)
    HAPPY                  = auto()
    EXCITED                = auto()
    WINKING                = auto()
    HEART_EYES             = auto()
    BLUSHING               = auto()
    SAD                    = auto()
    CRYING                 = auto()
    WORRIED                = auto()
    SURPRISED              = auto()
    CONFUSED               = auto()
    POUTY                  = auto()
    ANGRY                  = auto()
    DIZZY                  = auto()
    SICK_COLD              = auto()
    # C. System / HUD (6)
    BATTERY_FULL           = auto()
    BATTERY_LOW            = auto()
    GREET_HELLO            = auto()
    WEATHER_HUD            = auto()
    TELEMETRY_HUD          = auto()
    MUSIC_MODE             = auto()
    # D. Interactive / Pet (7)
    MEAL_BURGER            = auto()
    MEAL_NOODLES           = auto()
    AFFECTION_PET          = auto()
    INTERACTION_KISS_HUG   = auto()
    INTERACTION_BANG_SHOOT = auto()
    INTERACTION_RPS        = auto()
    BIRTHDAY_PARTY         = auto()
    # E. Festive Themes (6)
    FESTIVAL_MODAK         = auto()
    FESTIVAL_DIWALI        = auto()
    FESTIVAL_HOLI          = auto()
    FESTIVAL_SANTA         = auto()
    FESTIVAL_TRICOLOR      = auto()
    FESTIVAL_KITE          = auto()

# Category boundaries (inclusive start, exclusive end)
CATEGORIES: list[tuple[str, int, int]] = [
    ("A. Core Lifecycle",     0,  6),
    ("B. Emotions & Moods",    6, 20),
    ("C. System / HUD",       20, 26),
    ("D. Interactive / Pet",  26, 33),
    ("E. Festive Themes",     33, 39),
]
_ALL_STATES = list(EyeState)


# ---------------------------------------------------------------------------
# Particle
# ---------------------------------------------------------------------------
@dataclass
class Particle:
    x: float;  y: float
    vx: float; vy: float
    life: float;  max_life: float
    r: float
    colour: Tuple[int, int, int]
    shape:   str   = "circle"   # circle|heart|star|confetti|teardrop|flower
    angle:   float = 0.0
    va:      float = 0.0        # angular velocity deg/sec
    gravity: float = 80.0

    @property
    def alpha_ratio(self) -> float:
        return max(0.0, self.life / self.max_life)


# ---------------------------------------------------------------------------
# Face-frame descriptors
# ---------------------------------------------------------------------------
@dataclass
class EyeDesc:
    style:        str   = "pill"
    w:            float = 44.0
    h:            float = 68.0
    ox:           float = 0.0
    oy:           float = 0.0
    squint:       float = 0.0    # 0-1: symmetric top-lid squish
    inner_squint: float = 0.0   # asymmetric inner-top cut (ANGRY)
    colour: Optional[Tuple[int, int, int]] = None


@dataclass
class MouthDesc:
    style: str   = "smile"
    w:     float = 38.0
    h:     float = 12.0


@dataclass
class FaceFrame:
    left:  EyeDesc
    right: EyeDesc
    mouth: MouthDesc


def _E(**kw) -> EyeDesc:   return EyeDesc(**kw)
def _M(**kw) -> MouthDesc: return MouthDesc(**kw)


# ---------------------------------------------------------------------------
# FACE_STATES  — 30 entries
# ---------------------------------------------------------------------------
FACE_STATES: dict[EyeState, FaceFrame] = {

    # ── A. Core Lifecycle ─────────────────────────────────────────────
    EyeState.IDLE_NEUTRAL: FaceFrame(
        left  = _E(style="pill",      w=44, h=70),
        right = _E(style="pill",      w=44, h=70),
        mouth = _M(style="smile",     w=36, h=10),
    ),
    EyeState.LISTENING: FaceFrame(
        left  = _E(style="pill",      w=48, h=76),
        right = _E(style="pill",      w=48, h=76),
        mouth = _M(style="flat",      w=30, h=5),
    ),
    EyeState.THINKING: FaceFrame(
        left  = _E(style="pill",      w=36, h=52, ox=-5, squint=0.20),
        right = _E(style="pill",      w=46, h=66, ox=4),
        mouth = _M(style="dots",      w=44, h=8),
    ),
    EyeState.SPEAKING: FaceFrame(
        left  = _E(style="pill",      w=46, h=72),
        right = _E(style="pill",      w=46, h=72),
        mouth = _M(style="speak_open",w=30, h=18),
    ),
    EyeState.SLEEPY: FaceFrame(
        left  = _E(style="pill",      w=46, h=28, squint=0.55),
        right = _E(style="pill",      w=46, h=28, squint=0.55),
        mouth = _M(style="flat",      w=26, h=5),
    ),
    EyeState.DEEP_SLEEP: FaceFrame(
        left  = _E(style="flat",      w=48, h=8),
        right = _E(style="flat",      w=48, h=8),
        mouth = _M(style="flat",      w=18, h=4),
    ),

    # ── B. Emotions ───────────────────────────────────────────────────
    EyeState.HAPPY: FaceFrame(
        left  = _E(style="arc_happy", w=54, h=28),
        right = _E(style="arc_happy", w=54, h=28),
        mouth = _M(style="smile",     w=54, h=18),
    ),
    EyeState.EXCITED: FaceFrame(
        left  = _E(style="pill",      w=54, h=86),
        right = _E(style="pill",      w=54, h=86),
        mouth = _M(style="smile",     w=58, h=20),
    ),
    EyeState.WINKING: FaceFrame(
        left  = _E(style="wink",      w=48, h=12),
        right = _E(style="pill",      w=48, h=74),
        mouth = _M(style="smile",     w=42, h=13),
    ),
    EyeState.HEART_EYES: FaceFrame(
        left  = _E(style="heart",     w=50, h=50, colour=C_HEART),
        right = _E(style="heart",     w=50, h=50, colour=C_HEART),
        mouth = _M(style="smile",     w=38, h=11),
    ),
    EyeState.BLUSHING: FaceFrame(
        left  = _E(style="pill",      w=44, h=68, squint=0.10),
        right = _E(style="pill",      w=44, h=68, squint=0.10),
        mouth = _M(style="smile",     w=34, h=9),
    ),
    EyeState.SAD: FaceFrame(
        left  = _E(style="pill",      w=40, h=54, squint=0.35, oy=6),
        right = _E(style="pill",      w=40, h=54, squint=0.35, oy=6),
        mouth = _M(style="frown",     w=42, h=12),
    ),
    EyeState.CRYING: FaceFrame(
        left  = _E(style="pill",      w=38, h=48, squint=0.45, oy=8),
        right = _E(style="pill",      w=38, h=48, squint=0.45, oy=8),
        mouth = _M(style="frown",     w=38, h=12),
    ),
    EyeState.WORRIED: FaceFrame(
        left  = _E(style="pill",      w=40, h=58, squint=0.25, oy=4, ox=-4),
        right = _E(style="pill",      w=40, h=58, squint=0.25, oy=4, ox=4),
        mouth = _M(style="wavy",      w=38, h=10),
    ),
    EyeState.SURPRISED: FaceFrame(
        left  = _E(style="circle",    w=50, h=50),
        right = _E(style="circle",    w=50, h=50),
        mouth = _M(style="o",         w=28, h=28),
    ),
    EyeState.CONFUSED: FaceFrame(
        left  = _E(style="pill",      w=34, h=48, ox=-5, squint=0.30),
        right = _E(style="pill",      w=48, h=70),
        mouth = _M(style="question",  w=18, h=24),
    ),
    EyeState.POUTY: FaceFrame(
        left  = _E(style="pill",      w=46, h=44, squint=0.30),
        right = _E(style="pill",      w=46, h=44, squint=0.30),
        mouth = _M(style="frown",     w=36, h=9),
    ),
    EyeState.ANGRY: FaceFrame(
        left  = _E(style="angry",     w=46, h=60, inner_squint=0.60),
        right = _E(style="angry",     w=46, h=60, inner_squint=0.60),
        mouth = _M(style="flat",      w=34, h=6),
    ),
    EyeState.DIZZY: FaceFrame(
        left  = _E(style="x_mark",    w=46, h=46),
        right = _E(style="x_mark",    w=46, h=46),
        mouth = _M(style="squiggle",  w=46, h=11),
    ),
    EyeState.SICK_COLD: FaceFrame(
        left  = _E(style="pill",      w=42, h=46, squint=0.32, oy=4),
        right = _E(style="pill",      w=42, h=46, squint=0.32, oy=4),
        mouth = _M(style="flat",      w=22, h=5),
    ),

    # ── C. System / HUD ──────────────────────────────────────────────
    EyeState.BATTERY_FULL: FaceFrame(
        left  = _E(style="pill",      w=44, h=70),
        right = _E(style="pill",      w=44, h=70),
        mouth = _M(style="smile",     w=36, h=10),
    ),
    EyeState.BATTERY_LOW: FaceFrame(
        left  = _E(style="pill",      w=40, h=54, squint=0.35, oy=6),
        right = _E(style="pill",      w=40, h=54, squint=0.35, oy=6),
        mouth = _M(style="frown",     w=42, h=12),
    ),
    EyeState.GREET_HELLO: FaceFrame(
        left  = _E(style="arc_happy", w=54, h=28),
        right = _E(style="arc_happy", w=54, h=28),
        mouth = _M(style="smile",     w=52, h=16),
    ),
    EyeState.WEATHER_HUD: FaceFrame(
        left  = _E(style="pill",      w=44, h=62, oy=-12),
        right = _E(style="pill",      w=44, h=62, oy=-12),
        mouth = _M(style="smile",     w=34, h=10),
    ),
    EyeState.TELEMETRY_HUD: FaceFrame(
        left  = _E(style="pill",      w=42, h=60, oy=-12),
        right = _E(style="pill",      w=42, h=60, oy=-12),
        mouth = _M(style="smile",     w=32, h=10),
    ),
    EyeState.MUSIC_MODE: FaceFrame(
        left  = _E(style="arc_happy", w=48, h=26, oy=-8),
        right = _E(style="arc_happy", w=48, h=26, oy=-8),
        mouth = _M(style="open_smile", w=36, h=14),
    ),

    # ── D. Interactive / Pet ─────────────────────────────────────────
    EyeState.MEAL_BURGER: FaceFrame(
        left  = _E(style="pill",      w=52, h=82),
        right = _E(style="pill",      w=52, h=82),
        mouth = _M(style="smile",     w=56, h=18),
    ),
    EyeState.MEAL_NOODLES: FaceFrame(
        left  = _E(style="arc_happy", w=52, h=26),
        right = _E(style="arc_happy", w=52, h=26),
        mouth = _M(style="o",         w=26, h=20),
    ),
    EyeState.AFFECTION_PET: FaceFrame(
        left  = _E(style="arc_happy", w=54, h=28),
        right = _E(style="arc_happy", w=54, h=28),
        mouth = _M(style="smile",     w=54, h=20),
    ),
    EyeState.INTERACTION_KISS_HUG: FaceFrame(
        left  = _E(style="pill",      w=46, h=48, squint=0.28),
        right = _E(style="pill",      w=46, h=48, squint=0.28),
        mouth = _M(style="kiss",      w=22, h=14),
    ),
    EyeState.INTERACTION_BANG_SHOOT: FaceFrame(
        left  = _E(style="x_mark",    w=50, h=50),
        right = _E(style="x_mark",    w=50, h=50),
        mouth = _M(style="squiggle",  w=44, h=10),
    ),
    EyeState.INTERACTION_RPS: FaceFrame(
        left  = _E(style="pill",      w=44, h=70),
        right = _E(style="pill",      w=44, h=70),
        mouth = _M(style="flat",      w=28, h=5),
    ),
    EyeState.BIRTHDAY_PARTY: FaceFrame(
        left  = _E(style="pill",      w=54, h=86),
        right = _E(style="pill",      w=54, h=86),
        mouth = _M(style="smile",     w=58, h=20),
    ),

    # ── E. Festive Themes ────────────────────────────────────────────
    EyeState.FESTIVAL_MODAK: FaceFrame(
        left  = _E(style="arc_happy", w=48, h=28),
        right = _E(style="arc_happy", w=48, h=28),
        mouth = _M(style="smile",     w=38, h=13),
    ),
    EyeState.FESTIVAL_DIWALI: FaceFrame(
        left  = _E(style="pill",      w=44, h=68, colour=(255, 235, 120)),
        right = _E(style="pill",      w=44, h=68, colour=(255, 235, 120)),
        mouth = _M(style="smile",     w=40, h=14),
    ),
    EyeState.FESTIVAL_HOLI: FaceFrame(
        left  = _E(style="pill",      w=46, h=68, colour=(255, 120, 200)),
        right = _E(style="pill",      w=46, h=68, colour=(80, 230, 255)),
        mouth = _M(style="smile",     w=40, h=14),
    ),
    EyeState.FESTIVAL_SANTA: FaceFrame(
        left  = _E(style="arc_happy", w=48, h=28),
        right = _E(style="pill",      w=44, h=68),
        mouth = _M(style="smile",     w=38, h=13),
    ),
    EyeState.FESTIVAL_TRICOLOR: FaceFrame(
        left  = _E(style="pill",      w=44, h=68, colour=(255, 160, 40)),
        right = _E(style="pill",      w=44, h=68, colour=(80, 220, 90)),
        mouth = _M(style="smile",     w=38, h=12),
    ),
    EyeState.FESTIVAL_KITE: FaceFrame(
        left  = _E(style="pill",      w=44, h=68, oy=-4),
        right = _E(style="pill",      w=44, h=68, oy=-4),
        mouth = _M(style="smile",     w=38, h=12),
    ),
}


# ---------------------------------------------------------------------------
# Lerp helpers
# ---------------------------------------------------------------------------
def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _ease_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def _lerp_eye(src: EyeDesc, dst: EyeDesc, t: float) -> EyeDesc:
    snap = t >= 0.5
    return EyeDesc(
        style        = dst.style        if snap else src.style,
        w            = _lerp(src.w,            dst.w,            t),
        h            = _lerp(src.h,            dst.h,            t),
        ox           = _lerp(src.ox,           dst.ox,           t),
        oy           = _lerp(src.oy,           dst.oy,           t),
        squint       = _lerp(src.squint,       dst.squint,       t),
        inner_squint = _lerp(src.inner_squint, dst.inner_squint, t),
        colour       = dst.colour       if snap else src.colour,
    )


def _lerp_mouth(src: MouthDesc, dst: MouthDesc, t: float) -> MouthDesc:
    snap = t >= 0.5
    return MouthDesc(
        style = dst.style if snap else src.style,
        w     = _lerp(src.w, dst.w, t),
        h     = _lerp(src.h, dst.h, t),
    )


# ---------------------------------------------------------------------------
# Glow helpers
# ---------------------------------------------------------------------------
def _glow_rect(surf, rect, colour: tuple, cr: int) -> None:
    import pygame
    cr_int = max(2, int(cr))
    for extra, alpha in ((12, 10), (7, 22), (3, 48)):
        gr = rect.inflate(extra * 2, extra * 2)
        gs = pygame.Surface(gr.size, pygame.SRCALPHA)
        pygame.draw.rect(gs, (*colour, alpha),
                         pygame.Rect(0, 0, gr.width, gr.height),
                         border_radius=max(2, cr_int + extra))
        surf.blit(gs, gr.topleft)


def _glow_circle(surf, pos, r: int, colour: tuple) -> None:
    import pygame
    r_int = max(2, int(r))
    for extra, alpha in ((10, 10), (6, 22), (3, 48)):
        gr = r_int + extra
        gs = pygame.Surface((gr * 2, gr * 2), pygame.SRCALPHA)
        pygame.draw.circle(gs, (*colour, alpha), (gr, gr), gr)
        surf.blit(gs, (pos[0] - gr, pos[1] - gr))


# ---------------------------------------------------------------------------
# Mini particle shapes
# ---------------------------------------------------------------------------
def _draw_mini_heart(surf, cx: int, cy: int, r: int, colour: tuple) -> None:
    import pygame
    pts = []
    for i in range(61):
        t  = (i / 60) * 2 * math.pi
        hx = 16 * math.sin(t) ** 3
        hy = -(13 * math.cos(t) - 5 * math.cos(2*t)
               - 2  * math.cos(3*t) - math.cos(4*t))
        pts.append((cx + int(hx * r / 16), cy + int(hy * r / 16)))
    if len(pts) >= 3:
        pygame.draw.polygon(surf, colour, pts)


def _draw_mini_star(surf, cx: int, cy: int, r: int, colour: tuple,
                    points: int = 5) -> None:
    import pygame
    pts = []
    for i in range(points * 2):
        rad  = math.radians(i * 180 / points - 90)
        dist = r if i % 2 == 0 else max(2, r // 2)
        pts.append((cx + int(dist * math.cos(rad)),
                    cy + int(dist * math.sin(rad))))
    if len(pts) >= 3:
        pygame.draw.polygon(surf, colour, pts)


def _draw_mini_flower(surf, cx: int, cy: int, r: int, colour: tuple) -> None:
    import pygame
    petal_r = max(2, r // 3)
    for i in range(5):
        angle = math.radians(i * 72)
        px = cx + int(r * 0.62 * math.cos(angle))
        py = cy + int(r * 0.62 * math.sin(angle))
        pygame.draw.circle(surf, colour, (px, py), petal_r)
    # Centre yellow dot
    ctr = (255, 240, 80, colour[3] if len(colour) == 4 else 255)
    pygame.draw.circle(surf, ctr, (cx, cy), max(2, r // 4))


# ---------------------------------------------------------------------------
# Particle renderer
# ---------------------------------------------------------------------------
def _draw_particle(surf, p: Particle) -> None:
    import pygame
    alpha  = int(220 * p.alpha_ratio)
    colour = p.colour
    r      = max(2, int(p.r))
    x, y   = int(p.x), int(p.y)

    if p.shape == "circle":
        s = pygame.Surface((r*2+2, r*2+2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*colour, alpha), (r+1, r+1), r)
        surf.blit(s, (x-r-1, y-r-1))

    elif p.shape == "teardrop":
        sw, sh = r*3+2, r*4+2
        s  = pygame.Surface((sw, sh), pygame.SRCALPHA)
        tc = (*colour, alpha)
        pygame.draw.circle(s, tc, (sw//2, r+1), r)
        pts = [(sw//2 - r//2, r+1+r//2),
               (sw//2 + r//2, r+1+r//2),
               (sw//2,        r+1+r*2)]
        pygame.draw.polygon(s, tc, pts)
        surf.blit(s, (x - sw//2, y - r - 1))

    elif p.shape == "heart":
        s = pygame.Surface((r*4+4, r*4+4), pygame.SRCALPHA)
        _draw_mini_heart(s, r*2+2, r*2+2, r, (*colour, alpha))
        surf.blit(s, (x-r*2-2, y-r*2-2))

    elif p.shape == "star":
        s = pygame.Surface((r*4+4, r*4+4), pygame.SRCALPHA)
        _draw_mini_star(s, r*2+2, r*2+2, r, (*colour, alpha))
        surf.blit(s, (x-r*2-2, y-r*2-2))

    elif p.shape == "flower":
        s = pygame.Surface((r*4+4, r*4+4), pygame.SRCALPHA)
        _draw_mini_flower(s, r*2+2, r*2+2, r, (*colour, alpha))
        surf.blit(s, (x-r*2-2, y-r*2-2))

    elif p.shape == "confetti":
        s  = pygame.Surface((r*4+4, r*4+4), pygame.SRCALPHA)
        cx_, cy_ = r*2+2, r*2+2
        hw_, hh_ = r, max(1, r//2)
        rad = math.radians(p.angle % 360)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        corners = [(-hw_, -hh_), (hw_, -hh_), (hw_, hh_), (-hw_, hh_)]
        pts = [(cx_ + int(dx*cos_a - dy*sin_a),
                cy_ + int(dx*sin_a + dy*cos_a))
               for dx, dy in corners]
        pygame.draw.polygon(s, (*colour, alpha), pts)
        surf.blit(s, (x-r*2-2, y-r*2-2))


# ---------------------------------------------------------------------------
# Eye drawing functions
# ---------------------------------------------------------------------------
def _draw_pill(surf, cx: int, cy: int, eye: EyeDesc,
               blink_ratio: float) -> None:
    import pygame
    colour  = eye.colour or C_EYE
    eff_h   = max(4.0, eye.h * (1.0 - blink_ratio * 0.96))
    top_cut = eff_h * eye.squint * 0.5
    draw_h  = max(4.0, eff_h - top_cut)
    draw_w  = max(4.0, eye.w)
    cr      = max(2, int(min(draw_w, draw_h) // 2))
    ex = int(cx + eye.ox)
    ey = int(cy + eye.oy + top_cut / 2)
    rect = pygame.Rect(int(ex - draw_w/2), int(ey - draw_h/2),
                       int(draw_w), int(draw_h))
    _glow_rect(surf, rect, colour, cr)
    pygame.draw.rect(surf, colour, rect, border_radius=cr)
    if draw_h > 14 and draw_w > 12:
        pygame.draw.circle(surf, C_HILITE,
                           (int(ex - draw_w*0.20), int(ey - draw_h*0.24)),
                           max(2, int(draw_w * 0.11)))


def _draw_angry_pill(surf, cx: int, cy: int, eye: EyeDesc,
                     blink_ratio: float, is_left: bool) -> None:
    """Pill eye with angled inner-top lid mask — creates angry scowl effect."""
    import pygame
    colour  = eye.colour or C_EYE
    eff_h   = max(4.0, eye.h * (1.0 - blink_ratio * 0.96))
    top_cut = eff_h * eye.squint * 0.5
    draw_h  = max(4.0, eff_h - top_cut)
    draw_w  = max(4.0, eye.w)
    cr      = max(2, int(min(draw_w, draw_h) // 2))
    ex = int(cx + eye.ox)
    ey = int(cy + eye.oy + top_cut / 2)
    rect = pygame.Rect(int(ex - draw_w/2), int(ey - draw_h/2),
                       int(draw_w), int(draw_h))
    _glow_rect(surf, rect, colour, cr)
    pygame.draw.rect(surf, colour, rect, border_radius=cr)
    if draw_h > 14:
        pygame.draw.circle(surf, C_HILITE,
                           (int(ex - draw_w*0.20), int(ey - draw_h*0.24)),
                           max(2, int(draw_w * 0.11)))
    # Angular black mask over inner-top corner
    if eye.inner_squint > 0:
        cut_d = int(draw_h * eye.inner_squint * 0.75)
        top_y = int(ey - draw_h / 2)
        hw    = int(draw_w / 2)
        outer = int(draw_w * 0.25)
        if is_left:   # inner = right side
            pts = [(ex + hw, top_y),
                   (ex + hw, top_y + cut_d),
                   (ex - hw + outer, top_y)]
        else:          # inner = left side
            pts = [(ex - hw, top_y),
                   (ex - hw, top_y + cut_d),
                   (ex + hw - outer, top_y)]
        pygame.draw.polygon(surf, C_BG, pts)


def _draw_flat(surf, cx: int, cy: int, eye: EyeDesc) -> None:
    import pygame
    colour = eye.colour or C_EYE
    w, h   = max(10, eye.w), max(6, eye.h)
    cr     = max(2, int(h // 2))
    ex, ey = int(cx + eye.ox), int(cy + eye.oy)
    rect = pygame.Rect(int(ex - w/2), int(ey - h/2), int(w), int(h))
    _glow_rect(surf, rect, colour, cr)
    pygame.draw.rect(surf, colour, rect, border_radius=cr)



def _draw_arc_happy(surf, cx: int, cy: int, eye: EyeDesc) -> None:
    """  ^ parabolic arc — happy crescent  """
    import pygame
    colour  = eye.colour or C_EYE
    ex, ey  = int(cx + eye.ox), int(cy + eye.oy)
    n       = 32
    pts     = [(int(ex + (eye.w/2)*(2*i/n - 1)),
                int(ey - eye.h*(1 - (2*i/n - 1)**2)))
               for i in range(n+1)]
    thick   = max(5, int(eye.w * 0.11))
    if len(pts) >= 2:
        dim = tuple(max(0, c - 90) for c in colour)
        pygame.draw.lines(surf, dim, False, pts, thick + 10)
        pygame.draw.lines(surf, colour, False, pts, thick)
    for cap in (pts[0], pts[-1]):
        pygame.draw.circle(surf, colour, cap, thick // 2)


def _draw_wink(surf, cx: int, cy: int, eye: EyeDesc) -> None:
    import pygame
    colour  = eye.colour or C_EYE
    ex, ey  = int(cx + eye.ox), int(cy + eye.oy)
    n       = 22
    pts     = [(int(ex + eye.w/2*(2*i/n - 1)),
                int(ey + (eye.h/2)*math.sin(i/n * math.pi)))
               for i in range(n+1)]
    thick   = max(5, int(eye.w * 0.13))
    if len(pts) >= 2:
        dim = tuple(max(0, c - 90) for c in colour)
        pygame.draw.lines(surf, dim, False, pts, thick + 8)
        pygame.draw.lines(surf, colour, False, pts, thick)
    for cap in (pts[0], pts[-1]):
        pygame.draw.circle(surf, colour, cap, thick // 2)


def _draw_heart_eye(surf, cx: int, cy: int, eye: EyeDesc) -> None:
    import pygame
    colour = eye.colour or C_HEART
    r      = int(eye.w * 0.46)
    ex, ey = int(cx + eye.ox), int(cy + eye.oy)
    pts    = [(ex + int(16*math.sin(i/60*2*math.pi)**3 * r/16),
               ey + int(-(13*math.cos(i/60*2*math.pi)
                          - 5*math.cos(2*i/60*2*math.pi)
                          - 2*math.cos(3*i/60*2*math.pi)
                          -   math.cos(4*i/60*2*math.pi)) * r/16))
              for i in range(61)]
    if len(pts) >= 3:
        gw = int(eye.w * 3 + 4)
        gh = int(eye.h * 3 + 4)
        gs = pygame.Surface((gw, gh), pygame.SRCALPHA)
        ox, oy = gw//2, gh//2
        gpts = [(p[0]-ex+ox, p[1]-ey+oy) for p in pts]
        pygame.draw.polygon(gs, (*colour, 30), gpts)
        pygame.draw.polygon(gs, (*colour, 55),
                            [(x+2, y+2) for x, y in gpts])
        surf.blit(gs, (ex-ox, ey-oy))
        pygame.draw.polygon(surf, colour, pts)
        pygame.draw.circle(surf, C_HILITE,
                           (ex - r//5, ey - r//4), max(2, r//5))


def _draw_x_mark(surf, cx: int, cy: int, eye: EyeDesc,
                 angle_deg: float) -> None:
    import pygame
    colour = eye.colour or C_EYE
    r      = eye.w * 0.40
    ex, ey = int(cx + eye.ox), int(cy + eye.oy)
    thick  = max(5, int(r * 0.30))
    base   = math.radians(angle_deg % 360)
    for delta in (math.pi/4, -math.pi/4):
        a   = base + delta
        dx  = r * math.cos(a)
        dy  = r * math.sin(a)
        p1  = (int(ex - dx), int(ey - dy))
        p2  = (int(ex + dx), int(ey + dy))
        pygame.draw.line(surf, colour, p1, p2, thick)
        pygame.draw.circle(surf, colour, p1, thick // 2)
        pygame.draw.circle(surf, colour, p2, thick // 2)


def _draw_circle_eye(surf, cx: int, cy: int, eye: EyeDesc,
                     blink_ratio: float) -> None:
    import pygame
    colour = eye.colour or C_EYE
    r      = max(4, int(eye.w/2 * (1.0 - blink_ratio * 0.93)))
    ex, ey = int(cx + eye.ox), int(cy + eye.oy)
    _glow_circle(surf, (ex, ey), r, colour)
    pygame.draw.circle(surf, colour, (ex, ey), r)
    if r > 8:
        pygame.draw.circle(surf, C_HILITE,
                           (ex - r//4, ey - r//4), max(2, r//5))


# ---------------------------------------------------------------------------
# Mouth drawing functions
# ---------------------------------------------------------------------------
def _curve_pts(cx: float, cy: float, w: float, h: float,
               smile: bool, n: int = 32) -> list:
    """Generate polyline for a ∪ smile or ⌢ frown arc, both centred at (cx,cy)."""
    pts = []
    for i in range(n + 1):
        t  = i / n
        px = cx - w/2 + w*t
        if smile:
            py = cy - h/2 + h * math.sin(t * math.pi)   # dips DOWN
        else:
            py = cy + h/2 - h * math.sin(t * math.pi)   # rises UP
        pts.append((int(px), int(py)))
    return pts


def _draw_mouth_smile(surf, cx, cy, m: MouthDesc) -> None:
    import pygame
    pts   = _curve_pts(cx, cy, m.w, m.h, smile=True)
    thick = max(3, int(m.h * 0.45))
    pygame.draw.lines(surf, C_MOUTH, False, pts, thick)
    for cap in (pts[0], pts[-1]):
        pygame.draw.circle(surf, C_MOUTH, cap, thick // 2)


def _draw_mouth_frown(surf, cx, cy, m: MouthDesc) -> None:
    import pygame
    pts   = _curve_pts(cx, cy, m.w, m.h, smile=False)
    thick = max(3, int(m.h * 0.45))
    pygame.draw.lines(surf, C_MOUTH, False, pts, thick)
    for cap in (pts[0], pts[-1]):
        pygame.draw.circle(surf, C_MOUTH, cap, thick // 2)


def _draw_mouth_flat(surf, cx, cy, m: MouthDesc) -> None:
    import pygame
    w, h = max(8, m.w), max(4, m.h)
    rect = pygame.Rect(int(cx-w/2), int(cy-h/2), int(w), int(h))
    pygame.draw.rect(surf, C_MOUTH, rect, border_radius=max(2, int(h//2)))


def _draw_mouth_o(surf, cx, cy, m: MouthDesc, speak_ratio: float = 0.0) -> None:
    import pygame
    w  = max(8, m.w)
    h  = max(4, m.h * (0.30 + 0.70 * speak_ratio))
    cr = max(3, int(min(w/2, h/2)))
    pygame.draw.rect(surf, C_MOUTH,
                     pygame.Rect(int(cx-w/2), int(cy-h/2), int(w), int(h)),
                     border_radius=cr)


def _draw_mouth_dots(surf, cx, cy, m: MouthDesc) -> None:
    import pygame
    r       = max(3, int(m.h * 0.55))
    spacing = m.w / 3.0
    for i in range(3):
        pygame.draw.circle(surf, C_MOUTH,
                           (int(cx - m.w/2 + spacing*(i+0.5)), cy), r)


def _draw_mouth_squiggle(surf, cx, cy, m: MouthDesc) -> None:
    import pygame
    n   = 30
    pts = [(int(cx - m.w/2 + m.w*i/n),
            int(cy + m.h*0.5*math.sin(i/n*4*math.pi)))
           for i in range(n+1)]
    thick = max(3, int(m.h * 0.40))
    if len(pts) >= 2:
        pygame.draw.lines(surf, C_MOUTH, False, pts, thick)


def _draw_mouth_question(surf, cx, cy, m: MouthDesc) -> None:
    import pygame
    try:
        font = _get_font("consolas", max(14, int(m.h * 1.1)), bold=True)
        if font:
            txt  = font.render("?", True, C_MOUTH)
            surf.blit(txt, (cx - txt.get_width()//2, cy - txt.get_height()//2))
        else:
            pygame.draw.circle(surf, C_MOUTH, (cx, cy), max(4, int(m.h*0.5)), 3)
    except Exception:
        pygame.draw.circle(surf, C_MOUTH, (cx, cy), max(4, int(m.h*0.5)), 3)


def _draw_mouth_kiss(surf, cx, cy, m: MouthDesc) -> None:
    """Pursed kiss mouth  ( ˘ ³˘ )"""
    import pygame
    r  = max(5, int(m.w / 2.5))
    # Small oval
    pygame.draw.rect(surf, C_MOUTH,
                     pygame.Rect(cx-r, cy-max(3, r//2), r*2, max(6, r)),
                     border_radius=r//2)
    # Tiny arch above
    n   = 13
    pts = [(int(cx + m.w*0.35*(2*i/n - 1)),
            int(cy - r - 3 + 5*math.sin(i/n*math.pi)))
           for i in range(n+1)]
    thick = max(2, int(m.h * 0.22))
    if len(pts) >= 2:
        pygame.draw.lines(surf, C_MOUTH, False, pts, thick)


def _draw_mouth_wavy(surf, cx, cy, m: MouthDesc) -> None:
    """Anxious multi-frequency wavy mouth."""
    import pygame
    n   = 28
    pts = [(int(cx - m.w/2 + m.w*i/n),
            int(cy + m.h*0.5*(0.65*math.sin(i/n*5*math.pi)
                               + 0.35*math.sin(i/n*9*math.pi + 0.7))))
           for i in range(n+1)]
    thick = max(3, int(m.h * 0.38))
    if len(pts) >= 2:
        pygame.draw.lines(surf, C_MOUTH, False, pts, thick)


# ---------------------------------------------------------------------------
# Overlay renderers  (called after face draw, before scanlines)
# ---------------------------------------------------------------------------

def _ov_blush(surf, lx: int, rx: int, ey: int) -> None:
    """Pink blush ellipses under each eye."""
    import pygame
    for ex in (lx, rx):
        s = pygame.Surface((40, 22), pygame.SRCALPHA)
        pygame.draw.ellipse(s, (*C_BLUSH, 140), (0, 0, 40, 22))
        surf.blit(s, (ex - 20, ey + 38))


def _ov_excited_stars(surf, lx: int, rx: int, ey: int, t: float) -> None:
    """Twinkling corner sparkles."""
    import pygame
    positions = [(lx-54, ey-42), (lx-42, ey+46), (rx+54, ey-42), (rx+42, ey+46)]
    for i, (px, py) in enumerate(positions):
        pulse = 0.55 + 0.45 * math.sin(t * 3.8 + i * 1.3)
        r     = max(2, int(8 * pulse))
        s     = pygame.Surface((r*4+4, r*4+4), pygame.SRCALPHA)
        _draw_mini_star(s, r*2+2, r*2+2, r, (*C_STAR, 210))
        surf.blit(s, (px - r*2 - 2, py - r*2 - 2))


def _ov_sick(surf, cx: int, cy: int, t: float, w: int, h: int) -> None:
    """Subtle green tint + periodic 'Achoo!' burst."""
    import pygame
    tint = pygame.Surface((w, h), pygame.SRCALPHA)
    tint.fill((60, 180, 40, 18))
    surf.blit(tint, (0, 0))
    phase = t % 4.5
    if 0.0 < phase < 0.75:
        try:
            font = _get_font("consolas", 26, bold=True)
            if font:
                alpha = max(0, int(255 * (1 - phase / 0.75)))
                txt   = font.render("Achoo!", True, C_EYE)
                s     = pygame.Surface(txt.get_size(), pygame.SRCALPHA)
                s.blit(txt, (0, 0))
                s.set_alpha(alpha)
                surf.blit(s, (cx - txt.get_width()//2, cy - 56))
        except Exception:
            pass


def _ov_battery(surf, w: int, h: int, full: bool, t: float) -> None:
    """Battery HUD at top of screen."""
    import pygame
    bw, bh = 188, 72
    bx = w//2 - bw//2
    by = 18
    colour = C_BATTERY_G if full else C_BATTERY_R
    alpha  = 255 if full else (255 if int(t * 2) % 2 == 0 else 110)
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    # Outline
    pygame.draw.rect(s, (*colour, alpha), (bx, by, bw, bh), 3, border_radius=10)
    # Tip
    pygame.draw.rect(s, (*colour, alpha),
                     (bx + bw, by + bh//2 - 15, 12, 30), 0, border_radius=5)
    # Segments
    nseg = 3
    sw   = (bw - 24) // nseg
    for i in range(nseg):
        on  = full or i == 0
        sc  = (*colour, alpha) if on else (55, 55, 55, alpha)
        sx  = bx + 12 + i * (sw + 5)
        pygame.draw.rect(s, sc, (sx, by + 12, sw, bh - 24), 0, border_radius=5)
    # Label
    try:
        font  = _get_font("consolas", 17, bold=True)
        if font:
            label = "FULL" if full else "LOW !"
            txt   = font.render(label, True, (*colour, alpha))
            s.blit(txt, (w//2 - txt.get_width()//2, by + bh + 6))
    except Exception:
        pass
    surf.blit(s, (0, 0))


def _ov_speech_bubble(surf, cx: int, top_y: int, text: str, t: float) -> None:
    """Floating speech bubble above the face."""
    import pygame
    font = _get_font("consolas", 20, bold=True)
    if not font:
        return
    txt  = font.render(text, True, C_BG)
    tw, th = txt.get_size()
    pad  = 12
    bw   = tw + pad * 2
    bh   = th + pad * 2
    dy   = int(4 * math.sin(t * 1.8))
    bx   = cx - bw // 2
    by   = top_y - bh - 8 + dy
    s    = pygame.Surface((bw + 4, bh + 20), pygame.SRCALPHA)
    pygame.draw.rect(s, (*C_EYE, 230), (0, 0, bw, bh), border_radius=12)
    pygame.draw.polygon(s, (*C_EYE, 230),
                        [(bw//2 - 8, bh), (bw//2 + 8, bh), (bw//2, bh + 16)])
    s.blit(txt, (pad, pad))
    surf.blit(s, (bx, by))


def _ov_weather(surf, cx: int, by: int, t: float,
                city: str, temp_c: int, condition: str, desc: str) -> None:
    """Cyberpunk OLED Weather HUD with custom animated icons and [City] | [Temp]°C | [Condition] banner."""
    import pygame
    card_w, card_h = 360, 48
    bx = cx - card_w // 2
    # Semi-transparent dark card background
    card_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
    pygame.draw.rect(card_surf, (8, 18, 30, 220), (0, 0, card_w, card_h), border_radius=12)
    # Cyan border glow
    pygame.draw.rect(card_surf, (0, 229, 255, 160), (0, 0, card_w, card_h), 2, border_radius=12)
    surf.blit(card_surf, (bx, by))

    # Weather Icon (Centred at ix, iy)
    ix = bx + 30
    iy = by + 24
    cond = condition.lower()
    desc_l = (desc or "").lower()

    if cond in ("sunny", "clear") or "clear" in desc_l or "sun" in desc_l:
        # CLEAR / SUNNY: Glowing circular sun with small rotating rays
        pygame.draw.circle(surf, (255, 215, 45), (ix, iy), 8)
        ray_r = 13 + int(2 * math.sin(t * 3.5))
        for i in range(8):
            ang = math.radians(i * 45 + t * 45)
            rx1 = ix + int(10 * math.cos(ang))
            ry1 = iy + int(10 * math.sin(ang))
            rx2 = ix + int(ray_r * math.cos(ang))
            ry2 = iy + int(ray_r * math.sin(ang))
            pygame.draw.line(surf, (255, 230, 80), (rx1, ry1), (rx2, ry2), 2)

    elif cond in ("rainy", "drizzle") or "rain" in desc_l or "drizzle" in desc_l:
        # RAIN / DRIZZLE: Animated falling diagonal rain droplets
        pygame.draw.circle(surf, (140, 210, 255), (ix - 5, iy - 3), 7)
        pygame.draw.circle(surf, (180, 230, 255), (ix + 3, iy - 5), 8)
        pygame.draw.circle(surf, (140, 210, 255), (ix + 10, iy - 1), 5)
        pygame.draw.rect(surf, (140, 210, 255), (ix - 8, iy - 1, 20, 6), border_radius=3)
        # 3 Animated falling diagonal raindrops
        for k in range(3):
            dx = -5 + k * 6
            drop_y = iy + 6 + int((t * 28 + k * 9) % 14)
            pygame.draw.line(surf, (0, 229, 255), (ix + dx, drop_y), (ix + dx - 3, drop_y + 4), 2)

    elif cond == "stormy" or "storm" in desc_l or "thunder" in desc_l:
        # Thunderstorm: Cloud + yellow lightning
        pygame.draw.circle(surf, (140, 180, 220), (ix - 5, iy - 3), 7)
        pygame.draw.circle(surf, (180, 210, 240), (ix + 3, iy - 5), 8)
        pygame.draw.rect(surf, (140, 180, 220), (ix - 8, iy - 1, 20, 6), border_radius=3)
        pts = [(ix + 1, iy + 4), (ix - 2, iy + 9), (ix + 2, iy + 9), (ix - 1, iy + 16)]
        pygame.draw.lines(surf, (255, 230, 40), False, pts, 2)

    elif cond == "snowy" or "snow" in desc_l:
        # Snowy: Cloud + falling snowflakes
        pygame.draw.circle(surf, (200, 235, 255), (ix - 4, iy - 2), 8)
        pygame.draw.circle(surf, (230, 245, 255), (ix + 4, iy - 4), 9)
        pygame.draw.rect(surf, (200, 235, 255), (ix - 7, iy - 1, 20, 8), border_radius=3)
        for k in range(3):
            dx = -5 + k * 6
            drop_y = iy + 8 + int((t * 15 + k * 6) % 10)
            pygame.draw.circle(surf, (255, 255, 255), (ix + dx, drop_y), 2)

    else:
        # OVERCAST / CLOUDY: Cyan fluffy cloud outlines + light rain/mist dots
        pygame.draw.circle(surf, (0, 229, 255), (ix - 6, iy - 2), 8, 2)
        pygame.draw.circle(surf, (0, 229, 255), (ix + 3, iy - 6), 9, 2)
        pygame.draw.circle(surf, (0, 229, 255), (ix + 11, iy - 1), 6, 2)
        pygame.draw.line(surf, (0, 229, 255), (ix - 10, iy + 5), (ix + 14, iy + 5), 2)
        # Light rain/mist dots
        for k in range(3):
            dx = -4 + k * 5
            dot_y = iy + 8 + int((t * 12 + k * 4) % 6)
            pygame.draw.circle(surf, (0, 200, 255), (ix + dx, dot_y), 1)

    # Bottom HUD: Display "[City] | [Temp]°C | [Condition]"
    try:
        font_hud = _get_font("consolas", 15, bold=True)
        if font_hud:
            display_cond = (desc or condition).title()
            if len(display_cond) > 18:
                display_cond = display_cond[:16] + ".."
            hud_str = f"{city} | {temp_c}°C | {display_cond}"
            hud_txt = font_hud.render(hud_str, True, (0, 229, 255))
            surf.blit(hud_txt, (bx + 58, by + 15))
    except Exception:
        pass


_overlay_weather = _ov_weather


def _ov_telemetry(surf, cx: int, by: int, t: float,
                  cpu: float, ram: float, temp: float, battery: int,
                  is_charging: bool = False, is_alert: bool = False) -> None:
    """Cyberpunk OLED Telemetry HUD with CPU, RAM, Battery, and Temperature status pills."""
    import pygame
    card_w, card_h = 380, 48
    bx = cx - card_w // 2

    # Card background
    card_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
    pygame.draw.rect(card_surf, (8, 16, 28, 225), (0, 0, card_w, card_h), border_radius=12)
    border_color = (255, 70, 70, 200) if is_alert else (0, 229, 255, 160)
    pygame.draw.rect(card_surf, border_color, (0, 0, card_w, card_h), 2, border_radius=12)
    surf.blit(card_surf, (bx, by))

    try:
        font = _get_font("consolas", 13, bold=True)
        if font:
            # CPU
            cpu_col = (255, 120, 80) if cpu > 80 else (0, 229, 255)
            txt_cpu = font.render(f"CPU {int(cpu)}%", True, cpu_col)
            surf.blit(txt_cpu, (bx + 16, by + 16))

            # RAM
            ram_col = (255, 120, 80) if ram > 85 else (0, 229, 255)
            txt_ram = font.render(f"RAM {int(ram)}%", True, ram_col)
            surf.blit(txt_ram, (bx + 104, by + 16))

            # TEMP
            temp_col = (255, 70, 70) if (temp >= 65.0 or is_alert) else ((255, 210, 60) if temp >= 60.0 else (80, 255, 140))
            txt_temp = font.render(f"TEMP {int(temp)}°C", True, temp_col)
            surf.blit(txt_temp, (bx + 196, by + 16))

            # BATTERY
            bat_str = f"BAT {battery}%" + ("⚡" if is_charging else "")
            bat_col = (80, 255, 140) if battery > 20 else (255, 80, 60)
            txt_bat = font.render(bat_str, True, bat_col)
            surf.blit(txt_bat, (bx + 292, by + 16))
    except Exception:
        pass


def _ov_music(surf, cx: int, by: int, t: float, track_title: str) -> None:
    """Animated Equalizer Visualizer & Track Banner for Music Playback."""
    import pygame
    num_bars = 16
    bar_w = 12
    bar_gap = 5
    total_w = num_bars * (bar_w + bar_gap) - bar_gap
    start_x = cx - total_w // 2

    # Draw bouncing dynamic equalizer bars
    base_y = by + 26
    for i in range(num_bars):
        bx = start_x + i * (bar_w + bar_gap)
        # Compute dynamic height using multi-frequency harmonic synthesis
        h_mod = (
            math.sin(t * 7.0 + i * 0.75) * 0.4 +
            math.cos(t * 11.0 + i * 1.3) * 0.35 +
            math.sin(t * 4.0 - i * 0.5) * 0.25
        )
        bar_h = int(8 + (0.5 + 0.5 * h_mod) * 26)

        # Gradient color: Cyan at bottom to Neon Green/Yellow at top
        r = int(min(255, max(0, 30 + i * 12)))
        g = int(min(255, max(0, 200 + int(30 * math.sin(t * 5 + i)))))
        b = int(min(255, max(0, 255 - i * 10)))
        pygame.draw.rect(surf, (r, g, b), (bx, base_y - bar_h, bar_w, bar_h), border_radius=3)

    # Track Title Banner
    try:
        font_music = _get_font("consolas", 13, bold=True)
        if font_music:
            title_display = track_title or "Morphix Music Player"
            if len(title_display) > 28:
                title_display = title_display[:26] + ".."
            txt = font_music.render(f"Music: {title_display}", True, (0, 240, 255))
            surf.blit(txt, (cx - txt.get_width() // 2, by - 12))
    except Exception:
        pass


def _ov_burger(surf, cx: int, by: int, t: float) -> None:
    """Pixel burger with floating-bun bite animation."""
    import pygame
    bite    = 0.45 + 0.45 * math.sin(t * 2.4)
    layers  = [
        ((220, 175, 85),  62, 24),   # top bun
        ((255, 215, 55),  58,  8),   # cheese
        ((175,  85,  35), 54, 11),   # patty
        (( 95, 195,  80), 58,  8),   # lettuce
        ((220, 175, 85),  62, 18),   # bottom bun
    ]
    total_h = sum(h for _, _, h in layers) + len(layers) * 3
    y       = by - total_h // 2
    for i, (col, w, h) in enumerate(layers):
        yo = y - int(bite * 20) if i == 0 else y
        pygame.draw.rect(surf, col,
                         (cx - w//2, yo, w, h), border_radius=5)
        y += h + 3
    # Sesame seeds on top bun
    bun_y = by - total_h//2 - int(bite * 20)
    for sx, sy in [(cx-10, bun_y+7), (cx+5, bun_y+4), (cx-2, bun_y+14)]:
        pygame.draw.circle(surf, (200, 155, 65), (sx, sy), 3)


def _ov_noodles(surf, cx: int, by: int, t: float) -> None:
    """Noodle bowl with animated wavy noodles + steam."""
    import pygame
    bowl_rx, bowl_ry = 46, 24
    # Bowl body
    pygame.draw.ellipse(surf, (175, 135, 95),
                        (cx-bowl_rx, by-bowl_ry//2, bowl_rx*2, bowl_ry), 0)
    pygame.draw.ellipse(surf, (215, 175, 135),
                        (cx-bowl_rx, by-bowl_ry//2-7, bowl_rx*2, 20), 0)
    # Noodle lines
    for row in range(4):
        n   = 18
        pts = [(int(cx - bowl_rx + 10 + (bowl_rx*2-20)*i/n),
                int(by - bowl_ry//4 + row*7 +
                    int(5 * math.sin(i*0.85 + t*3.2 + row*1.6))))
               for i in range(n+1)]
        if len(pts) >= 2:
            pygame.draw.lines(surf, (240, 220, 155), False, pts, 3)
    # Steam puffs
    for i in range(3):
        wx    = cx - 14 + i * 14
        phase = t * 2.6 + i * 1.2
        wy    = by - bowl_ry//2 - 8 - int(14 * (math.sin(phase)*0.5 + 0.5))
        for k in range(3):
            sy    = wy - k * 7
            alpha = max(0, 160 - k * 50)
            s     = pygame.Surface((12, 12), pygame.SRCALPHA)
            pygame.draw.circle(s, (200, 200, 200, alpha), (6, 6), max(1, 4-k))
            surf.blit(s, (wx - 6, sy - 6))


def _ov_kiss_hearts(surf, cx: int, ey: int, t: float) -> None:
    """Pulsing centre heart + two small side hearts."""
    import pygame
    pulse = 0.70 + 0.30 * math.sin(t * 4.5)
    r     = int(30 * pulse)
    hy    = ey - 42
    s     = pygame.Surface((r*4+4, r*4+4), pygame.SRCALPHA)
    _draw_mini_heart(s, r*2+2, r*2+2, r, (*C_HEART, 215))
    surf.blit(s, (cx - r*2 - 2, hy - r*2 - 2))
    for sgn, phase in ((-1, 0.9), (1, 1.3)):
        r2  = int(14 * (0.55 + 0.45*math.sin(t*4.5 + phase)))
        hx2 = cx + sgn * int(44 * pulse)
        hy2 = hy + int(10 * math.sin(t*4.5 + phase))
        s2  = pygame.Surface((r2*4+4, r2*4+4), pygame.SRCALPHA)
        _draw_mini_heart(s2, r2*2+2, r2*2+2, r2, (*C_HEART, 155))
        surf.blit(s2, (hx2-r2*2-2, hy2-r2*2-2))


def _ov_rps(surf, cx: int, cy: int,
            choice: int, settled: bool, t: float) -> None:
    """Rock ✊ / Paper ✋ / Scissors ✌ overlay."""
    import pygame
    labels    = ["ROCK!", "PAPER!", "SCISSORS!"]
    icons_txt = ["[R]", "[P]", "[S]"]   # ASCII fallback
    idx       = choice if settled else int(t * 4) % 3
    font = _get_font("consolas", 30, bold=True)
    if font:
        txt = font.render(icons_txt[idx], True, C_EYE)
        surf.blit(txt, (cx - txt.get_width()//2, cy - txt.get_height()//2 + 12))
    if settled:
        lf = _get_font("consolas", 15, bold=True)
        if lf:
            lbl = lf.render(labels[choice], True, C_MOUTH)
            surf.blit(lbl, (cx - lbl.get_width()//2, cy + 44))


def _ov_party_hat(surf, cx: int, ey: int, t: float) -> None:
    """Party hat above the face."""
    import pygame
    tip_y  = int(ey - 82 - 7 * math.sin(t * 2.8))
    brim_y = tip_y + 58
    brim_w = 66
    pts    = [(cx, tip_y), (cx - brim_w//2, brim_y), (cx + brim_w//2, brim_y)]
    pygame.draw.polygon(surf, C_PARTY_HAT, pts)
    # Stripes
    for i in range(3):
        f      = (i + 1) / 4
        p1     = (int(cx + (cx - brim_w//2 - cx)*f), int(tip_y + (brim_y - tip_y)*f))
        p2     = (int(cx + (cx + brim_w//2 - cx)*f), int(tip_y + (brim_y - tip_y)*f))
        pygame.draw.line(surf, (255, 220, 80), p1, p2, 3)
    pygame.draw.polygon(surf, (255, 140, 75), pts, 2)
    # Brim ellipse
    pygame.draw.ellipse(surf, (255, 140, 75),
                        (cx - brim_w//2 - 4, brim_y - 6, brim_w + 8, 14))
    # Pompom
    pygame.draw.circle(surf, (255, 240, 100), (cx, tip_y), 8)


def _ov_modak(surf, cx: int, my: int, ot: float) -> None:
    """Steamed golden modak dumpling with saffron folds and steam."""
    import pygame
    sw = math.sin(ot * 3.0) * 3
    y = my - 12
    # Steam wave
    pygame.draw.line(surf, (200, 240, 255), (cx, y - 22), (int(cx + sw), y - 30), 2)
    # Dumpling body
    body_pts = [
        (cx, y - 18),
        (cx + 13, y + 2),
        (cx + 9, y + 14),
        (cx - 9, y + 14),
        (cx - 13, y + 2),
    ]
    pygame.draw.polygon(surf, (255, 242, 185), body_pts)
    pygame.draw.polygon(surf, (245, 166, 35), body_pts, 2)
    # Pleats / folds
    pygame.draw.line(surf, (230, 140, 20), (cx, y - 16), (cx, y + 14), 2)
    pygame.draw.line(surf, (230, 140, 20), (cx, y - 16), (cx - 7, y + 12), 2)
    pygame.draw.line(surf, (230, 140, 20), (cx, y - 16), (cx + 7, y + 12), 2)


def _ov_tilak(surf, cx: int, ey: int) -> None:
    """Traditional Chandan Tilak on the forehead between eyes."""
    import pygame
    # Red vermillion line
    pygame.draw.rect(surf, (255, 23, 68), (cx - 2, ey - 48, 4, 18), border_radius=2)
    # Golden Chandan dot in center
    pygame.draw.circle(surf, (255, 215, 0), (cx, ey - 38), 3)


def _ov_diwali_diyas(surf, lx: int, rx: int, my: int, ot: float) -> None:
    """Twin glowing terracotta diyas with animated flickering flames."""
    import pygame
    flame_flicker = int(math.sin(ot * 12.0) * 3)
    for dx in (lx - 25, rx + 25):
        dy = my + 10
        # Clay base
        pygame.draw.arc(surf, (184, 85, 30), (dx - 12, dy, 24, 14), 3.14, 0.0, 3)
        pygame.draw.line(surf, (220, 120, 50), (dx - 12, dy + 7), (dx + 12, dy + 7), 2)
        # Flame glow aura
        aura_s = pygame.Surface((32, 32), pygame.SRCALPHA)
        pygame.draw.circle(aura_s, (255, 200, 0, 45), (16, 16), 12 + flame_flicker)
        surf.blit(aura_s, (dx - 16, dy - 16))
        # Flame polygon
        flame_pts = [
            (dx, dy - 12 + flame_flicker),
            (dx + 5, dy - 2),
            (dx, dy + 2),
            (dx - 5, dy - 2),
        ]
        pygame.draw.polygon(surf, (255, 235, 59), flame_pts)
        pygame.draw.circle(surf, (255, 110, 0), (dx, dy - 2), 2)


def _ov_santa_hat(surf, lx: int, ey: int, ot: float) -> None:
    """Mini red Santa hat on top of the left eye with pom-pom and snowflakes."""
    import pygame
    hx = lx - 5
    hy = ey - 48
    hat_pts = [(hx - 18, hy + 18), (hx + 18, hy + 18), (hx - 12, hy - 14)]
    pygame.draw.polygon(surf, (229, 57, 53), hat_pts)
    pygame.draw.rect(surf, (255, 255, 255), (hx - 22, hy + 14, 44, 9), border_radius=4)
    pygame.draw.circle(surf, (255, 255, 255), (hx - 14, hy - 16), 6)
    # Gentle snowflake particles
    for i in range(5):
        sx = int((lx - 70 + i * 40 + ot * 12) % 360) + 60
        sy = int((ey - 40 + i * 25 + ot * 18) % 180) + 20
        pygame.draw.circle(surf, (200, 235, 255), (sx, sy), 2)


def _ov_tricolor_badge(surf, cx: int, cy: int) -> None:
    """Indian Tiranga rosette ribbon badge."""
    import pygame
    bx = cx
    by = cy + 18
    w, h = 36, 12
    pygame.draw.rect(surf, (255, 111, 0), (bx - w//2, by - h//2, w, 4), border_top_left_radius=3, border_top_right_radius=3)
    pygame.draw.rect(surf, (255, 255, 255), (bx - w//2, by - h//2 + 4, w, 4))
    pygame.draw.circle(surf, (13, 71, 161), (bx, by), 2)
    pygame.draw.rect(surf, (46, 125, 50), (bx - w//2, by - h//2 + 8, w, 4), border_bottom_left_radius=3, border_bottom_right_radius=3)


def _ov_holi_splash(surf, cx: int, cy: int, ot: float) -> None:
    """Colorful gulal powder burst dots."""
    import pygame
    colors = [(233, 30, 99), (255, 234, 0), (0, 229, 255), (118, 255, 3), (255, 87, 34)]
    offsets = [(-60, -35), (60, -35), (-80, 20), (80, 20), (0, -45), (-40, 45), (40, 45)]
    for i, (ox, oy) in enumerate(offsets):
        c = colors[i % len(colors)]
        r = int(4 + 2 * math.sin(ot * 4.0 + i))
        pygame.draw.circle(surf, c, (cx + ox, cy + oy), r)


def _ov_kite(surf, cx: int, ey: int, ot: float) -> None:
    """Flying diamond kite with string."""
    import pygame
    kx = cx + 85
    ky = ey - 35 + int(math.sin(ot * 3.0) * 5)
    pts = [(kx, ky - 14), (kx + 12, ky), (kx, ky + 14), (kx - 12, ky)]
    pygame.draw.polygon(surf, (0, 229, 255), pts)
    pygame.draw.polygon(surf, (255, 64, 129), pts, 2)
    pygame.draw.line(surf, (255, 255, 255), (kx, ky - 14), (kx, ky + 14), 1)
    pygame.draw.line(surf, (255, 255, 255), (kx - 12, ky), (kx + 12, ky), 1)
    pygame.draw.lines(surf, (255, 215, 0), False, [(kx, ky + 14), (kx - 4, ky + 20), (kx + 3, ky + 26)], 2)


def _ov_festival_hud(surf, cx: int, y: int, title: str) -> None:
    """Bottom HUD pill rendering festival greeting title."""
    if not title:
        return
    import pygame
    try:
        font = pygame.font.SysFont("consolas", 12, bold=True)
    except Exception:
        return

    txt_surf = font.render(title, True, (255, 220, 90))
    tw, th = txt_surf.get_width(), txt_surf.get_height()
    bx = cx - (tw + 16) // 2
    by = y - th // 2 - 1

    pill = pygame.Rect(bx, by, tw + 16, th + 4)
    pygame.draw.rect(surf, (20, 15, 35), pill, border_radius=6)
    pygame.draw.rect(surf, (255, 180, 0), pill, width=1, border_radius=6)
    surf.blit(txt_surf, (bx + 8, by + 2))


def _ov_festival(surf, cx: int, ey: int, my: int, ot: float,
                 theme: str, title: str, w: int, h: int, lx: int, rx: int) -> None:
    """Dispatches and renders the active festival procedural overlay."""
    th = str(theme).upper()
    if "MODAK" in th:
        _ov_tilak(surf, cx, ey)
        _ov_modak(surf, cx, my, ot)
    elif "DIWALI" in th or "DIYA" in th:
        _ov_diwali_diyas(surf, lx, rx, my, ot)
    elif "SANTA" in th or "CHRISTMAS" in th:
        _ov_santa_hat(surf, lx, ey, ot)
    elif "TRICOLOR" in th or "TIRANGA" in th:
        _ov_tricolor_badge(surf, cx, my)
    elif "HOLI" in th or "SPLASH" in th:
        _ov_holi_splash(surf, cx, ey, ot)
    elif "KITE" in th or "SANKRANTI" in th:
        _ov_kite(surf, cx, ey, ot)
    elif "NEW_YEAR" in th or "BIRTHDAY" in th:
        _ov_party_hat(surf, cx, ey, ot)

    if title:
        _ov_festival_hud(surf, cx, h - 20, title)


# ---------------------------------------------------------------------------
# EyeAnimator  — core renderer
# ---------------------------------------------------------------------------
class EyeAnimator:
    WINDOW_W = 480
    WINDOW_H = 240
    FPS      = 60

    EYE_L_X  = 0.325
    EYE_R_X  = 0.675
    EYE_Y    = 0.44
    MOUTH_Y  = 0.72

    TRANSITION_DUR   = 0.30
    BLINK_CLOSE_TIME = 0.055
    BLINK_OPEN_TIME  = 0.095
    IDLE_BLINK_MIN   = 3.2
    IDLE_BLINK_MAX   = 7.0
    MAX_PARTICLES    = 160

    def __init__(self, width: int = WINDOW_W, height: int = WINDOW_H) -> None:
        self.width  = width
        self.height = height
        self._state = EyeState.IDLE_NEUTRAL
        neutral     = FACE_STATES[EyeState.IDLE_NEUTRAL]

        self._src_l = dataclasses.replace(neutral.left)
        self._src_r = dataclasses.replace(neutral.right)
        self._src_m = dataclasses.replace(neutral.mouth)
        self._cur_l = dataclasses.replace(neutral.left)
        self._cur_r = dataclasses.replace(neutral.right)
        self._cur_m = dataclasses.replace(neutral.mouth)
        self._dst_l = dataclasses.replace(neutral.left)
        self._dst_r = dataclasses.replace(neutral.right)
        self._dst_m = dataclasses.replace(neutral.mouth)

        self._lerp_t   = 1.0
        self._lerp_dur = self.TRANSITION_DUR

        # Blink
        self._blink_t     = 0.0
        self._blink_phase = 0
        self._next_blink  = self._rand_blink()

        # Speaking
        self._speak_phase = 0.0
        self._speak_amp   = 0.0

        # ZZZ / dizzy / think
        self._zzz_t       = 0.0
        self._dizzy_angle = 0.0
        self._think_t     = 0.0

        # Overlay & state timers
        self._overlay_t    = 0.0
        self._sick_t       = 0.0
        self._battery_t    = 0.0
        self._rps_t        = 0.0
        self._rps_settled  = False
        self._rps_choice   = 0

        # Particles
        self._particles:   List[Particle] = []
        self._spawn_timer  = 0.0
        self._bang_spawned = False

        # Weather overlay info
        self._weather_city      = "Auto"
        self._weather_temp      = 25
        self._weather_condition = "sunny"
        self._weather_desc      = "Clear sky"

        # Music & Telemetry overlay info
        self._music_title       = "Morphix Music Player"
        self._music_playing     = False
        self._telemetry_cpu     = 15.0
        self._telemetry_ram     = 40.0
        self._telemetry_temp    = 42.0
        self._telemetry_battery = 100
        self._telemetry_charging = False
        self._telemetry_alert   = False

        # Seasonal / Indian Festival overlay info
        self._festival_theme    = "NONE"
        self._festival_title    = ""

        # Pre-cached fonts to avoid per-frame SysFont allocations (Pi 4B optimization)
        self.font_zzz       = _get_font("consolas", 19, bold=True)
        self.font_hud       = _get_font("consolas", 15, bold=True)
        self.font_festival  = _get_font("consolas", 12, bold=True)
        self.font_music     = _get_font("consolas", 13, bold=True)
        self.font_telemetry = _get_font("consolas", 13, bold=True)

        self._lock = threading.Lock()

    def set_festival_theme(self, theme: Any = "NONE", title: str = "") -> None:
        """Sets the active seasonal or Indian festival theme."""
        with self._lock:
            if hasattr(theme, "value"):
                self._festival_theme = str(theme.value)
            elif theme is None:
                self._festival_theme = "NONE"
            else:
                self._festival_theme = str(theme)
            self._festival_title = str(title) if title else ""

    def set_weather_data(self, city: str = "Auto", temp: float = 25.0,
                         condition: str = "sunny", desc: str = "Clear sky") -> None:
        with self._lock:
            self._weather_city      = city
            self._weather_temp      = int(round(temp))
            self._weather_condition = condition.lower()
            self._weather_desc      = desc

    def set_music_data(self, title: str = "Music", is_playing: bool = True) -> None:
        with self._lock:
            self._music_title   = title
            self._music_playing = is_playing

    def set_telemetry_data(self, cpu: float = 15.0, ram: float = 40.0,
                           temp: float = 42.0, battery: int = 100,
                           is_charging: bool = False, is_alert: bool = False) -> None:
        with self._lock:
            self._telemetry_cpu      = cpu
            self._telemetry_ram      = ram
            self._telemetry_temp     = temp
            self._telemetry_battery  = battery
            self._telemetry_charging = is_charging
            self._telemetry_alert    = is_alert

    def set_state(self, state: EyeState) -> None:
        with self._lock:
            if state == self._state:
                return
            self._state = state
            frame = FACE_STATES[state]
            self._src_l = dataclasses.replace(self._cur_l)
            self._src_r = dataclasses.replace(self._cur_r)
            self._src_m = dataclasses.replace(self._cur_m)
            self._dst_l = dataclasses.replace(frame.left)
            self._dst_r = dataclasses.replace(frame.right)
            self._dst_m = dataclasses.replace(frame.mouth)
            self._lerp_t      = 0.0
            self._speak_amp   = 1.0 if state == EyeState.SPEAKING else 0.0
            self._speak_phase = 0.0
            self._zzz_t       = 0.0
            self._dizzy_angle = 0.0
            self._think_t     = 0.0
            self._overlay_t   = 0.0
            self._sick_t      = 0.0
            self._battery_t   = 0.0
            self._rps_t       = 0.0
            self._rps_settled = False
            self._rps_choice  = 0
            self._spawn_timer = 0.0
            self._bang_spawned = False
            self._particles.clear()

    def trigger_blink(self) -> None:
        with self._lock:
            if self._blink_phase == 0:
                self._blink_t     = 0.0
                self._blink_phase = 1

    def get_state(self) -> EyeState:
        return self._state

    # ------------------------------------------------------------------
    def _rand_blink(self) -> float:
        return random.uniform(self.IDLE_BLINK_MIN, self.IDLE_BLINK_MAX)

    # ------------------------------------------------------------------
    # Particle spawners
    # ------------------------------------------------------------------
    def _spawn_tears(self) -> None:
        ax_l = int(self.width  * self.EYE_L_X)
        ax_r = int(self.width  * self.EYE_R_X)
        ay   = int(self.height * self.EYE_Y)
        for ex in (ax_l, ax_r):
            self._particles.append(Particle(
                x=ex + random.uniform(-7, 7), y=ay + 28,
                vx=random.uniform(-6, 6),     vy=random.uniform(28, 55),
                life=1.3, max_life=1.3,
                r=random.uniform(3, 5),
                colour=C_TEAR, shape="teardrop", gravity=35.0,
            ))

    def _spawn_pet_effect(self) -> None:
        cx = self.width // 2
        cy = int(self.height * 0.5)
        self._particles.append(Particle(
            x=random.uniform(cx-100, cx+100),
            y=random.uniform(cy-20, cy+30),
            vx=random.uniform(-18, 18),
            vy=-random.uniform(22, 52),
            life=2.0, max_life=2.0,
            r=random.uniform(7, 14),
            colour=random.choice([C_HEART, (255,220,80), (140,255,120), (200,100,255)]),
            shape=random.choice(["heart", "flower", "star"]),
            gravity=-8.0,
        ))

    def _spawn_bang_stars(self) -> None:
        cx = self.width  // 2
        cy = int(self.height * 0.50)
        for i in range(20):
            ang   = i * (360 / 20)
            speed = random.uniform(65, 160)
            self._particles.append(Particle(
                x=cx, y=cy,
                vx=speed * math.cos(math.radians(ang)),
                vy=speed * math.sin(math.radians(ang)),
                life=1.6, max_life=1.6,
                r=random.uniform(4, 10),
                colour=random.choice([(255,220,80),(255,160,60),(255,80,80),(255,255,180)]),
                shape="star",
                gravity=28.0,
                angle=random.uniform(0, 360),
                va=random.uniform(-200, 200),
            ))

    def _spawn_confetti(self) -> None:
        for _ in range(random.randint(1, 3)):
            self._particles.append(Particle(
                x=random.uniform(20, self.width - 20),
                y=random.uniform(-12, 15),
                vx=random.uniform(-45, 45),
                vy=random.uniform(25, 85),
                life=random.uniform(2.5, 4.0),
                max_life=4.0,
                r=random.uniform(4, 7),
                colour=random.choice(_CONFETTI_COLS),
                shape="confetti",
                angle=random.uniform(0, 360),
                va=random.uniform(-220, 220),
                gravity=22.0,
            ))

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------
    def update(self, dt: float) -> None:
        with self._lock:
            # Transition lerp
            if self._lerp_t < 1.0:
                self._lerp_t = min(1.0, self._lerp_t + dt / self._lerp_dur)
                t = _ease_out(self._lerp_t)
                self._cur_l = _lerp_eye(self._src_l,   self._dst_l, t)
                self._cur_r = _lerp_eye(self._src_r,   self._dst_r, t)
                self._cur_m = _lerp_mouth(self._src_m, self._dst_m, t)

            # Auto-blink
            if self._state not in (EyeState.DEEP_SLEEP, EyeState.WINKING,
                                   EyeState.DIZZY,      EyeState.INTERACTION_BANG_SHOOT):
                self._next_blink -= dt
                if self._next_blink <= 0:
                    self._blink_phase = 1
                    self._blink_t     = 0.0
                    self._next_blink  = self._rand_blink()

            # Blink phases
            if self._blink_phase == 1:
                self._blink_t += dt / self.BLINK_CLOSE_TIME
                if self._blink_t >= 1.0:
                    self._blink_t     = 0.0
                    self._blink_phase = 2
            elif self._blink_phase == 2:
                self._blink_t += dt / self.BLINK_OPEN_TIME
                if self._blink_t >= 1.0:
                    self._blink_t     = 0.0
                    self._blink_phase = 0

            # Speaking oscillation
            if self._state == EyeState.SPEAKING:
                self._speak_phase += dt * 5.5
                self._speak_amp = 1.0
            else:
                self._speak_amp = max(0.0, self._speak_amp - dt * 4.0)

            # State timers
            self._overlay_t += dt
            if self._state == EyeState.DEEP_SLEEP:
                self._zzz_t += dt
            if self._state == EyeState.DIZZY or self._state == EyeState.INTERACTION_BANG_SHOOT:
                self._dizzy_angle += dt * 200.0
            if self._state == EyeState.THINKING:
                self._think_t += dt
            if self._state == EyeState.SICK_COLD:
                self._sick_t += dt
            if self._state == EyeState.BATTERY_LOW:
                self._battery_t += dt
            if self._state == EyeState.INTERACTION_RPS:
                self._rps_t += dt
                if not self._rps_settled and self._rps_t >= 2.5:
                    self._rps_settled = True
                    self._rps_choice  = random.randint(0, 2)

            # Particle spawning
            self._spawn_timer -= dt
            if self._spawn_timer <= 0:
                if self._state == EyeState.CRYING:
                    self._spawn_timer = random.uniform(0.25, 0.60)
                    self._spawn_tears()
                elif self._state == EyeState.AFFECTION_PET:
                    self._spawn_timer = random.uniform(0.18, 0.45)
                    self._spawn_pet_effect()
                elif self._state == EyeState.BIRTHDAY_PARTY:
                    self._spawn_timer = random.uniform(0.06, 0.16)
                    self._spawn_confetti()
                else:
                    self._spawn_timer = 0.5

            if (self._state == EyeState.INTERACTION_BANG_SHOOT
                    and not self._bang_spawned):
                self._spawn_bang_stars()
                self._bang_spawned = True

            # Update particles
            for p in self._particles:
                p.x   += p.vx * dt
                p.y   += p.vy * dt
                p.vy  += p.gravity * dt
                p.angle += p.va * dt
                p.life  -= dt
            self._particles = [p for p in self._particles if p.life > 0]
            if len(self._particles) > self.MAX_PARTICLES:
                self._particles = self._particles[-self.MAX_PARTICLES:]

    # ------------------------------------------------------------------
    # draw
    # ------------------------------------------------------------------
    def draw(self, surface) -> None:
        import pygame
        surface.fill(C_BG)

        with self._lock:
            # Blink ratio
            if self._blink_phase == 1:
                br = self._blink_t
            elif self._blink_phase == 2:
                br = 1.0 - self._blink_t
            else:
                br = 0.0

            ax_l = int(self.width  * self.EYE_L_X)
            ax_r = int(self.width  * self.EYE_R_X)
            ay   = int(self.height * self.EYE_Y)
            my   = int(self.height * self.MOUTH_Y)
            cx   = self.width // 2

            speak_ratio = (0.5 + 0.5 * math.sin(self._speak_phase)) * self._speak_amp

            # Sick shiver (±5px oscillation)
            cur_l = self._cur_l
            cur_r = self._cur_r
            if self._state == EyeState.SICK_COLD:
                shiv = 5 * math.sin(self._sick_t * 18)
                cur_l = dataclasses.replace(self._cur_l, ox=self._cur_l.ox + shiv)
                cur_r = dataclasses.replace(self._cur_r, ox=self._cur_r.ox + shiv)

            # Draw eyes
            self._render_eye(surface, ax_l, ay, cur_l, br, self._dizzy_angle, True)
            self._render_eye(surface, ax_r, ay, cur_r, br, self._dizzy_angle, False)

            # Draw mouth
            self._render_mouth(surface, cx, my, self._cur_m, speak_ratio)

            # Draw particles
            for p in self._particles:
                _draw_particle(surface, p)

            # State-specific overlays
            self._draw_overlay(surface, ax_l, ax_r, ay, my, cx)

            # ZZZ
            if self._state == EyeState.DEEP_SLEEP:
                self._draw_zzz(surface, ax_r + 56, ay - 26)

        # Scanlines
        scan = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for y in range(0, self.height, 4):
            pygame.draw.line(scan, (0, 0, 0, 14), (0, y), (self.width, y))
        surface.blit(scan, (0, 0))

    # ------------------------------------------------------------------
    def _render_eye(self, surf, cx, cy, eye: EyeDesc,
                    br: float, dizzy_angle: float, is_left: bool) -> None:
        s = eye.style
        if   s == "pill":      _draw_pill(surf, cx, cy, eye, br)
        elif s == "angry":     _draw_angry_pill(surf, cx, cy, eye, br, is_left)
        elif s == "flat":      _draw_flat(surf, cx, cy, eye)
        elif s == "arc_happy": _draw_arc_happy(surf, cx, cy, eye)
        elif s == "wink":      _draw_wink(surf, cx, cy, eye)
        elif s == "heart":     _draw_heart_eye(surf, cx, cy, eye)
        elif s == "x_mark":    _draw_x_mark(surf, cx, cy, eye, dizzy_angle)
        elif s == "circle":    _draw_circle_eye(surf, cx, cy, eye, br)

    def _render_mouth(self, surf, cx, cy, m: MouthDesc,
                      speak_ratio: float) -> None:
        s = m.style
        if   s == "smile":      _draw_mouth_smile(surf, cx, cy, m)
        elif s == "frown":      _draw_mouth_frown(surf, cx, cy, m)
        elif s == "flat":       _draw_mouth_flat(surf, cx, cy, m)
        elif s in ("o", "speak_open"):
                                _draw_mouth_o(surf, cx, cy, m, speak_ratio)
        elif s == "dots":       _draw_mouth_dots(surf, cx, cy, m)
        elif s == "squiggle":   _draw_mouth_squiggle(surf, cx, cy, m)
        elif s == "question":   _draw_mouth_question(surf, cx, cy, m)
        elif s == "kiss":       _draw_mouth_kiss(surf, cx, cy, m)
        elif s == "wavy":       _draw_mouth_wavy(surf, cx, cy, m)

    def _draw_overlay(self, surf, lx, rx, ey, my, cx) -> None:
        ot  = self._overlay_t
        h   = self.height
        w   = self.width

        if   self._state == EyeState.BLUSHING:
            _ov_blush(surf, lx, rx, ey)
        elif self._state == EyeState.EXCITED:
            _ov_excited_stars(surf, lx, rx, ey, ot)
        elif self._state == EyeState.SICK_COLD:
            _ov_sick(surf, cx, h//2, self._sick_t, w, h)
        elif self._state == EyeState.BATTERY_FULL:
            _ov_battery(surf, w, h, True,  self._battery_t)
        elif self._state == EyeState.BATTERY_LOW:
            _ov_battery(surf, w, h, False, self._battery_t)
        elif self._state == EyeState.GREET_HELLO:
            _ov_speech_bubble(surf, cx, ey - 10, "hello  !", ot)
        elif self._state == EyeState.WEATHER_HUD:
            _ov_weather(surf, cx, h - 58, ot,
                        self._weather_city, self._weather_temp,
                        self._weather_condition, self._weather_desc)
        elif self._state == EyeState.TELEMETRY_HUD:
            _ov_telemetry(surf, cx, h - 58, ot,
                          self._telemetry_cpu, self._telemetry_ram,
                          self._telemetry_temp, self._telemetry_battery,
                          self._telemetry_charging, self._telemetry_alert)
        elif self._state == EyeState.MUSIC_MODE:
            _ov_music(surf, cx, h - 42, ot, self._music_title)
        elif self._state == EyeState.MEAL_BURGER:
            _ov_burger(surf, cx, h - 38, ot)
        elif self._state == EyeState.MEAL_NOODLES:
            _ov_noodles(surf, cx, h - 35, ot)
        elif self._state == EyeState.INTERACTION_KISS_HUG:
            _ov_kiss_hearts(surf, cx, ey, ot)
        elif self._state == EyeState.INTERACTION_RPS:
            _ov_rps(surf, cx, h//2 + 8,
                    self._rps_choice, self._rps_settled, self._rps_t)
        elif self._state == EyeState.BIRTHDAY_PARTY:
            _ov_party_hat(surf, cx, ey, ot)
        elif self._state == EyeState.FESTIVAL_MODAK:
            _ov_festival(surf, cx, ey, my, ot, "MODAK", "Ganpati Bappa Morya! 🥟", w, h, lx, rx)
        elif self._state == EyeState.FESTIVAL_DIWALI:
            _ov_festival(surf, cx, ey, my, ot, "DIWALI", "Happy Diwali! 🪔", w, h, lx, rx)
        elif self._state == EyeState.FESTIVAL_HOLI:
            _ov_festival(surf, cx, ey, my, ot, "HOLI", "Happy Holi! 🎨", w, h, lx, rx)
        elif self._state == EyeState.FESTIVAL_SANTA:
            _ov_festival(surf, cx, ey, my, ot, "SANTA", "Merry Christmas! 🎄", w, h, lx, rx)
        elif self._state == EyeState.FESTIVAL_TRICOLOR:
            _ov_festival(surf, cx, ey, my, ot, "TRICOLOR", "Jai Hind! 🇮🇳", w, h, lx, rx)
        elif self._state == EyeState.FESTIVAL_KITE:
            _ov_festival(surf, cx, ey, my, ot, "KITE", "Happy Makar Sankranti! 🪁", w, h, lx, rx)

        # Draw dynamic seasonal / festival overlay if active
        if self._festival_theme and self._festival_theme != "NONE":
            _ov_festival(surf, cx, ey, my, ot, self._festival_theme, self._festival_title, w, h, lx, rx)

    def _draw_zzz(self, surf, x: int, y: int) -> None:
        font = getattr(self, "font_zzz", None) or _get_font("consolas", 19, bold=True)
        if not font:
            return
        letters  = ["z", "Z", "Z"]
        offsets  = [(0, 0), (13, -18), (28, -36)]
        base_a   = int(90 + 70 * math.sin(self._zzz_t * 1.4))
        float_y  = int(self._zzz_t * 4.5) % 32
        for i, (letter, (ox, oy)) in enumerate(zip(letters, offsets)):
            s = font.render(letter, True, C_ZZZ)
            s.set_alpha(max(30, base_a - i * 25))
            surf.blit(s, (x + ox, y + oy - float_y))


# ---------------------------------------------------------------------------
# EyeAnimatorThread — non-blocking daemon thread  (interface UNCHANGED)
# ---------------------------------------------------------------------------
class EyeAnimatorThread:
    """
    Runs EyeAnimator inside a Pygame daemon thread.

    Usage::

        eye = EyeAnimatorThread()
        eye.start()
        eye.set_state(EyeState.LISTENING)
        eye.stop()
    """

    def __init__(self,
                 title:  str = "Morphix — OLED Eyes",
                 width:  int = EyeAnimator.WINDOW_W,
                 height: int = EyeAnimator.WINDOW_H) -> None:
        self.title  = title
        self.width  = width
        self.height = height
        self._animator: Optional[EyeAnimator]      = None
        self._thread:   Optional[threading.Thread] = None
        self._stop_evt  = threading.Event()
        self._ready_evt = threading.Event()

    def set_state(self, state: EyeState) -> None:
        if self._animator:
            self._animator.set_state(state)

    def set_festival_theme(self, theme: Any = "NONE", title: str = "") -> None:
        if self._animator:
            self._animator.set_festival_theme(theme, title)

    def set_weather_data(self, city: str = "Auto", temp: float = 25.0,
                         condition: str = "sunny", desc: str = "Clear sky") -> None:
        if self._animator:
            self._animator.set_weather_data(city, temp, condition, desc)

    def set_music_data(self, title: str = "Music", is_playing: bool = True) -> None:
        if self._animator:
            self._animator.set_music_data(title, is_playing)

    def set_telemetry_data(self, cpu: float = 15.0, ram: float = 40.0,
                           temp: float = 42.0, battery: int = 100,
                           is_charging: bool = False, is_alert: bool = False) -> None:
        if self._animator:
            self._animator.set_telemetry_data(cpu, ram, temp, battery, is_charging, is_alert)

    def trigger_blink(self) -> None:
        if self._animator:
            self._animator.trigger_blink()

    def get_state(self) -> Optional[EyeState]:
        return self._animator.get_state() if self._animator else None

    def start(self, wait_ready: bool = True, timeout: float = 5.0) -> bool:
        self._stop_evt.clear()
        self._ready_evt.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="MorphixEyeDisplay"
        )
        self._thread.start()
        if wait_ready:
            ok = self._ready_evt.wait(timeout=timeout)
            if not ok:
                logger.warning("EyeAnimatorThread not ready within %.1fs", timeout)
            return ok
        return True

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        try:
            import pygame
        except ImportError:
            logger.error("pygame not installed — pip install pygame")
            self._ready_evt.set()
            return
        try:
            pygame.init()
            pygame.display.set_caption(self.title)
            screen = pygame.display.set_mode((self.width, self.height))
            clock  = pygame.time.Clock()
            self._animator = EyeAnimator(self.width, self.height)
            logger.info("EyeAnimatorThread: window ready (%dx%d)",
                        self.width, self.height)
            self._ready_evt.set()
            prev = time.time()
            while not self._stop_evt.is_set():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._stop_evt.set()
                now  = time.time()
                dt   = now - prev
                prev = now
                self._animator.update(dt)
                self._animator.draw(screen)
                pygame.display.flip()
                clock.tick(EyeAnimator.FPS)
        except Exception as exc:
            logger.exception("EyeAnimatorThread crashed: %s", exc)
        finally:
            try:
                import pygame as _pg
                _pg.quit()
            except Exception:
                pass
            self._ready_evt.set()
            logger.info("EyeAnimatorThread: exited.")
