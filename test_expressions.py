"""
Morphix Desktop Companion - Expression & Eye Animation Viewer
Run this script to see Morphix's expressions live in a Pygame window!
Press SPACE or RIGHT ARROW to jump to next expression, or ESC to quit.
"""

import sys
import os
import time

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath("."))

try:
    import pygame
except ImportError:
    print("\n[!] Pygame is not installed. Please run: pip install pygame\n")
    sys.exit(1)

from src.display.eye_animator import EyeAnimator, EyeState

def main():
    pygame.init()
    width, height = 480, 320
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Morphix - Expression & Eye Preview")
    clock = pygame.time.Clock()

    animator = EyeAnimator(width, height)
    
    # List of all available states
    states = list(EyeState)
    current_idx = 0
    animator.set_state(states[current_idx])

    print("\n=======================================================")
    print("         MORPHIX EXPRESSION VISUALIZER RUNNING          ")
    print("=======================================================")
    print(" Controls:")
    print("   [SPACE] or [RIGHT ARROW] -> Next Expression")
    print("   [LEFT ARROW]             -> Previous Expression")
    print("   [B]                      -> Trigger Blink")
    print("   [ESC] or Close Window    -> Exit")
    print("=======================================================\n")
    print(f"Current Expression [1/{len(states)}]: {states[current_idx].name}")

    running = True
    last_auto_switch = time.time()
    auto_cycle_interval = 3.0  # seconds per expression if untouched
    auto_cycle = True
    prev_time = time.time()

    font = pygame.font.SysFont("consolas", 18, bold=True)

    while running:
        dt = time.time() - prev_time
        prev_time = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_SPACE, pygame.K_RIGHT):
                    current_idx = (current_idx + 1) % len(states)
                    animator.set_state(states[current_idx])
                    auto_cycle = False
                    print(f"[{current_idx+1}/{len(states)}] -> {states[current_idx].name}")
                elif event.key == pygame.K_LEFT:
                    current_idx = (current_idx - 1) % len(states)
                    animator.set_state(states[current_idx])
                    auto_cycle = False
                    print(f"[{current_idx+1}/{len(states)}] -> {states[current_idx].name}")
                elif event.key == pygame.K_b:
                    animator.trigger_blink()
                elif event.key == pygame.K_a:
                    auto_cycle = not auto_cycle
                    print(f"Auto cycle: {'ON' if auto_cycle else 'OFF'}")

        # Auto-switch expressions every few seconds
        if auto_cycle and (time.time() - last_auto_switch > auto_cycle_interval):
            current_idx = (current_idx + 1) % len(states)
            animator.set_state(states[current_idx])
            last_auto_switch = time.time()
            print(f"[{current_idx+1}/{len(states)}] -> {states[current_idx].name}")

        animator.update(dt)
        animator.draw(screen)

        # Draw current state name overlay at the top
        state_label = f"{current_idx+1}/{len(states)}: {states[current_idx].name}"
        text_surf = font.render(state_label, True, (0, 255, 200))
        screen.blit(text_surf, (15, 12))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    print("\nPreview closed. Goodbye!\n")

if __name__ == "__main__":
    main()
