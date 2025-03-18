import pygame
import sys
import math

class Visualizer:
    def __init__(self, width=800, height=600, margin=50, fps=30):
        pygame.init()
        self.width = width
        self.height = height
        self.margin = margin
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Planet Wars Visualization")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 14)
        self.fps = fps  # initial simulation speed

    def _scale_position(self, x, y, planets):
        xs = [p.X() for p in planets]
        ys = [p.Y() for p in planets]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        if max_x == min_x:
            max_x += 1
        if max_y == min_y:
            max_y += 1
        scaled_x = self.margin + (x - min_x) / (max_x - min_x) * (self.width - 2 * self.margin)
        scaled_y = self.margin + (y - min_y) / (max_y - min_y) * (self.height - 2 * self.margin)
        return int(scaled_x), int(scaled_y)

    def draw(self, env):
        # Fill background
        self.screen.fill((0, 0, 0))
        planets = env.pw._planets

        # Draw planets
        for planet in planets:
            x, y = self._scale_position(planet.X(), planet.Y(), planets)
            if planet.Owner() == 0:
                color = (128, 128, 128)  # neutral: gray
            elif planet.Owner() == 1:
                color = (91, 184, 227)      # agent: blue
            elif planet.Owner() == 2:
                color = (227, 91, 111)      # enemy: red
            else:
                color = (255, 255, 255)
            radius = (planet.GrowthRate() + 3) * 2
            pygame.draw.circle(self.screen, color, (x, y), radius)
            text = self.font.render(str(planet.NumShips()), True, (255, 255, 255))
            self.screen.blit(text, (x - text.get_width() // 2, y - text.get_height() // 2))
        
        # Draw fleets as small circles moving along their paths
        for fleet in env.pw._fleets:
            src_planet = env.pw._planets[fleet._source_planet]
            dst_planet = env.pw._planets[fleet._destination_planet]
            src_pos = self._scale_position(src_planet.X(), src_planet.Y(), planets)
            dst_pos = self._scale_position(dst_planet.X(), dst_planet.Y(), planets)
            if fleet.TotalTripLength() > 0:
                progress = 1 - (fleet.TurnsRemaining() / fleet.TotalTripLength())
            else:
                progress = 1
            x = int(src_pos[0] + progress * (dst_pos[0] - src_pos[0]))
            y = int(src_pos[1] + progress * (dst_pos[1] - src_pos[1]))
            if fleet.Owner() == 0:
                color = (128, 128, 128)
            elif fleet.Owner() == 1:
                color = (91, 184, 227)
            elif fleet.Owner() == 2:
                color = (227, 91, 111)
            else:
                color = (255, 255, 255)
            pygame.draw.circle(self.screen, color, (x, y), 6)
        
        # Display current turn (if available) and current simulation FPS.
        if hasattr(env, 'current_turn'):
            turn_text = self.font.render(f"Turn: {env.current_turn}", True, (255, 255, 255))
            self.screen.blit(turn_text, (10, 10))
        fps_text = self.font.render(f"Speed (FPS): {self.fps}", True, (255, 255, 255))
        self.screen.blit(fps_text, (self.width - fps_text.get_width() - 10, 10))

    def update(self):
        pygame.display.flip()
        self.clock.tick(self.fps)
        # Process events to allow speed adjustments
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # Adjust speed with UP/DOWN keys
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.fps = min(3600, self.fps + 5)
                elif event.key == pygame.K_DOWN:
                    self.fps = max(5, self.fps - 5)
                elif event.key == pygame.K_SPACE:
                    if self.fps > 0:
                        self.fps = 0
                    else:
                        self.fps = 30

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    pass
