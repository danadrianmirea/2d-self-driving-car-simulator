"""
Self Driving Car Simulator using PyGame module and NEAT Algorithm
"""
import sys
import os
import math
import pickle
import pygame
import neat

# Pygame initialization
pygame.init()
pygame.display.set_caption("SELF DRIVING CAR SIMULATOR")
WINDOW_SIZE = 1280, 720
SCREEN = pygame.display.set_mode(WINDOW_SIZE)
CAR_SIZE = 30, 51
CAR_CENTER = 195, 290
DELTA_DISTANCE = 10
DELTA_ANGLE = 5
TRACK_NAME="track3"
WHITE_COLOR = (255, 255, 255, 255)
TRACK = pygame.image.load(TRACK_NAME+'.png').convert_alpha()
TRACK_COPY = TRACK.copy()
FONT = pygame.font.SysFont("bahnschrift", 25)
CLOCK = pygame.time.Clock()

# Checkpoint settings
CHECKPOINT_DIR = "checkpoints_"+TRACK_NAME
CHECKPOINT_PREFIX = "neat-checkpoint-"
CHECKPOINT_FREQUENCY = 10
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

GENERATION = 0  # Tracks current generation


def translate_point(point, angle, distance):
    """
    Get the new coordinates of a given point w.r.t. an angle and distance from that point.
    """
    radians = math.radians(angle)
    return int(point[0] + distance * math.cos(radians)), \
           int(point[1] + distance * math.sin(radians))


class Car:
    """
    Implementation of the self-driving car
    """
    def __init__(self):
        self.corners = []
        self.edge_points = []
        self.edge_distances = []
        self.travelled_distance = 0
        self.angle = 0
        self.car_center = CAR_CENTER
        self.car = pygame.image.load("car.png").convert_alpha()
        self.car = pygame.transform.scale(self.car, CAR_SIZE)
        self.crashed = False
        self.update_sensor_data()

    def display_car(self):
        """
        Rotate the car and display it on the screen
        """
        rotated_car = pygame.transform.rotate(self.car, self.angle)
        rect = rotated_car.get_rect(center=self.car_center)
        SCREEN.blit(rotated_car, rect.topleft)

    def crash_check(self):
        """
        Check if any corner of the car goes out of the track
        """
        for corner in self.corners:
            if TRACK.get_at(corner) == WHITE_COLOR:
                return True
        return False

    def update_sensor_data(self):
        """
        Update the points on the edge of the track and the distances between the points and the center of the car
        """
        angles = [360 - self.angle, 90 - self.angle, 180 - self.angle]
        angles = [math.radians(i) for i in angles]
        edge_points = []
        edge_distances = []
        for angle in angles:
            distance = 0
            edge_x, edge_y = self.car_center
            while TRACK_COPY.get_at((edge_x, edge_y)) != WHITE_COLOR:
                edge_x = int(self.car_center[0] + distance * math.cos(angle))
                edge_y = int(self.car_center[1] + distance * math.sin(angle))
                distance += 1
            edge_points.append((edge_x, edge_y))
            edge_distances.append(distance)
        self.edge_points = edge_points
        self.edge_distances = edge_distances

    def display_edge_points(self):
        """
        Display lines from the center of the car to the edges on the track
        """
        for point in self.edge_points:
            pygame.draw.line(SCREEN, (0, 255, 0), self.car_center, point)
            pygame.draw.circle(SCREEN, (0, 255, 0), point, 5)

    def update_position(self):
        """
        Update the new position of the car
        """
        self.car_center = translate_point(
            self.car_center, 90 - self.angle, DELTA_DISTANCE)
        self.travelled_distance += DELTA_DISTANCE
        dist = math.sqrt(CAR_SIZE[0]**2 + CAR_SIZE[1]**2) / 2
        corners = []
        corners.append(translate_point(
            self.car_center, 60 - self.angle, dist))
        corners.append(translate_point(
            self.car_center, 120 - self.angle, dist))
        corners.append(translate_point(
            self.car_center, 240 - self.angle, dist))
        corners.append(translate_point(
            self.car_center, 300 - self.angle, dist))
        self.corners = corners


def run(genomes, config):
    """
    Runs the game for a specific generation of NEAT Algorithm
    """
    global GENERATION
    GENERATION += 1
    models = []
    cars = []

    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        models.append(net)
        genome.fitness = 0
        cars.append(Car())

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        running_cars = 0

        SCREEN.blit(TRACK, (0, 0))

        for i, car in enumerate(cars):
            if not car.crashed:
                running_cars += 1
                output = models[i].activate(car.edge_distances)
                choice = output.index(max(output))
                if choice == 0:
                    car.angle += DELTA_ANGLE
                elif choice == 1:
                    car.angle -= DELTA_ANGLE
                car.update_position()
                car.display_car()
                car.crashed = car.crash_check()
                car.update_sensor_data()
                genomes[i][1].fitness += car.travelled_distance
                car.display_edge_points()

        if running_cars == 0:
            return
        msg = "Generation: {}, Running Cars: {}".format(GENERATION, running_cars)
        text = FONT.render(msg, True, (0, 0, 0))
        SCREEN.blit(text, (0, 0))
        pygame.display.update()
        CLOCK.tick(60)  # Run at 60 FPS


# Load from checkpoint if available
checkpoint_file = None
if os.path.exists(CHECKPOINT_DIR):
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith(CHECKPOINT_PREFIX)]
    if checkpoint_files:
        # Sort checkpoints numerically based on their suffix
        checkpoint_files.sort(key=lambda x: int(x.split('-')[-1]))
        checkpoint_file = os.path.join(CHECKPOINT_DIR, checkpoint_files[-1])  # Load the latest checkpoint

if checkpoint_file:
    print(f"Loading from checkpoint: {checkpoint_file}")
    population = neat.Checkpointer.restore_checkpoint(checkpoint_file)
    GENERATION = int(checkpoint_file.split('-')[-1])  # Extract generation from filename
else:
    neat_config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                     "config.txt")
    population = neat.Population(neat_config)

# Add reporters to the population
population.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
population.add_reporter(stats)
checkpointer = neat.Checkpointer(CHECKPOINT_FREQUENCY, filename_prefix=os.path.join(CHECKPOINT_DIR, CHECKPOINT_PREFIX))
population.add_reporter(checkpointer)

# Run the simulation
population.run(run, 500)
