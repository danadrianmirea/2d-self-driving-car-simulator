import sys
import os
import math
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

# Pygame initialization
pygame.init()
pygame.display.set_caption("SELF DRIVING CAR SIMULATOR")
WINDOW_SIZE = 1280, 720
SCREEN = pygame.display.set_mode(WINDOW_SIZE)
CAR_SIZE = 30, 51
CAR_CENTER = 195, 290
DELTA_DISTANCE = 10
DELTA_ANGLE = 5
TRACK_NAME = "track3"
WHITE_COLOR = (255, 255, 255, 255)
TRACK = pygame.image.load(TRACK_NAME + ".png").convert_alpha()
FONT = pygame.font.SysFont("bahnschrift", 25)
CLOCK = pygame.time.Clock()

# PyTorch model for the car's neural network
class CarNN(nn.Module):
    def __init__(self):
        super(CarNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # Three possible actions: straight, left, right
        )

    def forward(self, x):
        return self.model(x)


def translate_point(point, angle, distance):
    radians = math.radians(angle)
    return int(point[0] + distance * math.cos(radians)), \
           int(point[1] + distance * math.sin(radians))


class Car:
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
        rotated_car = pygame.transform.rotate(self.car, self.angle)
        rect = rotated_car.get_rect(center=self.car_center)
        SCREEN.blit(rotated_car, rect.topleft)

    def crash_check(self):
        for corner in self.corners:
            if TRACK.get_at(corner) == WHITE_COLOR:
                return True
        return False

    def update_sensor_data(self):
        angles = [360 - self.angle, 90 - self.angle, 180 - self.angle]
        angles = [math.radians(i) for i in angles]
        edge_points = []
        edge_distances = []
        for angle in angles:
            distance = 0
            edge_x, edge_y = self.car_center
            while TRACK.get_at((edge_x, edge_y)) != WHITE_COLOR:
                edge_x = int(self.car_center[0] + distance * math.cos(angle))
                edge_y = int(self.car_center[1] + distance * math.sin(angle))
                distance += 1
            edge_points.append((edge_x, edge_y))
            edge_distances.append(distance)
        self.edge_points = edge_points
        self.edge_distances = edge_distances

    def display_edge_points(self):
        for point in self.edge_points:
            pygame.draw.line(SCREEN, (0, 255, 0), self.car_center, point)
            pygame.draw.circle(SCREEN, (0, 255, 0), point, 5)

    def update_position(self):
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


def train(genomes, model, optimizer, device):
    cars = [Car() for _ in range(len(genomes))]
    criterion = nn.MSELoss()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.blit(TRACK, (0, 0))
        running_cars = 0

        for i, car in enumerate(cars):
            if not car.crashed:
                running_cars += 1
                # Prepare input data
                inputs = torch.tensor(car.edge_distances, dtype=torch.float32).unsqueeze(0).to(device)
                output = model(inputs)
                choice = torch.argmax(output).item()

                if choice == 0:
                    car.angle += DELTA_ANGLE
                elif choice == 1:
                    car.angle -= DELTA_ANGLE

                car.update_position()
                car.display_car()
                car.crashed = car.crash_check()
                car.update_sensor_data()

                # Reward the model
                target = torch.zeros(3).to(device)
                target[choice] = 1.0  # Reward the chosen action
                loss = criterion(output, target.unsqueeze(0))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if running_cars == 0:
            running = False

        pygame.display.update()
        CLOCK.tick(30)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CarNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for generation in range(100):
        print(f"Generation {generation}")
        train(range(10), model, optimizer, device)
