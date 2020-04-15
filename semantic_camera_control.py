#!/usr/bin/env python
import glob
import os
import sys
import numpy as np

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from KeyboardControl import KeyboardControl
from LaneDetection import LaneDetection
import argparse
import logging
import random
import pygame
import time
import cv2

from smv2_drive import DriveController


class CarAgent(object):
    _isMotorConnected = False
    _positionFeedback = 0
    _feedbackAngle = 0
    _motorSteering = 0

    def __init__(self):
        self.image_height = 480
        self.image_width = 640

        self.camera_surface = None
        self.semantic_surface = None

        pygame.init()

        # Original
        self.pixels = {0: [0, 0, 0],        # None
                       1: [128, 64, 0],     # Buildings
                       2: [128, 0, 64],     # Fences
                       3: [128, 256, 64],   # Other
                       4: [128, 64, 256],   # Pedestrians
                       5: [64, 192, 256],   # Poles
                       6: [192, 256, 256],  # RoadLines
                       7: [192, 64, 128],   # Roads
                       8: [64, 256, 128],   # Sidewalks
                       9: [0, 128, 192],    # Vegetation
                       10: [0, 256, 64],    # Vehicles
                       11: [256, 128, 64],  # Walls
                       12: [128, 192, 64]   # TrafficSigns
                       }

        self.actor_list = []  # List of actors spawned in the environment
        # Set basic logging config
        logging.basicConfig(
            format='%(levelname)s: %(message)s', level=logging.INFO)

        # Create client to carla server running in localhost and port 2000. Timeout of 2 seconds
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)

        self._lane_detection = LaneDetection()
        self._display = pygame.display.set_mode(
            (self.image_width*2, self.image_height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        ###########################################################

        self._controller = DriveController("/dev/ttyUSB0", 1, False)

        ##### Register Callbacks #####
        self._controller.logCallback = self._logCallback
        self._controller.errorCallback = self._errorCallback
        self._controller.readingCallback = self._readingCallback
        self._controller.connectedCallback = self._connectedCallback

        # Connect and start
        self._controller.connect()

        print("WAITING TO CONNECT......")
        while True:
            time.sleep(0.2)
            if self._isMotorConnected:
                print("Connected !")
                break

        # Motor is connected

        # # turn motor to home
        self._controller.setAddedConstantTorque(350)
        self._goToAngle(0, 1000)

        self._controller.setAddedConstantTorque(150)

        while True:
            time.sleep(0.5)  # keep this high to debounce
            print(self._feedbackAngle)
            if abs(self._feedbackAngle) < 20:
                print("Motor to Zero!")
                break

        time.sleep(1)

        ###############################################

        self.world = self.client.get_world()  # Get the current world from the server
        # Get the blueprint library
        self.blueprint_library = self.world.get_blueprint_library()

        car_blueprint = self.blueprint_library.filter('vehicle.tesla.model3')[
            0]  # Get the blueprint for Tesla Model 3
        logging.info('Obtained blueprint : ' + str(car_blueprint))

        self.world_map = self.world.get_map()  # Get the current map in the environment
        # Get the list of possible spawn points for the map
        spawn_points = self.world_map.get_spawn_points()
        # Get a random spawn point from the list of spawn points
        spawn_point = random.choice(spawn_points)

        logging.info('Spawn point : ' + str(spawn_point))

        # Spawn the car at the spawn location
        self.car = self.world.spawn_actor(car_blueprint, spawn_point)
        logging.info('Spawned car : ' + str(self.car))
        self.world.player = self.car
        self.actor_list.append(self.car)  # Add car to the list of actors

        # Apply a controller for the car
        # self.car.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))

        # Creating a blueprint for the camera and changing the size of the image captured by the camera
        camera_blueprint = self.blueprint_library.find('sensor.camera.rgb')
        camera_blueprint.set_attribute(
            'image_size_x', str(self.image_width))
        camera_blueprint.set_attribute(
            'image_size_y', str(self.image_height))
        # Change the field of view of the camera
        camera_blueprint.set_attribute('fov', '110')

        logging.info('Camera blueprint : ' + str(camera_blueprint))

        # Set the spawn point of the camera relative to the car
        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
        logging.info('Spawn location for camera : ' + str(spawn_point))

        # Attach the camera sensor to the car
        camera = self.world.spawn_actor(
            camera_blueprint, spawn_point, self.car)
        logging.info('Camera actor : ' + str(camera))

        self.actor_list.append(camera)  # Add the camera to the actor list

        # Process the images coming from the camera, images are instances of carla.Image
        camera.listen(lambda image: self.process_image(image))

        # Creating blueprint for semantic segmentation camera
        semantic_camera_blueprint = self.blueprint_library.find(
            'sensor.camera.semantic_segmentation')
        semantic_camera_blueprint.set_attribute(
            'image_size_x', str(self.image_width))
        semantic_camera_blueprint.set_attribute(
            'image_size_y', str(self.image_height))
        semantic_camera_blueprint.set_attribute('fov', '150')

        logging.info('Semantic camera blueprint : ' +
                     str(semantic_camera_blueprint))

        # Set spawn point for semantic camera relative to car
        spawn_point = carla.Transform(carla.Location(x=2.5, z=3.0))
        logging.info('Spawn location for semantic camera : ' +
                     str(spawn_point))

        # Attach the semantic camera sensor to the car
        semantic_camera = self.world.spawn_actor(
            semantic_camera_blueprint, spawn_point, self.car)
        logging.info('Semantic camera actor : ' + str(semantic_camera))

        # Add the camera to the actor list
        self.actor_list.append(semantic_camera)

        # Process the images coming from the camera, images are instances of carla.Image
        semantic_camera.listen(
            lambda image: self.process_semantic_image(image))

        self.front_camera_image = None
        self.semantic_camera_image = None

    def run(self):
        controller = KeyboardControl(self.world, False)

        clock = pygame.time.Clock()
        while not controller.parse_events(self.client, self.world, clock):
            clock.tick_busy_loop(60)
            pygame.display.flip()

            self.world.wait_for_tick()

            if self.camera_surface is not None:
                self._display.blit(self.camera_surface, (0, 0))

            if self.semantic_surface is not None:
                self._display.blit(self.semantic_surface,
                                   (self.image_width, 0))

            if self.semantic_camera_image is not None:
                cv2.imshow("V:", self.semantic_camera_image)
                cv2.waitKey(1)

            print(self._motorSteering * -20)
            self._goToAngle(self._motorSteering * -20, 1000, 3, 0.05)

    def destroy(self):
        for actor in self.actor_list:
            logging.info('Destroying actor : ' + str(actor))
            actor.destroy()  # Delete all the actors before shutting down
        pygame.quit()

        ######################
        # nullptr ALL CALLBACKS before calling desctructor
        # Very very important to mitigate deadlock while exiting
        ######################
        self._controller.logCallback = None
        self._controller.errorCallback = None
        self._controller.readingCallback = None
        self._controller.connectedCallback = None

        print('All cleaned up... Shutting down....')

    def process_image(self, image):
        i = np.array(image.raw_data)  # Raw data from the image
        i = i.reshape((self.image_height, self.image_width, 4))
        i = i[:, :, :3]

        self.camera_surface = pygame.surfarray.make_surface(i.swapaxes(0, 1))
        self.front_camera_image = i

    def process_semantic_image(self, image):
        i = np.array(image.raw_data)
        i = i.reshape((self.image_height, self.image_width, 4))
        labels = np.copy(i[:, :, :3])

        for key in self.pixels.keys():
            value = self.pixels[key]
            labels[labels[:, :, 2] == key] = value

        self.semantic_surface = pygame.surfarray.make_surface(
            labels.swapaxes(0, 1))

        labels = np.copy(i[:, :, :3])
        labels[labels[:, :, 2] == 7] = [255, 255, 255]

        ###################################################

        # 1, Undistort image
        # Camera distortion is ignored

        img = np.asarray(labels)

        # 2,Apply Perspective transform
        warped = self._lane_detection.perspective_transform(img)

        # # 3,Apply Lane edges detection
        # binary = self._lane_detection.lane_edge_detection(warped)

        # # 4,Fit polynomial equation, calculate curvature and offset
        # lane_image = self._lane_detection.fit_and_cal_curvature_offset(binary)

        # # 5,Draw final image
        # self.result_image = self._lane_detection.unwarp_found_region(img, lane_image)

        ###################################################

        # Call Canny Edge Detection here.
        cannyed_image = cv2.Canny(warped, 100, 200)

        lines = cv2.HoughLinesP(
            cannyed_image,
            rho=4,
            theta=np.pi / 180,
            threshold=100,
            lines=np.array([]),
            minLineLength=40,
            maxLineGap=5
        )

        if lines is not None:
            angle = 0.0
            count = 0
            for line in lines:
                for x1, y1, x2, y2 in line:
                    angle += np.arctan2(y2 - y1, x2 - x1)
                    count += 1

                    cv2.line(warped, (x1, y1), (x2, y2),
                             color=[0, 0, 255], thickness=3)

            heading_slope = angle / count

            if abs(heading_slope) < 1:
                heading_slope = 0

            self._motorSteering = heading_slope

        # Uncomment this to see the laterial controller image
        # self.semantic_camera_image = warped

    def _goToAngle(self, pos, max_torque=2000, Kp=1.6, Kd=0.1):
        self._controller.setAbsoluteSetpoint(
            int((pos/360)*10000), max_torque, Kp, Kd)

    ################# Event Callbacks #################

    def _logCallback(self, obj):
        # logType
        # message
        print("LOG>", obj.message)
        pass

    def _connectedCallback(self, obj):
        # isConnected
        print("CON>", obj.isConnected)
        self._isMotorConnected = obj.isConnected

    def _errorCallback(self, obj):
        # bool trackingError,
        # bool driveFault
        print("Error>", obj.trackingError, obj.driveFault)

    def _readingCallback(self, obj):
        # int posSetpoint
        # int posFeedback
        # int torqueSetpoint
        # print("Reading >", obj.posSetpoint,
        #       obj.posFeedback, obj.torqueSetpoint)

        self._positionFeedback = obj.posFeedback
        self._feedbackAngle = (obj.posFeedback / 10000)*360
        self._feedbackSteer = (self._feedbackAngle / 360)*2

        # print("Reading>", self._positionFeedback, self._feedbackAngle)


if __name__ == "__main__":
    agent = CarAgent()
    try:
        agent.run()
    except KeyboardInterrupt:
        print('\n Keyboard Interruption... Program terminating....')
    finally:
        agent.destroy()
        print('\n Done')
