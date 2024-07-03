#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example of automatic vehicle control from client side."""

from __future__ import print_function

import argparse
import collections
import datetime
import glob
import logging
import math
import os
import numpy.random as random
import re
import sys
import weakref
import csv

import rospy
from std_msgs.msg import Float64
from configparser import ConfigParser

if sys.version_info >= (3, 0):
    from configparser import ConfigParser
else:
    from ConfigParser import RawConfigParser as ConfigParser

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================
def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        if int_generation in [1, 2, 3]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================
class World(object):
    """ Class representing the surrounding environment """
    def __init__(self, carla_world, hud, actor_filter, mode='automatic', args=None):
        self._args = args
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = actor_filter
        self.mode = mode
        
        if mode == 'automatic':
            self._actor_generation = getattr(self._args, 'generation', 'all')
            self.recording_enabled = False
            self.recording_start = 0
        else:
            pass
        self.restart()
        self.world.on_tick(hud.on_world_tick)
    
    def restart(self):
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        
        if self.mode == 'automatic':
            blueprint_list = get_actor_blueprints(self.world, self._actor_filter, self._actor_generation)
            if not blueprint_list:
                raise ValueError("Couldn't find any blueprints with the specified filters")
            blueprint = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        else:
            blueprint = self.world.get_blueprint_library().find('vehicle.tesla.model3')

        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        if self.mode =='automatic':
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            if self.mode == 'automatic':
                self.modify_vehicle_physics(self.player)
        
        if self.mode == 'automatic' and hasattr(self, '_args') and self._args.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()
        
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)
    
    def next_weather(self, reverse=False):
        """Get next weather setting"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def modify_vehicle_physics(self, actor):
        """Modify vehicle physics if actor is a vehicle"""
        if self.mode == 'automatic':  # This method is specific to automatic control
            try:
                physics_control = actor.get_physics_control()
                physics_control.use_sweep_wheel_collision = True
                actor.apply_physics_control(physics_control)
            except Exception:
                pass
    
    def tick(self, clock):
        """Method for every tick"""
        self.hud.tick(self, clock)

    def render(self, display):
        """Render world"""
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        """Destroy sensors, specific to automatic control if used"""
        if self.mode == 'automatic':
            if hasattr(self, 'camera_manager') and self.camera_manager.sensor is not None:
                self.camera_manager.sensor.destroy()
                self.camera_manager.sensor = None
                self.camera_manager.index = None
        else:
            pass

    def destroy(self):
        """Destroys all actors"""
        if self.mode =='automatic':
            actors = [
                getattr(self, 'camera_manager', None) and self.camera_manager.sensor,
                getattr(self, 'collision_sensor', None) and self.collision_sensor.sensor,
                getattr(self, 'lane_invasion_sensor', None) and self.lane_invasion_sensor.sensor,
                getattr(self, 'gnss_sensor', None) and self.gnss_sensor.sensor,
                self.player]
            for actor in actors:
                if actor is not None:
                    actor.destroy()
        else:
            sensors = [
                self.camera_manager.sensor,
                self.collision_sensor.sensor,
                self.lane_invasion_sensor.sensor,
                self.gnss_sensor.sensor]
            for sensor in sensors:
                if sensor is not None:
                    sensor.stop()
                    sensor.destroy()
            if self.player is not None:
                self.player.destroy()
    

# ==============================================================================
# -- IntegratedControl -----------------------------------------------------------
class IntegratedControl(object):
    def __init__(self, world, start_in_autopilot):
        self.world = world
        self.mode = world.mode  # Start with the world's current mode
        self._autopilot_enabled = start_in_autopilot
        self.steering_angle = 0.0  # 자율 주행 모드에서 스티어링 각도 초기화

        # 제어 초기화
        self.initialize_control()
        self.apply_initial_steering()  # 초기 자율 주행 스티어링 적용
        self.world.player.set_autopilot(self._autopilot_enabled)  # 자율주행 모드 적용
        
        # Initialize notification
        world.hud.notification(f"Mode: {self.mode}. Press 'H' or '?' for help.", seconds=4.0)

        # Common setup for both modes
        self._setup_controls()

        # Initialize joystick and read config
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 0:
            self._joystick = pygame.joystick.Joystick(0)
            self._joystick.init()
        else:
            print("Joystick not found.")

        self._parser = ConfigParser()
        self._parser.read('wheel_config.ini')
        self._load_config()
        print("Initialized with mode:", self.mode, "Autopilot:", self._autopilot_enabled)

        # ROS 퍼블리셔 초기화
        self.steering_publisher = rospy.Publisher('/carla/steering_angle', Float64, queue_size=10)

    def apply_control(self):
        try:
            if self.mode == 'automatic':
                control = carla.VehicleControl(steer=self.steering_angle)
                self.world.player.apply_control(control)
                rospy.loginfo(f"Applying steering angle: {self.steering_angle}")
                print(f"Applying steering angle: {self.steering_angle}")
        except Exception as e:
            rospy.logerr(f"Error in apply_control: {e}")
            print(f"Error in apply_control: {e}")

    def apply_initial_steering(self):
        if self._autopilot_enabled:
            self.steering_angle = 0.0  # 예시 초기 각도
            self.apply_control()
            rospy.loginfo(f"Initial steering angle applied: {self.steering_angle}")
            print(f"Initial steering angle applied: {self.steering_angle}")

    def initialize_control(self):
        # 초기 제어 상태 설정: 방향 속도 등
        pass

    def _setup_controls(self):
        # Set up controls for carla.Vehicle
        if isinstance(self.world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self.world.player.set_autopilot(self._autopilot_enabled)
        # Additional setup for other actor types can go here

    def _load_config(self):
        # Load control config from wheel_config.ini
        self._steer_idx = int(self._parser.get('G29 Racing Wheel', 'steering_wheel'))
        self._throttle_idx = int(self._parser.get('G29 Racing Wheel', 'throttle'))
        self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
        self._reverse_idx = int(self._parser.get('G29 Racing Wheel', 'reverse'))
        self._handbrake_idx = int(self._parser.get('G29 Racing Wheel', 'handbrake'))
        self._takeover_idx = int(self._parser.get('G29 Racing Wheel', 'takeover'))

    def parse_events(self, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.JOYBUTTONDOWN:
                self.handle_joystick_button(event)
            elif event.type == pygame.KEYUP:
                self.handle_keyboard(event)

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._parse_vehicle_wheel()
                self._control.reverse = self._control.gear < 0
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())
            world.player.apply_control(self._control)
        else:
            if self.mode == 'automatic':
                self._control.steer = self.steering_angle  # 자율 주행 모드에서 ROS로 받은 스티어링 각도 적용
                print(f"Applying steering angle: {self._control.steer}")  # 적용된 스티어링 값 출력
                world.player.apply_control(self._control)

    def toggle_mode(self):
        # 모드 전환 로직
        if self.mode == 'manual':
            self.mode = 'automatic'
            self._autopilot_enabled = True
            self.apply_initial_steering()
            self.world.hud.notification('Switched to automatic mode.')
        else:
            self.mode = 'manual'
            self._autopilot_enabled = False
            self.initialize_manual_control()
            self.world.hud.notification('Switched to manual mode.')

        self.world.player.set_autopilot(self._autopilot_enabled)
        self.world.mode = self.mode  # World 객체의 모드도 업데이트
        print(f"Mode toggled to: {self.mode}, Autopilot: {self._autopilot_enabled}")

    def initialize_manual_control(self):
        # 수동 제어를 위한 초기 차량 제어 상태 설정
        if isinstance(self.world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._control.steer = 0.0  # 조향 초기화
            self._control.throttle = 0.0  # 가속 초기화
            self._control.brake = 0.0  # 브레이크 초기화
            self._control.hand_brake = False  # 핸드브레이크 비활성화
            self._control.reverse = False  # 기어를 전진 상태로 설정
            self._control.manual_gear_shift = False  # 수동 기어 변경 비활성화(필요한 경우 변경)
            self.world.player.apply_control(self._control)
        elif isinstance(self.world.player, carla.Walker):
            # 보행자 제어를 위한 초기화(필요한 경우)
            pass

    def handle_joystick_button(self, event):
        button = event.button
        print(f"Joystick Button Pressed: {event.button}")
        if button == 0:
            self.world.restart()
        elif button == 1:
            self.world.hud.toggle_info()
        elif button == 2:
            self.world.camera_manager.toggle_camera()
        elif button == 3:
            self.world.next_weather()
        elif button == 23:
            self._control.gear = 1 if self._control.reverse else -1
        elif button == self._takeover_idx:
            self.toggle_mode()  # 모드 전환 메소드 호출

    def handle_keyboard(self, event):
        key = event.key
        print(f"Keyboard Pressed: {event.key}")
        if event.key == K_BACKSPACE:
            self.world.restart()
        elif event.key == K_F1:
            self.world.hud.toggle_info()
        elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
            self.world.hud.help.toggle()
        elif event.key == K_TAB:
            self.world.camera_manager.toggle_camera()
        elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
            self.world.next_weather(reverse=True)
        elif event.key == K_c:
            self.world.next_weather()
        elif event.key == K_BACKQUOTE:
            self.world.camera_manager.next_sensor()
        elif event.key > K_0 and event.key <= K_9:
            self.world.camera_manager.set_sensor(event.key - 1 - K_0)
        elif event.key == K_r:
            self.world.camera_manager.toggle_recording()
        if isinstance(self._control, carla.VehicleControl):
            if event.key == K_q:
                self._control.gear = 1 if self._control.reverse else -1
            elif event.key == K_m:
                self._control.manual_gear_shift = not self._control.manual_gear_shift
                self._control.gear = self.world.player.get_control().gear
                self.world.hud.notification('%s Transmission' %
                                    ('Manual' if self._control.manual_gear_shift else 'Automatic'))
            elif self._control.manual_gear_shift and event.key == K_COMMA:
                self._control.gear = max(-1, self._control.gear - 1)
            elif self._control.manual_gear_shift and event.key == K_PERIOD:
                self._control.gear = self._control.gear + 1
            elif event.key == K_p:
                self._autopilot_enabled = not self._autopilot_enabled
                self.world.player.set_autopilot(self._autopilot_enabled)
                self.world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_vehicle_wheel(self):
        if self.mode == 'automatic':
            print(f"Applying steering angle: {self.steering_angle}")  # Apply the ROS received steering angle
            self._control.steer = self.steering_angle
        else:
            numAxes = self._joystick.get_numaxes()
            jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
            jsButtons = [float(self._joystick.get_button(i)) for i in range(self._joystick.get_numbuttons())]

            K1 = 1.0
            steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])

            K2 = 1.6
            throttleCmd = K2 + (2.05 * math.log10(-0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
            throttleCmd = max(0, min(throttleCmd, 1))

            brakeCmd = 1.6 + (2.05 * math.log10(-0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
            brakeCmd = max(0, min(brakeCmd, 1))

            self._control.steer = steerCmd
            self._control.brake = brakeCmd
            self._control.throttle = throttleCmd
            self._control.hand_brake = bool(jsButtons[self._handbrake_idx])

    def _parse_walker_keys(self, keys, milliseconds):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = 5.556 if pygame.key.get_mods() & KMOD_SHIFT else 2.778
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================
class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        
        # 로그 파일 설정
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_dir = os.path.expanduser("~/home/mintlab01/carla/Data/simulation")
        os.makedirs(log_dir, exist_ok=True)
        self.log_file_path = os.path.join(log_dir, f"hud_log_{current_time}.csv")
        
        with open(self.log_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Server FPS', 'Client FPS', 'Vehicle', 'Map', 'Simulation Time', 
                             'Speed', 'Heading', 'Location', 'GNSS', 'Height', 'Throttle', 'Steer', 
                             'Brake', 'Reverse', 'Hand brake', 'Manual', 'Gear', 'Collision', 
                             'Number of vehicles', 'Nearby vehicles'])

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        """HUD method for every tick"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        control = world.player.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % transform.location.z,
            '']
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -2.0, 2.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                ('Manual:', control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != world.player.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))
        
        # Save HUD data to log file
        with open(self.log_file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            row = [self.server_fps, clock.get_fps(), get_actor_display_name(world.player, truncate=20),
                   world.map.name.split('/')[-1], str(datetime.timedelta(seconds=int(self.simulation_time))),
                   3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2), transform.rotation.yaw, 
                   f'({transform.location.x:.1f}, {transform.location.y:.1f})', 
                   f'({world.gnss_sensor.lat:.6f}, {world.gnss_sensor.lon:.6f})', transform.location.z,
                   control.throttle if isinstance(control, carla.VehicleControl) else '',
                   control.steer if isinstance(control, carla.VehicleControl) else '',
                   control.brake if isinstance(control, carla.VehicleControl) else '',
                   control.reverse if isinstance(control, carla.VehicleControl) else '',
                   control.hand_brake if isinstance(control, carla.VehicleControl) else '',
                   control.manual_gear_shift if isinstance(control, carla.VehicleControl) else '',
                   control.gear if isinstance(control, carla.VehicleControl) else '',
                   max_col, len(vehicles),
                   ', '.join([f'{dist:.1f}m {get_actor_display_name(vehicle, truncate=22)}' for dist, vehicle in sorted(vehicles) if dist <= 200])]
            writer.writerow(row)

    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """Notification text"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """Render for HUD class"""
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================
class FadingText(object):
    """ Class for fading text """

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================
class HelpText(object):
    """ Helper class for text render"""

    def __init__(self, font, width, height):
        """Constructor method"""
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle on or off the render help"""
        self._render = not self._render

    def render(self, display):
        """Render help text method"""
        if self._render:
            display.blit(self.surface, self.pos)

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================
class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================
class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================
class GnssSensor(object):
    """ Class for GNSS sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================
class CameraManager(object):
    def __init__(self, parent_actor, hud, mode='manual'):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self.mode = mode

        # CameraManager 클래스의 __init__ 메서드에서 카메라 변환 설정 부분
        # 모드에 따라 다른 카메라 변환 설정
        if self.mode == 'automatic':
            bound_x = 0.5 + self._parent.bounding_box.extent.x
            bound_y = 0.5 + self._parent.bounding_box.extent.y
            bound_z = 0.5 + self._parent.bounding_box.extent.z
            attachment = carla.AttachmentType
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), attachment.Rigid),
                (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), attachment.Rigid),
                (carla.Transform(carla.Location(x=+1.9*bound_x, y=+1.0*bound_y, z=1.2*bound_z)), attachment.Rigid),
                (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y, z=4.6*bound_z), carla.Rotation(pitch=6.0)), attachment.Rigid),
                (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), attachment.Rigid),
                # Add the driver view camera transformation
                (carla.Transform(carla.Location(x=0.4, y=-0.3, z=1.1), carla.Rotation(pitch=0.0)), attachment.Rigid)  # Driver view
            ]

        else:  # 수동주행 모드의 카메라 변환 설정
            self._camera_transforms = [
                carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
                carla.Transform(carla.Location(x=1.6, z=1.7)),
                # Add the driver view camera transformation
                carla.Transform(carla.Location(x=0.4, y=-0.3, z=1.1), carla.Rotation(pitch=0.0))  # Driver view
            ]

        self.transform_index = 2  # Set to the driver view by default
        
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '50')
            item.append(bp)
        
        self.index = None
        self.set_sensor(0, notify=False)

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor based on the current mode."""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
            force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))

        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None

            if self.mode == 'automatic':
                transform = self._camera_transforms[self.transform_index][0]
                attachment_type = self._camera_transforms[self.transform_index][1]
            else:
                transform = self._camera_transforms[self.transform_index]
                attachment_type = carla.AttachmentType.Rigid

            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                transform,
                attach_to=self._parent,
                attachment_type=attachment_type)

            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))

        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=assignment-from-no-return
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================
def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    rospy.init_node('carla_steering_publisher', anonymous=True)
    steering_publisher = rospy.Publisher('/carla/steering_angle', Float64, queue_size=10)

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        display_info = pygame.display.Info()
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.FULLSCREEN)

        hud = HUD(args.width, args.height)

        world = World(client.get_world(), hud, args.filter, mode='automatic', args=args)
        controller = IntegratedControl(world, start_in_autopilot=True)

        # 초기 스티어링 각도 설정
        controller.apply_initial_steering()

        clock = pygame.time.Clock()
        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(world, clock):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

            if controller.mode == 'automatic':
                control = world.player.get_control()
                steering_publisher.publish(control.steer)
                rospy.loginfo(f"Published steering angle: {control.steer}")
                print(f"Published steering angle: {control.steer}")

    except Exception as e:
        rospy.logerr(f'Error during game loop: {e}')
        print(f'Error during game loop: {e}')
    finally:
        if world is not None:
            world.destroy()
        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================
if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser(description='CARLA Manual Control Client')
    argparser.add_argument(
        '--host', metavar='H', default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '--port', metavar='P', default=2000, type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res', metavar='WIDTHxHEIGHT', default='1920x1080',
        help='window resolution (default: 1920x1080)')
    argparser.add_argument(
        '--filter', metavar='PATTERN', default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation', metavar='G', default='2',
        help='restrict to certain actor generation (default: "2")')
    argparser.add_argument(
        '--sync', action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--loop', action='store_true',
        help='Sets a new random destination upon reaching the previous one')
    argparser.add_argument(
        '--agent', choices=['Behavior', 'Basic'], default='Behavior',
        help='select which agent to run (default: Behavior)')
    argparser.add_argument(
        '--behavior', choices=['cautious', 'normal', 'aggressive'], default='normal',
        help='Choose one of the possible agent behaviors (default: normal)')
    argparser.add_argument(
        '--autopilot', action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--debug', action='store_true',
        help='enable debug output')

    args = argparser.parse_args()

    # Parse width and height from resolution argument
    try:
        args.width, args.height = [int(x) for x in args.res.split('x')]
    except AttributeError:
        args.width, args.height = 1920, 1080

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
