import pygame as pg
import pymunk # simulates in C -> fast 
import numpy as np
import skimage.measure as measure # for 2d max pooling (pip install scikit-image)
import random

# constants
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (3, 186, 252)
LIGHT_GREY = (70, 84, 105)
RED = (252, 3, 65)

class Person():
    def __init__(self, world, pathfinder, x_init, y_init, collision_radius=10):
        # pathfinder for following the shortest path to a given goal (e.g. a building)
        self.pf = pathfinder


        # physical particle (circle) object initialization attributes
        self.x = x_init
        self.y = y_init
        self.collision_radius = collision_radius
        self.body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        self.body.position = x_init, y_init
        
        # initial velocity
        self.body.velocity = random.uniform(-100, 100), random.uniform(-100, 100)
        
        self.shape = pymunk.Circle(self.body, self.collision_radius)
        self.shape.density = 1
        self.shape.elasticity = 1
        
        # setup attributes
        self.infected = False
        # TODO:
        # - subject
        # - goal_location (drawn from random distribuition, with probabilities dependend on subject)
        # -> i.e. physicists are more likely to go to the physics building
        # - age (drawn from random distribuition)
        self.target_building = random.randint(0, 4) # [inklusive der 2. Zahl!!]; target_building = INDEX des buildings in pf.targets
        
        # add the person to the simulation
        world.add(self.body, self.shape)
    
    def infect(self, n_people):
        self.shape.density = 0.9
        self.infected = True
        
    def update_velocity(self, timestep):
        # hyperparameters
        velocity_multiplier = 30
        vel_update_rate = 0.01 # how much of the new velocity gets injected
        
        # make dependent on current velocity (self.body.velocity) and planned path
        x, y = self.body.position
                              # XXXround(x, -1) / 10
        # x = round(x, -1) / 10 # round to nearest 10 and normalize to [0, 80]
        # y = round(y, -1) / 10 # round to nearest 10 and normalize to [0, 80]
        discrete_position = (int(x), int(y))
        x_velocity, y_velocity = self.pf.get_direction(discrete_position, target_building=self.target_building)
        
        # scale the velocity by the velocity multiplier
        x_velocity = velocity_multiplier * x_velocity
        y_velocity = velocity_multiplier * y_velocity
        
        old_velocity = self.body.velocity
        additive_x_noise = 2 * random.random() - 1 # [-1 to 1], mean: 0
        additive_y_noise = 2 * random.random() - 1 # [-1 to 1], mean: 0
        new_x_velocity = (1-vel_update_rate) * old_velocity[0] + vel_update_rate * x_velocity + additive_x_noise
        new_y_velocity = (1-vel_update_rate) * old_velocity[1] + vel_update_rate * y_velocity + additive_y_noise
        
        self.body.velocity = (new_x_velocity, new_y_velocity)
    
    def draw(self, screen):
        x, y = self.body.position
        discrete_position = (int(x), int(y))
        color = RED if self.infected else BLUE
        pg.draw.circle(screen, color, discrete_position, self.collision_radius)


class Wall():
    def __init__(self, world, start_pos, end_pos, thickness=3):
        """
        Initializes a wall object.
        For our simulation, we added the constraint that walls have
        to be symmetrical to either the x-axis or y-axis (no arbitrary lines).
        This allows the the pathfinding setup for people to be easier.
        Walls also cannot be just a dot (both x-values and both y-values are the same).
        """
        # ensure that wall is not a dot
        if (start_pos[0] == end_pos[0]) and (start_pos[1] == end_pos[1]):
            raise Exception("Value Error: Wall cannot be a dot (make it longer along one dimension).")
        
        # ensure that the wall is symmetrical to either the x-axis or the y-axis   
        if (start_pos[0] != end_pos[0]) and (start_pos[1] != end_pos[1]):
            raise Exception("Value Error: Wall's position values should match along one dimension.")
        
        # ensure that the thickness is an odd number so that it can be drawn appropriately
        if thickness % 2 == 0: # thickness is even
            raise Exception("Value Error: Wall's thickness should be an odd number.")
        
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.thickness = thickness
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC) # static body
        self.shape = pymunk.Segment(self.body, start_pos, end_pos, radius=thickness) # people might glitch through if not big enough
        self.shape.elasticity = 1
        world.add(self.body, self.shape)
    
    def get_pixels(self, use_buffer_px=True):
        """ Returns all pixels (coordinates) between the start and end of the wall 
        as (x,y) tuples. This includes the pixels to the sides of the straight line that 
        stem from the wall's thickness. 
        """
        # calculate how many pixels per side of the straight line (between start and endpoint)
        # belong to the wall
        # e.g. wall is 5 pixels thick -> 2px left of line + 1px middle + 2px right of line
        extra_pixels_per_side = self.thickness // 2
        
        # if we use a buffer of 1px, the wall will be 1px thicker in each direction
        # to avoid particles getting stuck on walls while following their path
        if use_buffer_px:
            extra_pixels_per_side += 1
        
        # determine all pixels of the world that the wall occupies
        wall_pixels = []
        if self.start_pos[0] == self.end_pos[0]:
            # line is up-down (parallel to y-axis)
            smaller_y = min(self.start_pos[1], self.end_pos[1])
            larger_y = max(self.start_pos[1], self.end_pos[1])            
            for y in range(smaller_y-extra_pixels_per_side, larger_y+extra_pixels_per_side+1):
                # add all of the wall's pixels to the wall_pixels list
                # (with the according thickness)
                for x in range(self.start_pos[0] - extra_pixels_per_side, 
                               self.start_pos[0] + extra_pixels_per_side + 1):                
                    wall_pixels.append((x, y))
        else:
            # line is left-right (parallel to x-axis)
            smaller_x = min(self.start_pos[0], self.end_pos[0])
            larger_x = max(self.start_pos[0], self.end_pos[0])
            for x in range(smaller_x-extra_pixels_per_side, larger_x+extra_pixels_per_side+1):
                # add all of the wall's pixels to the wall_pixels list
                # (with the according thickness)
                for y in range(self.start_pos[1] - extra_pixels_per_side, 
                               self.start_pos[1] + extra_pixels_per_side + 1):                
                    wall_pixels.append((x, y))   
        return wall_pixels
    
    def draw(self, screen):
        pg.draw.line(screen, RED, self.start_pos, self.end_pos, self.thickness)


class Train():
    def __init__(self, world, start_pos, wall_thickness=3):
        # state attributes
        self.door_is_open = False
        self.moving = True
        self.stopped_at_station = False
        
        # physical object initialization attributes
        self.start_pos = start_pos
        self.wall_thickness = wall_thickness
        
        # create the physical body object
        self.body = pymunk.Body(mass=100, body_type=pymunk.Body.KINEMATIC)
        
        # add segments (walls) to the physical body
        x, y = start_pos 
        self.wall1 = pymunk.Segment(self.body, (x, y), (x+20, y), radius=self.wall_thickness) # top-right
        self.wall2 = pymunk.Segment(self.body, (x, y), (x, y+80), radius=self.wall_thickness) # top-down(left)
        self.wall3 = pymunk.Segment(self.body, (x, y+80), (x+20, y+80), radius=self.wall_thickness) # bot-right
        self.door = pymunk.Segment(self.body, (x+20, y), (x+20, y+80), radius=self.wall_thickness) # top-down(right)
        self.segments = [self.wall1, self.wall2, self.wall3, self.door]
        
        # set the elasticity (bouncyness of other objects when they collide with the train)
        for segment in self.segments:
            segment.elasticity = 0.5
        
        # set the initial position and velocity of the physical body
        self.body.position = x, y
        self.body.velocity = (-1.1, 30)
        
        # add the train object to the simulation
        world.add(self.body)
        for segment in self.segments:
            world.add(segment)
        
    def update_state(self, world, timestep):
        """
        The movement/position of the train depends on current timestep and follows a loop.
        One cycle of the loop contains the following events (t is the first timestep of a cycle):
        t+0: train drives from the top of the map towards the bottom of the map.
        t+9k: train stops at the trainstation (velocity is set to 0) and opens it's door.
        t+13k: train closes it's door and resumes moving.
        t+36k: train respawns at the top of the map and the next cycle starts.
        """
        # t+9k: stop train at trainstation and open door
        if (timestep % 36_000) > 50 and (timestep % 9_000) <= 50 \
                    and not self.stopped_at_station and self.moving:
            self.moving = False
            self.stopped_at_station = True
            self.open_door(world)
            self.body.velocity = (0, 0) # stop moving
        
        # t+13k: resume train and close door
        if (timestep % 9_000) > 4000 and (timestep % 9_000) <= 4050 \
                    and self.stopped_at_station and not self.moving:
            self.moving = True
            self.close_door(world)
            self.body.velocity = (-1.1, 30) # resume moving
        
        # t+36k: respawn train at top
        if (timestep % 36_000) <= 50 and self.stopped_at_station and self.moving:
            self.stopped_at_station = False # reset stopped_at_station variable to False
            self.body.position = (70, 5) # respawn train at top
    
    def _get_door_coordinates(self) -> int:
        """ Returns the current discrete position of the train. """
        x_a, y_a = self.door.a
        x_a, y_a = int(x_a), int(y_a)
        x_b, y_b = self.door.b
        x_b, y_b = int(x_b), int(y_b)
        return x_a, y_a, x_b, y_b
        
    def close_door(self, world):
        """ Moves the train's right wall down to simulate closing the door. """
        x_a, x_b, y_a, y_b = self._get_door_coordinates()
        self.door.unsafe_set_endpoints(a=(x_a, y_a), b=(x_b, y_b+40))
        self.door_is_open = False
    
    def open_door(self, world):
        """ Moves the train's right wall up to simulate opening the door. """
        x_a, x_b, y_a, y_b = self._get_door_coordinates()        
        self.door.unsafe_set_endpoints(a=(x_a, y_a), b=(x_b, y_b-40))
        self.door_is_open = True
    
    def draw(self, screen):
        """ Draws a train image on the screen (in the train's current position). """
        # get the train's position
        x_float, y_float = self.body.position
        x, y = int(x_float), int(y_float)
        
        # load and display the train image
        train_img = pg.image.load('images/train_transparent.png')
        train_img = pg.transform.scale(train_img, (20, 80))
        screen.blit(train_img, (x+67, y))