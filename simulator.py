import pygame as pg
import pymunk # simulates in C -> fast 
import numpy as np
import skimage.measure as measure # for 2d max pooling (pip install scikit-image)
import random

from objects import Person, Wall, Train

# constants
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (3, 186, 252)
LIGHT_GREY = (70, 84, 105)
RED = (252, 3, 65)

class CovidSim():
    def __init__(self, n_people, infection_prob, draw_dots_for_debugging, FPS=60):
        # sim setup
        self.n_people = n_people
        self.pf = None # will be set later

        # visuals setup
        self.screen_size = 800
        self.screen = pg.display.set_mode((self.screen_size, self.screen_size))
        self.width, self.height = self.screen.get_size()
        self.clock = pg.time.Clock()
        self.FPS = FPS
        
        # some useful attributes
        self.draw_dots = draw_dots_for_debugging
        self.running = True
        
        # hyperparameters
        self.infection_prob = infection_prob
        
        # add logo and caption
        logo = pg.image.load('images/virus_logo.png')
        pg.display.set_icon(logo)
        pg.display.set_caption('COVID19-Sim')
        
        # create the simulated world
        self.world = pymunk.Space()
        
        # add screen borders as walls
        self.screen_borders = [
            Wall(world=self.world, start_pos=(0, 0), end_pos=(800, 0), thickness=1), # top-right
            Wall(world=self.world, start_pos=(0, 0), end_pos=(0, 800), thickness=1), # top-down (left)
            Wall(world=self.world, start_pos=(800, 0), end_pos=(800, 800), thickness=1), # top-down(right)
            Wall(world=self.world, start_pos=(0, 800), end_pos=(110, 800), thickness=1), # bot-right
            Wall(world=self.world, start_pos=(130, 800), end_pos=(800, 800), thickness=1)] # bot-right
        
        # create walls for buildings
        self.buildings = [
            self._create_tile(origin_pos=(630,490), tile_type='building_1'),
            self._create_tile(origin_pos=(620,450), tile_type='building_2'),
            self._create_tile(origin_pos=(700,530), tile_type='building_3'),
            self._create_tile(origin_pos=(690,430), tile_type='building_4'),
            self._create_tile(origin_pos=(700,310), tile_type='building_5'),
            self._create_tile(origin_pos=(630,310), tile_type='building_6'),
            #self._create_tile(origin_pos=(500,600), tile_type='building_7'),
            #self._create_tile(origin_pos=(500,600), tile_type='building_8'),
            #self._create_tile(origin_pos=(500,600), tile_type='building_9'),
            #self._create_tile(origin_pos=(500,600), tile_type='building_10'),
            #self._create_tile(origin_pos=(500,600), tile_type='building_11'),
            #self._create_tile(origin_pos=(500,600), tile_type='building_12'),
            #self._create_tile(origin_pos=(500,600), tile_type='building_13'),
            #self._create_tile(origin_pos=(500,600), tile_type='building_14'),
            #self._create_tile(origin_pos=(500,600), tile_type='building_14a'),
            #self._create_tile(origin_pos=(500,600), tile_type='building_15'),
            #self._create_tile(origin_pos=(500,600), tile_type='building_16'),
            #self._create_tile(origin_pos=(500,600), tile_type='building_17'),
            #self._create_tile(origin_pos=(500,600), tile_type='building_18'),
            #self._create_tile(origin_pos=(500,600), tile_type='building_19'),
            #self._create_tile(origin_pos=(500,600), tile_type='building_20'),
            #self._create_tile(origin_pos=(500,600), tile_type='building_21'),
            #self._create_tile(origin_pos=(500,600), tile_type='building_22'),
            #self._create_tile(origin_pos=(500,600), tile_type='building_23'),
            #self._create_tile(origin_pos=(500,600), tile_type='building_24'),
            #self._create_tile(origin_pos=(500,600), tile_type='building_25'),
            #self._create_tile(origin_pos=(500,600), tile_type='building_26'),
            #self._create_tile(origin_pos=(500,600), tile_type='building_27'),
            #self._create_tile(origin_pos=(500,600), tile_type='building_28'),
            #self._create_tile(origin_pos=(500,600), tile_type='building_29'),
            #self._create_tile(origin_pos=(500,600), tile_type='building_30'),
            #self._create_tile(origin_pos=(500,600), tile_type='building_31'),
        ]
    
    def collision_begin(self, arbiter, space, data):
        """ 
        This custom pre-collision-handling method handles infection spreading 
        when two persons collide.
        It also has to return True so the default PyMunk collision handler can handle 
        changes of physical attributes afterwards (e.g. updating the velocity).
        """
        involves_infected_person = False
        
        for shape in arbiter.shapes:
            # check if the collision involves an object that is not a person (e.g. a wall)
            if shape.__class__ == pymunk.shapes.Segment:
                return True
            
            # check if a person is infected
            if shape.density == 0.9:
                involves_infected_person = True
        
        if involves_infected_person:
            for shape in arbiter.shapes:
                if shape.density == 1.0: # 1.0 means not infected
                    # infection_prob: probability that infection status is shared when colliding
                    if random.random() < self.infection_prob:
                        shape.density = 0.9
        return True
    
    def run(self, seed=42):
        # setup
        random.seed(seed)

        # add people to the simulation
        self.people = [Person(world=self.world,
                              pathfinder=self.pf,
                              x_init=random.randint(0, self.screen_size),
                              y_init=random.randint(0, self.screen_size),
                              collision_radius=4,)
                       for i in range(self.n_people)]
        
    
        # define custom collision handler (collision might spread infection status)        
        self.handler = self.world.add_default_collision_handler()
        self.handler.begin = self.collision_begin   # each time two objects collide
                                                    # the custom collision_begin method is called
                                                    # for spreading the infection status 
        
        # infect one random person to start the epidemic
        random.choice(self.people).infect(self.n_people)
        
        # add a train to the simulation
        self.train = Train(world=self.world,
                           start_pos=(70, 5),
                           wall_thickness=3)
        
        # setup np-arrays for data

        while self.running:

            self.clock.tick(self.FPS)   # update pygame time
            self.world.step(1/self.FPS) # keeps rendered steps/s consistent (independent of self.FPS)
            
            # handle mouse and keyboard events
            self.events()
            
            # update the velocity of all people to navigate to their goal
            for person in self.people:
                person.update_velocity(pg.time.get_ticks())
            
            # update the train state and infections
            self.update()
            
            # render the map and all simulated objects
            if self.running:
                self.draw(self.draw_dots)
                            
            # save logs
            # evaluate later
    
    def events(self):
        for event in pg.event.get():
            
            # check if the exit button (top right "X") was pressed
            if event.type == pg.QUIT:
                pg.quit()
                self.running = False
            
            # check if a keyboard key was pressed
            if event.type == pg.KEYDOWN:
                # ESC key
                if event.key == pg.K_ESCAPE:
                    pg.quit()
                    self.running = False
    
    def update(self):
        # update train's state
        self.train.update_state(world=self.world, timestep=pg.time.get_ticks())
        
        # update infections
        for person in self.people:
            if person.shape.density == 0.9:
                person.infected = True
    
    def draw(self, draw_dots):
        # draw the golm-background image
        golm_img = pg.image.load('images/golm_test.png')
        golm_img = pg.transform.scale(golm_img, (self.screen_size, self.screen_size))
        self.screen.blit(golm_img, (0, 0))
        
        # draw train
        self.train.draw(self.screen)
        
        # draw people
        for person in self.people:
            person.draw(self.screen)
        
        # draw buildings
        for building in self.buildings:
            for wall in building:
                wall.draw(self.screen)
        
        # draw grid of dots for testing/debugging
        if draw_dots:
            for i in range(80):
                for j in range(80):
                    if i%10 == 0 and j%10 == 0:
                        pg.draw.circle(self.screen, RED, (i*10, j*10), 2) 
                    else:
                        pg.draw.circle(self.screen, LIGHT_GREY, (i*10, j*10), 2) 
        
        # update entire screen
        pg.display.flip() 
    
    
    def _create_tile(self, origin_pos, tile_type):
        """
        Takes the origin (top right) position and type of a building as input and
        returns a list of it's walls as static physical objects.
        """
        x, y = origin_pos
        
        if tile_type == 'house':
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x, y), end_pos=(x+80, y)),
                Wall(world=self.world, start_pos=(x, y), end_pos=(x, y+80)),
                Wall(world=self.world, start_pos=(x+80, y), end_pos=(x+80, y+80)),
                # create half-open wall
                Wall(world=self.world, start_pos=(x, y+80), end_pos=(x+20, y+80)),
                Wall(world=self.world, start_pos=(x+60, y+80), end_pos=(x+80, y+80))
            ]
        # RIGHT-OPEN, LONG
        if tile_type == 'building_1':
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x, y), end_pos=(x+20, y)), # top-right
                Wall(world=self.world, start_pos=(x, y), end_pos=(x, y+80)), # top-down(left)
                Wall(world=self.world, start_pos=(x, y+80), end_pos=(x+20, y+80)), # bot-right
                # create half-open wall
                Wall(world=self.world, start_pos=(x+20, y), end_pos=(x+20, y+30)), # top-down(right)
                Wall(world=self.world, start_pos=(x+20, y+60), end_pos=(x+20, y+80)),# top-down(right)
            ]
        # RIGHT-OPEN, SMALL SQUARE
        if tile_type == 'building_2':
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x, y), end_pos=(x+30, y)), # top-right
                Wall(world=self.world, start_pos=(x, y), end_pos=(x, y+30)), # top-down(left)
                Wall(world=self.world, start_pos=(x, y+30), end_pos=(x+30, y+30)), # bot-right
                # create half-open wall
                Wall(world=self.world, start_pos=(x+30, y+20), end_pos=(x+30, y+30)), # top-down(right)
            ]
        # LEFT-OPEN, LONG
        if tile_type == 'building_3':
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x, y), end_pos=(x+20, y)), # top-right
                Wall(world=self.world, start_pos=(x+20, y), end_pos=(x+20, y+80)), # top-down(right)
                Wall(world=self.world, start_pos=(x, y+80), end_pos=(x+20, y+80)), # bot-right
                # create half-open wall
                Wall(world=self.world, start_pos=(x, y), end_pos=(x, y+30)), # top-down(left)
                Wall(world=self.world, start_pos=(x, y+60), end_pos=(x, y+80)),# top-down(left)
            ]
        # LEFT-OPEN, MEDIUM
        if tile_type == 'building_4':
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x, y), end_pos=(x+40, y)), # top-right
                Wall(world=self.world, start_pos=(x+40, y), end_pos=(x+40, y+50)), # top-down(right)
                Wall(world=self.world, start_pos=(x, y+50), end_pos=(x+40, y+50)), # bot-right
                # create half-open wall
                Wall(world=self.world, start_pos=(x, y), end_pos=(x, y+20)), # top-down(left)
                Wall(world=self.world, start_pos=(x, y+40), end_pos=(x, y+50)),# top-down(left)
            ]
        # L-shape
        if tile_type == 'building_5':
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x, y), end_pos=(x+20, y)), # top-right
                Wall(world=self.world, start_pos=(x+20, y), end_pos=(x+20, y+60)), # top-down(right)
                # create half-open wall
                Wall(world=self.world, start_pos=(x, y), end_pos=(x, y+30)), # top-down(left)
                Wall(world=self.world, start_pos=(x, y+60), end_pos=(x, y+90)),# top-down(left)
                # create right complex
                Wall(world=self.world, start_pos=(x+20, y+60), end_pos=(x+60, y+60)), # bot-right
                Wall(world=self.world, start_pos=(x, y+90), end_pos=(x+60, y+90)), # bot-right
                Wall(world=self.world, start_pos=(x+60, y+60), end_pos=(x+60, y+90)), # top-down(right)
            ]
        # RIGHT-OPEN, SMALL SQUARE
        if tile_type == 'building_6':
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x, y), end_pos=(x+50, y)), # top-right
                Wall(world=self.world, start_pos=(x, y), end_pos=(x, y+20)), # top-down(left)
                Wall(world=self.world, start_pos=(x+50, y), end_pos=(x+50, y+20)), # top-down(right)
                # create half-open wall
                Wall(world=self.world, start_pos=(x, y+20), end_pos=(x+20, y+20)), # bot-right
                Wall(world=self.world, start_pos=(x+40, y+20), end_pos=(x+50, y+20)), # bot-right
            ]
        
        return tile_walls