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
YELLOW = (245, 203, 66)
RED = (252, 3, 65)

class CovidSim():
    def __init__(self, n_people, infection_prob=0.3, avg_incubation_time=5_000, avg_infectious_time=10_000, debug_mode=False, FPS=60):
        # sim setup
        self.n_people = n_people
        self.status_counts = [] # will be filled with 3-tuples of counts of how many people are healthy/infected/recovered at runtime
        self.pf = None # will be set later

        # visuals setup
        self.screen_size = 800
        self.width, self.height = (self.screen_size, self.screen_size)
        self.FPS = FPS
        
        # some useful attributes
        self.draw_dots = debug_mode
        self.draw_walls = debug_mode
        self.speedup_factor = 1
        self.running = True
        
        # hyperparameters
        self.infection_prob = infection_prob
        self.avg_incubation_time = avg_incubation_time
        self.avg_infectious_time = avg_infectious_time

        # setup screen_borders, buildings
        self.world = None
        self.screen_borders = None
        self.buildings = None
        self.create_world()
    

    def create_world(self):
        # create the simulated world
        self.world = pymunk.Space()
        
        # add screen borders as walls
        self.screen_borders = [
            Wall(world=self.world, start_pos=(0, 0), end_pos=(800, 0), thickness=1), # top-right
            Wall(world=self.world, start_pos=(0, 0), end_pos=(0, 800), thickness=1), # top-down (left)
            Wall(world=self.world, start_pos=(800, 0), end_pos=(800, 800), thickness=1), # top-down(right)

            # make two walls at the bottom to leave a hole for the train to pass through
            Wall(world=self.world, start_pos=(0, 800), end_pos=(110, 800), thickness=1), # bot-right
            Wall(world=self.world, start_pos=(130, 800), end_pos=(800, 800), thickness=1)] # bot-right
        
        # create walls for buildings
        self.buildings = [
            self._create_tile(origin_pos=(630,490), tile_type='building_1'),
            self._create_tile(origin_pos=(620,440), tile_type='building_2'),
            self._create_tile(origin_pos=(700,530), tile_type='building_3'),
            self._create_tile(origin_pos=(690,430), tile_type='building_4'),
            self._create_tile(origin_pos=(700,310), tile_type='building_5'),
            self._create_tile(origin_pos=(630,310), tile_type='building_6'),
            self._create_tile(origin_pos=(630,340), tile_type='building_7'),
            self._create_tile(origin_pos=(490,340), tile_type='building_8'),
            self._create_tile(origin_pos=(570,390), tile_type='building_9'),
            self._create_tile(origin_pos=(490,480), tile_type='building_10'),
            self._create_tile(origin_pos=(490,550), tile_type='building_11'),
            self._create_tile(origin_pos=(760,280), tile_type='building_12'),
            self._create_tile(origin_pos=(720,530), tile_type='building_13'),
            self._create_tile(origin_pos=(340,520), tile_type='building_14'),
            self._create_tile(origin_pos=(370,520), tile_type='building_14a'),
            self._create_tile(origin_pos=(280,600), tile_type='building_15'),
            self._create_tile(origin_pos=(280,560), tile_type='building_16'),
            self._create_tile(origin_pos=(280,520), tile_type='building_17'),
            #self._create_tile(origin_pos=(XXX), tile_type='building_18'),
            self._create_tile(origin_pos=(450,700), tile_type='building_19'),
            self._create_tile(origin_pos=(570,680), tile_type='building_20'),
            self._create_tile(origin_pos=(630,600), tile_type='BUD'),
            self._create_tile(origin_pos=(230,430), tile_type='IKMZ'),
            #self._create_tile(origin_pos=(XXX), tile_type='building_23'),
            self._create_tile(origin_pos=(490,620), tile_type='building_24'),
            self._create_tile(origin_pos=(260,100), tile_type='building_25'),
            self._create_tile(origin_pos=(390,100), tile_type='building_26'),
            self._create_tile(origin_pos=(350,290), tile_type='building_27'),
            self._create_tile(origin_pos=(250,280), tile_type='building_28'),
            self._create_tile(origin_pos=(350,340), tile_type='building_29'),
            #self._create_tile(origin_pos=(XXX), tile_type='building_30'),
            self._create_tile(origin_pos=(440,550), tile_type='building_31'),
            self._create_tile(origin_pos=(520,680), tile_type='building_35'),

            # this building is not visitable (because it doesn't have a number on the map)
            self._create_tile(origin_pos=(440,410), tile_type='building_36'),
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
            
            # check if a person is infectious
            if shape.density == 0.8: # (0.8 is the key for infectious)
                involves_infected_person = True
        
        if involves_infected_person:
            for shape in arbiter.shapes:
                if shape.density == 1.0: # (1.0 is the key for susceptible)
                    # infection_prob: probability that infection status is shared when colliding
                    if random.random() < self.infection_prob:
                        # set the density to 0.9 to signal that the person is now infected (infection status will be set in the next timestep)
                        shape.density = 0.9
        return True

    def get_status_counts(self):
        status_list = [person.status for person in self.people]
        susceptible_count = 0
        infected_count = 0
        infectious_count = 0
        removed_count = 0

        for status in status_list:
            if status == "susceptible":
                susceptible_count += 1
            elif status == "infected":
                infected_count += 1
            elif status == "infectious":
                infectious_count += 1
            else:
                removed_count += 1
        return susceptible_count, infected_count, infectious_count, removed_count
    
    def run(self, seed=42, speedup_factor=1, max_timestep=3000, return_data=False):
        # setup
        random.seed(seed)
        self.running = True
        self.speedup_factor = speedup_factor
        self.status_counts = [] # reset status counts

        # create the pygame-screen
        self.screen = pg.display.set_mode((self.screen_size, self.screen_size))
        self.clock = pg.time.Clock()

        # add logo and caption
        logo = pg.image.load('images/virus_logo.png')
        pg.display.set_icon(logo)
        pg.display.set_caption('COVID19-Sim')

        # create the particle simulation
        self.create_world()

        # add people to the simulation
        self.people = [Person(world=self.world,
                              pathfinder=self.pf,
                              init_min=0,
                              init_max=self.screen_size,
                              collision_radius=2)
                       for i in range(self.n_people)]
    
        # define custom collision handler (collision might spread infection status)        
        self.handler = self.world.add_default_collision_handler()
        self.handler.begin = self.collision_begin   # each time two objects collide
                                                    # the custom collision_begin method is called
                                                    # for spreading the infection status 
        
        # infect 3 random persons to start the epidemic
        for _ in range(3):
            random.choice(self.people).infect()
        
        # add a train to the simulation
        self.train = Train(world=self.world,
                           start_pos=(70, 5),
                           wall_thickness=3)

        timestep = 0
        while self.running:

            self.clock.tick(self.FPS)   # update pygame time
            self.world.step(self.speedup_factor/self.FPS) # keeps rendered steps/s consistent (independent of self.FPS)
            timestep += 1
            
            # handle mouse and keyboard events
            self.events()
            
            # update the velocity of all people to navigate to their goal
            for person in self.people:
                person.update_velocity(pg.time.get_ticks())
            
            # update the train state and infections
            self.update()
            
            # render the map and all simulated objects
            if self.running:
                self.draw()
                            
            # save counts of how many people are healthy/infected/recovered
            susceptible_count, infected_count, infectious_count, removed_count = self.get_status_counts()
            self.status_counts.append((susceptible_count, infected_count, infectious_count, removed_count))

            # check if maximum simulation time is reached
            if timestep >= max_timestep:
                break
            
        pg.quit()

        # return collected data
        if return_data:
            susceptible_counts = [status_tuple[0] for status_tuple in self.status_counts]
            infected_counts = [status_tuple[1] for status_tuple in self.status_counts]
            infectious_counts = [status_tuple[2] for status_tuple in self.status_counts]
            removed_counts = [status_tuple[3] for status_tuple in self.status_counts]
            return susceptible_counts, infected_counts, infectious_counts, removed_counts

    
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

        # update targets
        for person in self.people:
            person.update_target(timestep=pg.time.get_ticks())
        
        # update infections
        #for person in self.people:
        #    if person.shape.density == 0.9:
        #        person.status = "infected"

        # update status
        for person in self.people:
            person.update_infection_status(
                self.avg_incubation_time,
                self.avg_infectious_time,
                timestep=pg.time.get_ticks())

    
    def draw(self):
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
        if self.draw_walls:
            for building in self.buildings:
                for wall in building:
                    wall.draw(self.screen)
        
        # draw grid of dots for testing/debugging
        if self.draw_dots:
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
        if tile_type == 'building_2':
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x, y), end_pos=(x+20, y)), # top-right
                Wall(world=self.world, start_pos=(x, y), end_pos=(x, y+30)), # top-down(left)
                Wall(world=self.world, start_pos=(x, y+30), end_pos=(x+30, y+30)), # bot-right
            ]
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
        if tile_type == 'building_7':
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x, y+10), end_pos=(x+20, y+10)), # top-right
                Wall(world=self.world, start_pos=(x, y+10), end_pos=(x, y+70)), # top-down(left)
                Wall(world=self.world, start_pos=(x, y+70), end_pos=(x+20, y+70)), # bot-right
                # create half-open wall
                Wall(world=self.world, start_pos=(x+20, y+10), end_pos=(x+20, y+30)), # top-down(right)
                Wall(world=self.world, start_pos=(x+20, y+60), end_pos=(x+20, y+70)),# top-down(right)
            ]
        if tile_type == 'building_8':
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x, y), end_pos=(x+60, y)), # top-right
                Wall(world=self.world, start_pos=(x, y), end_pos=(x, y+30)), # top-down(left)
                Wall(world=self.world, start_pos=(x, y+30), end_pos=(x+20, y+30)), # bot-left (left of door)

                Wall(world=self.world, start_pos=(x+60, y), end_pos=(x+60, y-50)), # top-up
                Wall(world=self.world, start_pos=(x+60, y-50), end_pos=(x+90, y-50)), # top-up-right
                Wall(world=self.world, start_pos=(x+90, y-50), end_pos=(x+90, y+30)), # top-up-right-down
                Wall(world=self.world, start_pos=(x+70, y+30), end_pos=(x+90, y+30)), # right of door (bot-right)
            ]
        if tile_type == 'building_9':
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x, y+20), end_pos=(x+20, y+20)), # top-right
                Wall(world=self.world, start_pos=(x, y+20), end_pos=(x, y+70)), # top-down(left)
                Wall(world=self.world, start_pos=(x, y+70), end_pos=(x+20, y+70)), # bot-right
                # create half-open wall
                Wall(world=self.world, start_pos=(x+20, y+20), end_pos=(x+20, y+30)), # top-down(right)
                Wall(world=self.world, start_pos=(x+20, y+60), end_pos=(x+20, y+70)),# top-down(right)
            ]
        if tile_type == 'building_10':
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x, y), end_pos=(x+80, y)), # top-right
                Wall(world=self.world, start_pos=(x, y), end_pos=(x, y+40)), # top-down(left)
                Wall(world=self.world, start_pos=(x, y+40), end_pos=(x+30, y+40)), # bot-left (left of door)

                Wall(world=self.world, start_pos=(x+80, y), end_pos=(x+80, y-20)), # top-up
                Wall(world=self.world, start_pos=(x+80, y-20), end_pos=(x+100, y-20)), # top-up-right
                Wall(world=self.world, start_pos=(x+100, y-20), end_pos=(x+100, y+40)), # top-up-right-down
                Wall(world=self.world, start_pos=(x+70, y+40), end_pos=(x+100, y+40)), # right of door (bot-right)
            ]
        if tile_type == 'building_11':
            tile_walls = [
                # create half-open wall
                Wall(world=self.world, start_pos=(x+30, y), end_pos=(x+100, y)), # top-right

                # main walls
                Wall(world=self.world, start_pos=(x, y), end_pos=(x, y+40)), # top-down(left)
                Wall(world=self.world, start_pos=(x+100, y), end_pos=(x+100, y+40)), # top-down(right)
                Wall(world=self.world, start_pos=(x, y+40), end_pos=(x+100, y+40)), # bot-right
            ]
        if tile_type == 'building_12':
            tile_walls = [

                Wall(world=self.world, start_pos=(x, y+40), end_pos=(x, y+70)), # top-down(left)

                # create main walls
                Wall(world=self.world, start_pos=(x, y+10), end_pos=(x+20, y+10)), # top-right
                Wall(world=self.world, start_pos=(x, y+70), end_pos=(x+20, y+70)), # bot-right
                # create half-open wall
                Wall(world=self.world, start_pos=(x+20, y+10), end_pos=(x+20, y+70)), # top-down(right)
            ]
        if tile_type == 'building_13':
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x+20, y), end_pos=(x+30, y)), # top-right
                Wall(world=self.world, start_pos=(x, y), end_pos=(x, y+30)), # top-down(left)
                Wall(world=self.world, start_pos=(x, y+30), end_pos=(x+30, y+30)), # bot-right
                # create half-open wall
                Wall(world=self.world, start_pos=(x+30, y), end_pos=(x+30, y+30)), # top-down(right)
            ]
        if tile_type == 'building_14':
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x, y), end_pos=(x+20, y)), # top-right
                Wall(world=self.world, start_pos=(x+20, y), end_pos=(x+20, y+90)), # top-down(right)
                Wall(world=self.world, start_pos=(x, y+90), end_pos=(x+20, y+90)), # bot-right
                # create half-open wall
                Wall(world=self.world, start_pos=(x, y), end_pos=(x, y+20)), # top-down(left)
                Wall(world=self.world, start_pos=(x, y+50), end_pos=(x, y+90)),# top-down(left)
            ]
        if tile_type == 'building_14a':
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x+20, y), end_pos=(x+40, y)), # top-right
                Wall(world=self.world, start_pos=(x-10, y), end_pos=(x-10, y+30)), # top-down(left)
                Wall(world=self.world, start_pos=(x-10, y+30), end_pos=(x+40, y+30)), # bot-right
                # create half-open wall
                Wall(world=self.world, start_pos=(x+40, y), end_pos=(x+40, y+30)), # top-down(right)
            ]
        if tile_type == 'building_15':
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x, y), end_pos=(x+40, y)), # top-right
                Wall(world=self.world, start_pos=(x, y), end_pos=(x, y+20)), # top-down(left)
                # create half-open wall
                Wall(world=self.world, start_pos=(x, y+20), end_pos=(x+40, y+20)), # bot-right
            ]
        if tile_type == 'building_16':
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x, y), end_pos=(x+40, y)), # top-right
                Wall(world=self.world, start_pos=(x, y), end_pos=(x, y+20)), # top-down(left)
                # create half-open wall
                Wall(world=self.world, start_pos=(x, y+20), end_pos=(x+40, y+20)), # bot-right
            ]
        if tile_type == 'building_17':
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x, y), end_pos=(x+40, y)), # top-right
                Wall(world=self.world, start_pos=(x, y), end_pos=(x, y+20)), # top-down(left)
                # create half-open wall
                Wall(world=self.world, start_pos=(x, y+20), end_pos=(x+40, y+20)), # bot-right
            ]

        # building 18 is not on the map.

        if tile_type == 'building_19':
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x+30, y), end_pos=(x+50, y)), # top-right
                Wall(world=self.world, start_pos=(x, y), end_pos=(x, y+50)), # top-down(left)
                Wall(world=self.world, start_pos=(x, y+50), end_pos=(x+50, y+50)), # bot-right
                # create half-open wall
                Wall(world=self.world, start_pos=(x+50, y), end_pos=(x+50, y+50)), # top-down(right)
            ]
        if tile_type == 'building_20':
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x+20, y), end_pos=(x+40, y)), # top-right
                Wall(world=self.world, start_pos=(x-10, y), end_pos=(x-10, y+30)), # top-down(left)
                Wall(world=self.world, start_pos=(x-10, y+30), end_pos=(x+40, y+30)), # bot-right
                # create half-open wall
                Wall(world=self.world, start_pos=(x+40, y), end_pos=(x+40, y+30)), # top-down(right)
            ]
        if tile_type == 'BUD':
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x+30, y), end_pos=(x+40, y)), # top-right
                Wall(world=self.world, start_pos=(x, y), end_pos=(x, y+50)), # top-down(left)
                Wall(world=self.world, start_pos=(x, y+50), end_pos=(x+40, y+50)), # bot-right
                # create half-open wall
                Wall(world=self.world, start_pos=(x+40, y), end_pos=(x+40, y+50)), # top-down(right)
            ]
        if tile_type == 'IKMZ':
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x+30, y), end_pos=(x+70, y)), # top-right
                Wall(world=self.world, start_pos=(x, y), end_pos=(x, y+60)), # top-down(left)
                Wall(world=self.world, start_pos=(x, y+60), end_pos=(x+70, y+60)), # bot-right
                # create half-open wall
                Wall(world=self.world, start_pos=(x+70, y), end_pos=(x+70, y+60)), # top-down(right)
            ]
        if tile_type == 'building_24':
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x, y), end_pos=(x+80, y)), # top-right
                Wall(world=self.world, start_pos=(x, y), end_pos=(x, y+30)), # top-down(left)
                Wall(world=self.world, start_pos=(x, y+30), end_pos=(x+30, y+30)), # bot-left (left of door)

                Wall(world=self.world, start_pos=(x+80, y), end_pos=(x+80, y-30)), # top-up
                Wall(world=self.world, start_pos=(x+80, y-30), end_pos=(x+100, y-30)), # top-up-right
                Wall(world=self.world, start_pos=(x+100, y-30), end_pos=(x+100, y+30)), # top-up-right-down
                Wall(world=self.world, start_pos=(x+70, y+30), end_pos=(x+100, y+30)), # right of door (bot-right)
            ]
        if tile_type == 'building_35':
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x+30, y), end_pos=(x+40, y)), # top-right
                Wall(world=self.world, start_pos=(x, y), end_pos=(x, y+30)), # top-down(left)
                Wall(world=self.world, start_pos=(x, y+30), end_pos=(x+40, y+30)), # bot-right
                # create half-open wall
                Wall(world=self.world, start_pos=(x+40, y), end_pos=(x+40, y+30)), # top-down(right)
            ]
        if tile_type == 'building_25':
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x, y), end_pos=(x+90, y)), # top-right
                Wall(world=self.world, start_pos=(x, y), end_pos=(x, y+150)), # top-down(left)
                Wall(world=self.world, start_pos=(x+40, y+150), end_pos=(x+90, y+150)), # bot-right
                # create half-open wall
                Wall(world=self.world, start_pos=(x+90, y), end_pos=(x+90, y+150)), # top-down(right)
            ]
        if tile_type == 'building_26':
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x, y), end_pos=(x+40, y)), # top-right
                Wall(world=self.world, start_pos=(x, y), end_pos=(x, y+150)), # top-down(left)
                Wall(world=self.world, start_pos=(x, y+150), end_pos=(x+10, y+150)), # bot-right
                # create half-open wall
                Wall(world=self.world, start_pos=(x+40, y), end_pos=(x+40, y+150)), # top-down(right)
            ]
        if tile_type == 'building_27':
            tile_walls = [
                # create half-open wall
                Wall(world=self.world, start_pos=(x+30, y), end_pos=(x+100, y)), # top-right

                # main walls
                Wall(world=self.world, start_pos=(x-10, y), end_pos=(x-10, y+30)), # top-down(left)
                Wall(world=self.world, start_pos=(x+100, y), end_pos=(x+100, y+30)), # top-down(right)
                Wall(world=self.world, start_pos=(x-10, y+30), end_pos=(x+100, y+30)), # bot-right
            ]
        if tile_type == 'building_28':
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x, y+10), end_pos=(x+60, y+10)), # top-right
                Wall(world=self.world, start_pos=(x, y+10), end_pos=(x, y+90)), # top-down(left)
                Wall(world=self.world, start_pos=(x, y+90), end_pos=(x+60, y+90)), # bot-right
                # create half-open wall
                Wall(world=self.world, start_pos=(x+60, y+10), end_pos=(x+60, y+30)), # top-down(right)
                Wall(world=self.world, start_pos=(x+60, y+60), end_pos=(x+60, y+90)),# top-down(right)
                # wall in the middle
                Wall(world=self.world, start_pos=(x+30, y+30), end_pos=(x+30, y+60)),# top-down(right)
            ]
        if tile_type == 'building_29':
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x-10, y), end_pos=(x+80, y)), # top-right
                Wall(world=self.world, start_pos=(x-10, y), end_pos=(x-10, y+30)), # top-down(left)
                Wall(world=self.world, start_pos=(x-10, y+30), end_pos=(x+30, y+30)), # bot-left (left of door)

                Wall(world=self.world, start_pos=(x+80, y), end_pos=(x+80, y-20)), # top-up
                Wall(world=self.world, start_pos=(x+80, y-20), end_pos=(x+100, y-20)), # top-up-right
                Wall(world=self.world, start_pos=(x+100, y-20), end_pos=(x+100, y+30)), # top-up-right-down
                Wall(world=self.world, start_pos=(x+70, y+30), end_pos=(x+100, y+30)), # right of door (bot-right)
            ]
        if tile_type == 'building_31':
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x, y+10), end_pos=(x+20, y+10)), # top-right
                Wall(world=self.world, start_pos=(x, y+10), end_pos=(x, y+70)), # top-down(left)
                Wall(world=self.world, start_pos=(x, y+70), end_pos=(x+20, y+70)), # bot-right
                # create half-open wall
                Wall(world=self.world, start_pos=(x+20, y+10), end_pos=(x+20, y+30)), # top-down(right)
                Wall(world=self.world, start_pos=(x+20, y+60), end_pos=(x+20, y+70)),# top-down(right)
            ]
        if tile_type == 'building_36':
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x, y+10), end_pos=(x+20, y+10)), # top-right
                Wall(world=self.world, start_pos=(x, y+10), end_pos=(x, y+60)), # top-down(left)
                Wall(world=self.world, start_pos=(x, y+60), end_pos=(x+20, y+60)), # bot-right
                # create half-open wall
                Wall(world=self.world, start_pos=(x+20, y+10), end_pos=(x+20, y+30)), # top-down(right)
                Wall(world=self.world, start_pos=(x+20, y+50), end_pos=(x+20, y+60)),# top-down(right)
            ]
        return tile_walls