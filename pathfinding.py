import pygame as pg
import pymunk # simulates in C -> fast 
import numpy as np
import skimage.measure as measure # for 2d max pooling (pip install scikit-image)
import random

class Node():
    def __init__(self, coordinates, distance=None):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.distance = distance 
    
    def coordinates(self):
        return (self.x, self.y)

    def get_neighbors(self, world_array) -> list:
        neighbors = []
        for x_delta in [-1, 0, 1]:
            for y_delta in [-1, 0, 1]:                    
                # calculate the neighbor-node's coordinates
                neighbor_x = self.x + x_delta
                neighbor_y = self.y + y_delta
                
                # make sure that the neighbor is a node in the world_array
                if neighbor_x < 0 or neighbor_x > 799 or neighbor_y < 0 or neighbor_y > 799: # XXX
                    continue
                
                # disregard if the neighbor node is a wall
                if world_array[neighbor_x, neighbor_y] == 1:
                    # print(f'{(neighbor_x, neighbor_y)} is a wall')
                    continue
                
                # exclude the origin node (self) from neighbors
                if x_delta == 0 and y_delta == 0:
                    continue
                    
                # check that the neighbor is in bounds
                if neighbor_x < 0 or neighbor_x > 799: # XXX (was: >79 /199/ 799)
                    continue
                if neighbor_y < 0 or neighbor_y > 799: # XXX
                    continue
                
                # append valid neighbor
                neighbor = Node(coordinates=(neighbor_x, neighbor_y))
                neighbors.append(neighbor)
        
        # return all neighbors as a list of nodes
        return neighbors
    
    def distance_to_neighbor(self, neighbor):
        """ Returns the euclidian distance to a neighboring cell (node). 
        From node "X", the distance is defined as follows:
        
        sqrt(2) | 1 | sqrt(2)
        --------|---|--------
           1    | X |   1
        --------|---|--------
        sqrt(2) | 1 | sqrt(2)
        
        """
        # check that both nodes are neighbors
        if not self.x in [neighbor.x + x_delta for x_delta in [-1, 0, 1]] \
            or not self.y in [neighbor.y + y_delta for y_delta in [-1, 0, 1]]:
                raise Exception(
                    f'Error: {self.coordinates()} and {neighbor.coordinates()} are not neighbors!')
        
        # check that they are not the same node
        if self.coordinates() == neighbor.coordinates():
            raise Exception(
                f'Error: {self.coordinates()} and {neighbor.coordinates()} are the same node!')
        
        # check if they are direct neighbors (-> distance=1) 
        if self.x == neighbor.x and self.y in [neighbor.y + y_delta for y_delta in [-1, 0, 1]] \
            or self.y == neighbor.y and self.x in [neighbor.x + x_delta for x_delta in [-1, 0, 1]]:
                return 1
    
        # if they are not direct neighbors, they are diagonal neighbors (-> distance=sqrt(2))
        return np.sqrt(2)
    
    
class Queue():

    def __init__(self, start_node):
        self.queue = [start_node]
        self.enqueued_coordinates = {start_node.coordinates()}
        
    def add_node(self, node) -> None:
        # insert node as first element if the node is not already in enqueued
        if node.coordinates() not in self.enqueued_coordinates:
            self.queue = [node] + self.queue
            self.enqueued_coordinates.add(node.coordinates())
    
    def remove_node(self) -> Node:
        # remove node from the set of enqueued nodes and return last node
        node = self.queue.pop()
        self.enqueued_coordinates.remove(node.coordinates())
        return node
    
    def has_elements(self) -> bool:
        # return True if the Queue has elements and False otherwise
        return bool(self.queue)


class Pathfinder():
    
    def __init__(self, sim, use_precomputed_heatmaps):        

        self.world_array = self.create_world_array(sim)

        self.targets = [(random.randint(0, 800), random.randint(0, 800)) for i in range(3)] # specify list of real targets here
        n_targets = len(self.targets)
        self.heatmap_tensor = None # will be initialized in the following lines 

        if use_precomputed_heatmaps:
            self.load_heatmap_tensor()

        else:
            # compute the heatmaps for all targets
            self.heatmap_tensor = np.empty((n_targets, 800, 800))
            print("computing heatmaps with shape", self.heatmap_tensor.shape, "[this may take a while!]")

            for i, target in enumerate(self.targets):
                print(f"creating heatmap... [{i+1}/{n_targets}]")

                # create the target node
                target_node = Node(coordinates=target, distance=0)

                # create the heatmap for that target node
                self.heatmap_tensor[i] = self.create_heatmap(target_node)
            self.save_heatmap_tensor()

        # add the instanciated pathfinder to the given simulator
        sim.pf = self
    

    def save_heatmap_tensor(self):
        np.save("heatmaps/heatmap_tensor.npy", self.heatmap_tensor)
        print("saved heatmap_tensor with shape", self.heatmap_tensor.shape)

    
    def load_heatmap_tensor(self):
        self.heatmap_tensor = np.load("heatmaps/heatmap_tensor.npy")
        print("using precomputed heatmaps with shape", self.heatmap_tensor.shape)


    def create_world_array(self, sim):
        # model the world as an array of 0s and 1s for pathfinding
        # 0 means you can go there
        # 1 means there is a wall
        world_array = np.zeros((800, 800), dtype=int)

        # add walls (change pixels that belong to a wall to '1') 
        for building in sim.buildings:
            for wall in building:
                for pixel in wall.get_pixels():
                    world_array[pixel] = 1

        # add screen borders as walls
        world_array[0,   :] = 1
        world_array[799, :] = 1
        world_array[:,   0] = 1
        world_array[:, 799] = 1

        # reduce the number of pixels to make pathfinding practical
        # new shape: (80, 80)
        world_array = measure.block_reduce(world_array, (1, 1), np.max) # XXX was: 4, 4
        print('world_array shape:', world_array.shape)
        return world_array
        
    
    def create_heatmap(self, target_node): # vector field
        """ Creates a heatmap starting from the target coordinates. """
        # create a Queue object that is used to store nodes while searching
        queue = Queue(target_node)

        # expand the heatmap as long as there are nodes with no distance to the target
        self.visited = dict() # dict that maps coordinates to the corresponding node
        
        while queue.has_elements():
            # print(f'queue #nodes: {len(queue.queue)}')
            # print(f'visited #nodes: {len(self.visited.keys())}')
            current_node = queue.remove_node()
            self.expand_heatmap(queue, current_node)
        
        heatmap = np.zeros((800, 800), dtype=int) # XXX evtl. int wegnehmen, war 200x200
        for coordinates, node in self.visited.items():
            heatmap[coordinates] = node.distance
            
        # make a vector field here by applying convolution
        self.heatmap = heatmap ########### needed for current simulation with only 1 heatmap # !!!!! später diese line rausnehmen
        return heatmap
    
    
    def expand_heatmap(self, queue, current_node):
        # print(f'current: {current_node.coordinates()}')
        
        # set the current node to visited
        self.visited[current_node.coordinates()] = current_node
        
        # get neighbors of node
        neighbors = current_node.get_neighbors(self.world_array)
        for i in range(len(neighbors)):
            # replace neighbors that have been visited before by the according 
            # nodes to preserve the distance and other attributes
            if neighbors[i].coordinates() in self.visited.keys():
                neighbors[i] = self.visited[neighbors[i].coordinates()]
        
        for neighbor in neighbors:
            
            # check if the node was not visited before
            if neighbor.coordinates() not in self.visited.keys():
                
                # set the neighbor's distance
                neighbor.distance = current_node.distance + current_node.distance_to_neighbor(neighbor)
                
                # append neighbor to queue
                queue.add_node(neighbor)
            
            # node has been visited before
            else: 
                # check if the current path's distance is shorter 
                # than the neighbors old distance to the target
                if neighbor.distance > current_node.distance + current_node.distance_to_neighbor(neighbor):
                    
                    # update the distance to the shorter distance
                    neighbor.distance = current_node.distance + current_node.distance_to_neighbor(neighbor)

                    
    def get_direction(self, current_position, target_building) -> tuple: # -> set direction as new velocity
        """
        Returns the (x,y) direction vector that follows the shortest path to the target.
        """
        current_node = Node(coordinates=current_position)
        neighbors = current_node.get_neighbors(self.world_array)
        
        # determine the neighbor with the shortest distance to the target
        best_neighbor = None
        best_distance = np.inf
        for neighbor in neighbors:
            # print(neighbor.coordinates(), heatmap[neighbor.coordinates()])
            
            ### neighbor_distance = self.heatmap[neighbor.coordinates()] ### old: from version with only one heatmap
            neighbor_distance = self.heatmap_tensor[target_building][neighbor.coordinates()]
            
            if neighbor_distance < best_distance:
                best_neighbor = neighbor
                best_distance = neighbor_distance
        
        # if no best neighbor was found (edgecase), return a random direction
        if best_neighbor is None:
            return (np.random.choice([-1,0,1]), np.random.choice([-1,0,1]))
        
        # calculate the direction vector to the neighbor with the shortest distance
        direction_x = best_neighbor.x - current_position[0]
        direction_y = best_neighbor.y - current_position[1]
        return direction_x, direction_y