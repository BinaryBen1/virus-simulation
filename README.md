## 
# <div align="center">Simulation of a Coronavirus Epidemic <br/> using a SIR-Model </div>
######  <div align="center">A university project by Till Zemann & Ben Kampmann.</div>

<!--
Insert here:
- GIF of the simulation
- graphs

GIF Example:
##### <div align="center">![simulation_gif](https://media.tenor.com/o656qFKDzeUAAAAC/rick-astley-never-gonna-give-you-up.gif) Description.
</div>
-->

## Collision handling and infection spread

When a collision between two shapes occurs, the PyMunk simulator first calls a pre_solve function (in our case the custom collision_begin function).
We check that the two participating shapes in the collision are both people (a collision could also be a person colliding with a wall). If one of the two participants is infected, and the other one isn't, the infection status will be shared with a infection probability p (currently `0.3`).

## Train schedule

The train follows a simple loop. In each cycle it drives from the top to the bottom of the map and then respawns at the top.
Each cycle takes __36k pygame-timesteps__ to complete and contains the following events, __relative to
the start of the cycle__ (i.e. 9k steps means start time $t_{\text{start}}$ of the cycle + 9k timesteps).

| Timestep | Event |
| ------ | ------ |
| $t_{start}$ | The train drives from the top of the map towards the bottom of the map with a velocity $v_x = -1.1 \text{ px/s}$ and $v_y = 30 \text{ px/s}$. |
| $t_{start} + 9k$ | The train stops at the trainstation (velocity is set to 0) and opens the door. |
| $t_{start} + 13k$ | The train closes it's door and resumes moving with the initial velocity of $v_x = -1.1 \text{ px/s}$ and $v_y = 30 \text{ px/s}$. |
| $t_{start} + 36k$ | The train respawns at the top of the map at $x = 70$ and $y = 5$. |



## Acknowledgements

We would like to thank Dr. Helge Todt for guidance on the project.

## References

- Deutsches Rotes Kreuz: Epidemien und Pandemien: Hilfe bei Infektionsausbrüchen. [Pandemie, Epidemie, Endemie Definitionen](https://www.drk.de/hilfe-weltweit/wann-wir-helfen/katastrophe/epidemien-pandemien/) (last accessed on 31-Oct-2022). 

- AtiByte: Pymunk physics in Pyglet - p11 - collision handler. [Link](https://www.youtube.com/watch?v=ZVDm2Xtp3Lw).

- Arachnid56: City builder tutorial series | Creating the world grid | pygame (#1). [Link](https://www.youtube.com/watch?v=wI_pvfwcPgQ).

- Ear Of Corn Programming: Coronavirus Simulator and Analysis | Pymunk/PyGame Projects. [Link](https://www.youtube.com/watch?v=yJK5J8a7NFs).

- PyData London: James Allen - Forecasting social inequality using agent-based modelling. [Link](https://www.youtube.com/watch?v=RglNX4c_dfc).

- [PyGame (game engine) documentation](https://www.pygame.org/docs/)

- [PyMunk (physics engine) documentation](http://www.pymunk.org/en/latest/pymunk.html)



Heil Scheffer

