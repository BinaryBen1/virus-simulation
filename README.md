## 
# <div align="center">Simulation of a Coronavirus Epidemic <br/> using a SIR-Model </div>

<!--
Insert here:
- GIF of the simulation
- graphs

GIF Example:
##### <div align="center">![simulation_gif](https://media.tenor.com/o656qFKDzeUAAAAC/rick-astley-never-gonna-give-you-up.gif) Description.
</div>
-->


## Train schedule

The train follows a simple loop, where it goes from the top to the bottom and then respawns at the top.
Each cycle of that loop takes __36k pygame-timesteps__ to complete and contains the following events, __relative to
the start of the cycle__ (i.e. 9k steps means start time $t_{\text{start}}$ of the cycle + 9k timesteps).

| Timestep | Event |
| ------ | ------ |
| $t_{start}$ | The train drives from the top of the map towards the bottom of the map with a velocity $v_x = -1.1 \text{ px/s}$ and $v_y = 30 \text{ px/s}$. |
| $t_{start} + 9k$ | The train stops at the trainstation (velocity is set to 0) and opens the door. |
| $t_{start} + 13k$ | The train closes it's door and resumes moving with the initial velocity of $v_x = -1.1 \text{ px/s}$ and $v_y = 30 \text{ px/s}$. |
| $t_{start} + 36k$ | The train respawns at the top of the map at $x = 70$ and $y = 5$. |



## Acknowledgements

- Deutsches Rotes Kreuz: Epidemien und Pandemien: Hilfe bei Infektionsausbrüchen. [Pandemie, Epidemie, Endemie Definitionen](https://www.drk.de/hilfe-weltweit/wann-wir-helfen/katastrophe/epidemien-pandemien/) (last accessed on 31-Oct-2022). 

- Arachnid56: City builder tutorial series | Creating the world grid | pygame (#1). [Link](https://www.youtube.com/watch?v=wI_pvfwcPgQ).

- Ear Of Corn Programming: Coronavirus Simulator and Analysis | Pymunk/PyGame Projects. [Link](https://www.youtube.com/watch?v=yJK5J8a7NFs).

- PyData London: James Allen - Forecasting social inequality using agent-based modelling. [Link](https://www.youtube.com/watch?v=RglNX4c_dfc).

- [PyGame (game engine) documentation](https://www.pygame.org/docs/)

- [PyMunk (physics engine) documentation](https://www.pymunk.org/en/latest/overview.html)