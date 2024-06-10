import pygame
import numpy as np
from numba import njit

pygame.init()
window_size = (800, 800)
window = pygame.display.set_mode(window_size, pygame.RESIZABLE)
pygame.display.set_caption('Life')
dt = 0.5

BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)

def draw(x, y, color, size):
    pygame.draw.rect(window, color, pygame.Rect(x, y, size, size))

def create_atoms(number, color):
    group = []
    for _ in range(number):
        x, y = np.random.randint(50, window_size[0] - 50, size=2)
        group.append({'x': x, 'y': y, 'vx': 0, 'vy': 0, 'color': color})
    return group

@njit(fastmath=True)
def calculate_forces(atoms1_np, atoms2_np, g):
    forces = np.zeros_like(atoms1_np[:, :2])
    for i, a in enumerate(atoms1_np):
        ax, ay = a[:2]
        for b in atoms2_np:
            bx, by = b[:2]
            dx, dy = ax - bx, ay - by
            d = np.sqrt(dx**2 + dy**2)
            if 0 < d < 80:
                F = (g * 1) / d
                fx, fy = F * dx, F * dy
                forces[i, 0] += fx
                forces[i, 1] += fy
    return forces

@njit(fastmath=True)
def update_positions(atoms_np, forces, dt, window_width, window_height):
    for i, atom in enumerate(atoms_np):
        atom[:2] += forces[i] * dt
        # Check for boundary conditions
        if atom[0] < 0:
            atom[0] = 0
            atom[2] *= -1
        elif atom[0] > window_width:
            atom[0] = window_width
            atom[2] *= -1
        if atom[1] < 0:
            atom[1] = 0
            atom[3] *= -1
        elif atom[1] > window_height:
            atom[1] = window_height
            atom[3] *= -1
n_atoms = 2000
yellow_atoms = create_atoms(n_atoms, YELLOW)
red_atoms = create_atoms(n_atoms, RED)
green_atoms = create_atoms(n_atoms, GREEN)
blue_atoms = create_atoms(n_atoms, BLUE)

# Convertir a numpy structures
yellow_atoms_np = np.array([(a['x'], a['y'], a['vx'], a['vy']) for a in yellow_atoms], dtype=np.float64)
red_atoms_np = np.array([(a['x'], a['y'], a['vx'], a['vy']) for a in red_atoms], dtype=np.float64)
green_atoms_np = np.array([(a['x'], a['y'], a['vx'], a['vy']) for a in green_atoms], dtype=np.float64)
blue_atoms_np = np.array([(a['x'], a['y'], a['vx'], a['vy']) for a in blue_atoms], dtype=np.float64)

atoms_np = np.concatenate((yellow_atoms_np, red_atoms_np, green_atoms_np, blue_atoms_np))
lista_interacciones = np.random.normal(0, 1.25, 16)
        
# Normalizar las interacciones
lista_interacciones = np.array(lista_interacciones) / np.linalg.norm(lista_interacciones)

#lista_interacciones = [-0.18667159, -0.07045975, -0.35406358, -0.10307926,  0.13242806, -0.27269096, -0.27197114, -0.25528531, -0.36593772,  0.28653189,  0.36690277,  0.23758196, -0.12712116, -0.24294398,  0.21949765,  0.25046315]
#lista_interacciones = [ 0.0528949,  -0.34422419,  0.14444683, -0.34023361,  0.05537774, -0.30859658, -0.3233754,   0.12076884, -0.312273,    0.09932678, -0.40732891, -0.28979029,  0.03535335, -0.23729318, -0.11779851,  0.30971413]
print(lista_interacciones)
clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.VIDEORESIZE:
            window_size = (event.w, event.h)
            window = pygame.display.set_mode(window_size, pygame.RESIZABLE)

    # Actualizar posiciones usando forces
    n_yellow = len(yellow_atoms)
    n_red = len(red_atoms)
    n_green = len(green_atoms)
    n_blue = len(blue_atoms)
    
    groups = [
        (yellow_atoms_np, n_yellow),
        (red_atoms_np, n_red),
        (green_atoms_np, n_green),
        (blue_atoms_np, n_blue)
    ]

    for i in range(4):
        for j in range(i, 4):
            atoms1_np, len1 = groups[i]
            atoms2_np, len2 = groups[j]
            forces = calculate_forces(atoms1_np, atoms2_np, lista_interacciones[i*4+j])
            update_positions(atoms1_np, forces, dt, window_size[0], window_size[1])
            if i != j:
                forces = calculate_forces(atoms2_np, atoms1_np, lista_interacciones[j*4+i])
                update_positions(atoms2_np, forces, dt, window_size[0], window_size[1])

    # Reflect updated positions in the original atom structures
    for i, (atoms, atoms_np) in enumerate(zip([yellow_atoms, red_atoms, green_atoms, blue_atoms],
                                              [yellow_atoms_np, red_atoms_np, green_atoms_np, blue_atoms_np])):
        for j, atom in enumerate(atoms):
            atom['x'], atom['y'], atom['vx'], atom['vy'] = atoms_np[j]

    window.fill(BLACK)
    # Dibujar atoms (manteniendo la lista original para dibujo)
    for atom in (yellow_atoms + red_atoms + green_atoms + blue_atoms):
        draw(atom['x'], atom['y'], atom['color'], 3)  # Cambiar tamaño a 1

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
