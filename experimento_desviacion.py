import numpy as np
from numba import njit
import matplotlib.pyplot as plt

# Parámetros iniciales
window_size = (800, 800)
dt = 0.5

def create_atoms(number):
    group = []
    for _ in range(number):
        x, y = np.random.randint(50, window_size[0] - 50, size=2)
        group.append({'x': x, 'y': y, 'vx': 0, 'vy': 0})
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

@njit(fastmath=True)
def calculate_kinetic_energy(atoms_np):
    kinetic_energy = 0.0
    for atom in atoms_np:
        vx, vy = atom[2], atom[3]
        kinetic_energy += 0.5 * (vx**2 + vy**2)
    return kinetic_energy

@njit(fastmath=True)
def calculate_potential_energy(atoms1_np, atoms2_np, g):
    potential_energy = 0.0
    for a in atoms1_np:
        ax, ay = a[:2]
        for b in atoms2_np:
            bx, by = b[:2]
            dx, dy = ax - bx, ay - by
            d = np.sqrt(dx**2 + dy**2)
            if 0 < d < 80:
                potential_energy += (g * 1) / d
    return potential_energy

n_atoms = 2000
yellow_atoms = create_atoms(n_atoms)
red_atoms = create_atoms(n_atoms)
green_atoms = create_atoms(n_atoms)
blue_atoms = create_atoms(n_atoms)

# Convertir a numpy structures
yellow_atoms_np = np.array([(a['x'], a['y'], a['vx'], a['vy']) for a in yellow_atoms], dtype=np.float64)
red_atoms_np = np.array([(a['x'], a['y'], a['vx'], a['vy']) for a in red_atoms], dtype=np.float64)
green_atoms_np = np.array([(a['x'], a['y'], a['vx'], a['vy']) for a in green_atoms], dtype=np.float64)
blue_atoms_np = np.array([(a['x'], a['y'], a['vx'], a['vy']) for a in blue_atoms], dtype=np.float64)

atoms_np = np.concatenate((yellow_atoms_np, red_atoms_np, green_atoms_np, blue_atoms_np))
desviaciones_estandar = np.linspace(0.1, 2, 10)  # Lista de desviaciones estándar para probar
energias = []

for desviacion in desviaciones_estandar:
    lista_interacciones = np.random.normal(0, desviacion, 16)
    lista_interacciones = np.array(lista_interacciones) / np.linalg.norm(lista_interacciones)

    energia_total = 0
    pasos = 1000  # Reducir el número de pasos para evaluar solo los primeros frames

    for step in range(pasos):
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

        kinetic_energy = calculate_kinetic_energy(atoms_np)
        potential_energy = 0
        for i in range(4):
            for j in range(i, 4):
                atoms1_np, _ = groups[i]
                atoms2_np, _ = groups[j]
                potential_energy += calculate_potential_energy(atoms1_np, atoms2_np, lista_interacciones[i*4+j])

        energia_total += kinetic_energy + potential_energy

    energia_promedio = energia_total / pasos
    energias.append(energia_promedio)

# Graficar la energía del sistema en función de la desviación estándar
plt.plot(desviaciones_estandar, energias)
plt.xlabel('Desviación Estándar')
plt.ylabel('Energía del Sistema')
plt.title('Energía del Sistema en función de la Desviación Estándar')
plt.show()
