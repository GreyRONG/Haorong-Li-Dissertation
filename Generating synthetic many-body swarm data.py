import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import math
import multiprocessing
from itertools import combinations

class CellListMap:
    def __init__(self, positions, domain_size, radius, velocities=None):
        self.positions = positions
        self.dim = positions.shape[1]
        self.domain_size = np.array([domain_size] * self.dim if np.isscalar(domain_size) else domain_size)
        self.radius = radius

        if velocities is None:
            self.aj = np.random.uniform(-1, 1, size=self.positions.shape)
        else:
            self.aj = velocities

        self.cell_size, self.cell_num = self._compute_raw_size_and_cell_num(self.domain_size[0], radius)
        self.cell_size = np.array([self.cell_size] * self.dim)
        self.cell_num = np.array([self.cell_num] * self.dim)

        self.cells = defaultdict(list)
        self.build_cells()

    def _compute_raw_size_and_cell_num(self, domain_size_scalar, radius):
        min_cells = math.ceil(domain_size_scalar / radius)  
        for n_cells in range(min_cells, int(domain_size_scalar) + 1):
            cell_size = domain_size_scalar / n_cells
            if cell_size >= radius and abs(round(cell_size * n_cells) - domain_size_scalar) < 1e-8:
                return cell_size, n_cells
        raise ValueError("Failed to find proper cell_size.")

    def get_cell_index(self, position):
        return tuple((position // self.cell_size).astype(int))

    def build_cells(self):
        self.cells.clear()
        for i, pos in enumerate(self.positions):
            cell_idx = self.get_cell_index(pos)
            self.cells[cell_idx].append(i)

    def update_velocity(self, mode='1', min_speed=0.1):
        with multiprocessing.Pool() as pool:
            if mode == '1' or mode == '2':
                args = [(i, self.positions, self.aj, self.cells, self.cell_num, self.domain_size, self.radius, self.dim)
                        for i in range(len(self.positions))]
                if mode == '1':
                    new_velocities = pool.starmap(_compute_new_velocity, [(arg, min_speed) for arg in args])
                elif mode == '2':
                    new_velocities = pool.starmap(_compute_new_velocity_2, [(arg, min_speed) for arg in args])

            elif mode == '3':
                args = [(i, self.positions) for i in range(len(self.positions))]
                new_velocities = pool.starmap(
                    _compute_3body_velocity,
                    [(arg, self.aj, self.domain_size, self.radius, self, min_speed) for arg in args]
                )

            elif mode == '4':
                args = [(i, self.positions) for i in range(len(self.positions))]
                new_velocities = pool.starmap(
                    _compute_3body_velocity_mode4,
                    [(arg, self.aj, self.domain_size, self.radius, self, min_speed) for arg in args]
                )

            else:
                raise ValueError(f"Unknown velocity update mode: {mode}")

        self.aj = np.array(new_velocities)

    def update_position_1(self, t=1.0):
        with multiprocessing.Pool() as pool:
            args = [(i, self.positions, self.aj, t, self.domain_size)
                    for i in range(len(self.positions))]
            new_positions = pool.map(_compute_new_position_1, args)
        self.positions = np.array(new_positions)
        self.positions %= self.domain_size
        self.build_cells()

    def update_position_2(self, t=1.0, mode='velocity', repel_dist=2.0):
        with multiprocessing.Pool() as pool:
            args = [(i, self.positions, self.aj, t, self.domain_size, self.radius,
                    self.cells, self.cell_num, self.dim, mode, repel_dist)
                    for i in range(len(self.positions))]
            new_positions = pool.map(_compute_new_position, args)
        self.positions = np.array(new_positions)
        self.positions %= self.domain_size
        self.build_cells()

    def update_position_3(self, t=1.0):
        N = len(self.positions)

        args_mean = [(i, self.positions, self.domain_size) for i in range(N)]
        with multiprocessing.Pool() as pool:
            mean_positions = pool.map(_compute_triplet_mean_position, args_mean)

        mean_positions = np.array(mean_positions)

        new_velocities = mean_positions - self.positions
        deltas = new_velocities.copy()
        deltas = np.where(deltas > 0.5 * self.domain_size, deltas - self.domain_size, deltas)
        deltas = np.where(deltas < -0.5 * self.domain_size, deltas + self.domain_size, deltas)

        new_velocities = deltas

        self.aj = np.array([_apply_min_speed(v, 0.1) for v in new_velocities])

        new_positions = (self.positions + t * self.aj) % self.domain_size
        self.positions = new_positions
        self.build_cells()

def velocity_update_selector(cell_map, mode='1', min_speed=0.1):
    cell_map.update_velocity(mode=mode, min_speed=min_speed)

def position_update_selector(cell_map, mode='velocity', t=1.0, repel_dist=2.0):
    if mode == '1':
        cell_map.update_position_1(t=t)
    elif mode == '2':
        cell_map.update_position_1(t=t)
    
    elif mode == '3':
        cell_map.update_position_1(t=t)

    else:
        raise ValueError(f"Unknown update mode: {mode}")

def _apply_min_speed(v, min_speed):
    norm = np.linalg.norm(v)
    if norm < min_speed:
        return v / norm * min_speed if norm > 1e-8 else np.random.uniform(-1, 1, size=v.shape) * min_speed
    return v

def _compute_new_velocity(args, min_speed):
    i, positions, velocities, cells, cell_num, domain_size, radius, dim = args
    pos = positions[i]
    cell_idx = tuple((pos // (domain_size[0] / cell_num[0])).astype(int))

    neighbors = []
    for offset in np.ndindex(*(3,) * dim):
        neighbor_cell = tuple((np.array(cell_idx) + np.array(offset) - 1) % cell_num)
        for j in cells.get(neighbor_cell, []):
            delta = np.abs(positions[j] - pos)
            delta = np.where(delta > 0.5 * domain_size, domain_size - delta, delta)
            distance = np.linalg.norm(delta)
            if j != i and distance <= radius:
                neighbors.append(j)

    if neighbors:
        avg_neighbor_velocity = np.mean(velocities[neighbors], axis=0)
        return 0.5 * (velocities[i] + avg_neighbor_velocity)
    else:
        avg_neighbor_velocity = np.zeros_like(velocities[i])

    new_v = 0.5 * (velocities[i] + avg_neighbor_velocity)
    return _apply_min_speed(new_v, min_speed)

def _compute_new_velocity_2(args, min_speed):
    i, positions, velocities, cells, cell_num, domain_size, radius, dim = args
    pos = positions[i]
    cell_idx = tuple((pos // (domain_size[0] / cell_num[0])).astype(int))

    repel_radius = radius / 2
    repel_vector = np.zeros_like(pos)
    attract_vector = np.zeros_like(pos)
    neighbors = []

    for offset in np.ndindex(*(3,) * dim):
        neighbor_cell = tuple((np.array(cell_idx) + np.array(offset) - 1) % cell_num)
        for j in cells.get(neighbor_cell, []):
            if j == i:
                continue
            delta = positions[j] - pos
            delta = np.where(delta > 0.5 * domain_size, delta - domain_size, delta)
            delta = np.where(delta < -0.5 * domain_size, delta + domain_size, delta)
            dist = np.linalg.norm(delta)
            if dist <= radius:
                neighbors.append(j)
                if dist < repel_radius and dist > 1e-5:
                    repel_vector -= (delta / dist) * (repel_radius - dist) / repel_radius
                elif repel_radius < dist < radius:
                    attract_vector += (delta / dist) * (dist - repel_radius) / (radius - repel_radius)

    avg_neighbor_velocity = np.mean(velocities[neighbors], axis=0) if neighbors else np.zeros_like(velocities[i])

    repel_strength = 1.0
    attract_strength = 1.0
    total_vector = repel_strength * repel_vector + attract_strength * attract_vector

    new_v = (3*velocities[i] + 3*avg_neighbor_velocity + 1*total_vector) / 7.0
    return _apply_min_speed(new_v, min_speed)

def _compute_new_position(args):
    (i, positions, velocities, t, domain_size, radius,
     cells, cell_num, dim, mode, repel_dist) = args

    pos = positions[i]

    if mode == 'velocity':
        return (pos + t * velocities[i]) % domain_size

    elif mode == 'repel_near':
        return (pos + t * velocities[i]) % domain_size

def _compute_new_position_1(args):
    i, positions, velocities, t, domain_size = args
    return (positions[i] + t * velocities[i]) % domain_size

def _compute_triplet_mean_position(args):
    i, positions, domain_size = args
    N = len(positions)
    pos_i = positions[i]

    deltas = positions - pos_i
    deltas = np.where(deltas > 0.5 * domain_size, deltas - domain_size, deltas)
    deltas = np.where(deltas < -0.5 * domain_size, deltas + domain_size, deltas)
    dists = np.linalg.norm(deltas, axis=1)

    nearest_ids = np.argsort(dists)[0:2]
    triplet = positions[[i, *nearest_ids]]
    return np.mean(triplet, axis=0)

def _compute_3body_velocity(args, velocities, domain_size, radius, cell_map, min_speed):
    i, positions = args
    pos_i = positions[i]
    dim = positions.shape[1]

    deltas = positions - pos_i
    deltas = np.where(deltas > 0.5 * domain_size, deltas - domain_size, deltas)
    deltas = np.where(deltas < -0.5 * domain_size, deltas + domain_size, deltas)
    dists = np.linalg.norm(deltas, axis=1)

    cand = np.where((dists > 1e-8) & (dists <= radius))[0]
    if cand.size < 2:
        return velocities[i]

    nearest_order = np.argsort(dists[cand])
    nearest_ids = cand[nearest_order[:2]]

    triplet_ids = [i, *nearest_ids]
    triplet_positions = positions[triplet_ids]
    triplet_velocities = velocities[triplet_ids]

    mean_pos = np.mean(triplet_positions, axis=0)
    forces = mean_pos - triplet_positions
    forces = np.where(forces > 0.5 * domain_size, forces - domain_size, forces)
    forces = np.where(forces < -0.5 * domain_size, forces + domain_size, forces)
    avg_force = np.mean(forces, axis=0)

    avg_neighbor_velocity = np.mean(triplet_velocities[1:], axis=0)
    total_vector = np.zeros_like(pos_i)
    if radius is not None and cell_map is not None:
        repel_radius = radius / 3
        repel_vector = np.zeros_like(pos_i)
        attract_vector = np.zeros_like(pos_i)

        for j in cand:
            delta = positions[j] - pos_i
            delta = np.where(delta > 0.5 * domain_size, delta - domain_size, delta)
            delta = np.where(delta < -0.5 * domain_size, delta + domain_size, delta)
            dist = np.linalg.norm(delta)
            if dist < repel_radius and dist > 1e-5:
                repel_vector -= (delta / dist) * (repel_radius - dist) / repel_radius
            elif repel_radius < dist < radius:
                attract_vector += (delta / dist) * (dist - repel_radius) / (radius - repel_radius)

        total_vector = 1.0 * repel_vector + 1.0 * attract_vector

    new_v = (3 * velocities[i] + 1 * avg_neighbor_velocity + 1 * total_vector + 1 * avg_force) / 6.0
    return _apply_min_speed(new_v, min_speed)

def _compute_3body_velocity_mode4(args, velocities, domain_size, radius, cell_map, min_speed):
    i, positions = args
    pos_i = positions[i]
    dim = positions.shape[1]

    neighbors = get_neighbors_within_radius(i, cell_map, radius)
    if len(neighbors) < 2:
        return _compute_3body_velocity(args, velocities, domain_size, radius, cell_map, min_speed)

    repel_radius = radius / 3
    repel_vector = np.zeros_like(pos_i)
    attract_vector = np.zeros_like(pos_i)

    for j in neighbors:
        delta = positions[j] - pos_i
        delta = np.where(delta > 0.5 * domain_size, delta - domain_size, delta)
        delta = np.where(delta < -0.5 * domain_size, delta + domain_size, delta)
        dist = np.linalg.norm(delta)
        if dist < repel_radius and dist > 1e-5:
            repel_vector += (delta / dist) * (repel_radius - dist) 
            attract_vector += (delta / dist) * (dist - repel_radius)

    total_vector = 5.0 * repel_vector + 1.0 * attract_vector
    pair_sum = np.zeros_like(pos_i)
    pair_count = 0

    for j, k in combinations(neighbors, 2):
        rel_j = positions[j] - pos_i
        rel_k = positions[k] - pos_i
        rel_j = np.where(rel_j > 0.5 * domain_size, rel_j - domain_size, rel_j)
        rel_j = np.where(rel_j < -0.5 * domain_size, rel_j + domain_size, rel_j)
        rel_k = np.where(rel_k > 0.5 * domain_size, rel_k - domain_size, rel_k)
        rel_k = np.where(rel_k < -0.5 * domain_size, rel_k + domain_size, rel_k)

        midpoint_vec = 0.5 * (rel_j + rel_k)
        avg_pair_velocity = 0.5 * (velocities[j] + velocities[k])
        pair_v = (6 * velocities[i] + 1 * avg_pair_velocity + 1 * total_vector + 1 * midpoint_vec) / 9.0
        pair_sum += pair_v
        pair_count += 1

    new_v = pair_sum / pair_count
    return _apply_min_speed(new_v, min_speed)

def find_neighbors_for_id(args):
    idx, positions, box_size, radius = args
    positions = positions % box_size
    cell_map = CellListMap(positions, box_size, radius)
    pos = positions[idx]
    cell_idx = cell_map.get_cell_index(pos)
    neighbors = []
    for offset in np.ndindex(*(3,) * cell_map.dim):
        neighbor_cell = tuple((np.array(cell_idx) + np.array(offset) - 1) % cell_map.cell_num)
        for j in cell_map.cells.get(neighbor_cell, []):
            if j == idx:
                continue
            delta = np.abs(positions[j] - pos)
            delta = np.where(delta > 0.5 * box_size, box_size - delta, delta)
            if np.sum(delta ** 2) <= radius ** 2:
                neighbors.append(j)
    return (idx, sorted(neighbors))

def get_triplet_partner_ids(idx, positions, domain_size):
    pos_i = positions[idx]
    deltas = positions - pos_i
    deltas = np.where(deltas > 0.5 * domain_size, deltas - domain_size, deltas)
    deltas = np.where(deltas < -0.5 * domain_size, deltas + domain_size, deltas)
    dists = np.linalg.norm(deltas, axis=1)
    nearest_ids = np.argsort(dists)[1:3]
    return nearest_ids.tolist()

def get_neighbors_within_radius(idx, cell_map, radius):
    positions = cell_map.positions
    domain_size = cell_map.domain_size
    pos_i = positions[idx]
    dim = positions.shape[1]

    cell_idx = tuple((pos_i // (domain_size[0] / cell_map.cell_num[0])).astype(int))

    neighbors = []
    for offset in np.ndindex(*(3,) * dim):
        neighbor_cell = tuple((np.array(cell_idx) + np.array(offset) - 1) % cell_map.cell_num)
        for j in cell_map.cells.get(neighbor_cell, []):
            if j == idx:
                continue
            delta = positions[j] - pos_i
            delta = np.where(delta > 0.5 * domain_size, delta - domain_size, delta)
            delta = np.where(delta < -0.5 * domain_size, delta + domain_size, delta)
            if np.linalg.norm(delta) <= radius:
                neighbors.append(j)
    return sorted(neighbors)

def collect_display_links(cell_map, mode, target_ids, domain_size, radius):
    links = []
    if mode in ('1', '2'):
        args_list = [(tid, cell_map.positions, domain_size, radius) for tid in target_ids]
        with multiprocessing.Pool() as pool:
            results = pool.map(find_neighbors_for_id, args_list)
        for tid, neighs in results:
            for nid in neighs:
                links.append((tid, nid))
    elif mode == '3':
        for tid in target_ids:
            pos_i = cell_map.positions[tid]
            ds = cell_map.positions - pos_i
            ds = np.where(ds > 0.5 * cell_map.domain_size, ds - cell_map.domain_size, ds)
            ds = np.where(ds < -0.5 * cell_map.domain_size, ds + cell_map.domain_size, ds)
            dists = np.linalg.norm(ds, axis=1)
            cand = np.where((dists > 1e-8) & (dists <= radius))[0]
            if cand.size >= 2:
                order = np.argsort(dists[cand])
                nbrs = cand[order[:2]]
                for pid in nbrs:
                    links.append((tid, pid))
    elif mode == '4':
        added = set()
        for tid in target_ids:
            neighs = get_neighbors_within_radius(tid, cell_map, radius)
            for j, k in combinations(neighs, 2):
                tri_edges = [(tid, j), (tid, k), (j, k)]
                for a, b in tri_edges:
                    key = tuple(sorted((a, b)))
                    if key not in added:
                        added.add(key)
                        links.append((a, b))
    else:
        pass
    return links

def draw_velocity_triangle(ax, x, y, vx, vy, radius, color='blue', alpha=0.9):
    a = 0.08 * radius
    b = 0.05 * radius
    triangle_coords = np.array([[a, 0], [-a, b], [-a, -b]])
    angle = math.atan2(vy, vx)
    rotation_matrix = np.array([[math.cos(angle), -math.sin(angle)],
                                [math.sin(angle),  math.cos(angle)]])
    rotated_triangle = triangle_coords @ rotation_matrix.T + np.array([x, y])
    triangle = Polygon(rotated_triangle, closed=True, color=color, alpha=alpha)
    ax.add_patch(triangle)

def draw_background_grid(ax, cell_map, colors=None, alpha=0.3):
    if colors is None:
        colors = ['#FFDDC1', '#C1E1FF', '#D5FFC1', '#E1C1FF']
    for i in range(cell_map.cell_num[0]):
        for j in range(cell_map.cell_num[1]):
            x = i * cell_map.cell_size[0]
            y = j * cell_map.cell_size[1]
            color_index = (i % 2) * 2 + (j % 2)
            rect = Rectangle((x, y), cell_map.cell_size[0], cell_map.cell_size[1],
                             color=colors[color_index], alpha=alpha, zorder=0)
            ax.add_patch(rect)

if __name__ == '__main__':
    np.random.seed(5)
    N = 100
    domain_size = 100.0
    radius = 10.0
    steps = 100
    positions = np.random.rand(N, 2) * domain_size
    velocities = np.random.uniform(-1, 1, size=positions.shape)
    cell_map = CellListMap(positions, domain_size, radius, velocities=velocities)

    V_MODE = '4'
    P_MODE = '1'
    TARGET_IDS = [10]

    for step in range(steps):
        velocity_update_selector(cell_map, mode=V_MODE, min_speed=6)
        position_update_selector(cell_map, mode=P_MODE, t=0.3)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, domain_size)
        ax.set_ylim(0, domain_size)
        ax.set_aspect('equal')
        ax.set_xticks(np.arange(0, domain_size + 1, 10))
        ax.set_yticks(np.arange(0, domain_size + 1, 10))
        draw_background_grid(ax, cell_map)

        for i in range(len(cell_map.positions)):
            x, y = cell_map.positions[i]
            vx, vy = cell_map.aj[i]
            draw_velocity_triangle(ax, x, y, vx, vy, cell_map.radius)

        for tid in TARGET_IDS:
            x1, y1 = cell_map.positions[tid]
            vx1, vy1 = cell_map.aj[tid]
            draw_velocity_triangle(ax, x1, y1, vx1, vy1, cell_map.radius, color='red', alpha=0.95)
            ax.plot([], [], color='red', marker='^', linestyle='None', label=f'Target {tid}')

        links = collect_display_links(cell_map, V_MODE, TARGET_IDS, domain_size, radius)

        for i, j in links:
            x1, y1 = cell_map.positions[i]
            x2, y2 = cell_map.positions[j]
            ax.plot([x1, x2], [y1, y2], color='red', linewidth=1.2, alpha=0.9)

        ax.legend(loc='upper right', fontsize=10)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(f'Step {step}: Particle Visualization')
        plt.grid(True)
        plt.savefig(f"update{step}.png", dpi=300, bbox_inches=None, pad_inches=0.1)

        plt.close()
