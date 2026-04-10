"""
objects_handler.py - Interpretacao semantica da grade decodificada.

Portagem fiel de AppArq-master/src/objects_handler.cpp para Python.
Detecta paredes, janelas, portas, elementos hospedados e mobiliario
a partir da grade 37x37 de simbolos e formata o resultado como JSON
para o Revit.
"""

import json
from dataclasses import dataclass, field


EDGE = "A"
WALL = "B"
VOID = "X"
EMPTY = "-"


@dataclass
class Wall:
    beg: tuple  # (linha, coluna)
    end: tuple  # (linha, coluna)
    id: int

    def __lt__(self, other):
        return (self.beg, self.end) < (other.beg, other.end)

    def __eq__(self, other):
        return (self.beg, self.end) == (other.beg, other.end)

    def __hash__(self):
        return hash((self.beg, self.end))


class ObjectsHandler:
    """Detecta elementos arquitetonicos a partir de uma grade de simbolos."""

    def __init__(self, objects_path):
        with open(objects_path, "r", encoding="utf-8") as f:
            objects_json = json.load(f)

        self.wall_types = {}
        for val in objects_json.get("paredes", {}).values():
            color = val["cor"][0]
            self.wall_types[color] = val

        self.building_types = {}
        for val in objects_json.get("janelas", {}).values():
            key = (val["cor"][0], val["cor"][1])
            self.building_types[key] = val
        for val in objects_json.get("portas", {}).values():
            key = (val["cor"][0], val["cor"][1])
            self.building_types[key] = val

        self.hosted_types = {}
        for val in objects_json.get("hospedados", {}).values():
            color = val["cor"][0]
            self.hosted_types[color] = val

        self.furniture_types = {}
        for val in objects_json.get("mobiliario", {}).values():
            key = (val["cor"][0], val["cor"][1])
            self.furniture_types[key] = val

        self.walls = []
        self.building_elements = []
        self.hosted_elements = []
        self.furniture_elements = []

    def _get_symbol(self, grid, i, j):
        """Retorna o simbolo em (i,j), tratando '?' como vazio."""
        s = grid[i][j]
        return EMPTY if s == "?" else s

    # ------------------------------------------------------------------
    # Deteccao de paredes
    # ------------------------------------------------------------------

    def _find_walls(self, grid):
        N = len(grid)
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        wall_id = 0
        seen = set()

        for i in range(N):
            for j in range(N):
                if self._get_symbol(grid, i, j) != EDGE:
                    continue

                for di, dj in moves:
                    ci, cj = i + di, j + dj
                    if ci < 0 or cj < 0 or ci >= N or cj >= N:
                        continue

                    sym = self._get_symbol(grid, ci, cj)

                    if sym == WALL:
                        # Traca o segmento de parede.
                        beg = (i, j)
                        false_wall = False
                        while self._get_symbol(grid, ci, cj) not in (EDGE, VOID):
                            ci += di
                            cj += dj
                            if ci < 0 or cj < 0 or ci >= N or cj >= N:
                                false_wall = True
                                break
                        if false_wall:
                            continue

                        if self._get_symbol(grid, ci, cj) == EDGE:
                            end = (ci, cj)
                        else:
                            end = (ci - di, cj - dj)

                        # Normaliza a direcao.
                        if beg[0] == end[0]:  # horizontal
                            w = Wall(beg, end, wall_id) if beg[1] < end[1] else Wall(end, beg, wall_id)
                        else:  # vertical
                            w = Wall(beg, end, wall_id) if beg[0] < end[0] else Wall(end, beg, wall_id)

                        key = (w.beg, w.end)
                        if key not in seen:
                            seen.add(key)
                            self.walls.append(w)
                            wall_id += 1

                    else:
                        # Verifica se ha abertura de porta/janela na parede.
                        ni, nj = ci + di, cj + dj
                        if ni < 0 or nj < 0 or ni >= N or nj >= N:
                            continue

                        color1 = self._get_symbol(grid, ci, cj)
                        color2 = self._get_symbol(grid, ni, nj)
                        idx = (color1, color2)
                        idx2 = (color2, color1)

                        if idx in self.building_types or idx2 in self.building_types:
                            beg = (i, j)
                            false_wall = False
                            while self._get_symbol(grid, ci, cj) not in (EDGE, VOID):
                                ci += di
                                cj += dj
                                if ci < 0 or cj < 0 or ci >= N or cj >= N:
                                    false_wall = True
                                    break
                            if false_wall:
                                continue

                            if self._get_symbol(grid, ci, cj) == EDGE:
                                end = (ci, cj)
                            else:
                                end = (ci - di, cj - dj)

                            if beg[0] == end[0]:
                                w = Wall(beg, end, wall_id) if beg[1] < end[1] else Wall(end, beg, wall_id)
                            else:
                                w = Wall(beg, end, wall_id) if beg[0] < end[0] else Wall(end, beg, wall_id)

                            key = (w.beg, w.end)
                            if key not in seen:
                                seen.add(key)
                                self.walls.append(w)
                                wall_id += 1

    # ------------------------------------------------------------------
    # Deteccao de janelas e portas ao longo das paredes
    # ------------------------------------------------------------------

    def _find_building_elements(self, grid):
        for wall in self.walls:
            beg, end = wall.beg, wall.end

            if beg[0] == end[0]:  # parede horizontal
                row = beg[0]
                j = beg[1] + 1
                while j < end[1] - 1:
                    arr = (self._get_symbol(grid, row, j),
                           self._get_symbol(grid, row, j + 1))

                    found = False
                    for k in range(2):
                        test = arr if k == 0 else (arr[1], arr[0])
                        if test in self.building_types:
                            val = dict(self.building_types[test])
                            if k == 0:
                                val["position"] = [row, j]
                            else:
                                val["position"] = [row, j + 1]
                            val["hostWall"] = wall.id
                            val["orientation"] = 90 if k == 0 else 270
                            self.building_elements.append(val)
                            j += 1
                            found = True
                            break
                    j += 1

            else:  # parede vertical
                col = beg[1]
                i = beg[0] + 1
                while i < end[0] - 1:
                    arr = (self._get_symbol(grid, i, col),
                           self._get_symbol(grid, i + 1, col))

                    found = False
                    for k in range(2):
                        test = arr if k == 0 else (arr[1], arr[0])
                        if test in self.building_types:
                            val = dict(self.building_types[test])
                            if k == 0:
                                val["position"] = [i, col]
                            else:
                                # Mantido comportamento original do C++ (usa col+1 mesmo para vertical).
                                val["position"] = [i, col + 1]
                            val["hostWall"] = wall.id
                            val["orientation"] = 0 if k == 0 else 180
                            i += 1
                            found = True
                            break
                    i += 1

    # ------------------------------------------------------------------
    # Deteccao de elementos hospedados (pecas sanitarias adjacentes a paredes)
    # ------------------------------------------------------------------

    def _find_hosted_elements(self, grid):
        N = len(grid)
        is_wall = [[False] * N for _ in range(N)]

        for wall in self.walls:
            if wall.beg[0] == wall.end[0]:  # horizontal
                row = wall.beg[0]
                for j in range(wall.beg[1], wall.end[1] + 1):
                    is_wall[row][j] = True
            else:  # vertical
                col = wall.beg[1]
                for i in range(wall.beg[0], wall.end[0] + 1):
                    is_wall[i][col] = True

        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        visited = set()

        for wall in self.walls:
            if wall.beg[0] == wall.end[0]:  # horizontal
                row = wall.beg[0]
                for j in range(wall.beg[1], wall.end[1]):
                    # Verifica acima.
                    if row - 1 >= 0:
                        sym = self._get_symbol(grid, row - 1, j)
                        if sym in self.hosted_types and not is_wall[row - 1][j] and (row - 1, j) not in visited:
                            not_host = False
                            for di, dj in moves:
                                ni, nj = row - 1 + di, j + dj
                                if 0 <= ni < N and 0 <= nj < N:
                                    if not is_wall[ni][nj] and self._get_symbol(grid, ni, nj) != EMPTY:
                                        not_host = True
                            if not not_host:
                                visited.add((row - 1, j))
                                elem = dict(self.hosted_types[sym])
                                elem["position"] = [row - 1, j]
                                elem["hostWall"] = wall.id
                                self.hosted_elements.append(elem)

                    # Verifica abaixo.
                    if row + 1 < N:
                        sym = self._get_symbol(grid, row + 1, j)
                        if sym in self.hosted_types and not is_wall[row + 1][j] and (row + 1, j) not in visited:
                            not_host = False
                            for di, dj in moves:
                                ni, nj = row + 1 + di, j + dj
                                if 0 <= ni < N and 0 <= nj < N:
                                    if not is_wall[ni][nj] and self._get_symbol(grid, ni, nj) != EMPTY:
                                        not_host = True
                            if not not_host:
                                visited.add((row + 1, j))
                                elem = dict(self.hosted_types[sym])
                                elem["position"] = [row + 1, j]
                                elem["hostWall"] = wall.id
                                self.hosted_elements.append(elem)

            else:  # vertical
                col = wall.beg[1]
                for i in range(wall.beg[0], wall.end[0]):
                    # Verifica a direita.
                    if col + 1 < N:
                        sym = self._get_symbol(grid, i, col + 1)
                        if sym in self.hosted_types and not is_wall[i][col + 1] and (i, col + 1) not in visited:
                            not_host = False
                            for di, dj in moves:
                                ni, nj = i + di, col + 1 + dj
                                if 0 <= ni < N and 0 <= nj < N:
                                    if not is_wall[ni][nj] and self._get_symbol(grid, ni, nj) != EMPTY:
                                        not_host = True
                            if not not_host:
                                visited.add((i, col + 1))
                                elem = dict(self.hosted_types[sym])
                                elem["position"] = [i, col + 1]
                                elem["hostWall"] = wall.id
                                self.hosted_elements.append(elem)

                    # Verifica a esquerda.
                    if col - 1 >= 0:
                        sym = self._get_symbol(grid, i, col - 1)
                        if sym in self.hosted_types and not is_wall[i][col - 1] and (i, col - 1) not in visited:
                            not_host = False
                            for di, dj in moves:
                                ni, nj = i + di, col - 1 + dj
                                if 0 <= ni < N and 0 <= nj < N:
                                    if not is_wall[ni][nj] and self._get_symbol(grid, ni, nj) != EMPTY:
                                        not_host = True
                            if not not_host:
                                visited.add((i, col - 1))
                                elem = dict(self.hosted_types[sym])
                                elem["position"] = [i, col - 1]
                                elem["hostWall"] = wall.id
                                self.hosted_elements.append(elem)

    # ------------------------------------------------------------------
    # Deteccao de mobiliario (pares de simbolos fora das paredes)
    # ------------------------------------------------------------------

    def _find_furniture_elements(self, grid):
        N = len(grid)
        is_wall = [[False] * N for _ in range(N)]

        for wall in self.walls:
            if wall.beg[0] == wall.end[0]:
                row = wall.beg[0]
                for j in range(wall.beg[1], wall.end[1] + 1):
                    is_wall[row][j] = True
            else:
                col = wall.beg[1]
                for i in range(wall.beg[0], wall.end[0] + 1):
                    is_wall[i][col] = True

        for i in range(N):
            for j in range(N):
                if is_wall[i][j]:
                    continue

                # Verifica vizinho a direita.
                if j + 1 < N and not is_wall[i][j + 1]:
                    arr = (self._get_symbol(grid, i, j),
                           self._get_symbol(grid, i, j + 1))
                    for k in range(2):
                        test = arr if k == 0 else (arr[1], arr[0])
                        if test in self.furniture_types:
                            val = dict(self.furniture_types[test])
                            val["position"] = [[i, j], [i, j + 1]]
                            val["orientation"] = 90 if k == 0 else 270
                            self.furniture_elements.append(val)
                            is_wall[i][j] = is_wall[i][j + 1] = True
                            break

                if is_wall[i][j]:
                    continue

                # Verifica vizinho abaixo.
                if i + 1 < N and not is_wall[i + 1][j]:
                    arr = (self._get_symbol(grid, i, j),
                           self._get_symbol(grid, i + 1, j))
                    for k in range(2):
                        test = arr if k == 0 else (arr[1], arr[0])
                        if test in self.furniture_types:
                            val = dict(self.furniture_types[test])
                            val["position"] = [[i, j], [i + 1, j]]
                            val["orientation"] = 0 if k == 0 else 180
                            self.furniture_elements.append(val)
                            is_wall[i][j] = is_wall[i + 1][j] = True
                            break

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def find_objects_in_grid(self, grid):
        """Executa todas as etapas de deteccao na grade de simbolos.

        Args:
            grid: Lista 37x37 de strings de simbolos.
        """
        self.walls.clear()
        self.building_elements.clear()
        self.hosted_elements.clear()
        self.furniture_elements.clear()

        self._find_walls(grid)
        self._find_hosted_elements(grid)
        self._find_building_elements(grid)
        self._find_furniture_elements(grid)

    def format_objects_json(self):
        """Formata os elementos detectados como dict no esquema JSON do Revit."""
        root = {}

        # Paredes
        wall_list = []
        for w in sorted(self.walls):
            wall_list.append({
                "Coordinate": [
                    {"x": w.beg[0], "y": w.beg[1]},
                    {"x": w.end[0], "y": w.end[1]},
                ]
            })
        root["WallProperties"] = wall_list

        # Janelas e portas
        window_list = []
        door_list = []
        for elem in self.building_elements:
            entry = {
                "Coordinate": {"x": elem["position"][0], "y": elem["position"][1]},
                "Type": elem["nome"],
                "Rotation": elem["orientation"],
            }
            if elem["tipo"] == "Janela":
                window_list.append(entry)
            else:
                door_list.append(entry)
        root["WindowProperties"] = window_list
        root["DoorProperties"] = door_list

        # Elementos hospedados
        hosted_list = []
        for elem in self.hosted_elements:
            hosted_list.append({
                "Coordinate": {"x": elem["position"][0], "y": elem["position"][1]},
                "Type": elem["nome"],
            })
        root["HostedProperties"] = hosted_list

        # Mobiliario
        furniture_list = []
        for elem in self.furniture_elements:
            pos = elem["position"]
            furniture_list.append({
                "Coordinate": [
                    {"x": pos[0][0], "y": pos[0][1]},
                    {"x": pos[1][0], "y": pos[1][1]},
                ],
                "Type": elem["nome"],
                "Rotation": elem["orientation"],
            })
        root["FurnitureProperties"] = furniture_list

        return root

    def summary(self):
        """Retorna um resumo legivel dos elementos detectados."""
        return (
            f"Paredes: {len(self.walls)}, "
            f"Janelas/Portas: {len(self.building_elements)}, "
            f"Hospedados: {len(self.hosted_elements)}, "
            f"Mobiliario: {len(self.furniture_elements)}"
        )
