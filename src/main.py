import pygame
import random
import heapq
import sys
from abc import ABC, abstractmethod


# ==========================
# CLASSES DE PLAYER
# ==========================
class BasePlayer(ABC):
    def __init__(self, start_pos, max_battery=70):
        self.position = start_pos
        self.max_battery = max_battery
        self.battery = max_battery
        self.score = 0
        self.cargo = []
        self.delivered_packages = []

    @abstractmethod
    def move_to(self, new_pos):
        pass

    @abstractmethod
    def recharge(self):
        pass

    @abstractmethod
    def pick_package(self, package_pos):
        pass

    @abstractmethod
    def deliver_packages(self, goal_positions):
        pass

    @abstractmethod
    def escolher_alvo(self, world):
        pass


class DefaultPlayer(BasePlayer):
    def __init__(self, position):
        super().__init__(position)

    def move_to(self, new_pos):
        self.position = new_pos
        if self.battery > 0:
            self.battery -= 1
            self.score -= 1
        else:
            self.score -= 5

    def recharge(self):
        self.battery = 100

    def pick_package(self, package_pos):
        if self.position == package_pos:
            self.cargo.append(package_pos)
            return True
        return False

    def deliver_packages(self, goal_positions):
        """Entrega os pacotes que estão nas posições de entrega especificadas"""
        self.delivered_packages.append(self.cargo)
        self.score += 50
        return self.delivered_packages

    def escolher_alvo(self, world):
        current_pos = tuple(self.position)

        maze = world.maze

        def caminho_valido(alvo):
            return len(maze.astar(current_pos, tuple(alvo))) > 0

        def distancia_para(alvo):
            return maze.heuristic(current_pos, tuple(alvo))

        # Se tiver pacotes na carga - Prioridade Máxima
        if self.cargo:
            # 1. Tentar entregar primeiro
            entregas_validas = [g for g in world.goals if caminho_valido(g)]

            if entregas_validas:
                # Encontra a entrega mais próxima
                entrega = min(entregas_validas, key=lambda g: distancia_para(g))
                dist_entrega = distancia_para(entrega)

                # Verifica se tem energia para entrega + margem de segurança
                if (
                    self.battery >= dist_entrega + 10
                ):  # Reserva 10 unidades para emergências
                    return entrega

                # Se não tem energia suficiente, verifica se pode recarregar antes
                elif world.recharger and caminho_valido(world.recharger):
                    dist_recarga = distancia_para(world.recharger)
                    if self.battery >= dist_recarga:
                        return world.recharger

            # 2. Se não pode entregar, tenta recarregar
            if world.recharger and caminho_valido(world.recharger):
                dist_recarga = distancia_para(world.recharger)
                if self.battery < 30:  # Se a bateria estiver abaixo de 30%
                    return world.recharger

            # 3. Se não pode fazer nada, fica parado
            return list(current_pos)
        # Se não tiver pacotes - Prioridade de coleta
        if world.packages:
            pacotes_validos = [p for p in world.packages if caminho_valido(p)]

            if pacotes_validos:
                pacote = min(pacotes_validos, key=lambda p: distancia_para(p))
                dist_pacote = distancia_para(pacote)

                # Calcula energia necessária: ir até o pacote + possível entrega
                energia_necessaria = dist_pacote + 20  # Margem para entrega futura

                if self.battery >= energia_necessaria:
                    return pacote

                # Se não tem energia, verifica recarga
                elif world.recharger and caminho_valido(world.recharger):
                    dist_recarga = distancia_para(world.recharger)
                    if self.battery >= dist_recarga:
                        return world.recharger

            return list(current_pos)  # Fica parado se não puder coletar

        # Recarga emergencial (bateria <= 30%)
        if self.battery <= 30 and world.recharger and caminho_valido(world.recharger):
            dist_recarga = distancia_para(world.recharger)
            if self.battery >= dist_recarga:
                return world.recharger

        return None


# ==========================
# CLASSE WORLD (MUNDO)
# ==========================
class World:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        # Removida a referência ao Maze aqui
        self.maze = None  # Será definido posteriormente
        self.maze_size = 30
        self.width = 600
        self.height = 600
        self.block_size = self.width // self.maze_size

        # Cria uma matriz 2D para planejamento de caminhos:
        # 0 = livre, 1 = obstáculo
        self.map = [[0 for _ in range(self.maze_size)] for _ in range(self.maze_size)]
        # Geração de obstáculos com padrão de linha (assembly line)
        self.generate_obstacles()
        # Gera a lista de paredes a partir da matriz
        self.walls = []
        for row in range(self.maze_size):
            for col in range(self.maze_size):
                if self.map[row][col] == 1:
                    self.walls.append((col, row))

        # Número total de itens (pacotes) a serem entregues
        self.total_items = 5

        # Geração dos locais de coleta (pacotes)
        self.packages = []
        # Aqui geramos 5 locais para coleta
        while len(self.packages) < self.total_items:
            x = random.randint(0, self.maze_size - 1)
            y = random.randint(0, self.maze_size - 1)
            if self.map[y][x] == 0 and [x, y] not in self.packages:
                self.packages.append([x, y])

        # Geração dos locais de entrega (metas)
        self.goals = []
        while len(self.goals) < self.total_items - 1:
            x = random.randint(0, self.maze_size - 1)
            y = random.randint(0, self.maze_size - 1)
            if (
                self.map[y][x] == 0
                and [x, y] not in self.goals
                and [x, y] not in self.packages
            ):
                self.goals.append([x, y])

        # Cria o jogador usando a classe DefaultPlayer (pode ser substituído por outra implementação)
        self.player = self.generate_player()

        # Coloca o recharger (recarga de bateria) próximo ao centro (região 3x3)
        self.recharger = self.generate_recharger()

        # Inicializa a janela do Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Delivery Bot")

        # Carrega imagens para pacote, meta e recharger a partir de arquivos
        self.package_image = pygame.image.load("assets/cargo.png")
        self.package_image = pygame.transform.scale(
            self.package_image, (self.block_size, self.block_size)
        )

        self.goal_image = pygame.image.load("assets/operator.png")
        self.goal_image = pygame.transform.scale(
            self.goal_image, (self.block_size, self.block_size)
        )

        self.recharger_image = pygame.image.load("assets/charging-station.png")
        self.recharger_image = pygame.transform.scale(
            self.recharger_image, (self.block_size, self.block_size)
        )

        # Cores utilizadas para desenho (caso a imagem não seja usada)
        self.wall_color = (100, 100, 100)
        self.ground_color = (255, 255, 255)
        self.player_color = (0, 255, 0)
        self.path_color = (200, 200, 0)

    def generate_obstacles(self):
        """
        Gera obstáculos com sensação de linha de montagem:
         - Cria vários segmentos horizontais curtos com lacunas.
         - Cria vários segmentos verticais curtos com lacunas.
         - Cria um obstáculo em bloco grande (4x4 ou 6x6) simulando uma estrutura de suporte.
        """
        # Barragens horizontais curtas:
        for _ in range(7):
            row = random.randint(5, self.maze_size - 6)
            start = random.randint(0, self.maze_size - 10)
            length = random.randint(5, 10)
            for col in range(start, start + length):
                if random.random() < 0.7:
                    self.map[row][col] = 1

        # Barragens verticais curtas:
        for _ in range(7):
            col = random.randint(5, self.maze_size - 6)
            start = random.randint(0, self.maze_size - 10)
            length = random.randint(5, 10)
            for row in range(start, start + length):
                if random.random() < 0.7:
                    self.map[row][col] = 1

        # Obstáculo em bloco grande: bloco de tamanho 4x4 ou 6x6.
        block_size = random.choice([4, 6])
        max_row = self.maze_size - block_size
        max_col = self.maze_size - block_size
        top_row = random.randint(0, max_row)
        top_col = random.randint(0, max_col)
        for r in range(top_row, top_row + block_size):
            for c in range(top_col, top_col + block_size):
                self.map[r][c] = 1

    def generate_player(self):
        # Cria o jogador em uma célula livre que não seja de pacote ou meta.
        while True:
            x = random.randint(0, self.maze_size - 1)
            y = random.randint(0, self.maze_size - 1)
            if (
                self.map[y][x] == 0
                and [x, y] not in self.packages
                and [x, y] not in self.goals
            ):
                return DefaultPlayer([x, y])

    def generate_recharger(self):
        # Coloca o recharger próximo ao centro
        center = self.maze_size // 2
        while True:
            x = random.randint(center - 1, center + 1)
            y = random.randint(center - 1, center + 1)
            if (
                self.map[y][x] == 0
                and [x, y] not in self.packages
                and [x, y] not in self.goals
                and [x, y] != self.player.position
            ):
                return [x, y]

    def can_move_to(self, pos):
        x, y = pos
        if 0 <= x < self.maze_size and 0 <= y < self.maze_size:
            return self.map[y][x] == 0
        return False

    def draw_world(self, path=None):
        self.screen.fill(self.ground_color)
        # Desenha os obstáculos (paredes)
        for x, y in self.walls:
            rect = pygame.Rect(
                x * self.block_size,
                y * self.block_size,
                self.block_size,
                self.block_size,
            )
            pygame.draw.rect(self.screen, self.wall_color, rect)
        # Desenha os locais de coleta (pacotes) utilizando a imagem
        for pkg in self.packages:
            x, y = pkg
            self.screen.blit(
                self.package_image, (x * self.block_size, y * self.block_size)
            )
        # Desenha os locais de entrega (metas) utilizando a imagem
        for goal in self.goals:
            x, y = goal
            self.screen.blit(
                self.goal_image, (x * self.block_size, y * self.block_size)
            )
        # Desenha o recharger utilizando a imagem
        if self.recharger:
            x, y = self.recharger
            self.screen.blit(
                self.recharger_image, (x * self.block_size, y * self.block_size)
            )
        # Desenha o caminho, se fornecido
        if path:
            for pos in path:
                x, y = pos
                rect = pygame.Rect(
                    x * self.block_size + self.block_size // 4,
                    y * self.block_size + self.block_size // 4,
                    self.block_size // 2,
                    self.block_size // 2,
                )
                pygame.draw.rect(self.screen, self.path_color, rect)
        # Desenha o jogador (retângulo colorido)
        x, y = self.player.position
        rect = pygame.Rect(
            x * self.block_size, y * self.block_size, self.block_size, self.block_size
        )
        pygame.draw.rect(self.screen, self.player_color, rect)
        pygame.display.flip()


# ==========================
# CLASSE MAZE: Lógica do jogo e planejamento de caminhos (A*)
# ==========================
class Maze:
    def __init__(self, seed=None):
        self.world = World(seed)
        # Agora conectamos o Maze ao World depois de criado
        self.world.maze = self  # Esta é a linha crucial que faltava
        self.running = True
        self.score = 0
        self.steps = 0
        self.delay = 100
        self.path = []
        self.num_deliveries = 0  # contagem de entregas realizadas

    def heuristic(self, a, b):
        # Distância de Manhattan
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def is_valid(self, pos):
        x, y = pos
        return (
            0 <= x < self.world.maze_size
            and 0 <= y < self.world.maze_size
            and self.world.map[y][x] == 0
        )

    def astar(self, start, goal):
        start = tuple(start)
        goal = tuple(goal)

        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        open_set = []
        heapq.heappush(open_set, (self.heuristic(start, goal), start))

        came_from = {}
        gscore = {start: 0}
        fscore = {start: self.heuristic(start, goal)}
        closed_set = set()

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                path = []
                while current in came_from:
                    path.append(list(current))
                    current = came_from[current]
                path.reverse()
                return path

            closed_set.add(current)

            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)

                if not self.is_valid(neighbor):
                    continue

                tentative_g = gscore[current] + 1

                if neighbor in closed_set and tentative_g >= gscore.get(
                    neighbor, float("inf")
                ):
                    continue

                if tentative_g < gscore.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g
                    fscore[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (fscore[neighbor], neighbor))

        return []  # Nenhum caminho encontrado

    def game_loop(self):
        while self.running:
            if self.num_deliveries >= self.world.total_items:
                self.running = False
                break

            target = self.world.player.escolher_alvo(self.world)
            if target is None:
                self.running = False
                break

            self.path = self.astar(self.world.player.position, target)
            if not self.path:
                print("[[Nenhum caminho encontrado para o alvo]]", target)
                self.running = False
                break

            for pos in self.path:
                self.world.player.move_to(pos)
                self.steps += 1

                if self.world.recharger and pos == self.world.recharger:
                    self.world.player.recharge()
                    print("[[Bateria recarregada!]]")

                self.world.draw_world(self.path)
                pygame.time.wait(self.delay)

            if self.world.player.position == target:
                if target in self.world.packages:
                    if self.world.player.pick_package(target):
                        self.world.packages.remove(target)
                        print(
                            f"\n[Pacote coletado em {target}. Cargo: {len(self.world.player.cargo)}]"
                        )

                elif target in self.world.goals:
                    entregues = self.world.player.deliver_packages([target])

                    if entregues:
                        self.num_deliveries += 1
                        print(
                            f"\n[{len(self.world.player.cargo)} pacote(s) entregue(s). Total: {self.num_deliveries}/{self.world.total_items}]"
                        )
                        self.world.player.cargo = []

                    if (
                        self.num_deliveries >= self.world.total_items
                        and not self.world.player.cargo
                    ):
                        self.running = False

            print(
                f"\n[Passos: {self.steps}, Pontuação: {self.world.player.score}, Cargo: {len(self.world.player.cargo)}, Bateria: {self.world.player.battery}, Entregas: {self.num_deliveries}]"
            )


# ==========================
# PONTO DE ENTRADA PRINCIPAL
# ==========================
def main():
    pygame.init()

    maze = Maze(seed=42)  # ou 5 para outro layout
    maze.game_loop()  # Inicia o loop do jogo

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
