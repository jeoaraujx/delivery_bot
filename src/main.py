import pygame
import random
import heapq
import sys
from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np


# ==========================
# CLASSES DE PLAYER (mantidas iguais)
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
    def __init__(self, position, max_capacity=5):
        super().__init__(position)
        self.max_capacity = max_capacity

    def move_to(self, new_pos):
        self.position = new_pos
        if self.battery > 0:
            self.battery -= 1
            self.score -= 1
        else:
            self.score -= 5

    def recharge(self):
        self.battery = self.max_battery

    def pick_package(self, package_pos):
        if self.position == package_pos and len(self.cargo) < self.max_capacity:
            self.cargo.append(package_pos)
            return True
        elif len(self.cargo) >= self.max_capacity:
            print("Capacidade máxima de pacotes atingida!")
        return False

    def deliver_packages(self, goal_positions):
        for goal_pos in goal_positions:
            if self.position == goal_pos and self.cargo:
                num_delivered = len(self.cargo)
                self.delivered_packages.extend(self.cargo)
                self.score += 50 * num_delivered
                self.cargo = []
                return goal_pos
        return None

    def escolher_alvo(self, world):
        current_pos = tuple(self.position)
        maze = world.maze

        def caminho_valido(alvo):
            return len(maze.astar(current_pos, tuple(alvo))) > 0

        def distancia_real(alvo):
            return len(maze.astar(current_pos, tuple(alvo)))

        def melhor_alvo(alvos, prioridade):
            alvos_validos = [a for a in alvos if caminho_valido(a)]
            if not alvos_validos:
                return None

            alvos_com_pontuacao = []
            for alvo in alvos_validos:
                dist = distancia_real(alvo)
                custo = dist
                if prioridade == "entrega":
                    ganho = 50 * len(self.cargo)
                else:
                    ganho = 50

                if self.battery < dist:
                    if world.recharger and caminho_valido(world.recharger):
                        dist_recarga = distancia_real(world.recharger)
                        if self.battery >= dist_recarga:
                            custo = dist_recarga + distancia_real(alvo)
                        else:
                            continue
                    else:
                        continue

                score_liquido = ganho - custo
                alvos_com_pontuacao.append((score_liquido, dist, alvo))

            if not alvos_com_pontuacao:
                return None

            alvos_com_pontuacao.sort(key=lambda x: (-x[0], x[1]))
            return alvos_com_pontuacao[0][2]

        if self.cargo and len(self.cargo) < self.max_capacity:
            for pkg in world.packages:
                if pkg not in self.cargo:
                    dist_pacote = distancia_real(pkg)
                    melhor_entrega = melhor_alvo(world.goals, "entrega")
                    if melhor_entrega:
                        dist_entrega = len(
                            maze.astar(tuple(pkg), tuple(melhor_entrega))
                        )
                        bateria_necessaria = dist_pacote + dist_entrega
                        if world.recharger:
                            dist_recarga = len(
                                maze.astar(
                                    tuple(melhor_entrega), tuple(world.recharger)
                                )
                            )
                            bateria_necessaria += dist_recarga

                        if self.battery >= bateria_necessaria:
                            return pkg

        if self.cargo:
            melhor_entrega = melhor_alvo(world.goals, "entrega")
            if melhor_entrega:
                dist_entrega = distancia_real(melhor_entrega)
                dist_recarga = distancia_real(world.recharger)

                if self.battery >= dist_entrega + (
                    dist_recarga if world.recharger else 0
                ):
                    return melhor_entrega
                elif world.recharger and caminho_valido(world.recharger):
                    return world.recharger

        if world.packages and len(self.cargo) < self.max_capacity:
            melhor_pacote = melhor_alvo(world.packages, "coleta")
            if melhor_pacote:
                dist_pacote = distancia_real(melhor_pacote)

                if world.goals and world.recharger and caminho_valido(world.recharger):
                    melhor_goal = melhor_alvo(world.goals, "entrega")
                    if melhor_goal:
                        dist_goal = distancia_real(melhor_goal)
                        dist_recarga = distancia_real(world.recharger)

                        if self.battery >= dist_pacote + dist_goal + dist_recarga:
                            return melhor_pacote

                if self.battery >= dist_pacote + 15:
                    return melhor_pacote

        if self.battery <= 30 and world.recharger and caminho_valido(world.recharger):
            dist_recarga = distancia_real(world.recharger)
            if self.battery >= dist_recarga:
                return world.recharger


# ==========================
# CLASSE WORLD (mantida igual)
# ==========================
class World:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.maze = None
        self.maze_size = 30
        self.width = 600
        self.height = 600
        self.block_size = self.width // self.maze_size

        self.map = [[0 for _ in range(self.maze_size)] for _ in range(self.maze_size)]
        self.generate_obstacles()
        self.walls = []
        for row in range(self.maze_size):
            for col in range(self.maze_size):
                if self.map[row][col] == 1:
                    self.walls.append((col, row))

        self.total_items = 5
        self.packages = []
        while len(self.packages) < self.total_items:
            x = random.randint(0, self.maze_size - 1)
            y = random.randint(0, self.maze_size - 1)
            if self.map[y][x] == 0 and [x, y] not in self.packages:
                self.packages.append([x, y])

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

        self.player = self.generate_player()
        self.recharger = self.generate_recharger()

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Delivery Bot")

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

        self.wall_color = (100, 100, 100)
        self.ground_color = (255, 255, 255)
        self.player_color = (0, 255, 0)
        self.path_color = (200, 200, 0)

    def generate_obstacles(self):
        for _ in range(7):
            row = random.randint(5, self.maze_size - 6)
            start = random.randint(0, self.maze_size - 10)
            length = random.randint(5, 10)
            for col in range(start, start + length):
                if random.random() < 0.7:
                    self.map[row][col] = 1

        for _ in range(7):
            col = random.randint(5, self.maze_size - 6)
            start = random.randint(0, self.maze_size - 10)
            length = random.randint(5, 10)
            for row in range(start, start + length):
                if random.random() < 0.7:
                    self.map[row][col] = 1

        block_size = random.choice([4, 6])
        max_row = self.maze_size - block_size
        max_col = self.maze_size - block_size
        top_row = random.randint(0, max_row)
        top_col = random.randint(0, max_col)
        for r in range(top_row, top_row + block_size):
            for c in range(top_col, top_col + block_size):
                self.map[r][c] = 1

    def generate_player(self):
        while True:
            x = random.randint(0, self.maze_size - 1)
            y = random.randint(0, self.maze_size - 1)
            if (
                self.map[y][x] == 0
                and [x, y] not in self.packages
                and [x, y] not in self.goals
            ):
                return DefaultPlayer([x, y], max_capacity=5)

    def generate_recharger(self):
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
        for x, y in self.walls:
            rect = pygame.Rect(
                x * self.block_size,
                y * self.block_size,
                self.block_size,
                self.block_size,
            )
            pygame.draw.rect(self.screen, self.wall_color, rect)
        for pkg in self.packages:
            x, y = pkg
            self.screen.blit(
                self.package_image, (x * self.block_size, y * self.block_size)
            )
        for goal in self.goals:
            x, y = goal
            self.screen.blit(
                self.goal_image, (x * self.block_size, y * self.block_size)
            )
        if self.recharger:
            x, y = self.recharger
            self.screen.blit(
                self.recharger_image, (x * self.block_size, y * self.block_size)
            )
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
        x, y = self.player.position
        rect = pygame.Rect(
            x * self.block_size, y * self.block_size, self.block_size, self.block_size
        )
        pygame.draw.rect(self.screen, self.player_color, rect)
        pygame.display.flip()


# ==========================
# CLASSE MAZE (modificada para análise)
# ==========================
class Maze:
    def __init__(self, seed=None, headless=False):
        self.seed = seed
        self.world = World(seed)
        self.world.maze = self
        self.running = True
        self.score = 0
        self.steps = 0
        self.delay = 100
        self.path = []
        self.num_deliveries = 0
        self.headless = headless

    def heuristic(self, a, b):
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

        return []

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

                if not self.headless:
                    self.world.draw_world(self.path)
                    pygame.time.wait(self.delay)

            if self.world.player.position == target:
                if target in self.world.packages:
                    if self.world.player.pick_package(target):
                        self.world.packages.remove(target)
                        print(
                            f"\n[Pacote coletado em {target}. Cargo: {len(self.world.player.cargo)}/{self.world.player.max_capacity}]"
                        )

                elif target in self.world.goals:
                    delivered_goal = self.world.player.deliver_packages([target])

                    if delivered_goal is not None:
                        self.num_deliveries += (
                            len(self.world.player.delivered_packages)
                            - self.num_deliveries
                        )
                        print(
                            f"\n[Pacote(s) entregue(s) em {delivered_goal}. Total: {self.num_deliveries}/{self.world.total_items}]"
                        )

                        if delivered_goal in self.world.goals:
                            self.world.goals.remove(delivered_goal)

                    if (
                        self.num_deliveries >= self.world.total_items
                        and not self.world.player.cargo
                    ):
                        self.running = False

            print(
                f"\n[Passos: {self.steps}, Pontuação: {self.world.player.score}, "
                f"Cargo: {len(self.world.player.cargo)}/{self.world.player.max_capacity}, "
                f"Bateria: {self.world.player.battery}, Entregas: {self.num_deliveries}]"
            )

        return {
            "seed": self.seed,
            "steps": self.steps,
            "score": self.world.player.score,
            "final_battery": self.world.player.battery,
            "deliveries": self.num_deliveries,
            "max_capacity": self.world.player.max_capacity,
            "battery_efficiency": (
                self.steps / self.world.player.max_battery
                if self.world.player.max_battery > 0
                else 0
            ),
            "delivery_efficiency": (
                self.num_deliveries / self.steps if self.steps > 0 else 0
            ),
        }


# ==========================
# CLASSE DE ANÁLISE ESTATÍSTICA
# ==========================
class SimulationAnalyzer:
    def __init__(self, num_simulations=100):
        self.num_simulations = num_simulations
        self.results = []

    def run_simulations(self):
        for seed in tqdm(range(self.num_simulations), desc="Running simulations"):
            try:
                maze = Maze(seed=seed, headless=True)
                result = maze.game_loop()
                self.results.append(result)
            except Exception as e:
                print(f"Error with seed {seed}: {str(e)}")

        return pd.DataFrame(self.results)

    def analyze_results(self, df):
        print("\n=== Estatísticas Descritivas ===")
        print(df.describe())

        print("\n=== Matriz de Correlação ===")
        corr_matrix = df.corr()
        print(corr_matrix)

        return corr_matrix

    def plot_results(self, df):
        plt.figure(figsize=(15, 10))

        # Gráfico 1
        plt.subplot(2, 2, 1)
        sns.histplot(df["score"], bins=20, kde=True)
        plt.title("Distribuição de Pontuações")
        plt.axvline(
            df["score"].max(),
            color="r",
            linestyle="--",
            label=f'Max: {df["score"].max()}',
        )
        plt.legend()

        # Gráfico 2
        plt.subplot(2, 2, 2)
        sns.scatterplot(data=df, x="steps", y="score", hue="deliveries")
        plt.title("Passos vs Pontuação")

        # Gráfico 3
        plt.subplot(2, 2, 3)
        sns.lineplot(data=df.sort_values("score"), x=range(len(df)), y="deliveries")
        plt.title("Entregas por Simulação (Ordenado por Score)")

        # Gráfico 4
        plt.subplot(2, 2, 4)
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
        plt.title("Matriz de Correlação")

        plt.tight_layout()
        plt.savefig("./assets/analise_resultados.png")  # Salva em vez de mostrar
        plt.close()

        # Gráfico adicional
        plt.figure(figsize=(10, 6))
        df_sorted = df.sort_values("score")
        df_sorted["cumulative_max"] = df_sorted["score"].cummax()
        plt.plot(df_sorted["score"], label="Score por Seed")
        plt.plot(
            df_sorted["cumulative_max"],
            label="Upper Bound de Score",
            linestyle="--",
            color="r",
        )
        plt.xlabel("Seed")
        plt.ylabel("Pontuação")
        plt.title("Evolução do Upper Bound de Pontuação")
        plt.legend()
        plt.savefig("./assets/evolucao_score.png")  # Salva em vez de mostrar
        plt.close()

        print("\nGráficos salvos como 'analise_resultados.png' e 'evolucao_score.png'")

    def find_optimal_parameters(self, df):
        top_5 = df.nlargest(5, "score")
        print("\n=== Top 5 Melhores Execuções ===")
        print(top_5)

        max_deliveries = df["deliveries"].max()
        theoretical_upper_bound = max_deliveries * 50
        actual_upper_bound = df["score"].max()

        print(f"\nUpper Bound Teórico: {theoretical_upper_bound}")
        print(f"Upper Bound Alcançado: {actual_upper_bound}")
        print(f"Eficiência: {actual_upper_bound/theoretical_upper_bound:.2%}")

        return top_5


# ==========================
# PONTO DE ENTRADA PRINCIPAL
# ==========================
def main():
    # Modo padrão: execução normal com visualização
    if len(sys.argv) == 1:
        pygame.init()
        maze = Maze(seed=5)
        maze.game_loop()
        pygame.quit()
        sys.exit()

    # Modo análise: execução múltipla sem visualização
    elif sys.argv[1] == "--analyze":
        num_simulations = 100 if len(sys.argv) < 3 else int(sys.argv[2])
        analyzer = SimulationAnalyzer(num_simulations)
        results_df = analyzer.run_simulations()
        results_df.to_csv("./assets/simulation_results.csv", index=False)
        corr_matrix = analyzer.analyze_results(results_df)
        analyzer.plot_results(results_df)
        top_performers = analyzer.find_optimal_parameters(results_df)


if __name__ == "__main__":
    main()
