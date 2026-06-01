from player import BasePlayer
from board import Board, Move
import math
import random


class AIPlayer(BasePlayer):
    def __init__(self, player_id: int, name: str = "AI"):
        super().__init__(player_id, name)
        self.max_depth = 3

    def choose_move(self, board: Board) -> Move:
        legal = board.legal_moves()

        if not legal:
            return Move(0)

        # 1. Win immediately if possible
        for move in legal:
            b = board.clone()
            b.apply_move(move.col, self.player_id)
            if b.check_winner() == self.player_id:
                return move

        # 2. Block opponent immediate win
        for move in legal:
            b = board.clone()
            b.apply_move(move.col, self.opp_id)
            if b.check_winner() == self.opp_id:
                return move

        # 3. Use minimax with alpha-beta pruning
        best_score = -math.inf
        best_moves = []

        ordered_moves = self.order_moves(board, legal)

        for move in ordered_moves:
            b = board.clone()
            b.apply_move(move.col, self.player_id)

            score = self.minimax(
                b,
                depth=self.max_depth - 1,
                alpha=-math.inf,
                beta=math.inf,
                maximizing=False
            )

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return random.choice(best_moves)

    def minimax(self, board: Board, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        terminal, winner = board.terminal_status()

        if terminal:
            if winner == self.player_id:
                return 1_000_000
            elif winner == self.opp_id:
                return -1_000_000
            else:
                return 0

        if depth == 0:
            return self.evaluate_board(board)

        legal = self.order_moves(board, board.legal_moves())

        if maximizing:
            value = -math.inf

            for move in legal:
                b = board.clone()
                b.apply_move(move.col, self.player_id)

                value = max(
                    value,
                    self.minimax(b, depth - 1, alpha, beta, False)
                )

                alpha = max(alpha, value)

                if alpha >= beta:
                    break

            return value

        else:
            value = math.inf

            for move in legal:
                b = board.clone()
                b.apply_move(move.col, self.opp_id)

                value = min(
                    value,
                    self.minimax(b, depth - 1, alpha, beta, True)
                )

                beta = min(beta, value)

                if alpha >= beta:
                    break

            return value

    def order_moves(self, board: Board, moves):
        center = board.W // 2
        return sorted(moves, key=lambda m: abs(m.col - center))

    def evaluate_board(self, board: Board) -> float:
        score = 0

        # Prefer center columns
        center = board.W // 2
        for r in range(board.H):
            for c in range(board.W):
                if board.grid[r][c] == self.player_id:
                    score += max(0, board.W // 2 - abs(c - center))
                elif board.grid[r][c] == self.opp_id:
                    score -= max(0, board.W // 2 - abs(c - center))

        # Score all possible groups of 5
        score += self.score_windows(board, self.player_id)
        score -= self.score_windows(board, self.opp_id) * 1.1

        return score

    def score_windows(self, board: Board, player: int) -> int:
        total = 0
        K = board.K
        H = board.H
        W = board.W
        g = board.grid

        directions = [
            (0, 1),   # horizontal
            (1, 0),   # vertical
            (1, 1),   # diagonal down-right
            (1, -1),  # diagonal down-left
        ]

        for r in range(H):
            for c in range(W):
                for dr, dc in directions:
                    cells = []

                    for i in range(K):
                        rr = r + dr * i
                        cc = c + dc * i

                        if 0 <= rr < H and 0 <= cc < W:
                            cells.append(g[rr][cc])

                    if len(cells) == K:
                        total += self.evaluate_window(cells, player)

        return total

    def evaluate_window(self, cells, player: int) -> int:
        opponent = 1 if player == 2 else 2

        player_count = cells.count(player)
        opponent_count = cells.count(opponent)
        empty_count = cells.count(0)

        if player_count > 0 and opponent_count > 0:
            return 0

        if player_count == 5:
            return 100_000
        elif player_count == 4 and empty_count == 1:
            return 10_000
        elif player_count == 3 and empty_count == 2:
            return 500
        elif player_count == 2 and empty_count == 3:
            return 50
        elif player_count == 1 and empty_count == 4:
            return 5

        return 0