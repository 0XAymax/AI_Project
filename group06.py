import time
from player import BasePlayer
from board import Board, Move

class IntelligentPlayer(BasePlayer):
    TIME_LIMIT = 0.9  # seconds (safe margin below the 1s hard limit)

    def __init__(self, player_id: int, name: str = "Intelligent"):
        super().__init__(player_id, name)
        
    def choose_move(self, board: Board) -> Move:
        """
        Chooses the best move using iterative deepening Minimax.
        
        Args:
            board (Board): The current state of the game board.
            
        Returns:
            Move: The chosen move.
        """
        start_time = time.perf_counter()
        best_overall_move = None
        
        my_id = self.player_id
        opp_id = self.opp_id
        legal_moves = self.ordered_moves(board)
        
        if not legal_moves:
            return Move(0)  # Should not happen in normal play
            
        # 1. First pass - check for immediate win
        for move in legal_moves:
            board.apply_move(move.col, my_id)
            is_terminal, winner = board.terminal_status()
            board.undo_move(move.col)
            if is_terminal and winner == my_id:
                return move
                
        # 2. Second pass - check for immediate block
        block_move = None
        for move in legal_moves:
            board.apply_move(move.col, opp_id)
            is_terminal, winner = board.terminal_status()
            board.undo_move(move.col)
            if is_terminal and winner == opp_id:
                block_move = move
                break
                
        if block_move is not None:
            return block_move
            
        best_overall_move = legal_moves[0]
        
        for depth in range(1, 11):
            if time.perf_counter() - start_time >= self.TIME_LIMIT:
                break
                
            best_score = float('-inf')
            best_move = None
            timeout_occurred = False
            
            for move in legal_moves:
                board.apply_move(move.col, my_id)
                # Find the score for the opponent's perspective, so maximizing_player is False
                score = self.minimax(board, depth - 1, float('-inf'), float('inf'), False, my_id, opp_id, start_time, self.TIME_LIMIT)
                board.undo_move(move.col)
                
                if score is None:
                    timeout_occurred = True
                    break
                    
                if score > best_score or best_move is None:
                    best_score = score
                    best_move = move
                    
            if not timeout_occurred and best_move is not None:
                best_overall_move = best_move
                
            if timeout_occurred:
                break
                
        return best_overall_move

    def ordered_moves(self, board: Board) -> list[Move]:
        """
        Returns a list of legal moves, sorted by their distance to the center column.
        Center moves are generally stronger, so exploring them first improves alpha-beta pruning.
        
        Args:
            board (Board): The current state of the game board.
            
        Returns:
            list[Move]: A sorted list of legal moves.
        """
        center = board.W // 2
        moves = board.legal_moves()
        return sorted(moves, key=lambda m: abs(m.col - center))

    def score_window(self, window: list[int], my_id: int, opp_id: int) -> float:
        """
        Calculates a heuristic score for a window of cells.
        
        Args:
            window (list[int]): A list of cell values.
            my_id (int): Player's ID.
            opp_id (int): Opponent's ID.
            
        Returns:
            float: The heuristic score for the window.
        """
        score = 0.0
        my_count = window.count(my_id)
        opp_count = window.count(opp_id)
        empty_count = window.count(0)
        
        if my_count == 5:
            score += 1000000
        elif my_count == 4 and empty_count == 1:
            score += 500
        elif my_count == 3 and empty_count == 2:
            score += 100
        elif my_count == 2 and empty_count == 3:
            score += 10
            
        if opp_count == 4 and empty_count == 1 and my_count == 0:
            score -= 800
        elif opp_count == 3 and empty_count == 2 and my_count == 0:
            score -= 150
            
        return score

    def score_board(self, board: Board, my_id: int, opp_id: int) -> float:
        """
        Calculates the full board's heuristic score.
        
        Args:
            board (Board): The game board.
            my_id (int): Player's ID.
            opp_id (int): Opponent's ID.
            
        Returns:
            float: Total heuristic score.
        """
        score = 0.0
        
        # Center column bonus
        center_col = board.W // 2
        center_count = sum(1 for r in range(board.H) if board.grid[r][center_col] == my_id)
        score += center_count * 3
        
        # Horizontal scoring
        for r in range(board.H):
            for c in range(board.W - board.K + 1):
                window = board.grid[r][c:c+board.K]
                score += self.score_window(window, my_id, opp_id)
                
        # Vertical scoring
        for c in range(board.W):
            col_array = [board.grid[r][c] for r in range(board.H)]
            for r in range(board.H - board.K + 1):
                window = col_array[r:r+board.K]
                score += self.score_window(window, my_id, opp_id)
                
        # Positive diagonal scoring (top-left to bottom-right)
        for r in range(board.H - board.K + 1):
            for c in range(board.W - board.K + 1):
                window = [board.grid[r+i][c+i] for i in range(board.K)]
                score += self.score_window(window, my_id, opp_id)
                
        # Negative diagonal scoring (bottom-left to top-right)
        for r in range(board.K - 1, board.H):
            for c in range(board.W - board.K + 1):
                window = [board.grid[r-i][c+i] for i in range(board.K)]
                score += self.score_window(window, my_id, opp_id)
                
        return score

    def evaluate(self, board: Board, my_id: int, opp_id: int) -> float:
        """
        Evaluates the current board state and returns a score.
        
        Args:
            board (Board): The board to evaluate.
            my_id (int): ID of the maximizing player.
            opp_id (int): ID of the minimizing player.
            
        Returns:
            float: The evaluation score.
        """
        return self.score_board(board, my_id, opp_id)

    def minimax(self, board: Board, depth: int, alpha: float, beta: float, maximizing_player: bool, my_id: int, opp_id: int, start_time: float, time_limit: float):
        """
        Minimax algorithm with Alpha-Beta pruning to find the best move.
        
        Args:
            board (Board): The current board state.
            depth (int): The current depth in the search tree.
            alpha (float): Alpha value for pruning.
            beta (float): Beta value for pruning.
            maximizing_player (bool): True if maximizing player's turn, False otherwise.
            my_id (int): ID of the maximizing player.
            opp_id (int): ID of the minimizing player.
            start_time (float): Time the search started.
            time_limit (float): Maximum allowed search time.
            
        Returns:
            float: The heuristic value of the state, or None if timed out.
        """
        if time.perf_counter() - start_time >= time_limit:
            return None
            
        is_terminal, winner = board.terminal_status()
        
        if is_terminal:
            if winner == my_id:
                return 100000 + depth
            elif winner == opp_id:
                return -100000 - depth
            else:
                return 0
                
        if depth == 0:
            return self.evaluate(board, my_id, opp_id)
            
        if maximizing_player:
            max_eval = float('-inf')
            for move in self.ordered_moves(board):
                board.apply_move(move.col, my_id)
                eval_score = self.minimax(board, depth - 1, alpha, beta, False, my_id, opp_id, start_time, time_limit)
                board.undo_move(move.col)
                
                if eval_score is None:
                    return None
                    
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in self.ordered_moves(board):
                board.apply_move(move.col, opp_id)
                eval_score = self.minimax(board, depth - 1, alpha, beta, True, my_id, opp_id, start_time, time_limit)
                board.undo_move(move.col)
                
                if eval_score is None:
                    return None
                    
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

