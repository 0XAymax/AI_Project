import random
import time
from player import BasePlayer
from board import Board, Move

class IntelligentPlayer(BasePlayer):
    """
    Intelligent Connect-5 agent using Minimax with Alpha-Beta pruning.
    Currently acts as a random player placeholder.
    """
    
    TIME_LIMIT = 0.9  # seconds (safe margin below the 1s hard limit)

    def __init__(self, player_id: int, name: str = "Intelligent"):
        """
        Initializes the IntelligentPlayer.
        
        Args:
            player_id (int): The ID of the player (1 or 2).
            name (str): The name of the player.
        """
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

    def evaluate(self, board: Board, my_id: int, opp_id: int) -> float:
        """
        Evaluates the current board state and returns a score.
        
        Args:
            board (Board): The board to evaluate.
            my_id (int): ID of the maximizing player.
            opp_id (int): ID of the minimizing player.
            
        Returns:
            float: The evaluation score (0 for now).
        """
        return 0.0

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
                return 100000
            elif winner == opp_id:
                return -100000
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

