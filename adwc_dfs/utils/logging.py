"""
Logging utilities for ADWC-DFS
Captures terminal output and saves to log files
"""

import sys
import os
from datetime import datetime
import logging


class TeeOutput:
    """
    Captures stdout/stderr and writes to both terminal and file
    """
    def __init__(self, log_file, mode='a'):
        """
        Initialize TeeOutput
        
        Args:
            log_file: Path to log file
            mode: File open mode ('a' for append, 'w' for write)
        """
        self.terminal = sys.stdout
        self.log = open(log_file, mode, encoding='utf-8')
    
    def write(self, message):
        """Write message to both terminal and file"""
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate write
    
    def flush(self):
        """Flush both streams"""
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        """Close log file"""
        self.log.close()


def setup_logger(log_dir='logs', prefix='training', capture_stdout=True):
    """
    Setup logging for training
    
    Args:
        log_dir: Directory to store log files
        prefix: Prefix for log filename
        capture_stdout: If True, capture stdout to log file
        
    Returns:
        log_file_path, tee_stdout (or None if not capturing)
    """
    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"{prefix}_{timestamp}.log"
    log_file_path = os.path.join(log_dir, log_filename)
    
    # Setup Python logger
    logger = logging.getLogger('adwc_dfs')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Capture stdout if requested
    tee_stdout = None
    if capture_stdout:
        tee_stdout = TeeOutput(log_file_path, mode='a')
        sys.stdout = tee_stdout
        sys.stderr = tee_stdout
    
    # Write header to log file
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"ADWC-DFS Training Log\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
    
    return log_file_path, tee_stdout


def close_logger(tee_stdout=None):
    """
    Close logger and restore stdout
    
    Args:
        tee_stdout: TeeOutput object to close
    """
    if tee_stdout is not None:
        # Restore original stdout
        sys.stdout = tee_stdout.terminal
        sys.stderr = tee_stdout.terminal
        tee_stdout.close()
    
    # Close all handlers
    logger = logging.getLogger('adwc_dfs')
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


def get_latest_log(log_dir='logs', prefix='training'):
    """
    Get path to the latest log file
    
    Args:
        log_dir: Directory containing log files
        prefix: Prefix of log files
        
    Returns:
        Path to latest log file or None
    """
    if not os.path.exists(log_dir):
        return None
    
    log_files = [
        f for f in os.listdir(log_dir) 
        if f.startswith(prefix) and f.endswith('.log')
    ]
    
    if not log_files:
        return None
    
    # Sort by modification time
    log_files.sort(key=lambda f: os.path.getmtime(os.path.join(log_dir, f)), reverse=True)
    
    return os.path.join(log_dir, log_files[0])
